"""latent.eval: prompts, scoring, accounting, interventions (scrambled, zero_latent), the step sweep and summarize."""
import inspect
import json

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent.accounting import inference_record, model_dims
from latent.eval import STEP_SWEEP, ExampleResult, _record, build_prompts, evaluate, scramble_donors, summarize
from latent.model import GenOutputs, LatentLM
from latent.tests.fake_tokenizer import FakeTokenizer

WORDS = "alex bob wumpus yumpus zumpus rompus tumpus lempus is a every not".split()


def make_examples(n=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        q = " ".join(rng.choice(WORDS, size=4 + i % 5)) + " ?"
        steps = [" ".join(rng.choice(WORDS, size=3)) + " ." for _ in range(1 + i % 3)]
        out.append({"question": q, "steps": steps, "answer": " ".join(rng.choice(WORDS, size=2)) + " ."})
    return out


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp("ev") / "val.json"
    path.write_text(json.dumps(make_examples()))
    t = FakeTokenizer()
    ids = D.add_latent_tokens(t)
    examples = D.load_examples(str(path))
    D.tokenize_examples(examples, t)  # grow the fake vocabulary before sizing the model
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(t) + 4, n_positions=256)).eval()
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], t.eos_token_id)
    return t, ids, examples, lm


def run(setup, **kw):
    t, ids, examples, lm = setup
    args = dict(arm="coconut", n_latent=2, batch_size=3, device="cpu", max_new_tokens=6)
    args.update(kw)
    return evaluate(lm, t, examples, **args)


def test_costs_and_accounting(setup):
    t, ids, examples, lm = setup
    res = run(setup)
    dims = model_dims(lm)
    v2 = "arm" in inspect.signature(inference_record).parameters
    assert [r.idx for r in res] == [e.idx for e in examples]
    for r, ex in zip(res, examples):
        q = len(t.encode(ex.question + "\n"))
        assert r.latent_steps == 2
        # registration 5.3: p = |q| + 1 (v2 accounting); v1 accounting puts <end> in the prefill (same totals)
        assert r.prefill_tokens == (q + 1 if v2 else q + 2)
        assert r.sequential_passes == 1 + r.latent_steps + r.emitted_tokens
        assert r.flops == _record(dims, "coconut", q, q + 4, 2, r.emitted_tokens).flops
        assert 0 <= r.emitted_tokens <= 6


@pytest.mark.parametrize("n", STEP_SWEEP)
def test_step_sweep(setup, n):
    t, ids, examples, lm = setup
    res = run(setup, n_latent=n)
    assert all(r.latent_steps == n for r in res)
    recs = build_prompts(D.tokenize_examples(examples, t), t, arm="coconut", n_latent=n)
    assert all(rec["input_ids"].count(ids["latent_id"]) == n for rec in recs)


def test_zero_latent_and_pause_and_cot_prompts(setup):
    t, ids, examples, lm = setup
    zero = run(setup, intervention="zero_latent", n_latent=4)
    assert all(r.latent_steps == 0 for r in zero)
    assert zero[0].text.count("<|latent|>") == 0 and "<|start-latent|>" in zero[0].text
    pause = run(setup, arm="pause", n_latent=3)
    assert all(r.latent_steps == 0 for r in pause)
    q0 = len(t.encode(examples[0].question + "\n"))
    assert pause[0].prefill_tokens == q0 + 2 + 3  # pause tokens are part of the single prefill pass (5.3)
    assert pause[0].sequential_passes == 1 + pause[0].emitted_tokens
    cot = run(setup, arm="cot", n_latent=3)
    assert "<|start-latent|>" not in cot[0].text and cot[0].prefill_tokens == q0


def test_stage_spec_matches_reference_generation_prompts(setup):
    t, ids, examples, lm = setup
    tok = D.tokenize_examples(examples, t)
    for pad in (True, False):
        spec = dict(stage=2, c_thought=2, max_latent_stage=6, pad_latent_to_max=pad)
        recs = build_prompts(tok, t, arm="coconut", stage_spec=spec)
        for rec, s in zip(recs, tok):
            cap = 6 if pad else min(6, len(s["steps_tokenized"]))
            assert rec["input_ids"].count(ids["latent_id"]) == min(cap, 2) * 2


def test_answer_scoring_follows_the_reference(setup):
    """A stub generate that emits '### <answer>' then junk after another '#' and a comma check."""
    t, ids, examples, lm = setup

    class Stub(LatentLM):
        def generate(self, input_ids, attention_mask, position_ids, *, max_new_tokens, mode="coconut",
                     scramble_bank=None, return_latents=False):
            toks = []
            for _ in range(input_ids.shape[0]):
                k = self.seen = getattr(self, "seen", -1) + 1  # records arrive in example order
                ans = t.encode("### " + examples[k % len(examples)].answer)
                toks.append(ans if k % 2 == 0 else ans[:-1])  # odd examples drop the final "." and are wrong
            n = [len(x) for x in toks]
            return GenOutputs(toks, n, [0] * len(n), [1 + k for k in n], None)

    stub = Stub(lm.base_causallm, lm.latent_id, lm.start_id, lm.end_id, lm.eos_id)
    for bs in (1, 3):
        stub.seen = -1
        res = evaluate(stub, t, examples, arm="coconut", n_latent=1, batch_size=bs, device="cpu", max_new_tokens=8)
        assert [r.correct for r in res] == [i % 2 == 0 for i in range(len(res))]
    assert res[0].pred == D.answer_extract(res[0].text) == D.answer_normalize(res[0].gold)


def test_scrambled_uses_donor_latents(setup):
    t, ids, examples, lm = setup
    clean = run(setup, n_latent=3, return_latents=True)
    scr = run(setup, n_latent=3, intervention="scrambled", return_latents=True, scramble_seed=1)
    donors = scramble_donors({r.idx: 3 for r in clean}, seed=1)
    by = {r.idx: r for r in clean}
    for r in scr:
        assert donors[r.idx] != r.idx
        assert torch.equal(r.latents, by[donors[r.idx]].latents)
        assert r.latent_steps == 3
    with pytest.raises(ValueError):
        run(setup, arm="pause", intervention="scrambled")


def test_scramble_donors_is_a_derangement_within_classes():
    counts = {i: (2 if i < 5 else 4) for i in range(9)}
    counts[9] = 0
    d = scramble_donors(counts, seed=3)
    assert d[9] == 9
    for i in range(9):
        assert d[i] != i and counts[d[i]] == counts[i]
    assert sorted(d[i] for i in range(5)) == list(range(5))
    assert d == scramble_donors(counts, seed=3)
    with pytest.raises(ValueError):
        scramble_donors({0: 1, 1: 2, 2: 2})


def _res(correct, emitted):
    return [ExampleResult(idx=i, correct=c, pred="", gold="", text="", prefill_tokens=10, latent_steps=2,
                          emitted_tokens=e, sequential_passes=3 + e, flops=1e6 * (1 + e))
            for i, (c, e) in enumerate(zip(correct, emitted))]


def test_summarize_bootstrap():
    rng = np.random.default_rng(0)
    correct = (rng.random(200) < 0.7).tolist()
    emitted = rng.integers(5, 20, 200).tolist()
    s = summarize(_res(correct, emitted))
    assert s["n"] == 200 and s["accuracy"] == pytest.approx(np.mean(correct))
    lo, hi = s["accuracy_ci95"]
    assert lo < s["accuracy"] < hi and 0.05 < hi - lo < 0.2
    assert s["emitted_tokens_median"] == float(np.median(emitted))
    assert s["sequential_passes_mean"] == pytest.approx(3 + np.mean(emitted))
    assert s == summarize(_res(correct, emitted))  # seeded
    assert summarize(_res([True] * 5, [1] * 5))["accuracy_ci95"] == (1.0, 1.0)
    assert summarize([]) == {"n": 0}


def test_max_new_tokens_is_required(setup):
    t, ids, examples, lm = setup
    with pytest.raises(TypeError):
        evaluate(lm, t, examples, arm="coconut", n_latent=1, batch_size=2, device="cpu")
