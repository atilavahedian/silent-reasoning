"""LatentLM.generate: batched greedy decoding over right-padded (and latent-aligned left-padded) batches equals
per-example decoding token for token; per-example eos; cost counts; equality with the reference Coconut.generate
(batch size 1) and with HF greedy generate for the plain causal LM."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent.model import LatentLM
from latent.tests.fake_tokenizer import FakeTokenizer
from latent.tests.reference_coconut import Coconut

NEVER = -1  # an eos id that never occurs


def tiny(seed=0, attn="eager"):
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=256, attn_implementation=attn)
    return GPT2LMHeadModel(cfg).eval()


@pytest.fixture(scope="module")
def setup():
    t = FakeTokenizer()
    ids = D.add_latent_tokens(t)
    return t, ids


def prompts(ids, counts, seed=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i, n in enumerate(counts):
        q = torch.randint(5, 60, (4 + 3 * (i % 4),), generator=g).tolist()
        x = q + [ids["start_id"]] + [ids["latent_id"]] * n + [ids["end_id"]]
        out.append({"input_ids": x, "attention_mask": [1] * len(x), "position_ids": list(range(len(x))), "idx": i})
    return out


def gen(lm, ids, recs, mode="coconut", bank=None, mnt=12, return_latents=False):
    b = D.Collator(1, ids["latent_id"])(recs)
    return lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=mnt, mode=mode,
                       scramble_bank=bank, return_latents=return_latents)


def assert_batched_equals_single(lm, ids, recs, **kw):
    batched = gen(lm, ids, recs, **kw)
    for i, r in enumerate(recs):
        bank = kw.get("bank")
        single = gen(lm, ids, [r], **{**kw, "bank": None if bank is None else bank[i:i + 1]})
        assert batched.tokens[i] == single.tokens[0], i
        assert batched.emitted_tokens[i] == single.emitted_tokens[0]
        assert batched.latent_steps[i] == single.latent_steps[0]
        assert batched.sequential_passes[i] == single.sequential_passes[0]
    return batched


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("mode", ["coconut", "pause"])
def test_batched_equals_per_example(setup, mode, attn):
    t, ids = setup
    lm = LatentLM(tiny(attn=attn), ids["latent_id"], ids["start_id"], ids["end_id"], NEVER)
    counts = (0, 1, 2, 3, 2, 0)
    out = assert_batched_equals_single(lm, ids, prompts(ids, counts), mode=mode)
    assert out.emitted_tokens == [12] * len(counts)
    expect_steps = list(counts) if mode == "coconut" else [0] * len(counts)
    assert out.latent_steps == expect_steps
    assert out.sequential_passes == [1 + s + 12 for s in expect_steps]


def test_per_example_eos(setup):
    t, ids = setup
    model = tiny(seed=1)
    recs = prompts(ids, (1, 2, 3, 0, 2))
    free = gen(LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], NEVER), ids, recs, mnt=16)
    eos = free.tokens[0][4]  # a token example 0 emits at step 5 becomes eos
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], eos)
    out = assert_batched_equals_single(lm, ids, recs, mnt=16)
    for i, seq in enumerate(free.tokens):
        stop = seq.index(eos) if eos in seq else len(seq)
        assert out.tokens[i] == seq[:stop]
        assert out.emitted_tokens[i] == stop
        assert out.sequential_passes[i] == 1 + out.latent_steps[i] + stop
    assert out.emitted_tokens[0] == free.tokens[0].index(eos) <= 4


@pytest.mark.parametrize("counts", [(0,), (1,), (3,)])
def test_matches_reference_generate_batch_one(setup, counts):
    t, ids = setup
    model = tiny(seed=2)
    rec = prompts(ids, counts)[0]
    x = torch.tensor([rec["input_ids"]])
    free = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], NEVER)
    ours_free = gen(free, ids, [rec], mnt=10).tokens[0]
    eos = ours_free[6]
    if eos == ours_free[0]:  # the reference never stops on the first token; keep the comparison clean
        eos = NEVER
    ref = Coconut(model, ids["latent_id"], ids["start_id"], ids["end_id"], eos)
    ours = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], eos)
    ref_tokens = ref.generate(x, torch.ones_like(x), max_new_tokens=10)
    assert ref_tokens[:x.shape[1]] == rec["input_ids"]
    assert gen(ours, ids, [rec], mnt=10).tokens[0] == ref_tokens[x.shape[1]:]


def test_pause_mode_matches_hf_greedy_generate(setup):
    """cot / no_cot / pause arms: the reference runs HF generate on the plain causal LM."""
    t, ids = setup
    model = tiny(seed=4)
    rec = prompts(ids, (2,))[0]
    x = torch.tensor([rec["input_ids"]])
    hf = model.generate(x, attention_mask=torch.ones_like(x), max_new_tokens=9, do_sample=False, eos_token_id=None,
                        pad_token_id=1)
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], NEVER)
    assert gen(lm, ids, [rec], mode="pause", mnt=9).tokens[0] == hf[0, x.shape[1]:].tolist()


@pytest.mark.parametrize("mnt", [64, 128])
def test_reference_max_new_tokens(setup, mnt):
    t, ids = setup
    lm = LatentLM(tiny(seed=5), ids["latent_id"], ids["start_id"], ids["end_id"], NEVER)
    out = gen(lm, ids, prompts(ids, (1, 2)), mnt=mnt)
    assert out.emitted_tokens == [mnt, mnt]
    assert all(len(x) == mnt for x in out.tokens)


def test_scrambled_generate_batched_equals_single_and_returns_bank(setup):
    t, ids = setup
    lm = LatentLM(tiny(seed=6), ids["latent_id"], ids["start_id"], ids["end_id"], NEVER)
    counts = (2, 3, 1)
    bank = torch.randn(3, 3, 32, generator=torch.Generator().manual_seed(0))
    out = assert_batched_equals_single(lm, ids, prompts(ids, counts), mode="scrambled", bank=bank)
    assert out.latent_steps == list(counts)
    lat = gen(lm, ids, prompts(ids, counts), mode="scrambled", bank=bank, return_latents=True).latents
    for i, n in enumerate(counts):
        assert torch.equal(lat[i], bank[i, :n])


def test_return_latents_in_generate_match_forward(setup):
    t, ids = setup
    lm = LatentLM(tiny(seed=7), ids["latent_id"], ids["start_id"], ids["end_id"], NEVER)
    recs = prompts(ids, (3, 1))
    g = gen(lm, ids, recs, return_latents=True)
    b = D.Collator(1, ids["latent_id"])(recs)
    f = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"], return_latents=True)
    for a, c in zip(g.latents, f.latents):
        torch.testing.assert_close(a, c.detach(), atol=1e-6, rtol=0)


@pytest.mark.parametrize("mode", ["coconut", "pause"])
def test_batched_equals_per_example_on_18_ragged_examples(setup, mode):
    """Registration D10: at least 16 examples with ragged latent counts (0 to 6) and ragged question lengths, with
    an eos that stops some examples early."""
    t, ids = setup
    model = tiny(seed=8)
    counts = (0, 1, 2, 3, 6, 5, 4, 3, 2, 1, 0, 6, 6, 2, 3, 1, 4, 0)
    recs = prompts(ids, counts, seed=9)
    free = gen(LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], NEVER), ids, recs, mode=mode,
               mnt=14)
    eos = free.tokens[5][3]
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], eos)
    out = assert_batched_equals_single(lm, ids, recs, mode=mode, mnt=14)
    assert len(out.tokens) == 18 and min(out.emitted_tokens) < 14
