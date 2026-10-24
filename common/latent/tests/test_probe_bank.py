"""latent.probe.patch: single-step banks (only step t differs), the scrambled path's blindness to everything but
the bank, recomputation of downstream latents, and the null bank reproducing the clean run."""
import json

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent.eval import build_prompts, evaluate
from latent.model import LatentLM
from latent.probe.patch import BankRunner, delta_bank, latents_of, single_step_bank
from latent.tests.fake_tokenizer import FakeTokenizer


WORDS = "alex bob wumpus yumpus zumpus rompus tumpus lempus is a every not".split()


def make_examples(n, seed=0):
    rng = np.random.default_rng(seed)
    return [{"question": " ".join(rng.choice(WORDS, size=4 + i % 5)) + " ?",
             "steps": [" ".join(rng.choice(WORDS, size=3)) + " ." for _ in range(1 + i % 3)],
             "answer": " ".join(rng.choice(WORDS, size=2)) + " ."} for i in range(n)]


@pytest.fixture(autouse=True, scope="module")
def _one_thread():
    """Tiny models run fastest single-threaded, and a shared CPU is often oversubscribed; restored afterwards."""
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp("bank") / "val.json"
    path.write_text(json.dumps(make_examples(7)))
    t = FakeTokenizer()
    ids = D.add_latent_tokens(t)
    ex = D.load_examples(str(path))
    D.tokenize_examples(ex, t)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(t) + 4, n_positions=256)).eval()
    return t, ids, ex, LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], t.eos_token_id).eval()


@pytest.mark.parametrize("t", range(4))
def test_single_step_bank_only_step_t_differs(t):
    torch.manual_seed(t)
    own, donor = torch.randn(4, 16), torch.randn(4, 16)
    own_copy = own.clone()
    b = single_step_bank(own, donor, t)
    assert torch.equal(own, own_copy)  # input untouched
    assert torch.equal(b[t], donor[t])
    keep = [k for k in range(4) if k != t]
    assert torch.equal(b[keep], own[keep])
    assert (b != own).any(dim=1).nonzero().flatten().tolist() == [t]


def test_delta_bank_only_step_t_differs():
    own = torch.randn(3, 8)
    d = torch.randn(8)
    b = delta_bank(own, 1, d)
    assert torch.allclose(b[1] - own[1], d) and torch.equal(b[[0, 2]], own[[0, 2]])


def test_bank_shape_mismatch_raises():
    with pytest.raises(ValueError):
        single_step_bank(torch.zeros(3, 4), torch.zeros(2, 4), 0)


def _batch(setup, n_latent):
    t, ids, ex, lm = setup
    recs = build_prompts(D.tokenize_examples(ex, t), t, arm="coconut", n_latent=n_latent)
    return D.Collator(t.pad_token_id, ids["latent_id"])(recs)


def test_scrambled_patch_changes_logits_only_from_step_t(setup):
    t_, ids, ex, lm = setup
    n = 3
    b = _batch(setup, n)
    own, _ = latents_of(lm, t_, ex, n_latent=n, batch_size=8, device="cpu", max_new_tokens=2)
    bank = torch.stack(own)
    first = int((b["input_ids"] == ids["latent_id"]).any(0).nonzero()[0])
    base = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"], mode="scrambled", scramble_bank=bank).logits
    for t in range(n):
        donor = torch.randn_like(bank)
        pb = torch.stack([single_step_bank(bank[i], donor[i], t) for i in range(len(ex))])
        lg = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"], mode="scrambled", scramble_bank=pb).logits
        p = first + t
        assert torch.allclose(lg[:, :p], base[:, :p], atol=1e-5)  # causal: nothing before latent t moves
        assert not torch.allclose(lg[:, p], base[:, p], atol=1e-4)


def test_null_bank_reproduces_clean_coconut(setup):
    t, ids, ex, lm = setup
    clean = evaluate(lm, t, ex, arm="coconut", n_latent=3, batch_size=4, device="cpu", max_new_tokens=5,
                     return_latents=True)
    runner = BankRunner(lm, t, ex, n_latent=3, batch_size=3, device="cpu", max_new_tokens=5)
    preds = runner.run([(r.idx, r.latents, 3) for r in clean])
    assert preds == [r.pred for r in clean]


def test_recompute_reproduces_clean_latents_and_propagates(setup):
    t, ids, ex, lm = setup
    n = 4
    b = _batch(setup, n)
    own, _ = latents_of(lm, t, ex, n_latent=n, batch_size=8, device="cpu", max_new_tokens=2)
    bank = torch.stack(own)
    runner = BankRunner(lm, t, ex, n_latent=n, batch_size=8, device="cpu", max_new_tokens=2)
    junk = bank.clone()
    junk[:, 1:] = torch.randn_like(junk[:, 1:])
    rec = runner.recompute(b, junk, [0] * len(ex))  # step 0 kept, steps 1..3 recomputed by the feedback rule
    assert torch.allclose(rec, bank, atol=1e-4)
    patched = bank.clone()
    patched[:, 1] += 3.0 * torch.randn_like(patched[:, 1])  # a constant shift would be cancelled by LayerNorm
    rec2 = runner.recompute(b, patched, [1] * len(ex))
    assert torch.equal(rec2[:, :2], patched[:, :2])
    assert not torch.allclose(rec2[:, 2:], bank[:, 2:], atol=1e-3)  # the patch propagates downstream
