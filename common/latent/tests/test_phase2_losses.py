"""C2 decoder and C3 hinge (docs/cards/C2 and C3 "Leak and bug answers"; registration A3.4): the decoder is absent
from the inference graph and its loss reaches the language model only through the latents; the hinge arithmetic on
a toy; the scrambled bank is detached; derangements have no fixed point, the two C3-foreign banks differ rowwise and
the training seeds never equal the evaluation seed; the C3-foreign hinge has a non-zero gradient."""
import copy
import inspect
import random

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import aux_losses as AX
from latent import data as D
from latent import eval as E
from latent import model as M
from latent import prosqa_gen as G
from latent import state_targets as S
from latent.model import LatentLM
from latent.tests.fake_tokenizer import FakeTokenizer
from latent.tests.reference_dataset import MyCollator
from latent.tests.test_model import ragged_features


def tiny(vocab=64, seed=0):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=vocab, n_positions=256,
                                      resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)).eval()


@pytest.fixture(scope="module")
def tok():
    t = FakeTokenizer()
    return t, D.add_latent_tokens(t)


def _gsm_targets(t, n):
    recs = [{"question": "q", "steps": [f"<<{i}+{k}={i + k}>>" for k in range(1, 4)], "answer": str(i + 3), "idx": i}
            for i in range(n)]
    return S.build_targets(recs, t)


def _batch(t, ids, counts):
    return MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(ragged_features(ids, counts)))


def test_decoder_is_not_in_the_inference_graph(tok):
    """No inference module imports the decoder; LatentLM holds no decoder; its state_dict is the base model's."""
    for mod in (M, E):
        assert "aux_losses" not in inspect.getsource(mod) and "StateDecoder" not in inspect.getsource(mod)
    t, ids = tok
    lm = LatentLM(tiny(), ids["latent_id"], ids["start_id"], ids["end_id"], 0)
    assert not any(isinstance(m, AX.StateDecoder) for m in lm.modules())
    assert all(k.startswith("base_causallm.") for k in lm.state_dict())


@pytest.mark.parametrize("kind", ["gsm", "prosqa"])
def test_state_loss_reaches_the_lm_only_through_the_latents(tok, kind):
    """d L_state / d theta equals the vector-Jacobian product through the fed-back latents; with the latents
    detached, no language-model parameter gets a gradient; the language-model CE does not depend on the targets."""
    t, ids = tok
    counts = (2, 3, 1)
    if kind == "gsm":
        targets = _gsm_targets(t, 3)
    else:
        recs = G.generate(3, depth=3, seed=2)
        targets = {i: S.record_targets(r, t) for i, r in enumerate(recs)}
    model = tiny(vocab=max(64, len(t) + 1))
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], 0)
    dec = AX.StateDecoder(32)
    b = _batch(t, ids, counts)
    out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    params = list(model.parameters())
    ls, n_sup, units = AX.state_loss(dec, out.latents, [0, 1, 2], targets, c_thought=1, emb=lm.embedding.weight)
    assert ls is not None and n_sup == sum(min(3, c) for c in counts) and units > 0
    g_direct = torch.autograd.grad(ls, params, retain_graph=True, allow_unused=True)
    lat = [x for x in out.latents if x.shape[0]]
    g_lat = torch.autograd.grad(ls, lat, retain_graph=True)
    g_vjp = torch.autograd.grad(lat, params, grad_outputs=g_lat, retain_graph=True, allow_unused=True)
    for a, c in zip(g_direct, g_vjp):
        if a is None or c is None:
            assert a is None or float(a.abs().sum()) == 0.0
            continue
        torch.testing.assert_close(a, c, atol=1e-6, rtol=1e-5)
    assert sum(float(g.abs().sum()) for g in g_direct if g is not None) > 0
    det = [x.detach() for x in out.latents]
    ls2, _, _ = AX.state_loss(dec, det, [0, 1, 2], targets, c_thought=1, emb=lm.embedding.weight)
    assert all(g is None for g in torch.autograd.grad(ls2, params, allow_unused=True))
    other = {i: targets[(i + 1) % 3] for i in range(3)}
    out2 = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    AX.state_loss(dec, out2.latents, [0, 1, 2], other, c_thought=1, emb=lm.embedding.weight)
    assert torch.equal(out.loss, out2.loss)


def test_state_loss_none_at_stage_zero_and_c_thought_two(tok):
    t, ids = tok
    targets = _gsm_targets(t, 2)
    dec = AX.StateDecoder(32)
    emb = torch.randn(len(t) + 4, 32)
    zero = [torch.zeros(0, 32), torch.zeros(0, 32)]
    assert AX.state_loss(dec, zero, [0, 1], targets, c_thought=1, emb=emb) == (None, 0, 0)
    six = [torch.randn(6, 32, requires_grad=True), torch.randn(4, 32, requires_grad=True)]
    ls, n_sup, _ = AX.state_loss(dec, six, [0, 1], targets, c_thought=2, emb=emb)
    assert n_sup == 3 + 2
    g = torch.autograd.grad(ls, six)
    assert [int((x.abs().sum(1) > 0).sum()) for x in g] == [3, 2]
    assert all(bool((x[0::2] == 0).all()) for x in g)          # only the second latent of each pair is decoded


def test_hinge_arithmetic_on_a_toy():
    # per-row mean CE over answer tokens: row 0 has two labelled tokens, row 1 one
    logits = torch.log(torch.tensor([[[0.5, 0.5], [0.25, 0.75], [0.9, 0.1]],
                                     [[0.5, 0.5], [0.5, 0.5], [0.2, 0.8]]]))
    labels = torch.tensor([[-100, 0, 1], [-100, -100, 1]])
    ce, n = AX.per_example_ce(logits, labels)
    torch.testing.assert_close(ce, torch.tensor([(-torch.log(torch.tensor(0.5)) - torch.log(torch.tensor(0.75))) / 2,
                                                 -torch.log(torch.tensor(0.5))]))
    assert n.tolist() == [2, 1]
    ref, alt = torch.tensor([1.0, 2.0, 0.5]), torch.tensor([1.1, 3.0, 0.4])
    valid = torch.tensor([True, True, True])
    # max(0, 0.2 - 0.1) = 0.1; max(0, 0.2 - 1.0) = 0; max(0, 0.2 + 0.1) = 0.3 -> mean 0.4 / 3
    torch.testing.assert_close(AX.hinge(ref, alt, valid, 0.2), torch.tensor(0.4 / 3))
    torch.testing.assert_close(AX.hinge(ref, alt, torch.tensor([True, True, False]), 0.2), torch.tensor(0.05))
    assert AX.hinge(ref, alt, torch.tensor([False] * 3), 0.2) is None
    r = ref.clone().requires_grad_(True)
    a = alt.clone().requires_grad_(True)
    AX.hinge(r, a, valid, 0.2).backward()
    assert r.grad.tolist() == pytest.approx([1 / 3, 0.0, 1 / 3]) and a.grad.tolist() == pytest.approx([-1 / 3, 0, -1 / 3])


def test_derangements_and_seeds():
    counts = [6] * 32 + [0, 3, 3, 4]
    seeds = {AX.train_derangement_seed(s, e, k, st) for s in range(3) for e in range(30) for k in range(50)
             for st in (0, 1)}
    assert AX.EVAL_SCRAMBLE_SEED == 0 == inspect.signature(E.evaluate).parameters["scramble_seed"].default
    assert AX.EVAL_SCRAMBLE_SEED not in seeds and len(seeds) == 3 * 30 * 50 * 2
    pi, valid = AX.batch_derangement(counts, AX.train_derangement_seed(0, 3, 0))
    assert all(pi[i] != i and counts[pi[i]] == counts[i] for i in range(len(counts)) if valid[i])
    assert valid[:32] == [True] * 32 and valid[32:] == [False, True, True, False]   # no latents; class of 1
    ev = E.scramble_donors({i: 6 for i in range(32)}, seed=AX.EVAL_SCRAMBLE_SEED)
    assert [pi[i] for i in range(32)] != [ev[i] for i in range(32)]     # training and evaluation permutations differ
    pb, vb = AX.batch_derangement(counts, AX.train_derangement_seed(0, 3, 0, 1), avoid=pi)
    assert all(pb[i] != pi[i] and pb[i] != i for i in range(len(counts)) if vb[i])
    assert vb[32:] == [False, False, False, False]                       # the foreign pair needs >= 3 members
    assert AX.batch_derangement(counts, 7) == AX.batch_derangement(counts, 7)


def test_bank_is_detached_and_from_other_examples(tok):
    t, ids = tok
    lm = LatentLM(tiny(), ids["latent_id"], ids["start_id"], ids["end_id"], 0)
    b = _batch(t, ids, (3, 3, 3, 3))
    out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    pi, valid = AX.batch_derangement([3] * 4, 11)
    bank = AX.build_bank(out.latents, pi, valid)
    assert not bank.requires_grad
    for i in range(4):
        assert pi[i] != i and torch.equal(bank[i], out.latents[pi[i]].detach())
    scr = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="scrambled", scramble_bank=bank)
    ce, _ = AX.per_example_ce(scr.logits, b["labels"])
    g = torch.autograd.grad(ce.sum(), out.latents, allow_unused=True)
    assert all(x is None for x in g)          # no gradient into any example's own latent computation


def test_foreign_hinge_has_a_gradient(tok):
    """A3.4 C3-foreign: max(0, m - (CE_B - CE_A)) between two banks has a non-zero parameter gradient on a toy
    (the withdrawn question-position placebo had none) and the two bank permutations differ rowwise."""
    t, ids = tok
    model = tiny()
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], 0)
    b = _batch(t, ids, (3, 3, 3, 3))
    out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    pa, va = AX.batch_derangement([3] * 4, AX.train_derangement_seed(0, 1, 0, 0))
    pb, vb = AX.batch_derangement([3] * 4, AX.train_derangement_seed(0, 1, 0, 1), avoid=pa)
    assert all(va) and all(vb) and all(x != y for x, y in zip(pa, pb))
    ce = []
    for p, v in ((pa, va), (pb, vb)):
        o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="scrambled",
               scramble_bank=AX.build_bank(out.latents, p, v))
        ce.append(AX.per_example_ce(o.logits, b["labels"])[0])
    h = AX.hinge(ce[0], ce[1], torch.tensor([True] * 4), 0.2)
    assert float(h.detach()) > 0
    g = torch.autograd.grad(h, list(model.parameters()), allow_unused=True)
    assert sum(float(x.abs().sum()) for x in g if x is not None) > 0
    # the withdrawn placebo: CE at a question position is identical under true and scrambled latents (causality)
    o_true = out.logits[:, :3]
    o_scr = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="scrambled",
               scramble_bank=AX.build_bank(out.latents, pa, va)).logits[:, :3]
    torch.testing.assert_close(o_true, o_scr, atol=1e-6, rtol=0)


def test_decoder_shapes_and_tied_readout():
    dec = AX.StateDecoder(16, hidden=768)
    assert dec.inp.out_features == 768 and isinstance(dec.act, torch.nn.GELU) and len(dec.tok_heads) == 4
    z, step = torch.randn(5, 16), torch.tensor([0, 1, 2, 40, 3])
    emb = torch.randn(30, 16, requires_grad=True)
    lg = dec.token_logits(z, step, emb.detach())
    assert lg.shape == (5, 4, 30)
    lg.sum().backward()
    assert emb.grad is None                       # the readout matrix is used detached
    sym = dec.symbol_logits(z, step, torch.randn(5, 7, 16), torch.ones(5, 7, dtype=torch.bool))
    assert sym.shape == (5, 7)
    random.seed(0)
