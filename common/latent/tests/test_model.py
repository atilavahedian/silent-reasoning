"""LatentLM.forward against the port of the official Coconut forward (tests/reference_coconut.py), plus the pause and
scrambled modes, the cache helper and the embedding initialisation. Random tiny GPT-2, CPU, fp32."""
import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent.model import LatentLM, _crop_cache, init_latent_embeddings
from latent.tests.fake_tokenizer import FakeTokenizer
from latent.tests.reference_coconut import Coconut
from latent.tests.reference_dataset import MyCollator

ATOL = 1e-5


def tiny(attn="eager", vocab=64, seed=0, dropout=0.0):
    torch.manual_seed(seed)
    cfg = GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=vocab, n_positions=256, resid_pdrop=dropout,
                     embd_pdrop=dropout, attn_pdrop=dropout, attn_implementation=attn)
    return GPT2LMHeadModel(cfg)


@pytest.fixture(scope="module")
def tok():
    t = FakeTokenizer()
    ids = D.add_latent_tokens(t)
    return t, ids


def ragged_features(ids, counts=(0, 1, 2, 3), seed=1, qlens=(5, 7, 4, 9)):
    g = torch.Generator().manual_seed(seed)
    feats = []
    for i, n in enumerate(counts):
        q = torch.randint(5, 60, (qlens[i % len(qlens)],), generator=g).tolist()
        tail = torch.randint(5, 60, (6 + i,), generator=g).tolist()
        x = q + [ids["start_id"]] + [ids["latent_id"]] * n + [ids["end_id"]] + tail
        feats.append({"input_ids": x, "labels": [-100] * (len(q) + n + 2) + tail, "attention_mask": [1] * len(x),
                      "position_ids": list(range(len(x)))})
    return feats


def ref_batch(t, ids, feats):
    return MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(feats))


def pair(model, ids):
    ours = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], 0)
    ref = Coconut(model, ids["latent_id"], ids["start_id"], ids["end_id"], 0)
    return ours, ref


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("counts", [(0, 1, 2, 3), (3, 0, 2, 1), (2, 2, 2, 2), (0, 0, 0, 0), (1,)])
def test_coconut_forward_matches_reference(tok, attn, counts):
    t, ids = tok
    model = tiny(attn)
    b = ref_batch(t, ids, ragged_features(ids, counts))
    if any(counts) and len(counts) > 1:
        assert (b["position_ids"][:, 0] == 0).all() and (b["attention_mask"] == 0).any()  # reference left pad
    ours, ref = pair(model, ids)
    o = ours(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
    r = ref(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
    assert o.logits.shape == r.logits.shape
    torch.testing.assert_close(o.logits, r.logits, atol=ATOL, rtol=0)
    torch.testing.assert_close(o.loss, r.loss, atol=ATOL, rtol=0)
    torch.testing.assert_close(o.inputs_embeds, r.inputs_embeds, atol=ATOL, rtol=0)


def test_coconut_gradients_match_reference(tok):
    """The fed-back hidden state is not detached in the reference; the gradient of every parameter must agree."""
    t, ids = tok
    feats = ragged_features(ids, (0, 1, 2, 3))
    grads = []
    for which in ("ours", "ref"):
        model = tiny("eager")
        ours, ref = pair(model, ids)
        m = ours if which == "ours" else ref
        b = ref_batch(t, ids, feats)
        m(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"]).loss.backward()
        grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
    for n in grads[0]:
        torch.testing.assert_close(grads[0][n], grads[1][n], atol=ATOL, rtol=1e-4, msg=n)


def test_feedback_is_not_detached(tok):
    t, ids = tok
    model = tiny()
    lm, _ = pair(model, ids)
    b = ref_batch(t, ids, ragged_features(ids, (2, 3)))
    out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    assert all(x.requires_grad and x.grad_fn is not None for x in out.latents)
    # d latent / d first-layer weights is non-zero: the latent is a function of the model's own hidden states
    g = torch.autograd.grad(out.latents[1].sum(), model.transformer.h[0].attn.c_attn.weight)[0]
    assert g.abs().sum() > 0


def test_return_latents_are_the_fed_back_hidden_states(tok):
    t, ids = tok
    model = tiny()
    lm, ref = pair(model, ids)
    counts = (0, 1, 2, 3)
    b = ref_batch(t, ids, ragged_features(ids, counts))
    out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    r = ref(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
    for i, n in enumerate(counts):
        assert out.latents[i].shape == (n, 32)
        pos = (b["input_ids"][i] == ids["latent_id"]).nonzero().squeeze(1)
        torch.testing.assert_close(out.latents[i], r.inputs_embeds[i, pos], atol=ATOL, rtol=0)
    assert lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"]).latents is None


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
def test_pause_is_the_plain_causal_lm(tok, attn):
    t, ids = tok
    model = tiny(attn)
    lm, _ = pair(model, ids)
    b = ref_batch(t, ids, ragged_features(ids, (0, 1, 2, 3)))
    o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="pause")
    h = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], position_ids=b["position_ids"],
              labels=b["labels"])
    torch.testing.assert_close(o.logits, h.logits, atol=ATOL, rtol=0)
    torch.testing.assert_close(o.loss, h.loss, atol=ATOL, rtol=0)
    lat = b["input_ids"] == ids["latent_id"]
    emb = model.get_input_embeddings().weight[ids["latent_id"]]
    assert torch.equal(o.inputs_embeds[lat], emb.expand(int(lat.sum()), -1))


def _bank(counts, d=32, seed=5):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(len(counts), max(counts), d, generator=g)


def test_scrambled_latent_inputs_equal_the_bank(tok):
    t, ids = tok
    model = tiny()
    lm, _ = pair(model, ids)
    counts = (0, 1, 2, 3)
    bank = _bank(counts)
    b = ref_batch(t, ids, ragged_features(ids, counts))
    o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="scrambled", scramble_bank=bank,
           return_latents=True)
    for i, n in enumerate(counts):
        pos = (b["input_ids"][i] == ids["latent_id"]).nonzero().squeeze(1)
        assert torch.equal(o.inputs_embeds[i, pos], bank[i, :n])
        assert torch.equal(o.latents[i], bank[i, :n])


def test_scrambled_is_blind_to_the_examples_own_question(tok):
    """Perturbing example 0's question changes nothing at its latent inputs (they stay the bank rows) and nothing
    in any other example; in coconut mode the same perturbation does change example 0's latent inputs."""
    t, ids = tok
    model = tiny()
    lm, _ = pair(model, ids)
    counts = (3, 2, 3)
    bank = _bank(counts)
    feats = ragged_features(ids, counts)
    feats2 = copy.deepcopy(feats)
    feats2[0]["input_ids"][:3] = [7, 8, 9]
    b1, b2 = ref_batch(t, ids, feats), ref_batch(t, ids, feats2)
    lat = b1["input_ids"] == ids["latent_id"]
    for mode in ("scrambled", "coconut"):
        o1 = lm(b1["input_ids"], b1["attention_mask"], b1["labels"], b1["position_ids"], mode=mode,
                scramble_bank=bank)
        o2 = lm(b2["input_ids"], b2["attention_mask"], b2["labels"], b2["position_ids"], mode=mode,
                scramble_bank=bank)
        same = torch.equal(o1.inputs_embeds[0][lat[0]], o2.inputs_embeds[0][lat[0]])
        assert same == (mode == "scrambled")
        torch.testing.assert_close(o1.logits[1:], o2.logits[1:], atol=ATOL, rtol=0)  # batch isolation


def test_scrambled_no_gradient_from_latent_positions_into_hidden_states(tok):
    t, ids = tok
    model = tiny()
    lm, _ = pair(model, ids)
    counts = (1, 2, 3)
    b = ref_batch(t, ids, ragged_features(ids, counts))
    lat = b["input_ids"] == ids["latent_id"]
    params = [p for p in model.parameters()]
    for mode, expect_grad in (("scrambled", False), ("coconut", True)):
        bank = _bank(counts).requires_grad_(True)
        o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode=mode, scramble_bank=bank)
        gs = torch.autograd.grad(o.inputs_embeds[lat].sum(), params + [bank], allow_unused=True)
        param_grad = sum(float(g.abs().sum()) for g in gs[:-1] if g is not None)
        assert (param_grad > 0) == expect_grad
        if mode == "scrambled":
            used = (torch.tensor(counts).unsqueeze(1) > torch.arange(max(counts)).unsqueeze(0)).float()
            assert torch.equal(gs[-1], used.unsqueeze(-1).expand_as(bank))  # d/d bank = 1 on used rows only


def test_scrambled_requires_a_bank(tok):
    t, ids = tok
    lm, _ = pair(tiny(), ids)
    b = ref_batch(t, ids, ragged_features(ids, (1, 2)))
    with pytest.raises(ValueError):
        lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="scrambled")


def test_misaligned_latents_raise(tok):
    t, ids = tok
    lm, _ = pair(tiny(), ids)
    feats = ragged_features(ids, (1, 2))
    x = torch.tensor([f["input_ids"][:12] for f in feats])  # no left-pad alignment: first latents differ in column
    assert (x == ids["latent_id"]).any(1).all()
    with pytest.raises(ValueError):
        lm(x, torch.ones_like(x), None, torch.arange(12).expand(2, 12))


def test_crop_cache_handles_legacy_tuples_and_dynamic_cache():
    k = torch.randn(1, 2, 7, 4)
    legacy = ((k, k + 1), (k + 2, k + 3))
    cropped = _crop_cache(legacy, 5)
    assert isinstance(cropped, tuple) and cropped[1][0].shape[2] == 5 and torch.equal(cropped[1][0], (k + 2)[:, :, :5])
    try:
        from transformers import DynamicCache
    except ImportError:  # pragma: no cover - very old transformers
        return
    dc = DynamicCache.from_legacy_cache(legacy)
    out = _crop_cache(dc, 5)
    assert out is dc and dc.get_seq_length() == 5
    assert _crop_cache(dc, 9) is dc and dc.get_seq_length() == 5
    assert _crop_cache(None, 3) is None


def test_coconut_forward_reuses_returned_cache_type(tok):
    """Works whatever cache type the installed transformers returns (DynamicCache on 4.57, tuples on old 4.x)."""
    t, ids = tok
    model = tiny()
    lm, _ = pair(model, ids)
    b = ref_batch(t, ids, ragged_features(ids, (2, 3)))
    run = lm._run(b["input_ids"], b["attention_mask"], b["position_ids"], "coconut", None, keep_cache=True)
    cache = run.cache
    n = cache.get_seq_length() if hasattr(cache, "get_seq_length") else cache[0][0].shape[2]
    assert n == b["input_ids"].shape[1]


def test_init_latent_embeddings_copies_the_init_row(tok):
    t, ids = tok
    model = tiny(vocab=len(t))
    for w in ["<<", "x", "y"]:
        t.encode(w)
    model.resize_token_embeddings(len(t))
    tgt = init_latent_embeddings(model, t, ids)
    assert tgt == t.convert_tokens_to_ids("<<")
    w = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    for k in ("latent_id", "start_id", "end_id"):
        assert torch.equal(w[ids[k]], w[tgt]) and torch.equal(head[ids[k]], head[tgt])  # input and tied head
    t2 = FakeTokenizer()
    ids2 = D.add_latent_tokens(t2)
    m2 = tiny(vocab=2)
    assert init_latent_embeddings(m2, t2, ids2) == t2.eos_token_id  # no "<<": eos fallback, grown to len(tokenizer)
    assert m2.get_input_embeddings().weight.shape[0] == len(t2)


def _per_example_loss(logits, labels):
    lp = torch.nn.functional.cross_entropy(logits[:, :-1].transpose(1, 2), labels[:, 1:], reduction="none")
    m = (labels[:, 1:] != -100).float()
    return (lp * m).sum(1) / m.sum(1)


@pytest.mark.parametrize("mode", ["coconut", "pause", "scrambled"])
def test_train_mode_batch_isolation_same_shape_bit_identical(tok, mode):
    """Registration D10: in train mode (dropout on, same seed), example i's loss and logits are bit-identical when a
    companion's tokens change (same lengths, so the same padding and shapes)."""
    t, ids = tok
    model = tiny(dropout=0.1).train()
    lm, _ = pair(model, ids)
    counts = (3, 1, 2, 3)
    bank = _bank(counts)
    feats = ragged_features(ids, counts)
    feats2 = copy.deepcopy(feats)
    for j in (1, 3):
        feats2[j]["input_ids"][:4] = [11, 12, 13, 14]
        feats2[j]["input_ids"][-3:] = [21, 22, 23]
        feats2[j]["labels"][-3:] = [21, 22, 23]
    outs = []
    for f in (feats, feats2):
        b = ref_batch(t, ids, f)
        torch.manual_seed(123)
        o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode=mode, scramble_bank=bank)
        outs.append((o.logits, _per_example_loss(o.logits, b["labels"])))
    for i in (0, 2):
        assert torch.equal(outs[0][0][i], outs[1][0][i])
        assert torch.equal(outs[0][1][i], outs[1][1][i])
    assert not torch.equal(outs[0][1][1], outs[1][1][1])


@pytest.mark.parametrize("mode", ["coconut", "pause", "scrambled"])
def test_batch_isolation_alone_vs_with_companions(tok, mode):
    """Registration A1 D10: B = 1 vs B = 2 and 8 (different padding): example 0's loss and the logits at its real
    positions agree within 1e-4 fp32."""
    t, ids = tok
    lm, _ = pair(tiny().train(), ids)  # dropout 0: the RNG stream differs with the batch shape
    counts = (2, 3, 0, 1, 3, 2, 1, 3)
    feats = ragged_features(ids, counts, seed=4, qlens=(6, 9, 4, 11))
    bank = _bank(counts)

    def run(B):
        b = ref_batch(t, ids, feats[:B])
        o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode=mode,
               scramble_bank=bank[:B])
        real = b["attention_mask"][0].bool()
        return o.logits[0][real], _per_example_loss(o.logits, b["labels"])[0]

    l1, loss1 = run(1)
    for B in (2, 8):
        lb, lossb = run(B)
        torch.testing.assert_close(lb, l1, atol=1e-4, rtol=0)
        torch.testing.assert_close(lossb, loss1, atol=1e-4, rtol=0)


def _calibrate_module():
    import importlib.util
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "experiments", "2026-10-17-silent-reasoning",
                        "code", "calibrate_mac.py")
    spec = importlib.util.spec_from_file_location("calibrate_mac", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibrate_parity_functions_run_on_cpu(tok):
    """The tiny-model parity (default mode) and the real-batch parity code path (--real, here on a tiny model and
    fake data) of calibrate_mac.py: CPU fp32 vs CPU float64 agree to fp32 precision."""
    cal = _calibrate_module()
    assert cal.parity("cpu")["max_abs_logit_diff"] < 1e-4
    t = FakeTokenizer()
    D.add_latent_tokens(t)
    exs = [D.Example(question=f"q{i} a b c d e ?", answer="a .", steps=["a b .", "c d .", "e a ."], idx=i)
           for i in range(4)]
    D.tokenize_examples(exs, t)
    model = tiny(vocab=len(t)).eval()
    for stage in (3, 6):
        r = cal.real_parity(model, t, exs, "cpu", stage, batch=4)
        assert r["n_latent"] == stage and r["max_abs_logit_diff"] < 1e-4 and r["loss_rel_diff"] < 1e-5
