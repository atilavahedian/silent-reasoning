"""Adversarial-review tests of the Phase 2 mechanism code (experiments/.../docs/review_phase2_code.md).

Functional proofs, not bookkeeping checks: (1) the C1b band never reads a latent column outside the band: large noise
written into the non-band caches at those columns (forward, tail and decode steps, the N-row column passes included)
changes nothing; (2) the scrambled band hands exactly scramble_bank[i, rank] to block a; (3) pause mode is one pass
with no feedback; (4) C3-foreign hinges only rows whose two banks are both foreign (through train._aux_step), and no
hinged row sees its own latents in either bank; (5) the C2 decoder and targets are absent from eval.py and
run_eval.py; (6) the review's mode-flag fix: a LatentLM over an .eval() base is in eval mode, and evaluate() runs in
eval mode under no_grad and restores the base's own mode; (7) the C2-shuffled donor map on ProsQA-hard-like classes.
"""
import copy
import inspect
import json
import os

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import aux_losses as AX
from latent import data as D
from latent import eval as E
from latent import state_targets as S
from latent import train as T
from latent.model import LatentLM
from latent.tests.fake_tokenizer import FakeTokenizer
from latent.tests.reference_dataset import MyCollator
from latent.tests.test_model import ragged_features

EXP = os.path.join(os.path.dirname(__file__), "..", "..", "..", "experiments", "2026-10-17-silent-reasoning")


def tiny(n_layer=4, seed=0, dropout=0.0, vocab=64):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(n_layer=n_layer, n_embd=32, n_head=2, vocab_size=vocab, n_positions=256,
                                      resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout)).eval()


@pytest.fixture(scope="module")
def tok():
    t = FakeTokenizer()
    return t, D.add_latent_tokens(t)


def lm_of(model, ids, band=None):
    return LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], 0, latent_band=band)


def batch_of(t, ids, counts, seed=1, qlens=(5, 7, 4, 9)):
    feats = ragged_features(ids, counts, seed=seed, qlens=qlens)
    return MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(feats))


# ------------------------------------------------------------------------------------------ (1) band blindness
@pytest.mark.parametrize("band,counts", [((1, 3), (0, 1, 2, 3)), ((2, 4), (3, 0, 2, 1)), ((1, 2), (2, 2, 2)),
                                         ((0, 2), (1, 3)), ((3, 4), (2, 1, 2))])
def test_band_noise_in_non_band_latent_columns_changes_nothing(tok, band, counts):
    """Non-band blocks hold exactly-zero key/value columns at latent positions (test_band). Every non-band _block
    call here gets those columns overwritten with large random noise before its attention runs, for the prefix,
    the column passes (rows without a latent at that column), the tail and the decode steps. Logits at every real
    non-latent position, the fed-back latents and the greedy generation must be bit-identical to the untouched run,
    and no admitted mask entry may point at such a column."""
    t, ids = tok
    model = tiny(n_layer=4, seed=2)
    b = batch_of(t, ids, counts)
    lm = lm_of(model, ids, band)
    a, bnd = band
    gen = torch.Generator().manual_seed(9)
    ref_out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
    ref_gen = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=4)
    orig = lm._block
    seen = {"noised": 0, "outside_calls": 0}

    def noisy_block(i, x, mask, kv):
        if kv is not None and not a <= i < bnd:
            k, v = kv
            zero = (k == 0).all(1).all(-1) & (v == 0).all(1).all(-1)   # [B', S]: latent columns of a non-band cache
            seen["outside_calls"] += 1
            if bool(zero.any()):
                keys = torch.cat([zero, torch.zeros(zero.shape[0], x.shape[1], dtype=torch.bool)], 1)  # + new cols
                assert not bool((mask[:, 0] & keys.unsqueeze(1)).any()), f"block {i}: a query admits a latent column"
                z = zero[:, None, :, None]
                kv = (torch.where(z, torch.randn(k.shape, generator=gen) * 1e3, k),
                      torch.where(z, torch.randn(v.shape, generator=gen) * 1e3, v))
                seen["noised"] += int(zero.sum())
        return orig(i, x, mask, kv)

    lm._block = noisy_block
    try:
        out = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], return_latents=True)
        g = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=4)
    finally:
        del lm._block
    assert seen["outside_calls"] > 0 and seen["noised"] > 0
    real = b["attention_mask"].bool() & (b["input_ids"] != ids["latent_id"])
    assert torch.equal(out.logits[real], ref_out.logits[real])
    assert torch.equal(out.loss, ref_out.loss)
    for x, y in zip(out.latents, ref_out.latents):
        assert torch.equal(x, y)
    assert g.tokens == ref_gen.tokens and g.latent_steps == list(counts)


# ------------------------------------------------------------------------------- (2) scrambled band injection
@pytest.mark.parametrize("band", [(1, 3), (2, 4), (0, 2)])
def test_scrambled_band_hands_exactly_the_bank_to_block_a(tok, band):
    """At every latent column the vector given to _band_column is scramble_bank[:, rank] and the hidden state entering
    block a on the latent rows is that vector itself (a > 0: no position embedding, no dropout, nothing of the
    example's own states; a == 0: bank + wpe, the Coconut placement). Changing example 0's question changes neither."""
    t, ids = tok
    counts = (3, 2, 3, 0)
    a = band[0]
    bank = torch.randn(len(counts), 3, 32, generator=torch.Generator().manual_seed(2))
    recorded = {}
    for variant in ("base", "perturbed"):
        model = tiny(n_layer=4, seed=4)
        lm = lm_of(model, ids, band)
        feats = ragged_features(ids, counts)
        if variant == "perturbed":
            feats[0]["input_ids"][:3] = [7, 8, 9]
        b = MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(feats))
        lat = b["input_ids"] == ids["latent_id"]
        first = int(lat.any(0).nonzero()[0])
        orig_col, orig_block = lm._band_column, lm._block
        cur, vals_seen, x_seen = {}, [], []

        def col(state, embeds, position_ids, j, latcol, vals):
            cur["j"], cur["latcol"] = j, latcol.clone()
            vals_seen.append((j, latcol.clone(), vals.detach().clone()))
            try:
                return orig_col(state, embeds, position_ids, j, latcol, vals)
            finally:
                cur.clear()

        def block(i, x, mask, kv):
            if cur and i == a:
                x_seen.append((cur["j"], cur["latcol"], x.detach().clone()[:, 0]))
            return orig_block(i, x, mask, kv)

        lm._band_column, lm._block = col, block
        try:
            o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="scrambled",
                   scramble_bank=bank, return_latents=True)
        finally:
            del lm._band_column, lm._block
        assert len(vals_seen) == 3 and len(x_seen) == 3
        wpe = model.transformer.wpe
        for (j, latcol, vals), (j2, latcol2, x) in zip(vals_seen, x_seen):
            p = j - first
            assert j == j2 and torch.equal(latcol, latcol2) and torch.equal(latcol, lat[:, j])
            assert torch.equal(vals[latcol], bank[latcol, p])
            want = bank[latcol, p] if a > 0 else bank[latcol, p] + wpe(b["position_ids"][latcol, j])
            assert torch.equal(x[latcol], want)
        for i, n in enumerate(counts):
            assert torch.equal(o.latents[i], bank[i, :n])
        recorded[variant] = [v for _, _, v in vals_seen]
    for u, v in zip(recorded["base"], recorded["perturbed"]):
        assert torch.equal(u, v)


# ----------------------------------------------------------------------------------------- (3) pause feedback
def test_pause_mode_is_one_pass_with_no_feedback(tok):
    t, ids = tok
    model = tiny(n_layer=2, seed=1)
    lm = lm_of(model, ids)
    counts = (0, 1, 2, 3)
    b = batch_of(t, ids, counts)
    calls, orig = [], lm._pass

    def spy(embeds, *a, **k):
        calls.append(embeds.shape[1])
        return orig(embeds, *a, **k)

    lm._pass = spy
    try:
        o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="pause", return_latents=True)
        g = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=3, mode="pause")
    finally:
        del lm._pass
    L = b["input_ids"].shape[1]
    assert calls[:2] == [L, L] and all(c == 1 for c in calls[2:])      # one whole-sequence pass, then one per token
    emb = model.get_input_embeddings().weight[ids["latent_id"]]
    for i, n in enumerate(counts):
        assert o.latents[i].shape[0] == n and torch.equal(o.latents[i], emb.expand(n, -1))
    assert g.latent_steps == [0] * 4 and g.sequential_passes == [1 + e for e in g.emitted_tokens]
    assert T.LATENT_FEEDBACK_ARMS == ("coconut", "coconut_no_curriculum")      # train.py: every other arm is "pause"
    assert E.gen_mode("pause") == "pause" and E.gen_mode("coconut") == "coconut"
    with pytest.raises(ValueError):
        lm_of(model, ids, (0, 1))(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"], mode="pause")


# ------------------------------------------------------------------------- (4) C3 / C3-foreign through _aux_step
@pytest.mark.parametrize("null,hinged", [("none", 5), ("foreign", 3)])
def test_c3_hinged_rows_never_see_their_own_latents(tok, null, hinged):
    """Latent-count classes {3: rows 0-2, 2: rows 3-4, 0: row 5}. C3 hinges every row with latents in a class of
    >= 2 (five rows); C3-foreign only the class of >= 3 (three rows: the pair class has no second derangement). For
    every hinged row both banks hold another example's latents, the two banks differ rowwise, and every bank is
    detached. Rows outside the hinge keep their own detached latents in the bank (defined pass, no gradient)."""
    t, ids = tok
    model = tiny(n_layer=2, seed=6)
    lm = lm_of(model, ids).train()
    counts = (3, 3, 3, 2, 2, 0)
    cb = MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(ragged_features(ids, counts, qlens=(5, 7, 4, 9, 6, 8))))
    batch = {k: v for k, v in cb.items() if k != "idx"}
    cfg = T.TrainConfig(name="c3", arm="coconut", dep_loss=True, dep_null=null, max_new_tokens=3)
    banks = []
    h = lm.register_forward_pre_hook(lambda m, a, kw: banks.append(kw.get("scramble_bank")), with_kwargs=True)
    try:
        out, total, logs, passes = T._aux_step(cfg, lm, batch, list(range(6)), "coconut", 1, 0, 0, None, None)
    finally:
        h.remove()
    assert passes == (3 if null == "foreign" else 2) and logs["dep_rows"] == hinged
    banks = [x for x in banks if x is not None]
    assert len(banks) == passes - 1 and all(not x.requires_grad for x in banks)
    own = [x.detach() for x in out.latents]
    pi_a, va = AX.batch_derangement(list(counts), AX.train_derangement_seed(cfg.seed, 0, 0, 0))
    if null == "foreign":
        pi_b, vb = AX.batch_derangement(list(counts), AX.train_derangement_seed(cfg.seed, 0, 0, 1), avoid=pi_a)
        valid = [x and y for x, y in zip(va, vb)]
    else:
        valid = va
    assert sum(valid) == hinged
    for i, n in enumerate(counts):
        for bank in banks:
            if valid[i]:
                assert not torch.equal(bank[i, :n], own[i]), f"row {i} hinged against its own latents"
                assert any(torch.equal(bank[i, :n], own[j]) for j in range(6) if j != i and counts[j] == n)
            elif n:   # unhinged rows (a pair class under C3-foreign): a defined pass from a same-class row, no hinge
                assert any(torch.equal(bank[i, :n], own[j]) for j in range(6) if counts[j] == n)
        if valid[i] and len(banks) == 2:
            assert not torch.equal(banks[0][i, :n], banks[1][i, :n])
    assert logs["dep_hinge"] > 0 and float(total.detach()) >= float(out.loss.detach()) - 1e-6
    g = torch.autograd.grad(total - out.loss, list(model.parameters()), allow_unused=True)
    assert sum(float(x.abs().sum()) for x in g if x is not None) > 0   # the hinge term has a parameter gradient


# ------------------------------------------------------------------------------- (5) C2 absent from inference
def test_c2_decoder_and_targets_never_reach_the_inference_path():
    for mod in (E,):
        src = inspect.getsource(mod)
        assert not any(w in src for w in ("aux_losses", "StateDecoder", "state_targets", "state_loss"))
    for script in ("run_eval.py", "read_oracle.py"):
        with open(os.path.join(EXP, "code", script)) as f:
            src = f.read()
        assert not any(w in src for w in ("aux_losses", "StateDecoder", "state_targets", "state_decoder"))
    params = inspect.signature(E.evaluate).parameters
    assert not any(w in p for p in params for w in ("decoder", "target", "state"))
    sig = inspect.signature(LatentLM.generate).parameters
    assert set(sig) == {"self", "input_ids", "attention_mask", "position_ids", "max_new_tokens", "mode",
                        "scramble_bank", "return_latents", "time_steps"}
    assert T.DECODER_FILES["ckpt_best_final.pt"] != "ckpt_best_final.pt"   # the decoder is a separate file


# ------------------------------------------------------------------- (6) mode flag fix and evaluate's discipline
def test_wrapper_inherits_the_base_mode_and_evaluate_restores_it(tok, tmp_path):
    t, ids = tok
    model = tiny(n_layer=4, seed=3, dropout=0.1)            # .eval() with real dropout rates
    lm = lm_of(model, ids, (1, 3))
    assert lm.training is False and model.training is False
    b = batch_of(t, ids, (2, 2, 2))
    o1 = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"])
    o2 = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"])
    assert torch.equal(o1.logits, o2.logits)                # no dropout anywhere, band path included
    assert lm_of(model.train(), ids).training is True and lm_of(model.eval(), ids).training is False
    exs = [{"question": f"alex bob is a wumpus {i} ?", "steps": ["a b .", "c d ."], "answer": "x ."} for i in range(6)]
    with open(tmp_path / "toy.json", "w") as f:
        json.dump(exs, f)
    examples = D.load_examples(str(tmp_path / "toy.json"))
    D.tokenize_examples(examples, t)
    model = tiny(n_layer=4, seed=3, dropout=0.1, vocab=len(t) + 4)
    seen = []
    hook = model.transformer.h[0].ln_1.register_forward_hook(     # ln_1 runs in the HF path and the band path
        lambda m, a, o: seen.append((lm.training, m.training, torch.is_grad_enabled())))
    for band in ((1, 3), None):
        lm = lm_of(model.eval(), ids, band)
        for start in (False, True):
            lm.train(start)
            seen.clear()
            E.evaluate(lm, t, examples, arm="coconut", n_latent=2, batch_size=4, device="cpu", max_new_tokens=3)
            assert seen and all(s == (False, False, False) for s in seen)   # eval mode, no_grad throughout
            assert lm.training is start and model.training is start          # the base's own mode comes back
    hook.remove()


# ---------------------------------------------------------------------------- (7) C2-shuffled donor structure
def test_c2_shuffled_donors_on_screen_like_classes():
    counts = {i: 5 + i % 3 for i in range(300)}            # ProsQA-hard: depths 5, 6, 7 only
    d = S.shuffled_donors(counts, T.STATE_NULL_SEED + 0)
    assert all(d[i] != i and counts[d[i]] == counts[i] for i in counts) and sorted(d.values()) == list(range(300))
    assert d == S.shuffled_donors(counts, T.STATE_NULL_SEED + 0) != S.shuffled_donors(counts, T.STATE_NULL_SEED + 1)
    # train.py draws the map once per run, before the epoch loop, so record i keeps donor d[i] in every epoch
    src = inspect.getsource(T.train)
    assert src.index("shuffled_donors") < src.index("for epoch in range(start_epoch, cfg.num_epochs)")
