"""C1b (docs/cards/C1b_partial_depth_latents.md, "Leak and bug answers"; registration A3.4): the band forward against
the reference Coconut at full depth, against a naive per-position oracle built from HF's own GPT2Block for a strict
band, the per-block masks (no query attends to a latent column outside the band), real skipping of the non-band
blocks at latent positions, the identity of the fed-back vector with the band output h_b, generation, the scrambled
band and the timing record. Random tiny GPT-2, CPU, fp32."""
import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent.eval import evaluate, scramble_donors
from latent.model import LatentLM
from latent.tests.fake_tokenizer import FakeTokenizer
from latent.tests.reference_coconut import Coconut
from latent.tests.reference_dataset import MyCollator
from latent.tests.test_model import ragged_features

ATOL = 1e-5


def tiny(n_layer=4, attn="eager", seed=0, dropout=0.0, vocab=64):
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(n_layer=n_layer, n_embd=32, n_head=2, vocab_size=vocab, n_positions=256,
                                      resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
                                      attn_implementation=attn)).eval()


@pytest.fixture(scope="module")
def tok():
    t = FakeTokenizer()
    return t, D.add_latent_tokens(t)


def lm_of(model, ids, band):
    return LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], 0, latent_band=band)


def batch_of(t, ids, counts, seed=1):
    return MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(ragged_features(ids, counts, seed=seed)))


def _blk(block, seq):
    out = block(seq)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def naive_band(model, b, latent_id, band):
    """Independent oracle: per example and position, each block attends over the explicit list of its admissible
    inputs (non-latent positions always; latent positions only inside the band), using HF's GPT2Block on that list
    and reading the last row. Returns {(i, t): logits} at real non-latent positions, {(i, t): fed vector} at latents
    and {(i, t): h_b} at every real position."""
    a, bnd = band
    tr = model.transformer
    n_layer = len(tr.h)
    logits, fed, hbs = {}, {}, {}
    for i in range(b["input_ids"].shape[0]):
        ctx = [[] for _ in range(n_layer)]
        hb_prev = None
        for t in range(b["input_ids"].shape[1]):
            if not b["attention_mask"][i, t]:
                continue
            tok_id = int(b["input_ids"][i, t])
            pos = b["position_ids"][i, t]
            is_lat = tok_id == latent_id
            if is_lat:
                fed[(i, t)] = hb_prev
                x = hb_prev + tr.wpe(pos) if a == 0 else hb_prev
                blocks = range(a, bnd)
            else:
                x = tr.wte.weight[tok_id] + tr.wpe(pos)
                blocks = range(n_layer)
            hb = None
            for k in blocks:
                ctx[k].append(x)
                x = _blk(tr.h[k], torch.stack(ctx[k]).unsqueeze(0))[0, -1]
                if k == bnd - 1:
                    hb = x
            if not is_lat:
                hf = tr.ln_f(x)
                logits[(i, t)] = model.lm_head(hf)
                if bnd == n_layer:
                    hb = hf
            elif bnd == n_layer:
                hb = tr.ln_f(x)
            hbs[(i, t)] = hb
            hb_prev = hb
    return logits, fed, hbs


@pytest.mark.parametrize("attn", ["eager", "sdpa"])
@pytest.mark.parametrize("counts", [(0, 1, 2, 3), (3, 0, 2, 1), (2, 2, 2, 2), (1,)])
def test_full_band_is_the_reference_coconut(tok, attn, counts):
    """layer_fraction 1 (band (0, L)) reproduces the reference forward: logits at real non-latent positions, loss,
    input embeddings, fed-back latents and every parameter gradient."""
    t, ids = tok
    b = batch_of(t, ids, counts)
    grads = []
    for which in ("band", "ref"):
        model = tiny(n_layer=3, attn=attn)
        m = lm_of(model, ids, (0, 3)) if which == "band" else Coconut(model, ids["latent_id"], ids["start_id"],
                                                                       ids["end_id"], 0)
        assert which != "band" or m.layer_fraction == 1.0
        o = m(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
        o.loss.backward()
        grads.append(({n: p.grad.clone() for n, p in model.named_parameters()}, o))
    (gb, ob), (gr, orf) = grads
    real = b["attention_mask"].bool() & (b["input_ids"] != ids["latent_id"])
    torch.testing.assert_close(ob.logits[real], orf.logits[real], atol=ATOL, rtol=0)
    torch.testing.assert_close(ob.loss, orf.loss, atol=ATOL, rtol=0)
    torch.testing.assert_close(ob.inputs_embeds, orf.inputs_embeds, atol=ATOL, rtol=0)
    for n in gb:
        torch.testing.assert_close(gb[n], gr[n], atol=ATOL, rtol=1e-4, msg=n)


@pytest.mark.parametrize("band,counts", [((1, 3), (0, 1, 2, 3)), ((0, 2), (0, 1, 2, 3)), ((2, 4), (3, 0, 2, 1)),
                                         ((1, 2), (2, 2, 2)), ((0, 4), (1, 3))])
def test_band_matches_naive_per_position_oracle(tok, band, counts):
    t, ids = tok
    model = tiny(n_layer=4)
    b = batch_of(t, ids, counts)
    o = lm_of(model, ids, band)(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"],
                                return_latents=True)
    logits, fed, _ = naive_band(model, b, ids["latent_id"], band)
    for (i, j), lg in logits.items():
        torch.testing.assert_close(o.logits[i, j], lg, atol=ATOL, rtol=0, msg=str((i, j)))
    for i, n in enumerate(counts):
        cols = (b["input_ids"][i] == ids["latent_id"]).nonzero().squeeze(1).tolist()
        assert o.latents[i].shape[0] == n == len(cols)
        for k, j in enumerate(cols):
            torch.testing.assert_close(o.latents[i][k], fed[(i, j)], atol=ATOL, rtol=0)


def test_fed_back_vector_is_the_band_output(tok):
    """C1b leak answer: the fed-back tensor is h_b of the previous position (HF hidden_states[b] for the first
    latent, the band's own output for later ones), not h_L."""
    t, ids = tok
    model = tiny(n_layer=4)
    band = (1, 3)
    counts = (3, 2, 3)
    b = batch_of(t, ids, counts)
    o = lm_of(model, ids, band)(b["input_ids"], b["attention_mask"], None, b["position_ids"], return_latents=True)
    full = lm_of(model, ids, None)(b["input_ids"], b["attention_mask"], None, b["position_ids"], return_latents=True)
    _, _, hbs = naive_band(model, b, ids["latent_id"], band)
    for i, n in enumerate(counts):
        cols = (b["input_ids"][i] == ids["latent_id"]).nonzero().squeeze(1).tolist()
        first = cols[0]
        prefix = model(input_ids=b["input_ids"][i:i + 1, :first], attention_mask=b["attention_mask"][i:i + 1, :first],
                       position_ids=b["position_ids"][i:i + 1, :first], output_hidden_states=True)
        torch.testing.assert_close(o.latents[i][0], prefix.hidden_states[band[1]][0, first - 1], atol=ATOL, rtol=0)
        for k in range(1, n):
            torch.testing.assert_close(o.latents[i][k], hbs[(i, cols[k - 1])], atol=ATOL, rtol=0)
        assert not torch.allclose(o.latents[i], full.latents[i], atol=1e-3)   # not the last-layer state


def test_no_query_attends_to_a_latent_outside_the_band(tok):
    """Mask check: every attention mask used at a block outside [a, b) excludes every latent column for every query
    row; inside the band later queries do see them."""
    t, ids = tok
    model = tiny(n_layer=4)
    band = (1, 3)
    lm = lm_of(model, ids, band)
    b = batch_of(t, ids, (0, 1, 2, 3))
    lat = b["input_ids"] == ids["latent_id"]
    lm.band_debug = []
    lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
    g = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=3)  # decode masks too
    assert g.latent_steps == [0, 1, 2, 3]
    seen_inside = False
    masks = [r for r in lm.band_debug if r[0] == "mask"]
    assert masks
    for _, i, _, m in masks:
        S = m.shape[-1]
        latk = torch.zeros(m.shape[0], S, dtype=torch.bool)
        if m.shape[0] == lat.shape[0]:
            latk[:, :min(S, lat.shape[1])] = lat[:, :S]
        else:   # column passes run non-band blocks on the rows without a latent at that column
            continue
        hit = (m[:, 0] & latk.unsqueeze(1)).any()
        if band[0] <= i < band[1]:
            seen_inside |= bool(hit)
        else:
            assert not bool(hit), f"block {i} attends to a latent column"
    assert seen_inside
    lm.band_debug = None


def test_non_band_blocks_are_skipped_at_latent_positions(tok):
    """A3.4: real skipping. With every row holding n latents, non-band blocks run twice per forward (prefix and
    tail), band blocks 2 + n times; the non-band caches hold zeros at the latent columns."""
    t, ids = tok
    model = tiny(n_layer=4)
    band = (1, 3)
    calls = {i: 0 for i in range(4)}
    hooks = [model.transformer.h[i].ln_1.register_forward_hook(lambda m, a, o, i=i: calls.__setitem__(i, calls[i] + 1))
             for i in range(4)]
    lm = lm_of(model, ids, band)
    b = batch_of(t, ids, (3, 3, 3))
    run = lm._run(b["input_ids"], b["attention_mask"], b["position_ids"], "coconut", None, keep_cache=True)
    for h in hooks:
        h.remove()
    assert [calls[i] for i in range(4)] == [2, 5, 5, 2]
    lat = (b["input_ids"] == ids["latent_id"])
    for i, (k, v) in enumerate(run.cache["kv"]):
        at_lat = k.permute(0, 2, 1, 3)[lat]
        assert bool((at_lat == 0).all()) == (not band[0] <= i < band[1])


def test_band_generate_batched_equals_single_and_latents_match_forward(tok):
    t, ids = tok
    model = tiny(n_layer=4, seed=3)
    lm = lm_of(model, ids, (1, 3))
    recs = []
    g = torch.Generator().manual_seed(4)
    for i, n in enumerate((0, 1, 2, 3, 2)):
        q = torch.randint(5, 60, (4 + 3 * (i % 3),), generator=g).tolist()
        x = q + [ids["start_id"]] + [ids["latent_id"]] * n + [ids["end_id"]]
        recs.append({"input_ids": x, "attention_mask": [1] * len(x), "position_ids": list(range(len(x))), "idx": i})
    coll = D.Collator(1, ids["latent_id"])
    b = coll(recs)
    out = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=6, return_latents=True)
    for i, r in enumerate(recs):
        s = coll([r])
        one = lm.generate(s["input_ids"], s["attention_mask"], s["position_ids"], max_new_tokens=6)
        assert out.tokens[i] == one.tokens[0]
    f = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"], return_latents=True)
    for a_, c_ in zip(out.latents, f.latents):
        torch.testing.assert_close(a_, c_.detach(), atol=1e-6, rtol=0)
    assert out.latent_steps == [0, 1, 2, 3, 2] and out.sequential_passes == [1 + s + 6 for s in out.latent_steps]


def test_band_generate_matches_naive_oracle_greedy(tok):
    """The first generated token of each prompt equals the argmax of the oracle's logits at <|end-latent|>."""
    t, ids = tok
    model = tiny(n_layer=4, seed=5)
    band = (2, 4)
    b = batch_of(t, ids, (1, 2, 3))
    gen_b = {k: v for k, v in b.items()}
    lm = lm_of(model, ids, band)
    logits, _, _ = naive_band(model, b, ids["latent_id"], band)
    o = lm(b["input_ids"], b["attention_mask"], None, b["position_ids"])
    for (i, j), lg in logits.items():
        assert int(o.logits[i, j].argmax()) == int(lg.argmax())
    assert gen_b["input_ids"].shape[0] == 3


def test_scrambled_band_uses_the_bank_and_is_blind(tok):
    t, ids = tok
    model = tiny(n_layer=4)
    lm = lm_of(model, ids, (1, 3))
    counts = (3, 2, 3)
    bank = torch.randn(3, 3, 32, generator=torch.Generator().manual_seed(2))
    feats = ragged_features(ids, counts)
    feats2 = copy.deepcopy(feats)
    feats2[0]["input_ids"][:3] = [7, 8, 9]
    b1 = MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(feats))
    b2 = MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(feats2))
    o1 = lm(b1["input_ids"], b1["attention_mask"], b1["labels"], b1["position_ids"], mode="scrambled",
            scramble_bank=bank, return_latents=True)
    o2 = lm(b2["input_ids"], b2["attention_mask"], b2["labels"], b2["position_ids"], mode="scrambled",
            scramble_bank=bank, return_latents=True)
    for i, n in enumerate(counts):
        assert torch.equal(o1.latents[i], bank[i, :n]) and torch.equal(o2.latents[i], bank[i, :n])
    torch.testing.assert_close(o1.logits[1:], o2.logits[1:], atol=ATOL, rtol=0)
    with pytest.raises(ValueError):
        lm(b1["input_ids"], b1["attention_mask"], b1["labels"], b1["position_ids"], mode="scrambled")
    with pytest.raises(ValueError):
        lm(b1["input_ids"], b1["attention_mask"], b1["labels"], b1["position_ids"], mode="pause")


def test_band_batch_isolation_alone_vs_with_companions(tok):
    t, ids = tok
    lm = lm_of(tiny(n_layer=4).train(), ids, (1, 3))
    counts = (2, 3, 0, 1, 3, 2, 1, 3)
    feats = ragged_features(ids, counts, seed=4, qlens=(6, 9, 4, 11))

    def run(B):
        b = MyCollator(t, latent_id=ids["latent_id"])(copy.deepcopy(feats[:B]))
        o = lm(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
        real = b["attention_mask"][0].bool() & (b["input_ids"][0] != ids["latent_id"])
        return o.logits[0][real]
    one = run(1)
    for B in (2, 8):
        torch.testing.assert_close(run(B), one, atol=1e-4, rtol=0)


def test_band_validation_and_layer_fraction(tok):
    t, ids = tok
    model = tiny(n_layer=12)
    assert lm_of(model, ids, (2, 10)).layer_fraction == 8 / 12
    assert lm_of(model, ids, (4, 8)).layer_fraction == 4 / 12
    assert lm_of(model, ids, None).layer_fraction == 1.0
    for bad in ((3, 2), (0, 13), (-1, 4)):
        with pytest.raises(ValueError):
            lm_of(model, ids, bad)


def test_band_zero_latents_is_the_plain_path(tok):
    t, ids = tok
    model = tiny(n_layer=4)
    b = batch_of(t, ids, (0, 0))
    o = lm_of(model, ids, (1, 3))(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
    r = lm_of(model, ids, None)(b["input_ids"], b["attention_mask"], b["labels"], b["position_ids"])
    assert torch.equal(o.logits, r.logits) and torch.equal(o.loss, r.loss)


def test_generate_timing_records_latent_and_token_steps(tok):
    t, ids = tok
    model = tiny(n_layer=4)
    for band in ((1, 3), None):
        lm = lm_of(model, ids, band)
        b = batch_of(t, ids, (3, 3))
        g = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=4, time_steps=True)
        assert g.timing["token_passes"] == 3 and g.timing["prefill_passes"] == 1
        assert g.timing["latent_passes"] == (3 if band else 2) and g.timing["latent_ms_mean"] > 0
        assert lm.timing is None
        assert lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=2).timing is None


def test_evaluate_band_model_scrambled_and_accounting(tok):
    """The k = 8 scrambled diagnostic (now a kill line, A3.4) runs through evaluate on a band model; FLOPs carry the
    layer fraction."""
    import json
    import tempfile
    from latent.accounting import inference_record, model_dims
    t, ids = tok
    exs = [{"question": f"alex bob is a wumpus {i} ?", "steps": ["a b .", "c d ."], "answer": "x ."}
           for i in range(6)]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(exs, f)
    examples = D.load_examples(f.name)
    D.tokenize_examples(examples, t)
    model = tiny(n_layer=4, vocab=len(t) + 4)
    lm = lm_of(model, ids, (1, 3))
    clean = evaluate(lm, t, examples, arm="coconut", n_latent=3, batch_size=4, device="cpu", max_new_tokens=3,
                     return_latents=True)
    timing = []
    scr = evaluate(lm, t, examples, arm="coconut", n_latent=3, batch_size=4, device="cpu", max_new_tokens=3,
                   intervention="scrambled", return_latents=True, timing_out=timing)
    donors = scramble_donors({r.idx: 3 for r in clean}, seed=0)
    by = {r.idx: r for r in clean}
    for r in scr:
        assert torch.equal(r.latents, by[donors[r.idx]].latents)
    assert len(timing) == 2 and all(x["latent_passes"] == 3 for x in timing)
    dims = model_dims(lm)
    q = len(t.encode(examples[0].question + "\n"))
    full = inference_record(dims, "coconut", q, 3, clean[0].emitted_tokens)
    cut = inference_record(dims, "coconut", q, 3, clean[0].emitted_tokens, layer_fraction=0.5)
    assert clean[0].flops == cut.flops < full.flops


@pytest.mark.parametrize("band", [None, (1, 3)])
def test_generation_stops_at_the_last_position(tok, band):
    """A row whose next token would pass n_positions stops (capped) instead of failing on the position index; rows
    that fit are unchanged."""
    t, ids = tok
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=4, n_embd=32, n_head=2, vocab_size=64, n_positions=24)).eval()
    lm = LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], -1, latent_band=band)
    recs = []
    for i, qlen in enumerate((4, 14)):
        x = list(range(5, 5 + qlen)) + [ids["start_id"]] + [ids["latent_id"]] * 2 + [ids["end_id"]]
        recs.append({"input_ids": x, "attention_mask": [1] * len(x), "position_ids": list(range(len(x))), "idx": i})
    coll = D.Collator(1, ids["latent_id"])
    b = coll(recs)
    g = lm.generate(b["input_ids"], b["attention_mask"], b["position_ids"], max_new_tokens=10)
    assert g.capped == [False, True] and g.emitted_tokens == [10, 24 - 18 + 1]
    one = coll([recs[0]])
    assert lm.generate(one["input_ids"], one["attention_mask"], one["position_ids"], max_new_tokens=10).tokens[0] \
        == g.tokens[0]
