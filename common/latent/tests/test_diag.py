"""latent.diag and code/run_diag.py (registration 7.10 items 4 and 5; A11.3 (1); A12.3): the Recorder sees what the
backbone is fed and what generate emits; the scrambled pass's fed vectors are the donors' own latents at distance > 0,
and an identity "derangement" is BUG; batch isolation holds on CPU and a cross-row leak shows in the same-shape swap;
generation divergences carry position and margins; the script writes hashed files for both numerics modes and refuses
any split but valid. Tiny random GPT-2, CPU, fp32."""
import csv
import hashlib
import json

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent import diag
from latent import eval as E
from latent import train as T
from latent.model import LatentLM
from latent.tests.fake_tokenizer import FakeTokenizer

WORDS = "alex bob wumpus yumpus zumpus rompus tumpus lempus is a every not".split()
N, BS, NL, MNT = 10, 4, 3, 8


def _records(n=N, seed=0):
    rng = np.random.default_rng(seed)
    return [{"question": " ".join(rng.choice(WORDS, size=4 + i % 5)) + " ?", "steps": ["a b .", "c d ."],
             "answer": " ".join(rng.choice(WORDS, 2)) + " ."} for i in range(n)]


@pytest.fixture(autouse=True)
def _restore_tf32():
    """run_diag.py sets TF32 through run_train.set_tf32 (legacy and fp32_precision flags); later tests in this process
    set legacy flags only, which torch refuses to mix with a "tf32" fp32_precision: put back the starting state."""
    b = torch.backends
    mods = [m for m in (b.cuda.matmul, b.cudnn, getattr(b.cudnn, "conv", None), getattr(b.cudnn, "rnn", None))
            if m is not None and hasattr(m, "fp32_precision")]
    saved, mp = [(m, m.fp32_precision) for m in mods], torch.get_float32_matmul_precision()
    yield
    for m, v in saved:
        m.fp32_precision = v
    torch.set_float32_matmul_precision(mp)


@pytest.fixture(scope="module")
def setup():
    t = FakeTokenizer()
    ids = D.add_latent_tokens(t)
    ex = [D.Example(question=r["question"], answer=r["answer"], steps=r["steps"], idx=k)
          for k, r in enumerate(_records())]
    D.tokenize_examples(ex, t)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(t) + 4,
                                       n_positions=256)).eval()
    return t, ex, LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], t.eos_token_id)


def test_recorder_sees_the_fed_vectors_and_the_emitted_streams(setup):
    t, ex, lm = setup
    recs = diag.prompt_records(t, ex, arm="coconut", n_latent=NL)
    own = E.evaluate(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT,
                     return_latents=True)
    with diag.Recorder(lm) as rec:
        scr = E.evaluate(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT,
                         intervention="scrambled", return_latents=True)
    assert "generate" not in lm.__dict__ and not lm.backbone._forward_pre_hooks and not lm.lm_head._forward_hooks
    clean, fed = rec.by_idx(recs, BS, "coconut"), rec.by_idx(recs, BS, "scrambled")
    donors = E.scramble_donors({i: NL for i in clean}, seed=0)
    for r_own, r_scr in zip(own, scr):
        i = r_own.idx
        assert torch.equal(clean[i]["fed"], r_own.latents)              # the own pass: the bank's source
        assert torch.equal(fed[i]["fed"], r_scr.latents)                # what the backbone got = what eval collected
        assert torch.equal(fed[i]["fed"], clean[donors[i]]["fed"])      # = the donor's own latent, every rank
        for row in (clean[i], fed[i]):                                  # step s's argmax is the s-th emitted token
            assert row["top2_ids"][:, 0].tolist() == row["stream"] and row["covered_once"]
    # 7.10 item 6: one prefill plus one pass per latent rank under own feedback; one pass when scrambled
    assert {c["prompt_passes"] for c in rec.calls if c["mode"] == "coconut"} == {1 + NL}
    assert {c["prompt_passes"] for c in rec.calls if c["mode"] == "scrambled"} == {1}
    with pytest.raises(RuntimeError, match="order"):
        rec.by_idx(list(reversed(recs)), BS, "coconut")


def test_scramble_distances_are_positive_and_identity_donors_are_bug(setup, monkeypatch):
    t, ex, lm = setup
    rows, ranks, s = diag.scramble(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                   max_new_tokens=MNT)
    assert s["bug"] == [] and s["fed_vectors"]["verdict"] == "ok"
    assert len(ranks) == N * NL and s["fed_vectors"]["zero_distance"] == 0 and s["fed_vectors"]["min_l2"] > 0
    assert s["fed_vectors"]["fed_equals_donor"] == N == s["fed_vectors"]["fed_equals_collected"]
    assert all(r["donor"] != r["idx"] for r in rows)
    assert s["first_answer_logits"]["changed"] == N           # a random model's answer logits see the latents
    assert s["own_repeat"] == {"latents_max": 0.0, "first_logits_max": 0.0}
    with pytest.raises(ValueError, match="latent-feedback"):
        diag.scramble(lm, t, ex, arm="pause", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT)
    # a derangement that is the identity feeds every example its own latent: zero distance at every rank, BUG
    monkeypatch.setattr(E, "scramble_donors", lambda counts, seed=0: {k: k for k in counts})
    rows, ranks, s = diag.scramble(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                   max_new_tokens=MNT)
    assert s["fed_vectors"]["verdict"] == "BUG" and s["fed_vectors"]["zero_distance"] == N * NL
    assert any("distances are 0" in b for b in s["bug"]) and any("own donor" in b for b in s["bug"])
    assert s["first_answer_logits"]["changed"] == 0
    assert any("BUG" in f for f in diag.flags({"off": {**_empty_iso_gen(), "bug": s["bug"]}}, BS))


def _empty_iso_gen():
    return {"isolation": diag.isolation_counts([]),
            "generation": {k: {"diverging": 0, "n": 0} for k in diag.SETS}}


@pytest.mark.parametrize("arm,n_latent", [("cot", 0), ("coconut", NL)])
def test_isolation_holds_and_a_cross_row_leak_shows_in_the_swap(setup, arm, n_latent):
    t, ex, lm = setup
    perm = diag.permutation(N, 0)
    rows = diag.isolation(lm, t, ex, arm=arm, n_latent=n_latent, batch_size=BS, device="cpu", perm=perm)
    c = diag.isolation_counts(rows)
    assert [r["idx"] for r in rows] == list(range(N))
    for k in diag.ISO_KEYS:
        assert c[k]["over_tol"] == 0, k
    assert c["logits_swap"]["n"] == N                           # every row's companions replaced at its batch's shape
    assert c["latents_nat"]["n"] == (N if n_latent else 0)
    assert all(r["width_nat"] >= r["prompt_len"] for r in rows)
    # row j changes row i: a hook mixing the batch mean into every row (a BatchNorm-like leak)
    h = lm.base_causallm.transformer.ln_f.register_forward_hook(lambda m, i, o: o + 0.5 * o.mean(0, keepdim=True))
    try:
        leak = diag.isolation_counts(diag.isolation(lm, t, ex, arm=arm, n_latent=n_latent, batch_size=BS,
                                                    device="cpu", perm=perm))
    finally:
        h.remove()
    assert leak["logits_swap"]["over_tol"] == N and leak["last_nat"]["over_tol"] == N
    fl = diag.flags({"off": {"isolation": leak, "generation": {k: {"diverging": 0, "n": N} for k in diag.SETS}}}, BS)
    assert any("row j changes row i" in f for f in fl) and any(f.startswith("tf32 off: 10 of 10") for f in fl)


def test_permutation_keeps_record_0_first():
    p = diag.permutation(300, 0)
    assert p[0] == 0 and sorted(p) == list(range(300)) and p == diag.permutation(300, 0) != list(range(300))


def test_first_divergence():
    assert diag.first_divergence([1, 2, 3], [1, 2, 3]) is None
    assert diag.first_divergence([1, 2, 3], [1, 5, 3]) == 1
    assert diag.first_divergence([1, 2], [1, 2, 0]) == 2
    assert diag.first_divergence([], [7]) == 0


def test_generation_divergence_position_and_margins(setup):
    """A hook that favours one token at every decode step only when B > 1: the B = 1 and B = 4 streams part at the
    first decode step wherever B = 1 emits another token there, with a large margin at B = 4 only."""
    t, ex, lm = setup
    perm = diag.permutation(N, 0)
    rows, res = diag.generation(lm, t, ex, arm="cot", n_latent=0, batch_size=BS, device="cpu", max_new_tokens=MNT,
                                perm=perm)
    g = diag.generation_counts(rows, res)
    assert g["nat"]["diverging"] == 0 == g["perm"]["diverging"] and g["hit_cap"]["b1"] == N
    assert g["correctness"]["b1"]["n"] == N and "accuracy_ci95" not in g["correctness"]["b1"]
    assert "accuracy_ci95" in diag.generation_counts(rows, res, n_boot=50)["correctness"]["nat"]
    tok = t.tok_to_id["bob"]

    def bump(m, i, o):
        if o.shape[0] > 1 and o.shape[1] == 1:
            o = o.clone()
            o[:, -1, tok] += 100.0
        return o
    h = lm.lm_head.register_forward_hook(bump)
    try:
        rows, res = diag.generation(lm, t, ex, arm="cot", n_latent=0, batch_size=BS, device="cpu",
                                    max_new_tokens=MNT, perm=perm)
    finally:
        h.remove()
    g = diag.generation_counts(rows, res)
    assert g["nat"]["diverging"] > 0
    for d in g["nat"]["divergences"]:
        assert d["pos"] == 1 and d["margin_batch"] > 50 > d["margin_b1"] >= 0
        assert int(d["top2_batch"].split("|")[0]) == tok and int(d["top2_b1"].split("|")[0]) != tok
        assert d["width_batch"] >= d["prompt_len"] > 0
    assert any("diverge" in f for f in diag.flags({"off": {"isolation": diag.isolation_counts([]), "generation": g}},
                                                  BS))


# --------------------------------------------------------------------------------------------- run_diag.py
@pytest.fixture()
def diagenv(tmp_path, monkeypatch):
    from latent.tests import test_screen_tools as sct
    t, model, ck = sct._fixture(tmp_path, n=8)

    def fake_load_base(model_id, dtype=None, device="cpu", resize=True):
        m = GPT2LMHeadModel(model.config)
        m.load_state_dict(model.state_dict())
        tok = FakeTokenizer()
        tok.id_to_tok, tok.tok_to_id = list(t.id_to_tok), dict(t.tok_to_id)
        return m, tok
    monkeypatch.setattr(T, "load_base", fake_load_base)
    mod = sct._mod("run_diag")

    def cfg(name, arm, **kw):
        d = dict(name=name, arm=arm, seed=0, model_id="toy", max_new_tokens=8,
                 train_path=str(tmp_path / "toy_train.json"), val_path=str(tmp_path / "toy_valid.json"),
                 eval_batch_size=3, device="cpu", save_dir=str(tmp_path / name), c_thought=1 if arm == "coconut" else 0,
                 max_latent_stage=2 if arm == "coconut" else 0, **kw)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        return str(p)
    return mod, cfg, str(ck), tmp_path


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_run_diag_cot_batch_writes_both_numerics_modes(diagenv):
    mod, cfg, ck, tmp = diagenv
    out = tmp / "d1"
    s = mod.main(["--config", cfg("prosqa_cot", "cot"), "--checkpoint", ck, "--mode", "cot_batch", "--out", str(out),
                  "--set", "seed=1", "--set", 'name="prosqa_cot_s1"', "--commit", "c0ffee"])
    assert json.loads((out / "summary.json").read_text()) == json.loads(json.dumps(s))
    assert (s["mode"], s["split"], s["run"], s["seed"], s["commit"], s["registered_read"]) == (
        "cot_batch", "valid", "prosqa_cot_s1", 1, "c0ffee", None)
    assert s["tf32_modes"] == ["off", "on"] and set(s["results"]) == set(s["numerics"]) == {"off", "on"}
    assert s["numerics"]["off"]["tf32_requested"] is False and s["numerics"]["on"]["tf32_requested"] is True
    assert s["split_file"].endswith("toy_valid.json") and s["split_sha256"] == _sha(tmp / "toy_valid.json")
    assert s["checkpoint_sha256"] == _sha(ck) and s["files"] == {"examples.csv": _sha(out / "examples.csv")}
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert len(rows) == 2 * 8 and {r["tf32"] for r in rows} == {"off", "on"}
    assert {"idx", "logits_nat", "last_perm", "logits_swap", "diverges_nat", "margin_b1_nat", "correct_b1",
            "correct_nat"} <= set(rows[0])
    assert not s["bug"] and s["batch_size"] == 3 and s["n"] == 8 and s["n_latent"] == 0
    g = s["results"]["on"]["generation"]
    assert g["nat"]["n"] == 8 and set(g["correctness"]) == {"b1", "nat", "perm"}
    one = mod.main(["--config", cfg("prosqa_cot", "cot"), "--checkpoint", ck, "--mode", "cot_batch", "--out",
                    str(tmp / "d2"), "--tf32", "off", "--batch-size", "4"])
    assert one["tf32_modes"] == ["off"] and one["batch_size"] == 4 and one["commit"] == mod.git_commit()


def test_run_diag_latent_path_writes_fed_vectors(diagenv):
    mod, cfg, ck, tmp = diagenv
    out = tmp / "l1"
    s = mod.main(["--config", cfg("prosqa_coconut", "coconut"), "--checkpoint", ck, "--mode", "latent_path",
                  "--out", str(out), "--tf32", "on"])
    assert s["n_latent"] == 2 and s["scramble_seed"] == 0 and s["bug"] is False
    sc = s["results"]["on"]["scramble"]
    assert sc["fed_vectors"]["zero_distance"] == 0 and sc["fed_vectors"]["verdict"] == "ok"
    ranks = list(csv.DictReader(open(out / "fed_vectors.csv")))
    assert len(ranks) == 8 * 2 and all(float(r["l2"]) > 0 for r in ranks)
    assert s["files"] == {"examples.csv": _sha(out / "examples.csv"), "fed_vectors.csv": _sha(out / "fed_vectors.csv")}
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert {"donor", "fed_min_l2", "first_maxabs_own_scr", "latents_nat", "latents_swap",
            "diverges_perm"} <= set(rows[0])


def test_run_diag_refuses_test_and_wrong_arm(diagenv):
    mod, cfg, ck, tmp = diagenv
    c = cfg("prosqa_cot", "cot")
    for split in ("test", "train", "VALID"):
        with pytest.raises(SystemExit, match="REFUSED"):
            mod.main(["--config", c, "--checkpoint", ck, "--mode", "cot_batch", "--split", split])
    with pytest.raises(SystemExit, match="latent-feedback"):
        mod.main(["--config", c, "--checkpoint", ck, "--mode", "latent_path", "--out", str(tmp / "x")])
    with pytest.raises(SystemExit, match="_valid.json"):
        mod.main(["--config", c, "--checkpoint", ck, "--mode", "cot_batch",
                  "--set", f'val_path="{tmp / "toy_test.json"}"'])
    assert not (tmp / "x").exists()
