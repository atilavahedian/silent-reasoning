"""latent.diag's donor-answer read and code/run_diag.py --mode donor_answer (A6.1 rule 4 (B)'s directional read on
GSM-50k; design note governance/designs/a6_1_directional_read.md, rev230; registered by critic entry 2026-10-20T17:36Z,
whose part (4) names these tests): T1 a hand example of the pure scorer (class, ambiguous pairs, the counts, the three
garbage readings, the 6.2-form interval against an independent implementation on the same draws); T2 the class rule
reads the donor's record only; T3 null blindness (the donor column is the registered derangement, the fed vectors
are the donors' own latents bit for bit, a self / own / non-donor vector is a BUG with no pass/fail line, the scorer
with two donor maps gives two hand-checked statistics); T4 batch isolation of the read's per-example columns; T5 the
identification check; T6 refusals and write-once; T7 determinism; T8 the data pin on the valid file of record (skipped
without it). Tiny random GPT-2 and FakeTokenizer, CPU."""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from latent import data as D
from latent import diag
from latent import eval as E
from latent import train as T
from latent.model import LatentLM
from latent.tests import test_diag as TD
from latent.tests.fake_tokenizer import FakeTokenizer

_restore_tf32 = TD._restore_tf32          # run_diag.py sets TF32 flags: put the starting state back after each test
ROW = "2026-10-20T17:36Z"
N, BS, NL, MNT = 12, 4, 6, 8              # 12 examples, eval batch 4, 6 latents (c_thought 2 x 3 stages), 8 tokens
WORDS = "alex bob wumpus yumpus zumpus rompus tumpus lempus".split()
EXP = Path(__file__).resolve().parents[3] / "experiments" / "2026-10-17-silent-reasoning"
VALID = str(EXP / "data" / "gsm_valid.json")


def _sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ------------------------------------------------------------------------------------------- T1 to T3 (pure)
HAND = [   # (steps, answer): idx 0 a 1-step donor in the class; 1 a 4-step donor whose step 3 (7) is not its answer;
    (["<<2+3=5>>"], "5"),                                                   # 6 a 4-step donor in the class (step 3
    (["<<1+1=2>>", "<<2*3=6>>", "<<6+1=7>>", "<<7*2=14>>"], "14"),          # = 16 = its answer); 3 a step not of the
    (["<<4*2=8>>", "<<8-1=7>>", "<<7*3=21>>"], "21"),                       # <<expr=c>> form; 5 commas normalised;
    (["<<3+3=6>>", "6 apples"], "6"),                                       # 7 shares 2's answer (an ambiguous pair)
    (["<<10/2=5>>", "<<5+4=9>>"], "9"),
    (["<<1,000+1=1,001>>"], "1,001"),
    (["<<2+2=4>>", "<<4+4=8>>", "<<8*2=16>>", "<<16+0=16>>"], "16"),
    (["<<20+1=21>>"], "21")]
DONORS = {0: 4, 1: 0, 2: 5, 3: 6, 4: 1, 5: 3, 6: 7, 7: 2}                  # a derangement (no fixed point)
PRED_OWN = {0: "5", 1: "3", 2: "1001", 3: "6", 4: "", 5: "1001", 6: "16", 7: "21"}
PRED_SCR = {0: "9", 1: "5", 2: "1001", 3: "16", 4: "9", 5: "1001", 6: "4", 7: "21"}


def _hand_examples(edit=None):
    recs = [list(x) for x in HAND]
    if edit:
        edit(recs)
    return [D.Example(question=f"q{i} ?", answer=a, steps=list(s), idx=i) for i, (s, a) in enumerate(recs)]


def _results(examples, preds):
    out = []
    for ex in examples:
        g = D.answer_normalize(ex.answer)
        out.append(E.ExampleResult(idx=ex.idx, correct=preds[ex.idx] == g, pred=preds[ex.idx], gold=g, text="x",
                                   prefill_tokens=0, latent_steps=NL, emitted_tokens=2, sequential_passes=0,
                                   flops=0.0))
    return out


def _indep_interval(d, n_boot=10_000, seed=0):
    """An independent 6.2-form interval: the same default_rng(seed) draws in ten chunks of 1000 rows."""
    g = np.random.default_rng(seed)
    m = len(d)
    means = np.concatenate([d[g.integers(0, m, size=(1000, m))].mean(1) for _ in range(n_boot // 1000)])
    se = float(np.sqrt(means.var(ddof=1)))
    return float(d.mean()) - 1.96 * se, float(d.mean()) + 1.96 * se, np.percentile(means, [2.5, 97.5])


def test_t1_hand_example():
    ex = _hand_examples()
    rows = diag.donor_answer_rows(ex, _results(ex, PRED_OWN), _results(ex, PRED_SCR), DONORS)
    by = {r["idx"]: r for r in rows}
    # the class: donors 4, 0 (1 step), 5 (commas), 6 (4 steps, step 3 = answer), 7, 2 are in; 1 (step 3 = 7, not
    # 14) and 3 (step 2 not of the form) are out; 7's donor 2 has 7's own answer 21: ambiguous, excluded
    assert [r["in_class"] for r in rows] == [True, True, True, True, False, False, True, True]
    assert [r["ambiguous"] for r in rows] == [False] * 7 + [True]
    assert [r["idx"] for r in rows if r["scored"]] == [0, 1, 2, 3, 6]
    assert (by[0]["donor_n_steps"], by[0]["donor_last_sup_step"]) == (2, 2)
    assert (by[1]["donor_n_steps"], by[1]["donor_last_sup_step"]) == (1, 1)            # a 1-step donor: step 1
    assert (by[3]["donor_n_steps"], by[3]["donor_last_sup_step"]) == (4, 3)            # a 4-step donor: step 3
    assert by[4]["donor_last_sup_step"] == 3 and not by[4]["in_class"]
    # per example (S = 0, 1, 2, 3, 6): d_scr 1 1 1 1 0; d_own 0 0 1 0 0; changed 1 1 0 1 1; g_chg 0 0 0 0 1
    S = [by[i] for i in (0, 1, 2, 3, 6)]
    assert [r["d_scr"] for r in S] == [True, True, True, True, False]
    assert [r["d_own"] for r in S] == [False, False, True, False, False]
    assert [r["changed"] for r in S] == [True, True, False, True, True]
    assert [r["g_chg"] for r in S] == [False, False, False, False, True]
    assert [r["g_scr"] for r in S] == [False, False, False, False, True]
    assert [r["g_own"] for r in S] == [False, True, False, False, False]
    assert [r["correct_own"] for r in S] == [True, False, False, True, True] and not any(r["correct_scr"] for r in S)
    assert by[0]["scr_eq_any_donor_step"] and by[2]["scr_eq_any_donor_step"] and not by[6]["scr_eq_any_donor_step"]
    assert by[7]["donor_gold_in_own_steps"] and sum(r["donor_gold_in_own_steps"] for r in rows if r["in_class"]) == 1
    s = diag.donor_answer_summary(rows)
    assert (s["n"], s["n_class"], s["n_ambiguous"], s["ambiguous_idx"], s["n_scored"]) == (8, 6, 1, [7], 5)
    assert (s["k_dir"], s["k_own"], s["k_changed"], s["k_g_chg"], s["k_g_scr"], s["k_g_own"]) == (4, 1, 4, 1, 1, 1)
    assert (s["r_dir"], s["r_own"], s["delta_dir"], s["g_chg"]) == (0.8, 0.2, 3 / 5, 0.2)
    assert (s["acc_own_S"], s["acc_scr_S"], s["drop_S"], s["delta_g"], s["k_net"]) == (0.6, 0.0, 0.6, 0.0, 3)
    assert s["delta_g"] == pytest.approx(s["drop_S"] - s["delta_dir"], abs=1e-15)   # ruling 3's identity on S
    assert s["g_scr"] == pytest.approx(1 - s["acc_scr_S"] - s["r_dir"], abs=1e-15)    # the gold-literal form
    assert not s["garbage_exceeds"] and not s["garbage_literal_exceeds"] and not s["garbage_excess_exceeds"]
    assert s["garbage_labels"]["literal"].startswith("gold-literal reading: 1 minus acc_scr(S) minus R_dir")
    assert (s["acc_own"], s["acc_scr"], s["k_acc_own"], s["k_acc_scr"]) == (5 / 8, 3 / 8, 5, 3)   # over all 8
    assert s["drop"] == 2 / 8
    assert (s["complement"]["n"], s["complement"]["k_dir"], s["complement"]["k_correct_scr"]) == (2, 0, 2)
    assert s["secondary"]["donor_gold_in_own_steps"] == 1
    for tag, d in (("", [1, 1, 0, 1, 0]), ("_g", [0, -1, 0, 0, 1]), ("_chg", [-1, -1, -1, -1, 1])):
        lo, hi, pct = _indep_interval(np.array(d, float))
        assert s[f"lo{tag}"] == pytest.approx(lo, abs=1e-12) and s[f"hi{tag}"] == pytest.approx(hi, abs=1e-12)
        assert (s[f"pct_lo{tag}"], s[f"pct_hi{tag}"]) == pytest.approx(tuple(pct), abs=1e-12)
    assert s["lo"] > 0 and diag.donor_answer_outcome(s, [], "passed") == "directional read present" and s["thin"]
    assert s["pass_line"] == "directional read present iff Delta_dir > 0 and the lower bound > 0"
    # garbage exceeded: two more scrambled predictions that changed and are not the donor's (G_chg 3 > R_dir 2 of 5)
    scr2 = {**PRED_SCR, 0: "8", 1: "7"}
    s2 = diag.donor_answer_summary(diag.donor_answer_rows(ex, _results(ex, PRED_OWN), _results(ex, scr2), DONORS))
    assert (s2["k_dir"], s2["k_g_chg"], s2["garbage_exceeds"]) == (2, 3, True)
    assert (s2["k_g_scr"], s2["garbage_literal_exceeds"]) == (3, True)
    assert s2["garbage_excess_exceeds"] == ((s2["k_g_scr"] - s2["k_g_own"]) > (s2["k_dir"] - s2["k_own"]))
    # outcome words: BUG first, then held, then the pass line on the 6.2-form interval
    assert diag.donor_answer_outcome(s, ["x"], "held") == "BUG"
    assert diag.donor_answer_outcome(s, [], "held") == "held: identification check failed"
    assert diag.donor_answer_outcome({**s, "lo": -0.01}, [], "passed") == "directional read absent (interval includes 0)"
    assert diag.donor_answer_outcome({**s, "k_dir": 1}, [], None) == "directional read absent (interval includes 0)"
    assert diag.donor_answer_outcome({**s, "lo": None}, [], "passed") == "directional read absent (interval includes 0)"
    assert diag.donor_answer_summary(rows) == s                          # deterministic


def test_t2_the_class_rule_reads_the_donor_record_only():
    from latent.state_targets import answer_in_last_target, record_targets
    one, three, four = (D.Example(question="q", answer=a, steps=s, idx=0) for s, a in (
        (["<<1+1=2>>"], "2"), (["<<1+1=2>>", "<<2+2=4>>", "<<4*3=12>>"], "12"),
        (["<<1+1=2>>", "<<2+2=4>>", "<<4*3=12>>", "<<12+1=13>>"], "13")))
    assert diag.donor_in_class(one) and diag.donor_in_class(three) and not diag.donor_in_class(four)
    assert diag.donor_in_class(D.Example(question="q", answer="12", steps=four.steps, idx=0))   # step 3 = 12
    assert not diag.donor_in_class(D.Example(question="q", answer="2", steps=[], idx=0))          # nothing supervised
    for e in (one, three, four):
        assert diag.donor_in_class(e) == bool(answer_in_last_target(record_targets(e, FakeTokenizer()), e.answer, 3))
    with pytest.raises(ValueError, match="GSM"):
        diag.donor_in_class(D.Example(question="q", answer="a", steps=["s"], idx=0,
                                      meta={"edges": [[0, 1]], "root": 0, "idx_to_symbol": ["a", "b"]}))
    ex = _hand_examples()
    base = {r["idx"]: r["in_class"] for r in diag.donor_answer_rows(ex, _results(ex, PRED_OWN), _results(ex, PRED_SCR),
                                                                      DONORS)}

    def edit(recs):          # record 4 (donor of 0; its own donor is 1) leaves the class: its answer is not 9
        recs[4][1] = "10"
    ex2 = _hand_examples(edit)
    got = {r["idx"]: r["in_class"] for r in diag.donor_answer_rows(ex2, _results(ex2, PRED_OWN),
                                                                     _results(ex2, PRED_SCR), DONORS)}
    assert got == {**base, 0: False}                  # only 0, whose DONOR is 4, moves; 4's own flag does not

    def edit1(recs):         # record 1 (donor of 4; its own donor is 0) enters the class: step 3's 7 is its answer
        recs[1][1] = "7"
    ex3 = _hand_examples(edit1)
    got3 = {r["idx"]: r["in_class"] for r in diag.donor_answer_rows(ex3, _results(ex3, PRED_OWN),
                                                                      _results(ex3, PRED_SCR), DONORS)}
    assert got3 == {**base, 4: True}                  # only 4, whose DONOR is 1, moves


def test_t3c_two_donor_maps_two_statistics():
    """The scorer is a function of the donor map it is given (no model, no bank, no recomputation): the same
    ExampleResults under another derangement give another, hand-checked, statistic."""
    ex = _hand_examples()
    own, scr = _results(ex, PRED_OWN), _results(ex, PRED_SCR)
    a = diag.donor_answer_summary(diag.donor_answer_rows(ex, own, scr, DONORS))
    cyc = {i: (i + 1) % 8 for i in range(8)}
    rows = diag.donor_answer_rows(ex, own, scr, cyc)
    b = diag.donor_answer_summary(rows)
    # cyc: donors 1 (out), 2, 3 (out), 4, 5, 6, 7, 0; no ambiguous pair; no prediction equals its donor's gold
    assert [r["in_class"] for r in rows] == [False, True, False, True, True, True, True, True]
    assert (b["n_class"], b["n_ambiguous"], b["n_scored"], b["k_dir"], b["k_own"]) == (6, 0, 6, 0, 0)
    assert b["delta_dir"] == 0.0 and diag.donor_answer_outcome(b, [], "passed").startswith("directional read absent")
    assert (a["n_scored"], a["k_dir"], a["k_own"]) == (5, 4, 1)
    for bad in ({**DONORS, 0: 9}, {k: v for k, v in DONORS.items() if k}):
        with pytest.raises(ValueError):
            diag.donor_answer_rows(ex, own, scr, bad)
    wrong = _results(ex, PRED_OWN)
    wrong[0].correct = False                                  # correct is not pred == gold: refused
    with pytest.raises(ValueError, match="correct"):
        diag.donor_answer_rows(ex, wrong, scr, DONORS)
    with pytest.raises(ValueError, match="gold"):
        diag.donor_answer_rows(_hand_examples(lambda r: r[0].__setitem__(1, "6")), own, scr, DONORS)


def test_t1_columns_carry_no_strings():
    ex = _hand_examples()
    rows = diag.donor_answer_rows(ex, _results(ex, PRED_OWN), _results(ex, PRED_SCR), DONORS)
    for r in rows:
        assert all(isinstance(v, (bool, int, float)) for v in r.values()), r
    assert set(rows[0]) <= set(diag.DONOR_COLUMNS)


# ------------------------------------------------------------------------------------------- the model passes
def _gsm_records(n=N, seed=0):
    """GSM-like toy records: 1 to 5 "<<a+v=c>>" steps, answers equal to the last step's result or 100 above it, one
    step of another form; the class (donor's step min(n, 3) result = its answer) is mixed."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        k = 1 + i % 5
        acc, steps = 0, []
        for v in (int(x) for x in rng.integers(1, 9, size=k)):
            steps.append(f"<<{acc}+{v}={acc + v}>>")
            acc += v
        if i % 7 == 6:
            steps[min(2, k - 1)] = "carry the one"
        out.append({"question": " ".join(rng.choice(WORDS, size=4 + i % 3)) + " ?", "steps": steps,
                    "answer": str(acc if i % 3 else acc + 100)})
    return out


def _examples(recs):
    return [D.Example(question=r["question"], answer=r["answer"], steps=list(r["steps"]), idx=k)
            for k, r in enumerate(recs)]


def _toy_model(recs):
    """FakeTokenizer with the latent tokens and the questions' and answers' pieces, and a tiny GPT-2 whose vocabulary
    is exactly that (no spare id the tokenizer cannot decode; the steps' "<<a+v=c>>" pieces, which a generation
    prompt never holds, get ids beyond it and are never generated, so predictions carry no "="); init 0.2, so the
    greedy streams depend on the fed latents (at the default 0.02 every stream is one repeated token)."""
    t = FakeTokenizer()
    ids = D.add_latent_tokens(t)
    for r in recs:
        t.encode(r["question"] + "\n")
        t.encode("### " + r["answer"])
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=len(t), n_positions=256,
                                       initializer_range=0.2)).eval()
    return t, ids, model


@pytest.fixture(scope="module")
def gsm():
    recs = _gsm_records()
    t, ids, model = _toy_model(recs)
    return t, _examples(recs), LatentLM(model, ids["latent_id"], ids["start_id"], ids["end_id"], t.eos_token_id)


def _tuned(t, ex, lm):
    """The same prompts (questions unchanged) with answers and steps rewritten from the model's own predictions, so
    that the read's booleans are mixed: record j's answer is the scrambled prediction of the example it donates to
    (that example's d_scr true) for even j, its own-pass prediction (its correct_own true) for odd j; even j get a
    1-step "<<0+0=a>>" (in the class), odd j four steps whose third result is not the answer (outside)."""
    res = {}
    diag.scramble(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT,
                  results_out=res)
    own, scr = {r.idx: r.pred for r in res["own"]}, {r.idx: r.pred for r in res["scr"]}
    donors = E.scramble_donors({e.idx: NL for e in ex}, seed=0)
    inv = {j: i for i, j in donors.items()}
    out = []
    for e in ex:
        j = e.idx
        if j % 2 == 0:
            a = scr[inv[j]]
            steps = [f"<<0+0={a}>>"]
        else:
            a = own[j]
            steps = ["<<1+1=2>>", "<<2+1=3>>", "<<3+1=4>>", f"<<4+0={a}>>"]
        out.append(D.Example(question=e.question, answer=a, steps=steps, idx=j))
    return out, own, scr, donors


def _expected(ex, own, scr, donors):
    """An independent per-example computation from the predictions (no diag code)."""
    import re
    rx = re.compile(r"^<<(.*)=([^=]*)>>$")

    def res(st):
        m = rx.match(st.strip())
        return None if not m or not m.group(2).strip() else m.group(2).strip().replace(",", "").strip()

    def norm(x):
        return x.replace(",", "").strip()
    by = {e.idx: e for e in ex}
    out = {}
    for e in ex:
        i, j = e.idx, donors[e.idx]
        dj = by[j]
        last = res(dj.steps[min(len(dj.steps), 3) - 1]) if dj.steps else None
        cls = last is not None and last == norm(dj.answer)
        gi, gj = norm(e.answer), norm(dj.answer)
        po, ps = norm(own[i]), norm(scr[i])
        out[i] = {"in_class": cls, "ambiguous": gi == gj, "scored": cls and gi != gj, "correct_own": po == gi,
                  "correct_scr": ps == gi, "d_own": po == gj, "d_scr": ps == gj, "changed": po != ps,
                  "g_chg": po != ps and ps != gj}
    return out


def _cols(rows):
    keep = ("in_class", "ambiguous", "scored", "correct_own", "correct_scr", "d_own", "d_scr", "changed", "g_chg",
            "g_own", "g_scr", "first_argmax_own", "first_argmax_scr", "emitted_own", "emitted_scr", "donor")
    return {r["idx"]: {k: r[k] for k in keep} for r in rows}


def test_t3_null_blindness_on_the_model(gsm, monkeypatch):
    """(a) the donor column is scramble_donors({idx: 6}, 0) and every fed vector is the donor's own latent bit for bit
    at every rank (the Recorder); the scorer's booleans equal an independent computation from the passes'
    predictions; (b) a self donor (the example's own latents) and a non-donor vector are BUG."""
    t, ex, lm = gsm
    ex2, own, scr, donors = _tuned(t, ex, lm)
    rows, ranks, s = diag.donor_answer(lm, t, ex2, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                       max_new_tokens=MNT, n_boot=1000)
    assert s["bug"] == [] and [r["donor"] for r in rows] == [donors[i] for i in range(N)]
    assert all(r["fed_equals_donor"] and r["fed_equals_collected"] and not r["donor_is_self"] for r in rows)
    assert len(ranks) == N * NL and min(r["l2"] for r in ranks) > 0 and s["scramble"]["fed_vectors"]["verdict"] == "ok"
    want = _expected(ex2, own, scr, donors)
    for r in rows:
        assert {k: r[k] for k in want[r["idx"]]} == want[r["idx"]], r["idx"]
    da = s["donor_answer"]
    assert da["k_dir"] >= 2 and da["n_class"] >= 4 and sum(r["correct_own"] for r in rows) >= 2   # not vacuous
    assert da["n_scored"] == sum(w["scored"] for w in want.values())
    assert da["k_dir"] == sum(w["d_scr"] for w in want.values() if w["scored"])
    assert [r["idx"] for r in rows] == list(range(N)) and list(rows[0]) == list(diag.DONOR_COLUMNS)
    # (b) a derangement that is the identity: every example is fed its own latents
    monkeypatch.setattr(E, "scramble_donors", lambda counts, seed=0: {k: k for k in counts})
    rows_b, _, sb = diag.donor_answer(lm, t, ex2, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                      max_new_tokens=MNT, n_boot=1000)
    assert any("own donor" in b for b in sb["bug"]) and any("distances are 0" in b for b in sb["bug"])
    assert diag.donor_answer_outcome(sb["donor_answer"], sb["bug"], "passed") == "BUG"


def test_t3b_a_non_donor_vector_is_a_bug(gsm, monkeypatch):
    """evaluate feeds one derangement while the record's donors are another: the fed vectors are not the recorded
    donors' latents (fed_equals_donor false), a BUG."""
    t, ex, lm = gsm
    orig, calls = E.scramble_donors, []

    def two_maps(counts, seed=0):
        calls.append(1)
        return orig(counts, seed=seed + (len(calls) == 1))      # evaluate's call: another seed
    monkeypatch.setattr(E, "scramble_donors", two_maps)
    rows, _, s = diag.donor_answer(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                   max_new_tokens=MNT, n_boot=1000)
    assert any("not the donor's own latent" in b for b in s["bug"]) and not all(r["fed_equals_donor"] for r in rows)


def test_t4_batch_isolation_of_the_read(gsm):
    """7.10 item 5 on the read's columns (fp32, CPU): the per-example booleans (changed and g_chg included), argmax
    and emitted counts at B = 1 equal those at B = 4 and at B = 4 over a permuted example order (every companion
    replaced, row i fixed in content); the own-vs-scrambled first-answer logit difference agrees within 1e-4; the
    same-shape companion swap of diag.isolation stays within 1e-4 on the own path; a cross-row leak is seen."""
    t, ex, lm = gsm
    ex2, _, _, _ = _tuned(t, ex, lm)
    kw = dict(arm="coconut", n_latent=NL, device="cpu", max_new_tokens=MNT, n_boot=1000)
    r4, _, s4 = diag.donor_answer(lm, t, ex2, batch_size=BS, **kw)
    r1, _, s1 = diag.donor_answer(lm, t, ex2, batch_size=1, **kw)
    perm = [ex2[k] for k in diag.permutation(N, 3)]
    rp, _, sp = diag.donor_answer(lm, t, perm, batch_size=BS, **kw)
    assert s4["bug"] == s1["bug"] == sp["bug"] == []
    a, b, c = _cols(r4), _cols(r1), _cols(rp)
    assert a == b == c and sum(x["d_scr"] for x in a.values()) >= 2 and sum(x["changed"] for x in a.values()) >= 2
    for x, y in ((r4, r1), (r4, sorted(rp, key=lambda r: r["idx"]))):
        for p, q in zip(x, y):
            assert abs(p["first_maxabs_own_scr"] - q["first_maxabs_own_scr"]) <= diag.TOL
    for k in ("k_dir", "k_own", "k_g_chg", "n_scored", "lo", "hi"):
        assert s4["donor_answer"][k] == s1["donor_answer"][k] == sp["donor_answer"][k]
    iso = diag.isolation_counts(diag.isolation(lm, t, ex2, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                               perm=diag.permutation(N, 0)))
    assert all(iso[k]["over_tol"] == 0 for k in diag.ISO_KEYS) and iso["last_swap"]["n"] == N
    h = lm.base_causallm.transformer.ln_f.register_forward_hook(lambda m, i, o: o + 0.5 * o.mean(0, keepdim=True))
    try:
        leak, _, _ = diag.donor_answer(lm, t, ex2, batch_size=BS, **kw)
        leak1, _, _ = diag.donor_answer(lm, t, ex2, batch_size=1, **kw)
    finally:
        h.remove()
    assert max(abs(p["first_maxabs_own_scr"] - q["first_maxabs_own_scr"]) for p, q in zip(leak, leak1)) > diag.TOL


def test_t5_correctness_equals_the_reads_outside_any_recorder(gsm):
    t, ex, lm = gsm
    rows, _, _ = diag.donor_answer(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu",
                                   max_new_tokens=MNT, n_boot=1000)
    own = E.evaluate(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT)
    scr = E.evaluate(lm, t, ex, arm="coconut", n_latent=NL, batch_size=BS, device="cpu", max_new_tokens=MNT,
                     intervention="scrambled", scramble_seed=0)
    assert [r["correct_own"] for r in rows] == [r.correct for r in own]
    assert [r["correct_scr"] for r in rows] == [r.correct for r in scr]
    assert [r["emitted_own"] for r in rows] == [r.emitted_tokens for r in own]


# ------------------------------------------------------------------------------------------- run_diag.py
@pytest.fixture()
def donorenv(tmp_path, monkeypatch):
    from latent.tests import test_screen_tools as sct
    recs = _gsm_records()
    for split in ("train", "valid"):
        (tmp_path / f"toy_{split}.json").write_text(json.dumps(recs))
    t, _, model = _toy_model(recs)
    ck = tmp_path / "ckpt.pt"
    torch.save(model.state_dict(), ck)

    def fake_load_base(model_id, dtype=None, device="cpu", resize=True):
        m = GPT2LMHeadModel(model.config)
        m.load_state_dict(model.state_dict())
        tok = FakeTokenizer()
        tok.id_to_tok, tok.tok_to_id = list(t.id_to_tok), dict(t.tok_to_id)
        return m, tok
    monkeypatch.setattr(T, "load_base", fake_load_base)
    mod = sct._mod("run_diag")

    def cfg(name="toy_c3_s0", arm="coconut", **kw):
        d = dict(name=name, arm=arm, seed=0, model_id="toy", max_new_tokens=MNT,
                 train_path=str(tmp_path / "toy_train.json"), val_path=str(tmp_path / "toy_valid.json"),
                 eval_batch_size=BS, device="cpu", save_dir=str(tmp_path / name), c_thought=2, max_latent_stage=3,
                 precision="bf16", dep_loss=True)
        d.update(kw)
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(d))
        return str(p)
    return mod, cfg, str(ck), tmp_path


def _run(mod, cfg_path, ck, out, *extra):
    return mod.main(["--config", cfg_path, "--checkpoint", ck, "--mode", "donor_answer", "--out", str(out),
                     "--set", "seed=0", *extra])


def test_run_diag_donor_answer_writes_the_record(donorenv):
    mod, cfg, ck, tmp = donorenv
    out = tmp / "jobdir" / "toy_c3_s0_valid_donor_answer"
    s = _run(mod, cfg(), ck, out, "--commit", "c0ffee")
    assert json.loads((out / "summary.json").read_text()) == json.loads(json.dumps(s))
    assert (s["mode"], s["critic_row"], s["critic_row_in_log"], s["registered_read"], s["split"], s["commit"]) == (
        "donor_answer", ROW, True, None, "valid", "c0ffee")
    assert s["answers"].startswith("A6.1 rule 4 (B)") and "2026-10-20T17:36Z" in s["answers"]
    assert (s["n"], s["batch_size"], s["n_latent"], s["n_latent_steps"], s["max_new_tokens"]) == (N, BS, 6, 3, MNT)
    assert (s["scramble_seed"], s["n_boot"], s["boot_seed"], s["tf32_modes"]) == (0, 10_000, 0, ["registered", "strict"])
    assert s["checkpoint_sha256"] == _sha(ck) and s["split_sha256"] == _sha(tmp / "toy_valid.json")
    assert s["data_pin"].startswith("not the valid file of record") and s["reference"] is None and s["dep_loss"]
    rg, st = s["numerics"]["registered"], s["numerics"]["strict"]
    assert (rg["autocast"], rg["tf32_requested"], rg["float32_matmul_precision"]) == ("bf16", True, "high")
    assert rg["torch_at_start"]["autocast_enabled"] and rg["torch_at_start"] == rg["torch_at_end"]
    assert (st["autocast"], st["tf32_requested"], st["float32_matmul_precision"]) == (None, False, "highest")
    assert not st["torch_at_start"]["autocast_enabled"]
    reg = s["results"]["registered"]["donor_answer"]
    assert reg["identification"]["status"] == "not run" and reg["outcome"] in diag.DONOR_OUTCOMES[2:]
    assert s["outcome"] == reg["outcome"] and s["bug"] is False
    svr = s["results"]["strict"]["strict_vs_registered"]
    assert set(svr) >= {"outcome_registered", "outcome_strict", "outcome_word_disagrees", "examples_whose_d_scr_differs"}
    assert s["results"]["registered"]["scramble"]["fed_vectors"]["zero_distance"] == 0
    assert s["line"].startswith(f"directional read (A6.1 rule 4 (B), donor-answer, toy valid, seed 0, toy_c3_s0 "
                                f"{_sha(ck)[:8]}): R_dir {reg['k_dir']} of {reg['n_scored']}")
    assert diag.DONOR_PASS_LINE in s["line"] and "G_chg" in s["line"] and "not run" in s["line"]
    assert "bf16 autocast over TF32 on (strict fp32 beside, diagnostic)" in s["labels"][0]
    assert "trained on A100-80GB; no test read; ProsQA-hard UNTESTED (A2.1)" in s["labels"][0]
    assert s["no_test_read"].startswith("No test read") and set(s["predictions"]) == set(mod.DONOR_PREDICTIONS)
    assert s["files"] == {"examples.csv": _sha(out / "examples.csv"), "fed_vectors.csv": _sha(out / "fed_vectors.csv")}
    rows = list(csv.DictReader(open(out / "examples.csv")))
    assert len(rows) == 2 * N and list(rows[0]) == ["numerics", *diag.DONOR_COLUMNS]
    assert [r["numerics"] for r in rows] == ["registered"] * N + ["strict"] * N
    ranks = list(csv.DictReader(open(out / "fed_vectors.csv")))
    assert len(ranks) == 2 * N * NL and list(ranks[0])[:4] == ["numerics", "idx", "rank", "donor"]
    text = (out / "summary.json").read_text() + (out / "examples.csv").read_text()
    for w in ("inert", "mechanism", "reproduces", "K2 would fire", "latents carry state"):
        assert w not in text
    for r in rows:                                     # ids, booleans and numbers only: no pred, gold or text
        for k, v in r.items():
            if k == "numerics":
                assert v in ("registered", "strict")
            elif v not in ("True", "False", ""):
                float(v)
    with pytest.raises(SystemExit, match="write-once"):
        _run(mod, cfg(), ck, out)
    ref = _run(mod, cfg(name="toy_coconut_s0", dep_loss=False), ck, tmp / "ref", "--tf32", "on")
    assert ref["reference"] == "reference arm (no hinge); diagnostic; decides nothing" and ref["tf32_modes"] == [
        "registered"] and "strict fp32 not run" in ref["labels"][0] and ref["labels"][1] == ref["reference"]
    assert ref["predictions"]["critic (2026-10-20T17:36Z (3), the registered prediction)"]["arm"].startswith(
        "Delta_dir +0.02")


def _write_read(d, name, run, ck_sha, split_sha, kind, correct, n_latent=NL):
    x = d / name
    x.mkdir(parents=True)
    with open(x / "examples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "correct", "cot_match"])
        for i, c in enumerate(correct):
            w.writerow([i, c, False])
    sm = {"run": run, "split": "valid", "checkpoint_sha256": ck_sha, "split_sha256": split_sha, "intervention": kind,
          "n_latent": n_latent, "examples_csv_sha256": _sha(x / "examples.csv")}
    if kind == "scrambled":
        sm["scramble_seed"] = 0
    (x / "summary.json").write_text(json.dumps(sm))


def test_t5_identification_check(donorenv):
    """Ruling 9: 0 of N on each read passes; any difference holds the record (outcome "held: identification check
    failed", no lift); reads of another checkpoint are "not run"; the default directory is the one holding --out."""
    mod, cfg, ck, tmp = donorenv
    s0 = _run(mod, cfg(), ck, tmp / "first")
    rows = [r for r in csv.DictReader(open(tmp / "first" / "examples.csv")) if r["numerics"] == "registered"]
    co, cs = [r["correct_own"] == "True" for r in rows], [r["correct_scr"] == "True" for r in rows]
    ck_sha, sp_sha = s0["checkpoint_sha256"], s0["split_sha256"]
    job = tmp / "job"
    _write_read(job, "toy_c3_s0_valid_none", "toy_c3_s0", ck_sha, sp_sha, "none", co)
    _write_read(job, "toy_c3_s0_valid_scrambled", "toy_c3_s0", ck_sha, sp_sha, "scrambled", cs)
    ok = _run(mod, cfg(), ck, job / "toy_c3_s0_valid_donor_answer")          # the job's other entries beside --out
    ident = ok["results"]["registered"]["donor_answer"]["identification"]
    assert (ident["status"], ident["none_diff"], ident["scrambled_diff"]) == ("passed", 0, 0)
    assert ident["entries"] == ["toy_c3_s0_valid_none", "toy_c3_s0_valid_scrambled"] and ok["outcome"] != "held: " \
        "identification check failed" and "0/0 of 12" in ok["line"]
    flipped = tmp / "flipped"
    _write_read(flipped, "toy_c3_s0_valid_none", "toy_c3_s0", ck_sha, sp_sha, "none", co)
    _write_read(flipped, "toy_c3_s0_valid_scrambled", "toy_c3_s0", ck_sha, sp_sha, "scrambled",
                [not c if i in (2, 5) else c for i, c in enumerate(cs)])
    held = _run(mod, cfg(), ck, tmp / "held", "--identify-dir", str(flipped))
    hi = held["results"]["registered"]["donor_answer"]["identification"]
    assert (hi["status"], hi["none_diff"], hi["scrambled_diff"], hi["scrambled_diff_idx"]) == ("held", 0, 2, [2, 5])
    assert held["outcome"] == "held: identification check failed" and any(f.startswith("held") for f in held["flags"])
    one = tmp / "one"
    _write_read(one, "toy_c3_s0_valid_none", "toy_c3_s0", ck_sha, sp_sha, "none", [not c if i == 0 else c
                                                                                    for i, c in enumerate(co)])
    _write_read(one, "toy_c3_s0_valid_scrambled", "toy_c3_s0", ck_sha, sp_sha, "scrambled", cs)
    reg = [{"idx": int(r["idx"]), "correct_own": r["correct_own"] == "True", "correct_scr": r["correct_scr"] == "True"}
           for r in rows]
    h1 = mod.identification(reg, "toy_c3_s0", ck_sha, sp_sha, NL, [str(one)])
    assert (h1["status"], h1["none_diff"]) == ("held", 1)                     # 1 of N holds (any difference)
    other = tmp / "other"
    _write_read(other, "toy_c3_s0_valid_none", "toy_c3_s0", "f" * 64, sp_sha, "none", co)
    _write_read(other, "toy_c3_s0_valid_scrambled", "toy_c3_s0", "f" * 64, sp_sha, "scrambled", cs)
    no = mod.identification([{"idx": i, "correct_own": a, "correct_scr": b} for i, (a, b) in enumerate(zip(co, cs))],
                            "toy_c3_s0", ck_sha, sp_sha, NL, [str(other), str(tmp / "nothing")])
    assert no["status"] == "not run" and "checkpoint_sha256" in no["why"]


def test_t6_refusals_before_the_model_loads(donorenv, monkeypatch):
    mod, cfg, ck, tmp = donorenv
    c = cfg()
    base = ["--checkpoint", ck, "--mode", "donor_answer", "--out", str(tmp / "x")]
    for extra, msg in ((["--split", "test"], "valid split only"), (["--tf32", "off"], "--tf32 off alone"),
                       (["--set", "latent_band=[0,2]"], "latent_band"), (["--n-boot", "100"], "10,000"),
                       (["--batch-size", "1"], "eval batch"), (["--set", "max_latent_stage=2"], "step min"),
                       (["--set", 'precision="fp8"'], "registered flag"), (["--extra-idx", "1"], "band_logits")):
        with pytest.raises(SystemExit, match=msg):
            mod.main(["--config", c, *base, *extra])
    with pytest.raises(SystemExit, match="latent-feedback"):
        mod.main(["--config", cfg(name="toy_cot_s0", arm="cot"), *base])
    with pytest.raises(SystemExit, match="identify-dir belongs"):
        mod.main(["--config", c, "--checkpoint", ck, "--mode", "latent_path", "--identify-dir", str(tmp)])
    prosqa = [{"question": "q ?", "steps": ["a"], "answer": "b", "edges": [[0, 1]], "root": 0,
               "idx_to_symbol": ["a", "b"]}] * 3
    (tmp / "pq_valid.json").write_text(json.dumps(prosqa))
    with pytest.raises(SystemExit, match="GSM read"):
        mod.main(["--config", cfg(val_path=str(tmp / "pq_valid.json")), *base])
    (tmp / "gsm50k_train.json").write_text("[]")
    with pytest.raises(SystemExit, match="not the GSM-50k valid file of record"):
        mod.main(["--config", cfg(name="gsm50k_c3_s0", train_path=str(tmp / "gsm50k_train.json")), *base])
    loaded = []
    monkeypatch.setattr(T, "build_model", lambda *a_, **k_: loaded.append(1))
    log = tmp / "critic_log.md"
    log.write_text("| Logged (UTC) | Ruling |\n|---|---|\n| 2026-10-20T17:12Z | another row |\n")
    monkeypatch.setattr(mod, "CRITIC_LOG", str(log))
    with pytest.raises(SystemExit, match="registered by critic entry 2026-10-20T17:36Z"):
        mod.main(["--config", c, *base])
    assert not loaded and not (tmp / "x").exists()


def test_t7_determinism(donorenv):
    mod, cfg, ck, tmp = donorenv
    a = _run(mod, cfg(), ck, tmp / "a")
    b = _run(mod, cfg(), ck, tmp / "b")
    for f in ("examples.csv", "fed_vectors.csv"):
        assert (tmp / "a" / f).read_bytes() == (tmp / "b" / f).read_bytes()
    assert json.dumps(a["results"]) == json.dumps(b["results"]) and a["line"] == b["line"]


def test_t3b_bug_record_has_no_pass_line(donorenv, monkeypatch):
    mod, cfg, ck, tmp = donorenv
    monkeypatch.setattr(E, "scramble_donors", lambda counts, seed=0: {k: k for k in counts})
    s = _run(mod, cfg(), ck, tmp / "bug", "--tf32", "on")
    assert s["bug"] is True and s["outcome"] == "BUG" and ": BUG: " in s["line"]
    assert "R_dir" not in s["line"] and diag.DONOR_PASS_LINE not in s["line"]
    assert any(f.startswith("registered: BUG") for f in s["flags"])


# ------------------------------------------------------------------------------------------- T8 data pin
@pytest.mark.skipif(not os.path.isfile(VALID), reason="data/gsm_valid.json absent")
def test_t8_data_pin_on_the_valid_file_of_record(tmp_path, monkeypatch):
    from latent.tests import test_screen_tools as sct
    mod = sct._mod("run_diag")
    assert _sha(VALID) == mod.DONOR_VALID_SHA256
    ex = D.load_examples(VALID)
    pre = mod.donor_preview(ex, 6, 3)
    assert pre == mod.DONOR_PIN
    assert pre["n_class"] == 295 and pre["ambiguous_idx"] == [15, 124, 125, 175, 470, 477] and pre["n_scored"] == 289
    donors = E.scramble_donors({e.idx: 6 for e in ex}, seed=0)
    blank = _results(ex, {e.idx: "" for e in ex})
    rows = diag.donor_answer_rows(ex, blank, blank, donors)
    s = diag.donor_answer_summary(rows, n_boot=1000)
    assert (s["n_class"], s["n_ambiguous"], s["n_scored"], s["complement"]["n"]) == (295, 6, 289, 205)
    assert s["secondary"]["donor_gold_in_own_steps"] == 14 and s["n_ambiguous_all"] == 9
    assert sum(diag.donor_in_class(e) for e in ex) == 295              # own step min(n, 3) = own answer, too
    # the pin is a refusal before the model loads (ruling 1): the GSM-50k C3 config on the file of record
    cfgp = os.path.join(os.path.dirname(VALID), "..", "configs", "screen", "gsm50k_c3.json")
    monkeypatch.setattr(mod, "DONOR_PIN", {**mod.DONOR_PIN, "n_class": 296})
    with pytest.raises(SystemExit, match="n_class 295 != 296"):
        mod.main(["--config", cfgp, "--checkpoint", VALID, "--mode", "donor_answer", "--out", str(tmp_path / "x"),
                  "--set", f'val_path="{VALID}"'])
    assert not (tmp_path / "x").exists()
