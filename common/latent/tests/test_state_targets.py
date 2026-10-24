"""C2 targets (docs/cards/C2_verifiable_state_supervision.md "Leak and bug answers"; registration A3.4 C2):
hand-written cases, a BFS cross-check on 200 real ProsQA records and a parser check on 1,000 real GSM8k-Aug records
(skipped when data/ is absent), the latent-to-step map, the answer statistic and the C2-shuffled derangement."""
import json
import os
import re

import pytest

from latent import prosqa_gen as G
from latent import state_targets as S
from latent.data import load_examples
from latent.tests.fake_tokenizer import FakeTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PROSQA = os.path.join(REPO, "data", "prosqa_train.json")
GSM = os.path.join(REPO, "data", "gsm_train.json")
GPT2_TOK = os.path.join(REPO, "common", "babylm", "recipes", "sprint1_port", "gpt2_tokenizer")


def test_frontiers_hand_written():
    # 0 = root (an entity), 1 = another entity; 0 -> 2 -> 3 -> 4, 1 -> 5 -> 6, plus a shortcut 0 -> 3 in the second
    edges = [[0, 2], [2, 3], [3, 4], [1, 5], [5, 6]]
    assert S.prosqa_frontiers(edges, 0, 7, 3) == [[2], [2, 3], [2, 3, 4]]
    assert S.prosqa_frontiers(edges + [[0, 3]], 0, 7, 3) == [[2, 3], [2, 3, 4], [2, 3, 4]]
    assert S.prosqa_frontiers(edges, 1, 7, 2) == [[5], [5, 6]]
    assert S.prosqa_frontiers([], 0, 3, 2) == [[], []]   # nothing reachable; the root is never a target


def test_gsm_parser_hand_written():
    cases = {"<<600*30/100=180>>": "180", "<<1+2=3>>": "3", "<<1/3=0.333>>": "0.333", "<<-2+1=-1>>": "-1",
             "<<5*2=10.5>>": "10.5", "<<5>>": None, "": None, "12": None, "<<4*3=>>": None, " <<2+2=4>> ": "4"}
    for step, want in cases.items():
        assert S.gsm_step_result(step) == want, step
    t = FakeTokenizer()
    assert S.gsm_result_tokens("180", t) == t.encode("180")
    assert S.gsm_result_tokens(None, t) is None
    assert S.gsm_result_tokens("1 2 3 4 5", t) is None   # 5 tokens > 4 heads: no target
    assert len(S.gsm_result_tokens("1 2 3 4", t)) == 4


def test_record_targets_and_supervised_steps():
    t = FakeTokenizer()
    gsm = {"question": "q ?", "steps": ["<<1+2=3>>", "<<3*4=12>>", "bad"], "answer": "12"}
    tg = S.record_targets(gsm, t)
    assert tg["kind"] == "gsm" and tg["results"] == ["3", "12", None] and tg["tokens"][2] is None
    rec = G.generate(1, depth=3, seed=5)[0]
    tp = S.record_targets(rec, t)
    assert tp["kind"] == "prosqa" and len(tp["frontiers"]) == 3 and tp["n_symbols"] == len(rec["idx_to_symbol"])
    assert all(len(x) >= 1 for x in tp["symbol_tokens"])
    # latent t (1-based) carries step t for c_thought 1; the second latent of pair t for c_thought 2
    assert S.supervised_steps(5, 3, 1) == [(1, 0), (2, 1), (3, 2)]
    assert S.supervised_steps(2, 6, 2) == [(1, 1), (2, 3)]          # no target beyond the steps
    assert S.supervised_steps(4, 4, 2) == [(1, 1), (2, 3)]          # no target beyond the stage's latents
    assert S.supervised_steps(4, 0, 1) == [] and S.supervised_steps(0, 3, 1) == []   # stage 0 / no steps
    # targets never read the answer field: changing the answer leaves them unchanged
    assert S.record_targets({**gsm, "answer": "999"}, t) == tg
    assert S.record_targets({**rec, "answer": "x"}, t) == tp


def test_answer_in_last_target_and_stats():
    t = FakeTokenizer()
    a = {"question": "q", "steps": ["<<1+2=3>>", "<<3*4=12>>"], "answer": "12", "idx": 0}
    b = {"question": "q", "steps": ["<<1+1=2>>", "<<2*2=4>>"], "answer": "5", "idx": 1}
    tg = S.build_targets([a, b], t)
    assert S.answer_in_last_target(tg[0], "12", 3) is True
    assert S.answer_in_last_target(tg[0], "12", 1) is False          # stage 1: the last supervised step is step 1
    assert S.answer_in_last_target(tg[1], "5", 3) is False
    assert S.answer_in_last_target(tg[0], "12", 0) is None
    st = S.target_stats([a, b], tg, 3)
    assert st["answer_in_last_target_frac"] == 0.5 and st["supervised_steps_final_regime"] == 4
    rec = G.generate(1, depth=3, seed=1)[0]
    tp = S.record_targets(rec, t)
    assert S.answer_in_last_target(tp, rec["answer"], 3) is True     # the depth-3 frontier holds the target
    assert S.answer_in_last_target(tp, rec["answer"], 2) is False    # the proof is a shortest path


def test_shuffled_donors_derangement_within_classes():
    counts = {i: (2 if i < 6 else 3 if i < 10 else 7) for i in range(11)}   # idx 10 is a singleton class
    d = S.shuffled_donors(counts, seed=4)
    assert all(d[i] != i for i in d) and sorted(d.values()) == sorted(d)
    for i in range(10):
        assert counts[d[i]] == counts[i] or i in (6, 7, 8, 9)            # class 3 absorbs the singleton 7
    assert {counts[d[10]]} <= {3, 7}
    assert d == S.shuffled_donors(counts, seed=4) and d != S.shuffled_donors(counts, seed=5)
    with pytest.raises(ValueError):
        S.shuffled_donors({0: 1}, seed=0)


def test_prosqa_frontiers_bfs_cross_check_on_200_real_records():
    if not os.path.exists(PROSQA):
        pytest.skip("data/prosqa_train.json absent")
    recs = json.load(open(PROSQA))[:200]
    t = FakeTokenizer()
    for r in recs:
        tg = S.record_targets(r, t)
        n = len(r["idx_to_symbol"])
        dist = G.bfs_dist(n, r["edges"], r["root"])
        L = len(r["steps"])
        assert len(tg["frontiers"]) == L
        for k, F in enumerate(tg["frontiers"], start=1):
            assert F == sorted(v for v, dd in dist.items() if 1 <= dd <= k)
            assert r["neg_target"] not in F and r["root"] not in F
            if k > 1:
                assert set(tg["frontiers"][k - 2]) <= set(F)
        # the proof's t-th step reaches a node first in F_t (shortest path): walk the proof by name
        names = r["idx_to_symbol"]
        for k, step in enumerate(r["steps"], start=1):
            node = names.index(step.rstrip(".").rsplit(" ", 1)[-1])
            assert node in tg["frontiers"][k - 1] and (k == 1 or node not in tg["frontiers"][k - 2])
        assert r["target"] in tg["frontiers"][-1]


def test_gsm_parser_on_1000_real_records():
    if not os.path.exists(GSM):
        pytest.skip("data/gsm_train.json absent")
    exs = load_examples(GSM, 1000)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(GPT2_TOK) if os.path.isdir(GPT2_TOK) else FakeTokenizer()
    except Exception:  # pragma: no cover - environment dependent
        tok = FakeTokenizer()
    n_steps = n_parsed = n_tok = n_num = 0
    for ex in exs:
        tg = S.record_targets(ex, tok)
        assert tg["kind"] == "gsm" and len(tg["results"]) == len(ex.steps)
        for s, res, toks in zip(ex.steps, tg["results"], tg["tokens"]):
            n_steps += 1
            if res is None:
                assert "=" not in s or not s.startswith("<<")
                continue
            n_parsed += 1
            assert res == s.strip()[2:-2].rsplit("=", 1)[1].strip()     # independent parse: after the last "="
            n_num += bool(re.fullmatch(r"[-+0-9.,/eE ]+", res))          # the data also has e.g. "78.947...≈79"
            if toks is not None:
                n_tok += 1
                assert 1 <= len(toks) <= 4
                dec = tok.decode(toks).strip()
                # the FakeTokenizer (used where the GPT-2 tokenizer directory is absent, e.g. on a VM whose code tarball
                # carries common/latent only) decodes pieces joined by spaces; the real tokenizer round-trips exactly
                assert dec == res if not isinstance(tok, FakeTokenizer) else dec.replace(" ", "") == res
    assert n_parsed / n_steps > 0.99 and n_tok / n_parsed > 0.98 and n_num / n_parsed > 0.99
