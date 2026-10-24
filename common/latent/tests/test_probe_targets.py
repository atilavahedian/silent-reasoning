"""latent.probe.extract: frontier sets (hand cases and BFS), candidates, GSM result parsing, step entities."""
import random

import pytest

from latent.data import Example
from latent.probe.extract import (bfs_dist, candidate_nodes, example_targets, frontier_sets, gsm_results,
                                  latent_hops, step_entity)
from latent.prosqa_gen import generate
from latent.tests.fake_tokenizer import FakeTokenizer


def test_frontier_hand_chain_and_diamond():
    # 0 -> 1 -> 2 -> 3, plus 0 -> 4, 4 -> 3 (diamond shortcut) and 5 -> 2 (other source)
    edges = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 3], [5, 2]]
    fr = frontier_sets(edges, 0, 6, 4)
    assert fr[0] == {1, 4}
    assert fr[1] == {1, 4, 2, 3}
    assert fr[2] == fr[1] and fr[3] == fr[1]  # saturates; 5 is never reachable
    assert 0 not in fr[3] and 5 not in fr[3]


def test_frontier_hand_empty_and_isolated_root():
    assert frontier_sets([[1, 2]], 0, 3, 3) == [frozenset()] * 3
    assert frontier_sets([], 0, 1, 2) == [frozenset(), frozenset()]


def test_frontier_cycle_safe():
    fr = frontier_sets([[0, 1], [1, 2], [2, 0]], 0, 3, 3)
    assert fr[2] == {1, 2}  # root excluded even when reachable again


@pytest.mark.parametrize("seed", range(20))
def test_frontier_matches_bfs_random(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 15)
    edges = [[rng.randrange(n), rng.randrange(n)] for _ in range(rng.randint(0, 30))]
    root = rng.randrange(n)
    dist = bfs_dist(n, edges, root)
    fr = frontier_sets(edges, root, n, 8)
    for t in range(1, 9):
        assert fr[t - 1] == {v for v, d in dist.items() if 1 <= d <= t and v != root}


def test_frontier_matches_bfs_on_prosqa_records():
    for rec in generate(30, depth=4, seed=3):
        n = len(rec["idx_to_symbol"])
        dist = bfs_dist(n, rec["edges"], rec["root"])
        fr = frontier_sets(rec["edges"], rec["root"], n, 6)
        assert rec["target"] in fr[3] and rec["target"] not in fr[2]  # proof depth 4 is the shortest path
        assert all(rec["neg_target"] not in f for f in fr)
        for t in range(1, 7):
            assert fr[t - 1] == {v for v, d in dist.items() if 1 <= d <= t}


def test_candidates_exclude_root_and_sources():
    edges = [[0, 2], [1, 2], [2, 3]]
    assert candidate_nodes(edges, 0, 5) == [2, 3]  # 1 (source) and 4 (isolated) never reachable


def test_latent_hops():
    assert [latent_hops(j, 1) for j in range(4)] == [1, 2, 3, 4]
    assert [latent_hops(j, 2) for j in range(6)] == [1, 1, 2, 2, 3, 3]


def test_gsm_results_parser():
    steps = ["<<16-3-4=9>>", "<<9*2=18>>", "<<60/100*5=3>>", "no equation", "<<1+1=2>> and <<2+2=4>>", "<<a = 1.5>>"]
    assert gsm_results(steps) == ["9", "18", "3", None, "2", "1.5"]


def test_gsm_results_negative_and_decimal():
    assert gsm_results(["<<5-8=-3>>", "<<10/4=2.5>>", "<<1,000*2=2000>>"]) == ["-3", "2.5", "2000"]


def test_step_entity():
    assert step_entity("Sally is a scrompus.") == "scrompus"
    assert step_entity("Every scrompus is a rempus.") == "rempus"
    assert step_entity("Every x is an impus.") == "impus"
    assert step_entity("nonsense") is None


def test_example_targets_prosqa_and_gsm():
    t = FakeTokenizer()
    rec = generate(1, depth=3, seed=0)[0]
    ex = Example(question=rec["question"], answer=rec["answer"], steps=rec["steps"], idx=7,
                 meta={k: rec[k] for k in ("idx_to_symbol", "edges", "root", "target", "neg_target")})
    r = example_targets(ex, 6, 1, t)
    assert r["task"] == "prosqa" and len(r["frontier"]) == 6 and r["hops"] == [1, 2, 3, 4, 5, 6]
    assert r["target"] in r["frontier"][2] and r["target"] not in r["frontier"][1]
    assert r["step_entities"][-1] == rec["idx_to_symbol"][rec["target"]]
    g = Example(question="q", answer="18", steps=["<<16-3-4=9>>", "<<9*2=18>>"], idx=0, meta={})
    rg = example_targets(g, 6, 2, t)
    assert rg["task"] == "gsm" and rg["results"] == ["9", "18"]
    assert rg["latent_step"] == [-1, 0, -1, 1, -1, -1]  # second latent of each pair carries step t
    assert rg["result_first_token"] == [t.encode("9")[0], t.encode("18")[0]]
