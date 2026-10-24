"""latent.prosqa_gen: validity of every generated proof (BFS), controls, seeding, and a comparison of a depth-3
regeneration with the released ProsQA validation set (skipped when data/prosqa_valid.json is absent)."""
import json
import os
import random
import statistics as st

import pytest

from latent import prosqa_gen as G

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
VALID = os.path.join(REPO, "data", "prosqa_valid.json")
KEYS = ["question", "answer", "steps", "idx_to_symbol", "edges", "root", "target", "neg_target"]


def _is_path(rec):
    """The proof as a walk over the record's own edge list, starting at the root and ending at the target."""
    names = rec["idx_to_symbol"]
    edges = {tuple(e) for e in rec["edges"]}
    cur = rec["root"]
    for i, s in enumerate(rec["steps"]):
        nxt = [v for (u, v) in edges if u == cur and s.endswith(f" is a {names[v]}.")
               and s.startswith(names[u] if i == 0 else f"Every {names[u]} ")]
        if len(nxt) != 1:
            return False
        cur = nxt[0]
    return cur == rec["target"]


@pytest.mark.parametrize("depth", range(3, 9))
def test_generated_records_valid(depth):
    recs = G.generate(200, depth=depth, seed=100 + depth)
    for r in recs:
        assert list(r) == KEYS
        assert len(r["steps"]) == depth
        assert _is_path(r)
        dist = G.bfs_dist(len(r["idx_to_symbol"]), r["edges"], r["root"])
        assert r["neg_target"] not in dist  # negative target unreachable from the root
        assert dist[r["target"]] == depth   # the proof is a shortest path
        assert G.validate_record(r) == []
        assert len({tuple(e) for e in r["edges"]}) == len(r["edges"])


def test_distractor_count_and_seed():
    recs = G.generate(50, depth=5, seed=3, n_distractor_edges=24)
    assert all(len(r["edges"]) == 5 + 24 for r in recs)
    assert G.generate(5, depth=4, seed=7) == G.generate(5, depth=4, seed=7)
    assert G.generate(5, depth=4, seed=7) != G.generate(5, depth=4, seed=8)
    with pytest.raises(ValueError):
        G.generate_record(random.Random(0), depth=4, n_distractor_edges=1)


def test_released_records_pass_validator():
    if not os.path.exists(VALID):
        pytest.skip("data/prosqa_valid.json absent")
    for r in json.load(open(VALID)):
        assert list(r) == KEYS
        assert G.validate_record(r) == []
        assert _is_path(r)


def test_depth3_matches_released_distributions():
    if not os.path.exists(VALID):
        pytest.skip("data/prosqa_valid.json absent")
    rel = [r for r in json.load(open(VALID)) if len(r["steps"]) == 3]  # 124 of 300 released records have depth 3
    gen = G.generate(300, depth=3, seed=0)
    assert all(len(r["steps"]) == 3 for r in gen)

    def stats(rs):
        return {"edges": st.mean(len(r["edges"]) for r in rs),
                "words": st.mean(len(r["question"].split()) for r in rs),
                "nodes": st.mean(len(r["idx_to_symbol"]) for r in rs)}
    a, b = stats(gen), stats(rel)
    for k in a:
        assert abs(a[k] - b[k]) / b[k] < 0.10, (k, a[k], b[k])
    names_rel = {n for r in rel for n in r["idx_to_symbol"]}
    assert {n for r in gen for n in r["idx_to_symbol"]} <= set(G.PERSONS) | set(G.CONCEPTS)
    assert names_rel <= set(G.PERSONS) | set(G.CONCEPTS)
