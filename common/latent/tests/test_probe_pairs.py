"""latent.probe.tasks: minimal pairs (graphs identical, answers flip), answer options, edit specs."""
import pytest

from latent.data import Example
from latent.probe.extract import bfs_dist, example_targets
from latent.probe.synthetic import make_reach_data
from latent.probe.tasks import answer_option, edit_specs, make_pair
from latent.prosqa_gen import generate, validate_record
from latent.tests.fake_tokenizer import FakeTokenizer

KEYS = ("idx_to_symbol", "edges", "root", "target", "neg_target")


def prosqa_examples(n=40, depth=3, seed=0):
    return [Example(question=r["question"], answer=r["answer"], steps=r["steps"], idx=i, meta={k: r[k] for k in KEYS})
            for i, r in enumerate(generate(n, depth=depth, seed=seed))]


def as_record(ex):
    return {"question": ex.question, "answer": ex.answer, "steps": ex.steps, **{k: ex.meta[k] for k in KEYS}}


def body(q):
    return q.rpartition(" Is ")[0]


@pytest.mark.parametrize("rule", ["target_swap", "root_swap"])
def test_prosqa_pairs_flip_answer_keep_graph(rule):
    made = 0
    for a in prosqa_examples():
        b = make_pair(a, rule, seed=0)
        if b is None:
            continue
        made += 1
        assert b.idx == a.idx
        assert b.meta["edges"] == a.meta["edges"] and b.meta["idx_to_symbol"] == a.meta["idx_to_symbol"]
        assert body(b.question) == body(a.question)  # identical graph text: only the query differs
        assert b.answer != a.answer
        assert answer_option(b.answer, "prosqa") != answer_option(a.answer, "prosqa")
        assert validate_record(as_record(b)) == [], validate_record(as_record(b))
        names, m = a.meta["idx_to_symbol"], a.meta
        n = len(names)
        if rule == "target_swap":
            assert b.meta["root"] == m["root"] and b.meta["neg_target"] == m["neg_target"]
            d = bfs_dist(n, m["edges"], m["root"])
            assert d[b.meta["target"]] == d[m["target"]]  # same depth
            # exactly one word of the question differs (T -> T')
            wa, wb = a.question.split(), b.question.split()
            assert len(wa) == len(wb) and sum(x != y for x, y in zip(wa, wb)) == 1
        else:
            assert b.meta["target"] == m["neg_target"] and b.meta["neg_target"] == m["target"]
            assert b.meta["root"] != m["root"]
    assert made >= 10


def test_prosqa_target_swap_skips_records_without_partner():
    a = prosqa_examples(1)[0]
    m = dict(a.meta)
    # a chain root -> 2 -> 3(target), 1 -> 4 (neg): no other node at depth 2
    m.update(idx_to_symbol=["Ann", "Bob", "wumpus", "lempus", "rompus"], edges=[[0, 2], [2, 3], [1, 4]], root=0,
             target=3, neg_target=4)
    ex = Example(question="Ann is a wumpus. Every wumpus is a lempus. Bob is a rompus. Is Ann a lempus or rompus?",
                 answer="Ann is a lempus.", steps=["Ann is a wumpus.", "Every wumpus is a lempus."], idx=0, meta=m)
    assert make_pair(ex, "target_swap") is None
    b = make_pair(ex, "root_swap")
    assert b is not None and b.answer == "Bob is a rompus." and b.question.endswith("Is Bob a lempus or rompus?")


def test_pairs_deterministic():
    a = prosqa_examples(5)[3]
    assert make_pair(a, "target_swap", seed=1).question == make_pair(a, "target_swap", seed=1).question


@pytest.mark.parametrize("rule", ["target_swap", "root_swap"])
def test_reach_pairs_flip(rule):
    made = 0
    for a in make_reach_data(40, seed=2):
        b = make_pair(a, rule, seed=0)
        if b is None:
            continue
        made += 1
        assert b.meta["edges"] == a.meta["edges"] and b.meta["idx_to_symbol"] == a.meta["idx_to_symbol"]
        assert b.question.rpartition(" from ")[0] == a.question.rpartition(" from ")[0]
        assert b.answer != a.answer and {a.answer, b.answer} == {"yes", "no"}
        if rule == "target_swap":
            assert b.meta["root"] == a.meta["root"] and b.meta["query"] != a.meta["query"]
        else:
            assert b.meta["query"] == a.meta["query"] and b.meta["root"] != a.meta["root"]
    assert made >= 20


def test_unknown_rule_raises():
    with pytest.raises(ValueError):
        make_pair(prosqa_examples(1)[0], "nope")


def test_answer_option():
    assert answer_option("Sally is a sterpus.", "prosqa") == "sterpus"
    assert answer_option("Sally is a sterpus .", "prosqa") == "sterpus"  # FakeTokenizer spacing
    assert answer_option("", "prosqa") is None
    assert answer_option("yes", "reach") == "yes" and answer_option("maybe", "reach") is None


def test_edit_specs():
    t = FakeTokenizer()
    a = prosqa_examples(1)[0]
    r = example_targets(a, 3, 1, t)
    sp = {s["kind"]: s for s in edit_specs(r)}
    names = a.meta["idx_to_symbol"]
    assert sp["add_neg"]["symbol"] == a.meta["neg_target"] and sp["add_neg"]["sign"] == 1.0
    assert sp["remove_target"]["symbol"] == a.meta["target"] and sp["remove_target"]["sign"] == -1.0
    assert sp["add_neg"]["predicted"] == sp["remove_target"]["predicted"] == names[a.meta["neg_target"]]
    for ex in make_reach_data(10, seed=0):
        (s,) = edit_specs(example_targets(ex, 3, 1, t))
        assert s["predicted"] == ("no" if ex.meta["reachable"] else "yes")
        assert s["sign"] == (-1.0 if ex.meta["reachable"] else 1.0)
