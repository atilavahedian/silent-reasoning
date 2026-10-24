"""latent.data against the port of the official Coconut dataset.py (tests/reference_dataset.py).

Every arm, stages 0..max_latent_stage+1, pad_latent_to_max both ways, c_thought 1 and 2, uniform_prob 0 and 0.3
(seeded), training and generation records, on 6 examples with 1 to 5 steps: record-level equality of input_ids,
labels, attention_mask, position_ids, idx, and equality of the collated batches against MyCollator.
"""
import copy
import itertools
import json
import os
import random

import pytest
import torch

from latent import data as D
from latent.tests import reference_dataset as R
from latent.tests.fake_tokenizer import FakeTokenizer

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

FIXTURE = [
    {"question": "Ann has 3 apples and buys 2 more. How many apples?", "steps": ["<<3+2=5>>"], "answer": "5"},
    {"question": "Every wumpus is a yumpus. Alex is a wumpus. Is Alex a yumpus or zumpus?",
     "steps": ["Alex is a wumpus.", "Every wumpus is a yumpus."], "answer": "Alex is a yumpus.",
     "idx_to_symbol": ["Alex", "wumpus", "yumpus", "zumpus"], "edges": [[0, 1], [1, 2]], "root": 0, "target": 2,
     "neg_target": 3},
    {"question": "A train goes 60 miles per hour for 2 hours then 30 for 1. How far?",
     "steps": ["<<60*2=120>>", "<<30*1=30>>", "<<120+30=150>>"], "answer": "1,050"},
    {"question": "Max is a lempus. Every lempus is a rompus. Every rompus is a tumpus. Every tumpus is a gorpus. "
                 "Is Max a gorpus?",
     "steps": ["Max is a lempus.", "Every lempus is a rompus.", "Every rompus is a tumpus.",
               "Every tumpus is a gorpus."], "answer": "True"},
    {"question": "Five numbers: 1 2 3 4 5. Sum them step by step.",
     "steps": ["<<1+2=3>>", "<<3+3=6>>", "<<6+4=10>>", "<<10+5=15>>", "<<15*1=15>>"], "answer": "15"},
    {"question": "Bob had 10 coins, lost 4, found 7. Coins now?",
     "steps": ["<<10-4=6>>", "<<6+7=13>>", "<<13*1=13>>"], "answer": "13"},
]
MAX_STAGES = (3, 6)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    path = tmp_path_factory.mktemp("fx") / "fixture.json"
    path.write_text(json.dumps(FIXTURE))
    tok = FakeTokenizer()
    ids = D.add_latent_tokens(tok)
    ref = R.get_dataset(str(path), tok)
    ours = D.tokenize_examples(D.load_examples(str(path)), tok)
    return {"path": str(path), "tok": tok, "ids": ids, "ref": ref, "ours": ours}


def _records_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert set(x) == set(y)
        for k in ("input_ids", "labels", "attention_mask", "position_ids", "idx"):
            assert x.get(k) == y.get(k), k


def _batches_equal(ba, bb):
    assert set(ba) == set(bb)
    for k in ba:
        assert ba[k].dtype == bb[k].dtype, k
        assert torch.equal(ba[k], bb[k]), k


def _reference_records(setup, arm, epoch, epochs_per_stage, c_thought, max_stage, pad, uniform_prob, seed, gen):
    cfg = R.arm_configs(arm, c_thought, max_stage, pad, uniform_prob)
    stage = R.scheduled_stage(cfg, epoch, epochs_per_stage)
    if arm == "coconut_no_curriculum":  # no reference code: the paper's "data from the last stage"
        stage = max_stage + 1
    nsm = R.no_special_marker(cfg)
    i = setup["ids"]
    if gen:
        return R.get_question_latent_dataset(stage, setup["ref"], cfg, i["start_id"], i["latent_id"], i["end_id"],
                                             no_special_marker=nsm)
    random.seed(seed)
    return R.get_cot_latent_dataset(stage, setup["ref"], cfg, i["start_id"], i["latent_id"], i["end_id"],
                                    no_special_marker=nsm)


def test_tokenize_matches_reference(setup):
    assert setup["ours"] == setup["ref"]


def test_load_examples_meta_and_idx(setup):
    ex = D.load_examples(setup["path"])
    assert [e.idx for e in ex] == list(range(6))
    assert ex[1].meta == {k: FIXTURE[1][k] for k in ("idx_to_symbol", "edges", "root", "target", "neg_target")}
    assert ex[0].meta == {}
    assert [e.idx for e in D.load_examples(setup["path"], max_size=2)] == [0, 1]


CASES = [(arm, ms, stage, pad, c, up, gen)
         for arm in D.ARMS for ms in MAX_STAGES for stage in range(ms + 2) for pad in (True, False)
         for c in (1, 2) for up in (0.0, 0.3) for gen in (False, True)
         if not (gen and up > 0)]


@pytest.mark.parametrize("arm,max_stage,stage,pad,c_thought,uniform_prob,gen", CASES)
def test_stage_dataset_and_collator_match_reference(setup, arm, max_stage, stage, pad, c_thought, uniform_prob, gen):
    eps = 5
    epoch = stage * eps + 2  # an epoch inside the stage
    seed = 1000 * stage + 7 * c_thought + int(pad) + (13 if uniform_prob else 0)
    ref = _reference_records(setup, arm, epoch, eps, c_thought, max_stage, pad, uniform_prob, seed, gen)
    i = setup["ids"]
    ours = D.build_stage_dataset(setup["ours"], stage=epoch // eps, arm=arm, c_thought=c_thought,
                                 max_latent_stage=max_stage, pad_latent_to_max=pad, uniform_prob=uniform_prob,
                                 latent_id=i["latent_id"], start_id=i["start_id"], end_id=i["end_id"],
                                 rng=random.Random(seed), for_generation=gen)
    _records_equal(ours, ref)

    tok = setup["tok"]
    tok.padding_side = "right"
    ref_coll = R.MyCollator(tok, latent_id=i["latent_id"], label_pad_token_id=-100)
    our_coll = D.Collator(tok.pad_token_id, i["latent_id"], label_pad=-100)
    for chunk in ([0, 1, 2, 3, 4, 5], [5, 0, 3], [1, 4], [2]):
        feats = [ours[j] for j in chunk]
        before = copy.deepcopy(feats)
        bo = our_coll(feats)
        assert feats == before  # the collator does not mutate its input
        br = ref_coll(copy.deepcopy([ref[j] for j in chunk]))
        _batches_equal(bo, br)


def test_uniform_prob_mixes_stages(setup):
    i = setup["ids"]
    kw = dict(arm="coconut", c_thought=1, max_latent_stage=3, pad_latent_to_max=True, latent_id=i["latent_id"],
              start_id=i["start_id"], end_id=i["end_id"])
    base = D.build_stage_dataset(setup["ours"] * 20, stage=2, uniform_prob=0.0, rng=random.Random(0), **kw)
    mixed = D.build_stage_dataset(setup["ours"] * 20, stage=2, uniform_prob=0.3, rng=random.Random(0), **kw)
    n_lat = lambda r: r["input_ids"].count(i["latent_id"])
    assert {n_lat(r) for r in base} == {2}
    assert len({n_lat(r) for r in mixed}) > 1


def test_label_mask_layout(setup):
    """-100 exactly on question, <start>, latents, <end>; supervised on remaining steps and answer."""
    i = setup["ids"]
    s = setup["ours"][4]  # 5 steps
    rec = D.build_stage_dataset([s], stage=2, arm="coconut", c_thought=2, max_latent_stage=3,
                                pad_latent_to_max=True, uniform_prob=0.0, latent_id=i["latent_id"],
                                start_id=i["start_id"], end_id=i["end_id"], rng=random.Random(0))[0]
    q = s["question_tokenized"]
    rest = list(itertools.chain.from_iterable(s["steps_tokenized"][2:])) + s["answer_tokenized"]
    assert rec["input_ids"] == q + [i["start_id"]] + [i["latent_id"]] * 4 + [i["end_id"]] + rest
    assert rec["labels"] == [-100] * (len(q) + 6) + rest
    assert rec["labels"][-1] == setup["tok"].eos_token_id

    cot = D.build_stage_dataset([s], stage=3, arm="cot", c_thought=1, max_latent_stage=3, pad_latent_to_max=True,
                                uniform_prob=0.0, latent_id=i["latent_id"], start_id=i["start_id"],
                                end_id=i["end_id"], rng=random.Random(0))[0]
    assert cot["labels"] == [-100] * len(q) + cot["input_ids"][len(q):]
    assert i["latent_id"] not in cot["input_ids"] and i["start_id"] not in cot["input_ids"]

    nocot = D.build_stage_dataset([s], stage=3, arm="no_cot", c_thought=1, max_latent_stage=3,
                                  pad_latent_to_max=True, uniform_prob=0.0, latent_id=i["latent_id"],
                                  start_id=i["start_id"], end_id=i["end_id"], rng=random.Random(0))[0]
    assert nocot["input_ids"] == q + s["answer_tokenized"]


def test_collator_alignment(setup):
    i = setup["ids"]
    recs = D.build_stage_dataset(setup["ours"], stage=1, arm="coconut", c_thought=1, max_latent_stage=3,
                                 pad_latent_to_max=True, uniform_prob=0.0, latent_id=i["latent_id"],
                                 start_id=i["start_id"], end_id=i["end_id"], rng=random.Random(0))
    b = D.Collator(setup["tok"].pad_token_id, i["latent_id"])(recs)
    cols = [(row == i["latent_id"]).nonzero()[0].item() for row in b["input_ids"]]
    assert len(set(cols)) == 1
    for r in range(len(recs)):
        n_left = cols[0] - recs[r]["input_ids"].index(i["latent_id"])
        assert b["attention_mask"][r, :n_left].sum() == 0
        assert (b["labels"][r, :n_left] == -100).all()
        assert b["position_ids"][r, n_left:n_left + 3].tolist() == [0, 1, 2]


def test_answer_extract():
    assert D.answer_extract("Q?\n<<1+1=2>>\n### 1,234 ") == "1234"
    assert D.answer_extract("Is Tom a lempus or scrompus?\n### Tom is a lempus.") == "Tom is a lempus."
    assert D.answer_normalize(" 1,050") == "1050"
    assert D.cot_extract("Q?\n<<1+1=2>>\n<<2+2=4>>\n### 4") == "<<1+1=2>>\n<<2+2=4>>"


@pytest.mark.parametrize("name", ["prosqa_valid", "prontoqa_valid", "gsm_valid"])
def test_real_data_matches_reference(name, tmp_path):
    path = os.path.join(REPO, "data", f"{name}.json")
    if not os.path.exists(path):
        pytest.skip(f"{path} absent (run code/fetch_data.py)")
    head = json.load(open(path))[:64]
    p = tmp_path / "head.json"
    p.write_text(json.dumps(head))
    tok = FakeTokenizer()
    i = D.add_latent_tokens(tok)
    ref = R.get_dataset(str(p), tok)
    ours = D.tokenize_examples(D.load_examples(str(p)), tok)
    assert ours == ref
    for arm in ("cot", "no_cot", "coconut"):
        for stage in (0, 2, 7):
            cfg = R.arm_configs(arm, 1, 6, True, 0.0)
            st = R.scheduled_stage(cfg, stage * 5, 5)
            random.seed(0)
            r = R.get_cot_latent_dataset(st, ref, cfg, i["start_id"], i["latent_id"], i["end_id"],
                                         no_special_marker=R.no_special_marker(cfg))
            o = D.build_stage_dataset(ours, stage=stage, arm=arm, c_thought=1, max_latent_stage=6,
                                      pad_latent_to_max=True, uniform_prob=0.0, latent_id=i["latent_id"],
                                      start_id=i["start_id"], end_id=i["end_id"], rng=random.Random(0))
            _records_equal(o, r)
