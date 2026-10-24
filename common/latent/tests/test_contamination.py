"""latent.contamination: 13-gram overlap on hand-made cases, and on the downloaded data when present."""
import os

import pytest

from latent import contamination as C
from latent.data import Example

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
WORDS = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen".split()


def rec(q, steps=()):
    return {"question": q, "steps": list(steps), "answer": "x"}


def test_exact_13gram_counts():
    train = [rec(" ".join(WORDS[:13]))]
    test = [rec(" ".join(WORDS[:13]).upper() + "!"),          # same 13 words, case and punctuation differ
            rec(" ".join(WORDS[1:14])),                        # shares only 12 words
            rec("short question"),
            rec("zero " + " ".join(WORDS[:13]) + " tail")]     # contains the 13-gram
    r = C.ngram_overlap(train, test)
    assert r["test_items_with_overlap"] == 2
    assert r["fraction"] == 0.5
    assert r["flagged_test_idx_head"] == [0, 3]
    assert r["test_items_shorter_than_n"] == 1


def test_steps_count_on_train_side_and_no_cross_record_spans():
    train = [rec("a b c", WORDS[:13])]  # the 13-gram lives in the steps
    assert C.ngram_overlap(train, [rec(" ".join(WORDS[:13]))])["fraction"] == 1.0
    split = [rec(" ".join(WORDS[:6])), rec(" ".join(WORDS[6:13]))]  # spans must not join two records
    assert C.ngram_overlap(split, [rec(" ".join(WORDS[:13]))])["fraction"] == 0.0
    ex = [Example(question=" ".join(WORDS[:13]), answer="x", steps=[], idx=0)]
    assert C.ngram_overlap(ex, ex)["fraction"] == 1.0


@pytest.mark.parametrize("task", ["prosqa", "prontoqa"])
def test_real_data_runs(task):
    tr, te = (os.path.join(REPO, "data", f"{task}_{s}.json") for s in ("train", "test"))
    if not (os.path.exists(tr) and os.path.exists(te)):
        pytest.skip(f"{task} data absent")
    r = C.overlap_files(tr, te)
    assert 0.0 <= r["fraction"] <= 1.0 and r["test_items"] > 0


def test_secondary_rate_uses_steps_and_answer():
    train = [rec(" ".join(WORDS[:13]))]
    test = [{"question": "q", "steps": WORDS[:7], "answer": " ".join(WORDS[7:13])}]
    idx = C.TrainIndex(train)
    assert idx.query(test, C.PRIMARY_TEST_FIELDS)["fraction"] == 0.0
    assert idx.query(test, C.SECONDARY_TEST_FIELDS)["fraction"] == 1.0
