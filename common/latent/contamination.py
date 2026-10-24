"""13-gram train/test overlap: the fraction of test items that share at least one word 13-gram with the training set.

Approach reused from the distillation-ruler experiment's registered contamination check
(experiments/2026-10-08-distillation-ruler, governance/ruler_registration.md; code in
common/sprint1/scripts/ruler_checks.py, `span_set` / the `span_overlap_32` block): hash every verbatim span of the
training pool into a set, then look up the evaluation spans. Differences, for this task: spans are 13 words (the
GPT-3 contamination convention named in the silent-reasoning plan) instead of 32 GPT-2 tokens, every train position
is a span start (stride 1, not 8), spans never cross a record boundary, the train side of a record is its question
followed by its steps, the test side is its question, and the result is reported per test item (any shared span)
rather than per span. A second, secondary rate uses question + steps + answer on the test side (registration A1).
Hashes are 64-bit, computed with numpy (a position-salted splitmix64 sum over word ids), so the
385k-record GSM8k-Aug pool fits in memory.

Normalization: lowercase, words = maximal runs of [a-z0-9] (punctuation and whitespace ignored), as in the GPT-3
13-gram analysis. Items shorter than 13 words cannot overlap and are counted in the denominator.

CLI: PYTHONPATH=common python3 -m latent.contamination TRAIN.json TEST.json [--n 13]
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np

_WORD = re.compile(r"[a-z0-9]+")
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


def normalize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _get(rec, key):
    return rec[key] if isinstance(rec, dict) else getattr(rec, key)


def _text(rec, fields) -> str:
    parts = []
    for f in fields:
        v = _get(rec, f)
        parts.extend(v if isinstance(v, list) else [v])
    return " ".join(parts)


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        x = x + _GOLDEN
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return x ^ (x >> np.uint64(31))


def _ngram_hashes(texts: list[str], vocab: dict, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(hashes, record index) of every word n-gram inside each text."""
    ids, rec = [], []
    for r, t in enumerate(texts):
        w = [vocab.setdefault(x, len(vocab)) for x in normalize(t)]
        ids.extend(w)
        rec.extend([r] * len(w))
    ids = np.asarray(ids, dtype=np.uint64)
    rec = np.asarray(rec, dtype=np.int64)
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, np.uint64), np.zeros(0, np.int64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h += _splitmix(ids[k:k + m] * np.uint64(1_000_003) + np.uint64(k))
    keep = rec[:m] == rec[n - 1:n - 1 + m]  # the span lies inside one record
    return h[keep], rec[:m][keep]


PRIMARY_TEST_FIELDS = ("question",)                       # registration 1.3 / A1: primary rate
SECONDARY_TEST_FIELDS = ("question", "steps", "answer")    # A1: second number


class TrainIndex:
    """Hashed set of every word n-gram of the training records (built once, queried for several test sets)."""

    def __init__(self, train_records, n: int = 13, train_fields=("question", "steps")):
        self.n, self.train_fields, self.vocab = n, tuple(train_fields), {}
        h, _ = _ngram_hashes([_text(r, train_fields) for r in train_records], self.vocab, n)
        self.hashes = np.unique(h)
        self.train_items = len(train_records)

    def query(self, test_records, test_fields=PRIMARY_TEST_FIELDS) -> dict:
        n = self.n
        vocab = dict(self.vocab)  # unseen test words get fresh ids that match nothing in train
        te, te_rec = _ngram_hashes([_text(r, test_fields) for r in test_records], vocab, n)
        hit = np.isin(te, self.hashes)
        n_test = len(test_records)
        flagged = np.unique(te_rec[hit])
        short = sum(1 for r in test_records if len(normalize(_text(r, test_fields))) < n)
        return {"n": n, "test_items": n_test, "test_items_with_overlap": int(len(flagged)),
                "fraction": float(len(flagged) / n_test) if n_test else 0.0, "test_items_shorter_than_n": short,
                "train_items": self.train_items, "train_unique_ngrams": int(len(self.hashes)),
                "test_ngrams": int(len(te)), "test_ngrams_shared": int(hit.sum()),
                "flagged_test_idx_head": [int(i) for i in flagged[:20]],
                "train_fields": list(self.train_fields), "test_fields": list(test_fields)}


def ngram_overlap(train_records, test_records, n: int = 13, train_fields=("question", "steps"),
                  test_fields=PRIMARY_TEST_FIELDS) -> dict:
    """Fraction of test items with at least one word n-gram (default 13) that also occurs in any training item.
    Records are dicts or latent.data.Example objects."""
    return TrainIndex(train_records, n, train_fields).query(test_records, test_fields)


def overlap_report(train_path: str, eval_paths: dict, n: int = 13) -> dict:
    """{name: {"primary": ..., "secondary": ...}} for each eval split against one train file: primary test side =
    question (registration 1.3 as amended by A1; the GSM8k-Aug 1% block applies to it), secondary test side =
    question + steps + answer."""
    with open(train_path) as f:
        idx = TrainIndex(json.load(f), n)
    out = {}
    for name, path in eval_paths.items():
        with open(path) as f:
            recs = json.load(f)
        out[name] = {"eval_file": path, "primary": idx.query(recs, PRIMARY_TEST_FIELDS),
                     "secondary": idx.query(recs, SECONDARY_TEST_FIELDS)}
    return {"train_file": train_path, "n": n, "train_fields": ["question", "steps"], "splits": out}


def overlap_files(train_path: str, test_path: str, n: int = 13, test_fields=PRIMARY_TEST_FIELDS) -> dict:
    with open(train_path) as f:
        train = json.load(f)
    with open(test_path) as f:
        test = json.load(f)
    return {"train": train_path, "test": test_path, **ngram_overlap(train, test, n=n, test_fields=test_fields)}


def main():
    ap = argparse.ArgumentParser(description="13-gram train/test overlap")
    ap.add_argument("train")
    ap.add_argument("test")
    ap.add_argument("--n", type=int, default=13)
    a = ap.parse_args()
    print(json.dumps(overlap_report(a.train, {"test": a.test}, a.n), indent=1))


if __name__ == "__main__":
    main()


__all__ = ["normalize", "TrainIndex", "ngram_overlap", "overlap_report", "overlap_files"]
