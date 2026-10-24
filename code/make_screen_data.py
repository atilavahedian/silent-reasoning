"""Phase 2 screen data (registration A3.4.5; PLAN.md amendment A1 item 2): ProsQA-hard and the GSM8k-Aug 50k subset.

    PYTHONPATH=common python3 code/make_screen_data.py [--data-dir data]

ProsQA-hard. `latent.prosqa_gen.generate_record` records from ONE stream `random.Random(0)` that produces train,
valid and test in that order (10,000 / 300 / 500). Per record, from that stream: the proof depth uniformly from
{5, 6, 7} (`rng.choice`), then the generator's default distractor draw d0 = clip(round(gauss(37 - (depth - 3), 6)),
15, 63) - depth (`rng.gauss`), and the record with n_distractor_edges = 2 x d0. A record the generator cannot place
(RuntimeError, or a d0 below the graph's mandatory edges: ValueError) is redrawn with a new d0 at the same depth (the
depth distribution stays uniform); the redraw count is reported. Every record must pass `prosqa_gen.validate_record`
(the proof walks the record's own edges, is a BFS shortest path, the negative target is unreachable from the root, the
question lists every edge) and an explicit BFS check; no question may repeat across splits (asserted).
**Context cap (deviation from A3.4.5, logged in the report):** GPT-2 has 1,024 positions; at the doubled distractor
count about 0.9% of records have questions over 887 GPT-2 tokens, and 0.16% of training sequences exceed 1,024 (a run
would stop on a position-index error), so a record whose question (+ "\n", the tokenisation of latent.data) exceeds
MAX_QUESTION_TOKENS = 1024 - 2 markers - 7 latents - 128 new tokens = 887 GPT-2 tokens is redrawn like an unplaceable one
(new draw, same depth), so every arm's prompt plus the registered generation budget fits; the count is reported.
Tokenizer: the GPT-2 BPE files at common/babylm/recipes/sprint1_port/gpt2_tokenizer (sha256 in the manifest). Same JSON schema
and key order as the released ProsQA files (question, answer, steps, idx_to_symbol, edges, root, target,
neg_target), written with json.dump (no indent), as the released files are.

GSM-50k. The first 50,000 records of `numpy.random.default_rng(0).permutation(len(gsm_train.json))` over
data/gsm_train.json in file order, written in permutation order to data/gsm50k_train.json; valid and test are the
oracle's data/gsm_valid.json and data/gsm_test.json (referenced, not copied). The manifest gets the sha256 of the
sorted index list, serialised as "\\n".join(str(i) for i in sorted(indices)) (UTF-8, no trailing newline).

Both append to configs/data_manifest.json under a top-level "screen" block (path, sha256, bytes, records, the rule,
realised statistics, 13-gram rates); the existing entries are not altered (checked: every pre-existing key is
re-read and compared after the write). Data stays in data/ (gitignored).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics as st
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common"))

import numpy as np  # noqa: E402

from latent import prosqa_gen as G  # noqa: E402
from latent.contamination import ngram_overlap  # noqa: E402

MANIFEST = os.path.join(EXP, "configs", "data_manifest.json")
GPT2_TOKENIZER = os.path.join(REPO, "common", "babylm", "recipes", "sprint1_port", "gpt2_tokenizer")
MAX_QUESTION_TOKENS = 1024 - 2 - 7 - 128
SPLITS = (("train", 10_000), ("valid", 300), ("test", 500))
DEPTHS = (5, 6, 7)
GSM_N = 50_000
SEED = 0


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def default_distractor_draw(rng: random.Random, depth: int) -> int:
    """The generator's default non-proof edge count for a record of this depth (prosqa_gen._try_record:
    total = clip(round(gauss(37 - (depth - 3), 6)), 15, 63); non-proof = total - depth)."""
    return min(63, max(15, round(rng.gauss(37.0 - (depth - 3), 6.0)))) - depth


def check_record(rec: dict) -> None:
    """validate_record plus the explicit BFS facts the registration names."""
    errs = G.validate_record(rec)
    if errs:
        raise AssertionError(f"invalid ProsQA-hard record: {errs}")
    dist = G.bfs_dist(len(rec["idx_to_symbol"]), rec["edges"], rec["root"])
    assert dist.get(rec["target"]) == len(rec["steps"]), "proof is not a BFS shortest path"
    assert rec["neg_target"] not in dist, "negative target reachable from the root"


def make_prosqa_hard(seed: int = SEED, splits=SPLITS, tokenizer=None, max_question_tokens=MAX_QUESTION_TOKENS):
    """Returns ({split: records}, info) from one random.Random(seed) stream (module docstring). tokenizer None
    disables the context cap."""
    rng = random.Random(seed)
    out, redraws, length_redraws, drawn = {}, 0, 0, []
    for name, n in splits:
        recs = []
        while len(recs) < n:
            depth = rng.choice(DEPTHS)
            while True:
                d0 = default_distractor_draw(rng, depth)
                try:
                    rec = G.generate_record(rng, depth=depth, n_distractor_edges=2 * d0)
                except (RuntimeError, ValueError):
                    redraws += 1
                    continue
                if tokenizer is not None and len(tokenizer.encode(rec["question"] + "\n")) > max_question_tokens:
                    length_redraws += 1
                    continue
                break
            check_record(rec)
            recs.append(rec)
            drawn.append(2 * d0)
        out[name] = recs
    seen = {}
    for name, recs in out.items():
        for r in recs:
            other = seen.setdefault(r["question"], name)
            assert other == name, f"duplicate question across {other} and {name}"
    dup_within = {name: len(recs) - len({r["question"] for r in recs}) for name, recs in out.items()}
    return out, {"redraws": redraws, "length_redraws": length_redraws, "requested_distractors_mean": st.mean(drawn),
                 "duplicates_within": dup_within}


def prosqa_stats(recs) -> dict:
    def ms(xs):
        return {"mean": round(st.mean(xs), 3), "sd": round(st.pstdev(xs), 3), "min": min(xs), "max": max(xs)}
    depth = [len(r["steps"]) for r in recs]
    return {"records": len(recs), "depth": ms(depth),
            "depth_counts": {str(d): depth.count(d) for d in sorted(set(depth))},
            "nodes": ms([len(r["idx_to_symbol"]) for r in recs]), "edges": ms([len(r["edges"]) for r in recs]),
            "distractor_edges": ms([len(r["edges"]) - len(r["steps"]) for r in recs]),
            "words_per_question": ms([len(r["question"].split()) for r in recs])}


def gsm_subset(train: list, n: int = GSM_N, seed: int = SEED):
    """First n of numpy.random.default_rng(seed).permutation(len(train)), in permutation order."""
    idx = np.random.default_rng(seed).permutation(len(train))[:n]
    return [train[int(i)] for i in idx], [int(i) for i in idx]


def index_sha256(indices) -> str:
    return hashlib.sha256("\n".join(str(i) for i in sorted(indices)).encode()).hexdigest()


def gsm_stats(recs) -> dict:
    steps = [len(r["steps"]) for r in recs]
    words = [len(r["question"].split()) for r in recs]
    return {"records": len(recs), "steps": {"mean": round(st.mean(steps), 3), "sd": round(st.pstdev(steps), 3),
                                            "counts": {str(k): steps.count(k) for k in sorted(set(steps))}},
            "words_per_question": {"mean": round(st.mean(words), 3), "sd": round(st.pstdev(words), 3)},
            "last_step_result_equals_answer_frac": round(sum(
                1 for r in recs if r["steps"] and r["steps"][-1].strip().endswith(">>")
                and r["steps"][-1].strip()[2:-2].rsplit("=", 1)[-1].replace(",", "").strip()
                == r["answer"].replace(",", "").strip()) / len(recs), 4)}


def _write(path, recs):
    with open(path, "w") as f:
        json.dump(recs, f)
    return {"path": os.path.relpath(path, REPO), "sha256": sha256_file(path), "bytes": os.path.getsize(path),
            "records": len(recs)}


def _overlap(train, test):
    r = ngram_overlap(train, test)
    return {k: r[k] for k in ("test_items", "test_items_with_overlap", "fraction")}


def update_manifest(block: dict, path: str = MANIFEST) -> None:
    """Adds / replaces the top-level "screen" block; every other key must read back unchanged."""
    with open(path) as f:
        text = f.read()
    man = json.loads(text)
    before = {k: v for k, v in man.items() if k != "screen"}
    man["screen"] = block
    with open(path, "w") as f:
        f.write(json.dumps(man, indent=1) + "\n")
    with open(path) as f:
        after = json.load(f)
    assert {k: v for k, v in after.items() if k != "screen"} == before, "manifest: existing entries changed"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data"))
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--tokenizer", default=GPT2_TOKENIZER, help="GPT-2 BPE files for the context cap")
    a = ap.parse_args(argv)
    t0 = time.time()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    assert tok.encode("Hello world") == [15496, 995] and len(tok) == 50257, "not the GPT-2 tokenizer"
    hard, info = make_prosqa_hard(tokenizer=tok)
    qtok = {name: [len(tok.encode(r["question"] + "\n")) for r in recs] for name, recs in hard.items()}
    longest = max(qt + sum(len(tok.encode(x + "\n")) for x in r["steps"]) + len(tok.encode("### " + r["answer"])) + 1
                  for name, recs in hard.items() for r, qt in zip(recs, qtok[name]))
    assert longest <= 1024 and max(max(v) for v in qtok.values()) <= MAX_QUESTION_TOKENS, longest
    files = [_write(os.path.join(a.data_dir, f"prosqa_hard_{name}.json"), recs) for name, recs in hard.items()]
    ov = {"vs_own_train": {s: _overlap(hard["train"], hard[s]) for s in ("valid", "test")}}
    released = os.path.join(a.data_dir, "prosqa_train.json")
    if os.path.exists(released):
        with open(released) as f:
            rel = json.load(f)
        ov["vs_prosqa_train"] = {s: _overlap(rel, hard[s]) for s in ("train", "valid", "test")}
    prosqa_block = {
        "rule": "registration A3.4.5: latent.prosqa_gen.generate_record, one random.Random(0) stream, train/valid/"
                "test in order 10000/300/500; depth = rng.choice((5, 6, 7)); n_distractor_edges = 2 x "
                "(clip(round(rng.gauss(37 - (depth - 3), 6)), 15, 63) - depth); unplaceable records and records "
                "over the context cap redrawn (new draw, same depth); every record validate_record + BFS; no "
                "cross-split duplicate question",
        "generator": "common/latent/prosqa_gen.py", "generator_sha256": sha256_file(G.__file__),
        "script": os.path.relpath(os.path.abspath(__file__), REPO), "script_sha256": sha256_file(__file__),
        "seed": SEED, "files": files, "redraws": info["redraws"], "length_redraws": info["length_redraws"],
        "context_cap": {"max_question_tokens": MAX_QUESTION_TOKENS,
                        "rule": "question + '\\n' in GPT-2 tokens <= 1024 - 2 markers - 7 latents - 128 new tokens; "
                                "deviation from A3.4.5 (GPT-2 has 1024 positions), redrawn at the same depth",
                        "tokenizer": os.path.relpath(a.tokenizer, REPO),
                        "tokenizer_sha256": {f: sha256_file(os.path.join(a.tokenizer, f))
                                             for f in sorted(os.listdir(a.tokenizer))},
                        "question_tokens": {name: {"mean": round(st.mean(v), 2), "max": max(v)}
                                            for name, v in qtok.items()},
                        "longest_cot_training_sequence_tokens": longest},
        "requested_distractors_mean": round(info["requested_distractors_mean"], 3),
        "duplicate_questions_within_split": info["duplicates_within"], "duplicate_questions_across_splits": 0,
        "stats": {name: prosqa_stats(recs) for name, recs in hard.items()}, "contamination_13gram": ov}
    print("prosqa_hard:", json.dumps({k: prosqa_block[k] for k in ("files", "redraws", "length_redraws", "context_cap",
                                                                    "stats")}), flush=True)

    with open(os.path.join(a.data_dir, "gsm_train.json")) as f:
        gtrain = json.load(f)
    sub, idx = gsm_subset(gtrain)
    gfile = _write(os.path.join(a.data_dir, "gsm50k_train.json"), sub)
    gref = {}
    for split in ("valid", "test"):
        p = os.path.join(a.data_dir, f"gsm_{split}.json")
        with open(p) as f:
            n = len(json.load(f))
        gref[split] = {"path": os.path.relpath(p, REPO), "sha256": sha256_file(p), "bytes": os.path.getsize(p),
                       "records": n}
    with open(os.path.join(a.data_dir, "gsm_test.json")) as f:
        gtest = json.load(f)
    gsm_block = {
        "rule": "registration 7.1 / A3.4.5: the first 50000 records of numpy.random.default_rng(0).permutation("
                "len(data/gsm_train.json)) over the file order, written in permutation order; valid and test are the "
                "oracle's files",
        "rng": "numpy.random.default_rng(0).permutation", "seed": SEED,
        "source": {"path": "data/gsm_train.json", "sha256": sha256_file(os.path.join(a.data_dir, "gsm_train.json")),
                   "records": len(gtrain)},
        "index_list_sha256": index_sha256(idx),
        "index_list_sha256_rule": "sha256 of '\\n'.join(str(i) for i in sorted(indices)), UTF-8, no trailing newline",
        "first_indices": idx[:5], "train": gfile, "valid": gref["valid"], "test": gref["test"],
        "stats": gsm_stats(sub), "contamination_13gram": {"test_primary": _overlap(sub, gtest)}}
    print("gsm50k:", json.dumps({k: gsm_block[k] for k in ("train", "index_list_sha256", "stats")}), flush=True)
    block = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seconds": round(time.time() - t0),
             "python": sys.version.split()[0], "numpy": np.__version__,
             "prosqa_hard": prosqa_block, "gsm50k": gsm_block}
    update_manifest(block, a.manifest)
    return block


if __name__ == "__main__":
    main()
