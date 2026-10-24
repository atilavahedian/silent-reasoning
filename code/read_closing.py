"""The closing plan's reads (PLAN.md A2.2 item 1; critic entries 2026-10-21T20:21Z (a) and (c)): two compilations, each
written once with its provenance, never a verdict.

    <venv>/bin/python code/read_closing.py det2 \
        [--det2-dir results/evals/det2-eval-acctc --det2-read <TS>] [--run-dir RUN=DIR ...] [--out DIR]
    <venv>/bin/python code/read_closing.py pareto [--out results/screens]

`det2` (row (a) (2) and (3)): the 13.6 same-seed pair table on ProsQA (prosqa_cot_s0 / det1 / det2) from the committed
test reads (k1-eval-main's prosqa_cot_s0 read, det1-eval-main, det2's eval job in det1-eval-main's form: the registered
read `<run>_test_none` and its batch-1 audit `<run>_test_none_audit`), every file hashed: per member the project (the
checkpoint's bucket), the checkpoint sha256, the test accuracy with its 95% interval (6.2's binomial form), the GPU type
of the read, the batch-1 audit's flips, and with --run-dir the resumes, the selected epoch, the GPU type of training and
the valid curve (13.2); the per-example agreement counts of every pair; and the two readings of sigma_task(ProsQA) the row
fixed: strict (of record; the pair = det1, det2): d_nd = |acc(det2) - 75.8|, sigma_task = sqrt(SD^2 + d_nd^2) if d_nd >
0.2 x SD (0.72, i.e. 4 or more of 500 net) else SD, SD = the three GPU cot seeds' SD of the final K1 table (3.60, floor
1.0); and "one run" (the author's, not amended; the pair = prosqa_cot_s0, det1) beside it. det2's accuracy enters no
cell, no seed pool and no sigma_cell; sigma_screen carries no determinism term (row (a) (3)). Without det2's read the
strict reading is "pending" and the table may be written only to a scratch --out (a dry run): the pair table of record
is written once with both members.

`pareto` (row (c)): the Pareto read as a compilation from the committed records (no VM, no new read of any checkpoint):
per task the screen table of record (GSM-50k the fourth table, ProsQA-hard the third; committed clean), the committed
test read (results/screens/test_read.json) and the eval directories they cite (committed clean; every per-example file
hashed against the table or the test read). Three panels per task and split (accuracy against mean emitted tokens E',
mean FLOPs with the reference cost beside, mean sequential passes), valid and test in separate tables, never on one
axis. Points: per latent arm the final selection (the table's cost row: exact means from the plain read), the
earlier-stage all-epoch best on valid from the training log's row (valid accuracy, val_emitted_mean, FLOPs in the first
table's range form by read_screen's cost helpers on the means: no per-example record on valid, no band), the step sweep
n in 0, 1, 2, 4, 8 of the final checkpoint (the table's sweep and the reads' per-example files); the tuned bar, no_cot,
pause and plain depth 355M (their valid FLOPs the tables' ranges); on test every row of the test read (the primary
selections and the secondary earlier-stage reads, per-example). Example-bootstrap bands (10,000 draws, default_rng(0),
percentile 95%) on the points with per-example records. Every label is critic entry (c) (3)'s, verified quoted in a row
logged at 2026-10-21T20:21Z. It says where each point sits and names the differences as gaps; it decides nothing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "common"))

import read_screen as RS  # noqa: E402

ROW = "2026-10-21T20:21Z"                          # critic entries (a), (b), (c) and the budget row
Z = 1.96
N_BOOT, BOOT_SEED, BOOT_CHUNK = RS.N_BOOT, RS.BOOT_SEED, RS.BOOT_CHUNK
TASK_ORDER = ("gsm50k", "prosqa_hard")
TITLE = {"gsm50k": "GSM-50k", "prosqa_hard": "ProsQA-hard"}
# Row (c) (4): "no verdict, WIN, tie, 'efficient', 'dominates', 'better', 'frontier' as a claim" (whole words), with
# A11.3's words (read_screen.FORBIDDEN) refused in any output of this tool.
PARETO_FORBIDDEN = re.compile(r"\b(verdicts?|wins?|won|ties?|tied|efficient|efficiency|dominates?|dominated|better|"
                              r"frontier)\b", re.IGNORECASE)

# ------------------------------------------------------------------------------------------------- det2, row (a)
DET1_K, DET1_SHA = 379, "24eb701e476e59a0fbcee5865a5fb8fb0e4769390f2238907cd7191d0e304481"    # row (a) (3)
S0_K = 375                                                          # prosqa_cot_s0's test read (75.0; the K1 table)
MEMBERS = (("s0", "prosqa_cot_s0", "k1-eval-main"), ("det1", "prosqa_cot_det1_s0", "det1-eval-main"),
           ("det2", "prosqa_cot_det2_s0", None))
TYPES_LABEL = ("pair read on two GPU types (A3.2 (ii) parity passed on both types; (iii) the type printed per member; "
               "the batch-1 audits beside each bound the eval-path flips, det1 {flips} of {n})")
DET2_LABELS = [
    "d_nd is the pair's difference as read",
    "det2's accuracy enters no cell, no seed pool and no sigma_cell",
    "sigma_screen on GSM-50k and ProsQA-hard (A3.4) carries no determinism term: det2 changes no line of any screen "
    "table of record; the ProsQA K1 table is unchanged",
    "the strict pair's figure is the one of record for every ProsQA verdict not yet written; rises, never falls"]

# --------------------------------------------------------------------------------------------- Pareto, row (c)
L_SEL = "final selection (3.3 item 1): n latents = the trained count"
L_EARLY = ("earlier-stage checkpoint, n latents = its stage; writes the remaining steps (E' as recorded: {g0} to {g1} "
           "tokens on GSM-50k test, {p0} to {p1} on ProsQA-hard test); secondary read (3.3 item 3); decides nothing")
L_SWEEP = "step sweep n = k of the final checkpoint (7.3; n = 8 out-of-distribution)"
L_BAR = "tuned bar: the pick (3 LRs)"
L_NOCOT = "no_cot (floor; the 124M plain-depth point)"
L_PAUSE = "pause (control)"
L_PD = "plain depth 355M (7.7)"
L_MEANS = "from the training log's means (range); no per-example record on valid"
L_NOBAND = "no band"
UNTESTED_73 = ["The 7.3 FLOP multiplier: UNTESTED (no budgeted-CoT read exists; it would be a new read of the bar "
               "checkpoint, outside (c)).", "The 7.3 budgeted-CoT curve: UNTESTED."]
TABLES = {"gsm50k": ("results/screens/gsm50k/gsm50k_screen_with_hinge.json", "the fourth GSM-50k table"),
          "prosqa_hard": ("results/screens/prosqa_hard/prosqa_hard_screen_with_plain_depth.json",
                          "the third ProsQA-hard table")}
PANELS = (("tokens", "mean emitted tokens E'", "emitted_mean"), ("flops", "mean FLOPs (5.1; GFLOPs)", "flops_mean"),
          ("passes", "mean sequential passes (5.4 (b): the matched-passes side)", "passes_mean"))


class ReadError(Exception):
    pass


# ----------------------------------------------------------------------------------------------- shared helpers
def row_has(log_text: str, ts: str, text: str, what: str, later: bool = False) -> None:
    """read_screen._row_has on a log text: `text` quoted ("...") in a critic entry logged at ts (or, with later, at or
    after ts); lead rows never count."""
    rows = [ln for ln in log_text.splitlines() if ln.startswith(f"| {ts} |")]
    if not rows:
        raise ReadError(f"no critic_log.md row {ts} ({what})")
    if later:
        rows = [ln for ln in log_text.splitlines() if re.match(r"\| \d{4}-\d\d-\d\dT\d\d:\d\dZ \| ", ln)
                and ln[2:19] >= ts and not ln[22:].startswith("**Lead")]
    if not any(f'"{text}"' in ln for ln in rows):
        raise ReadError(f"{what}: the text built from the files is not in critic entry {ts}"
                        f"{' or a later critic entry' if later else ''}: {text!r}")


def committed_clean(path: str, what: str) -> dict:
    st = RS.git_path_state(path)
    if st["in_repo"] and (st["dirty"] or not st["commit"]):
        raise ReadError(f"{path}: {what} is not committed clean (write-once results, A10.2)")
    return {"path": os.path.relpath(os.path.abspath(path), REPO) if st["in_repo"] else os.path.abspath(path),
            "commit": st["commit"]}


def interval(k: int, n: int) -> dict:
    """6.2's form for one accuracy: p +- 1.96 SE (binomial), in points (read_test.interval)."""
    p = k / n
    se = 100 * math.sqrt(p * (1 - p) / n)
    return {"k": k, "n": n, "points": 100 * p, "se_points": se, "ci95": [100 * p - Z * se, 100 * p + Z * se]}


def bands(cols: dict, n_boot: int = N_BOOT, seed: int = BOOT_SEED, chunk: int = BOOT_CHUNK) -> dict:
    """Example-bootstrap percentile bands (95%) of the means of several per-example columns on the same draws: numpy
    default_rng(seed), chunks of rng.integers(0, n, size=(k, n)), exactly read_screen.values_bootstrap's draws (so each
    column's band equals values_bootstrap(column)'s percentile interval)."""
    import numpy as np
    names = list(cols)
    X = np.asarray([cols[k] for k in names], dtype=float)
    n = X.shape[1]
    if n < 2:
        raise ValueError("per-example values of length >= 2 needed")
    rng = np.random.default_rng(seed)
    means = np.empty((len(names), n_boot))
    for s0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s0)
        idx = rng.integers(0, n, size=(k, n))
        for j in range(len(names)):
            means[j, s0:s0 + k] = X[j][idx].mean(1)
    out = {}
    for j, nm in enumerate(names):
        lo, hi = (float(x) for x in np.percentile(means[j], [2.5, 97.5]))
        out[nm] = [lo, hi]
    return out


def check_words(*texts: str, pareto: bool = True) -> None:
    RS.check_words(*texts)
    for tx in texts if pareto else ():
        m = PARETO_FORBIDDEN.search(tx)
        if m:
            raise SystemExit(f"REFUSED: output contains {m.group(0)!r} (critic entry {ROW} (c) (4): no verdict word)")


def _git_head() -> dict:
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", os.path.relpath(__file__, REPO)],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"head": head, "tool_uncommitted": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "tool_uncommitted": None}


def write_once(out: str, name: str, table: dict, md: str, prov: dict, inp: RS.Inputs, pareto: bool = True) -> None:
    paths = {k: os.path.join(out, f"{name}{suf}") for k, suf in (("json", ".json"), ("md", ".md"),
                                                                  ("prov", "_provenance.json"))}
    if any(os.path.exists(p) for p in paths.values()):
        raise SystemExit(f"REFUSED: {' / '.join(p for p in paths.values() if os.path.exists(p))} exists; written once")
    import platform
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "torch", "transformers"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    prov.update(generated_utc=table["generated_utc"], git=_git_head(), versions=versions, inputs=inp.listing(),
                outputs={"json": os.path.basename(paths["json"]), "md": os.path.basename(paths["md"])})
    js = json.dumps(table, indent=1, default=float)
    check_words(md, js, json.dumps(prov, indent=1), pareto=pareto)
    os.makedirs(out, exist_ok=True)
    for key, text in (("json", js + "\n"), ("md", md)):
        with open(paths[key], "x") as f:
            f.write(text)
    prov["outputs_sha256"] = {k: hashlib.sha256(open(paths[k], "rb").read()).hexdigest() for k in ("json", "md")}
    with open(paths["prov"], "x") as f:
        f.write(json.dumps(prov, indent=1) + "\n")


def _correct(rows: list[dict]) -> list[bool]:
    return [bool(r["correct"]) for r in sorted(rows, key=lambda r: r["idx"])]


# ------------------------------------------------------------------------------------------------- det2, row (a)
def read_member(inp: RS.Inputs, d: str, run: str, manifest: dict, log_text: str, sha: str | None = None,
                row: str | None = None) -> dict:
    """One member's registered test read and its batch-1 audit, det1-eval-main's form (JOB_RECEIPT.json, <run>_test_none
    and <run>_test_none_audit with summary.json and examples.csv), every file hashed; ReadError on any check."""
    rp = os.path.join(d, "JOB_RECEIPT.json")
    if not os.path.isfile(rp):
        raise ReadError(f"{rp} absent")
    rc = inp.read_json(rp, f"eval job receipt ({run})")
    specs = {e.get("name"): e for e in (rc.get("spec") or {}).get("evals") or []}
    done = {e.get("name"): e for e in rc.get("evals") or []}
    test = next((e for e in ((manifest.get("tasks") or {}).get("prosqa") or {}).get("files", [])
                 if e.get("path") == "data/prosqa_test.json"), None)
    if test is None:
        raise ReadError("the manifest has no data/prosqa_test.json entry (tasks.prosqa.files)")
    out = {"run": run, "dir": os.path.relpath(os.path.abspath(d), REPO) if os.path.abspath(d).startswith(REPO + os.sep)
           else os.path.abspath(d), "job": rc.get("job"), "attempt": rc.get("attempt")}
    reads = {}
    for tag, name in (("read", f"{run}_test_none"), ("audit", f"{run}_test_none_audit")):
        sp, dn = specs.get(name), done.get(name) or {}
        if sp is None or dn.get("status") not in ("done", "already_read") or dn.get("rc", 0) not in (0, None):
            raise ReadError(f"{d}: entry {name} absent from the receipt or not done")
        sm = inp.read_json(os.path.join(d, name, "summary.json"), f"test read ({run}, {tag})")
        csv_path = os.path.join(d, name, "examples.csv")
        csv_sha = inp.sha256(csv_path, f"test read ({run}, {tag}; per-example file)")
        rows = RS.parse_examples_csv(inp.read_text(csv_path, f"test read ({run}, {tag}; per-example file)"))
        k = sum(1 for r in rows if r["correct"])
        n = len(rows)
        probs = []
        if sm.get("examples_csv_sha256") != csv_sha:
            probs.append("examples.csv differs from the summary's hash")
        same_run = sm.get("run") == run or (sm.get("run") == run.removesuffix("_s0") and sm.get("seed") == 0)
        if not same_run or sm.get("split") != "test" or sm.get("intervention") != "none":   # K1 reads: config name
            probs.append(f"summary run {sm.get('run')}, split {sm.get('split')}, intervention {sm.get('intervention')}")
        if sm.get("split_sha256") != test.get("sha256"):
            probs.append("the test split's hash is not the manifest's")
        if [r["idx"] for r in rows] != list(range(n)) or int((sm.get("summary") or {}).get("n", -1)) != n:
            probs.append("not one row per test example in file order")
        if abs(float((sm.get("summary") or {}).get("accuracy", -1)) - k / max(n, 1)) > 1e-9:
            probs.append("the summary's accuracy is not the file's count")
        if sha is not None and sm.get("checkpoint_sha256") != sha:
            probs.append(f"checkpoint {str(sm.get('checkpoint_sha256'))[:12]}... is not {sha[:12]}...")
        if sp.get("checkpoint_sha256") != sm.get("checkpoint_sha256"):
            probs.append("the receipt's checkpoint is not the summary's")
        if tag == "audit" and not (sp.get("audit_rescore") is True and sp.get("batch_size") == 1):
            probs.append("the audit entry is not audit_rescore at batch 1")
        reg = sm.get("registered_read")
        if row is not None and reg != row:
            probs.append(f"registered read {reg}, not {row}")
        rrows = [ln for ln in log_text.splitlines() if ln.startswith(f"| {reg} |")]
        if not rrows or not any(str(sm.get("checkpoint_sha256")) in ln for ln in rrows):
            probs.append(f"no critic entry {reg} naming the checkpoint sha256")
        if probs:
            raise ReadError(f"{run} {tag} ({name}): " + "; ".join(probs))
        reads[tag] = {"entry": name, "k": k, "n": n, "rows": rows, "summary": sm, "csv_sha256": csv_sha}
    a, b = _correct(reads["read"]["rows"]), _correct(reads["audit"]["rows"])
    ck = reads["read"]["summary"]["checkpoint_sha256"]
    if reads["audit"]["summary"].get("checkpoint_sha256") != ck:
        raise ReadError(f"{run}: the audit reads another checkpoint")
    bucket = re.match(r"store://([^/]+)/", str((specs.get(f"{run}_test_none") or {}).get("checkpoint", "")))
    out.update(checkpoint_sha256=ck, project=(bucket.group(1).removesuffix("-bucket") if bucket else None),
               registered_read=reads["read"]["summary"].get("registered_read"),
               read_gpu=reads["read"]["summary"].get("gpu_type"), precision=reads["read"]["summary"].get("precision"),
               test=interval(reads["read"]["k"], reads["read"]["n"]), audit_k=reads["audit"]["k"],
               audit_flips=sum(x != y for x, y in zip(a, b)), correct=a)
    return out


def read_record(inp: RS.Inputs, d: str, run: str) -> dict:
    """--run-dir RUN=DIR: RUN_DONE.json (resumes, the selection), log.jsonl (the valid curve; rows of record, A8.2) and
    runs_index_row.csv (the training GPU type) when present."""
    rd = inp.read_json(os.path.join(d, "RUN_DONE.json"), f"run record ({run})")
    if rd.get("run") != run or rd.get("status") != "done":
        raise ReadError(f"{d}: RUN_DONE.json is not {run} done")
    rows, _ = RS.rows_of_record([json.loads(x) for x in inp.read_text(os.path.join(d, "log.jsonl"),
                                                                       f"run record ({run})").splitlines() if x.strip()])
    gpu = None
    p = os.path.join(d, "runs_index_row.csv")
    if os.path.isfile(p):
        gpu = next(csv.DictReader(io.StringIO(inp.read_text(p, f"run record ({run})")))).get("gpu_type")
    return {"resumes": int(rd.get("resumes", 0)), "selected_epoch": ((rd.get("best") or {}).get("final") or [None, None])[1],
            "checkpoint": (rd.get("checkpoints") or {}).get("ckpt_best_final.pt"), "train_gpu": gpu,
            "curve": [{"epoch": int(r["epoch"]), "val_acc": r["val_acc"]} for r in rows]}


def agreement(a: list[bool], b: list[bool]) -> dict:
    return {"both": sum(x and y for x, y in zip(a, b)), "only_a": sum(x and not y for x, y in zip(a, b)),
            "only_b": sum(y and not x for x, y in zip(a, b)), "neither": sum(not x and not y for x, y in zip(a, b)),
            "churn": sum(x != y for x, y in zip(a, b)), "net": sum(a) - sum(b), "n": len(a)}


def sigma_reading(k_a: int, k_b: int, n: int, sd: float) -> dict:
    """Row (a) (3): d_nd = |acc(a) - acc(b)| in points; sigma_task = sqrt(SD^2 + d_nd^2) if d_nd > 0.2 SD, else SD (floor
    1.0)."""
    d = abs(k_a - k_b) * 100 / n
    base = max(1.0, sd)
    line = 0.2 * sd
    return {"d_nd": d, "net": abs(k_a - k_b), "line": line, "over": d > line,
            "sigma_task": math.sqrt(base ** 2 + d ** 2) if d > line else base, "sd": sd}


def det2_table(a, inp: RS.Inputs) -> tuple[dict, str]:
    log_text = inp.read_text(a.critic_log, "rules (critic log)")
    manifest = inp.read_json(a.manifest, "data manifest (test split hash)")
    k1 = inp.read_json(a.k1_table, "the final K1 table (the ProsQA cot 3-seed SD)")
    row = next((r for r in k1.get("rows", []) if (r.get("task"), r.get("arm")) == ("prosqa", "cot")), None)
    if not row or len(row.get("seeds") or {}) != 3 or row.get("sd") is None:
        raise ReadError("the final K1 table has no ProsQA cot 3-seed SD")
    sd = float(row["sd"])
    dirs = {"s0": a.s0_dir, "det1": a.det1_dir, "det2": a.det2_dir}
    rec_dirs = dict(x.split("=", 1) for x in a.run_dir or [])
    members = {}
    for tag, run, _ in MEMBERS:
        if dirs[tag] is None:
            continue
        committed_clean(dirs[tag], f"{run}'s test read")
        m = read_member(inp, dirs[tag], run, manifest, log_text, sha=DET1_SHA if tag == "det1" else None,
                        row=a.det2_read if tag == "det2" else None)
        if run in rec_dirs:
            m["record"] = read_record(inp, rec_dirs[run], run)
            if m["record"]["checkpoint"] != m["checkpoint_sha256"]:
                raise ReadError(f"{run}: the read's checkpoint is not RUN_DONE's ckpt_best_final.pt")
            if tag == "det2" and m["record"]["resumes"] != 0:
                raise ReadError(f"{run}: {m['record']['resumes']} resumes: not a pair member (row (a) (1); A8.3)")
        members[tag] = m
    if members["det1"]["test"]["k"] != DET1_K or members["s0"]["test"]["k"] != S0_K:
        raise ReadError(f"det1 / prosqa_cot_s0 test counts {members['det1']['test']['k']} / {members['s0']['test']['k']} "
                        f"are not row (a) (3)'s {DET1_K} / {S0_K}")
    n = members["det1"]["test"]["n"]
    one_run = sigma_reading(members["det1"]["test"]["k"], members["s0"]["test"]["k"], n, sd)
    strict, labels = None, []
    if "det2" in members:
        strict = sigma_reading(members["det2"]["test"]["k"], DET1_K, n, sd)
        if members["det2"]["read_gpu"] != members["det1"]["read_gpu"]:
            lab = TYPES_LABEL.format(flips=members["det1"]["audit_flips"], n=n)
            row_has(log_text, ROW, lab, "row (a) (2)'s label on the pair's reads")
            labels.append(lab)
    pairs = {}
    for x, y in (("s0", "det1"), ("det1", "det2"), ("s0", "det2")):
        if x in members and y in members:
            pairs[f"{x}|{y}"] = agreement(members[x]["correct"], members[y]["correct"])
    for m in members.values():
        m.pop("correct")
    t = {"title": "The same-seed pair on ProsQA (13.6): prosqa_cot_s0 / det1 / det2, and sigma_task's two readings",
         "critic_row": f"{ROW} (a)", "members": members, "pairs": pairs, "sd_3seed": sd,
         "strict": strict, "one_run": one_run, "labels": labels + DET2_LABELS,
         "status": "complete" if strict else "pending: det2's read absent (dry run only)",
         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return t, det2_markdown(t)


def det2_markdown(t: dict) -> str:
    L = [f"# {t['title']}", "", f"Critic entry {t['critic_row']} parts (2) and (3); 1 seed per member (seed 0); test split "
         f"(500 items), each checkpoint read once (4.5). Labels: " + "; ".join(t["labels"]) + ".", "",
         "| member | run | project (bucket) | resumes | selected epoch | checkpoint | test (95% interval) | read on | "
         "trained on | batch-1 audit flips | registered read |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in ("s0", "det1", "det2"):
        m = t["members"].get(tag)
        if m is None:
            L.append(f"| {tag} | prosqa_cot_det2_s0 | | | | | pending (no read) | | | | |")
            continue
        r = m.get("record") or {}
        te = m["test"]
        L.append(f"| {tag} | {m['run']} | {m['project']} | {r.get('resumes', 'not read')} | "
                 f"{r.get('selected_epoch', 'not read')} | {m['checkpoint_sha256'][:12]}... | {te['points']:.1f} "
                 f"[{te['ci95'][0]:.1f}, {te['ci95'][1]:.1f}] ({te['k']}/{te['n']}) | {m['read_gpu']} | "
                 f"{r.get('train_gpu') or 'not read'} | {m['audit_flips']} of {te['n']} | {m['registered_read']} |")
    L += ["", "Per-example agreement (both correct / only the first / only the second / neither; churn = examples that "
          "differ in correctness; net = the count difference):", "", "| pair | both | only first | only second | "
          "neither | churn | net |", "|---|---|---|---|---|---|---|"]
    for k, p in t["pairs"].items():
        L.append(f"| {k.replace('|', ' / ')} | {p['both']} | {p['only_a']} | {p['only_b']} | {p['neither']} | "
                 f"{p['churn']} | {p['net']:+d} |")
    s, o = t["strict"], t["one_run"]
    L += ["", "## sigma_task(ProsQA), row (a) (3)", ""]
    if s:
        L.append(f"- Strict (of record; the pair = det1, det2): d_nd = |acc(det2) - 75.8| = {s['d_nd']:.1f} ({s['net']} "
                 f"of 500 net) {'>' if s['over'] else '<='} 0.2 x {s['sd']:.2f} = {s['line']:.2f}: sigma_task = "
                 + (f"sqrt({s['sd']:.2f}^2 + {s['d_nd']:.1f}^2) = {s['sigma_task']:.2f}" if s["over"] else
                    f"{s['sigma_task']:.2f} (the three GPU cot seeds' SD, floor 1.0; 6.1)"))
    else:
        L.append("- Strict (of record; the pair = det1, det2): pending: det2's registered read is absent.")
    L.append(f"- \"One run\" (the author's reading, not amended; the pair = prosqa_cot_s0, det1), printed beside: d_nd = "
             f"|75.8 - 75.0| = {o['d_nd']:.1f} ({o['net']} of 500 net), sigma_task {o['sigma_task']:.2f}.")
    curves = {tag: (m.get("record") or {}).get("curve") for tag, m in t["members"].items() if (m.get("record") or {})}
    if curves:
        eps = sorted({c["epoch"] for v in curves.values() for c in v})
        L += ["", "## Valid curves (13.2), val_acc by epoch", "", "| epoch | " + " | ".join(curves) + " |",
              "|---|" + "---|" * len(curves)]
        for e in eps:
            L.append(f"| {e} | " + " | ".join(next((f"{100 * c['val_acc']:.1f}" for c in v if c["epoch"] == e), "")
                                              for v in curves.values()) + " |")
    L += ["", f"Status: {t['status']}."]
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------------------------- Pareto, row (c)
def _csv_rows(inp: RS.Inputs, path: str, want_sha: str, what: str) -> list[dict]:
    if inp.sha256(path, what) != want_sha:
        raise ReadError(f"{path} does not hash to the {what}'s recorded sha256")
    return RS.parse_examples_csv(inp.read_text(path, what))


def _per_example_point(rows: list[dict]) -> dict:
    n = len(rows)
    cols = {"accuracy": [100.0 * r["correct"] for r in rows], "emitted_mean": [r["emitted_tokens"] for r in rows],
            "flops_mean": [r["flops"] for r in rows], "passes_mean": [r["sequential_passes"] for r in rows]}
    b = bands(cols)
    return {"n": n, "k": sum(1 for r in rows if r["correct"]), "emitted_mean": sum(cols["emitted_mean"]) / n,
            "flops_mean": math.fsum(cols["flops_mean"]) / n, "passes_mean": sum(cols["passes_mean"]) / n,
            "reference_flops_mean": math.fsum(r["reference_flops"] for r in rows) / n, "band": b}


def _rng(x) -> list:
    return [float(x[0]), float(x[1])] if isinstance(x, (list, tuple)) else [float(x), float(x)]


def pareto_task(inp: RS.Inputs, task: str, table: dict, test: dict, sc: RS.Screen) -> dict:
    """The points of one task (see the module docstring)."""
    n = table["n_valid"]
    runs = {r["key"]: r for r in table["runs"]}
    bar_key = table["bar"]["key"]
    cit = table.get("citations") or {}
    ev_dirs = [os.path.join(REPO, cit[k]["dir"]) for k in ("eval", "hinge_eval") if k in cit]
    for d in ev_dirs:
        committed_clean(d, "the eval reads")
    ent_sha = {e["name"]: e["examples_csv_sha256"] for blk in ("eval", "eval_hinge") for e in
               (table.get(blk) or {}).get("entries") or [] if e.get("ok")}

    def ev_rows(name):
        for d in ev_dirs:
            p = os.path.join(d, name, "examples.csv")
            if os.path.isfile(p):
                return _csv_rows(inp, p, ent_sha[name], f"valid read {name} (per-example file)")
        raise ReadError(f"{name}: no per-example file under {ev_dirs}")
    per = (table.get("eval") or {}).get("per_run") or {}
    valid, sentences, widths = [], [], []

    def fixed(key, label):
        r = runs[key]
        c = r["cost"]
        return {"id": f"{key}:selection", "run": r["run"], "key": key, "label": label, "kind": "log-mean range",
                "epoch": r["selected"]["epoch"], "n_latent": c["L"], "accuracy": r["points"], "k": r["k"], "n": n,
                "emitted_mean": c["emitted_mean"], "passes_mean": c["passes_mean"], "flops_mean": _rng(c["flops_mean"]),
                "reference_flops_mean": _rng(c["reference_flops_mean"]), "band": None,
                "labels": [L_MEANS, L_NOBAND]}
    valid.append(fixed(bar_key, L_BAR))
    valid.append(fixed("no_cot", L_NOCOT))
    if runs.get("plain_depth_355m", {}).get("status") == "read":
        valid.append(fixed("plain_depth_355m", L_PD))
    lat = [k for k, r in runs.items() if r.get("status") == "read" and not k.startswith("cot_lr")
           and k not in ("no_cot", "plain_depth_355m")]
    for key in lat:
        r = runs[key]
        c = r["cost"]
        none = (per.get(key) or {}).get("reads", {}).get("none")
        pe = _per_example_point(ev_rows(none["entry"])) if none else None
        fl = {"at 1": _rng(c["flops_mean"])}
        for k2, nm in (("cost_nominal", "nominal"), ("cost_measured", "measured")):
            if k2 in r:
                fl[f"{nm} {r[k2]['layer_fraction']:.4f}"] = _rng(r[k2]["flops_mean"])
        lf_read = float(r["cost_nominal"]["layer_fraction"]) if "cost_nominal" in r else 1.0   # C1b reads: the band
        sel = {"id": f"{key}:selection", "run": r["run"], "key": key, "label": L_PAUSE if key == "pause" else L_SEL,
               "kind": "per-example (plain read)" if pe else "log-mean range", "epoch": r["selected"]["epoch"],
               "n_latent": c["L"], "accuracy": r["points"], "k": r["k"], "n": n, "emitted_mean": c["emitted_mean"],
               "passes_mean": c["passes_mean"], "flops_mean": _rng(c["flops_mean"]), "flops_parts": fl,
               "reference_flops_mean": _rng(c["reference_flops_mean"]), "band": pe["band"] if pe else None,
               "labels": [] if pe else [L_NOBAND], "band_entry": none["entry"] if none else None,
               "band_k": pe["k"] if pe else None, "band_layer_fraction": lf_read}
        valid.append(sel)
        if r.get("earlier_stage_better"):
            b = r["best_all"]
            row = next(x for x in r["curve"] if int(x["epoch"]) == int(b["epoch"]))
            cfg = inp.read_json(os.path.join(EXP, next(s["config"] for s in RS.TASKS[task]["runs"] if s["key"] == key)),
                                "registered config")
            rec = {"config": cfg, "model_id": r["model_id"], "L": int(b["n_latent"]),
                   "M": RS.count_of(row["val_emitted_mean"], n, "val_emitted_mean"),
                   "feedback": cfg["arm"] in ("coconut", "coconut_no_curriculum")}
            cs = sc.costs(rec)
            fl = {"at 1": list(cs["flops_mean"])}
            for k2, nm in (("cost_nominal", "nominal"), ("cost_measured", "measured")):
                if k2 in r:
                    lf = r[k2]["layer_fraction"]
                    fl[f"{nm} {lf:.4f}"] = list(sc.costs(rec, layer_fraction=lf)["flops_mean"])
            lo, hi = cs["flops_mean"]
            w = (hi - lo) / ((hi + lo) / 2)
            widths.append({"run": r["run"], "width_frac": w})
            early = {"id": f"{key}:earlier", "run": r["run"], "key": key, "label": "earlier", "kind": "log-mean range",
                     "epoch": int(b["epoch"]), "stage": int(b["stage"]), "n_latent": int(b["n_latent"]),
                     "accuracy": 100 * b["val_acc"], "k": RS.count_of(b["val_acc"], n, "val_acc"), "n": n,
                     "emitted_mean": cs["emitted_mean"], "passes_mean": cs["passes_mean"],
                     "flops_mean": list(cs["flops_mean"]), "flops_parts": fl,
                     "reference_flops_mean": list(cs["reference_flops_mean"]), "band": None,
                     "labels": [L_MEANS, L_NOBAND], "range_width_frac": w}
            valid.append(early)
            sentences.append(("valid", sel, early))
        for nn, x in sorted(((int(k), v) for k, v in ((per.get(key) or {}).get("sweep") or {}).items()),
                            key=lambda kv: kv[0]):
            if x["plain_read"] or nn not in RS.SWEEP:
                continue
            pe2 = _per_example_point(ev_rows(x["entry"]))
            valid.append({"id": f"{key}:sweep{nn}", "run": r["run"], "key": key, "label": L_SWEEP, "kind":
                          "per-example", "epoch": r["selected"]["epoch"], "n_latent": nn, "ood": x["ood"],
                          "accuracy": x["points"], "k": x["k"], "n": n, "emitted_mean": x["emitted_mean"],
                          "passes_mean": x["passes_mean"], "flops_mean": [x["flops_mean"], x["flops_mean"]],
                          "reference_flops_mean": [pe2["reference_flops_mean"]] * 2, "band": pe2["band"],
                          "labels": [], "entry": x["entry"], "layer_fraction": lf_read})
    # test: every row of the test read, per-example
    rd = os.path.join(REPO, test["results_dir"]["dir"])
    committed_clean(rd, "the test reads")
    tpts = []
    by_run = {r["run"]: k for k, r in runs.items()}
    for row in test["rows"]:
        key = by_run.get(row["run"])
        if key is None or key.startswith("cot_lr") and key != bar_key:
            continue
        pe = _per_example_point(_csv_rows(inp, os.path.join(rd, row["entry"], "examples.csv"),
                                          row["examples_csv_sha256"], f"test read {row['entry']} (per-example file)"))
        if pe["k"] != row["test"]["k"]:
            raise ReadError(f"{row['entry']}: the per-example count {pe['k']} is not the test read's {row['test']['k']}")
        sm = inp.read_json(os.path.join(rd, row["entry"], "summary.json"), f"test read {row['entry']} (summary)")
        lf = float((sm.get("variant") or {}).get("layer_fraction", 1.0))
        sec = row["read"] == "secondary"
        label = ("earlier" if sec else L_BAR if key == bar_key else L_NOCOT if key == "no_cot" else
                 L_PD if key == "plain_depth_355m" else L_PAUSE if key == "pause" else L_SEL)
        tpts.append({"id": f"{key}:{'earlier' if sec else 'selection'}", "run": row["run"], "key": key, "label": label,
                     "kind": "per-example", "epoch": row["epoch"], "n_latent": row["n_latent"],
                     "accuracy": 100 * pe["k"] / pe["n"], "k": pe["k"], "n": pe["n"],
                     "emitted_mean": pe["emitted_mean"], "passes_mean": pe["passes_mean"],
                     "flops_mean": [pe["flops_mean"]] * 2, "reference_flops_mean": [pe["reference_flops_mean"]] * 2,
                     "layer_fraction": lf, "band": pe["band"], "labels": [], "entry": row["entry"]})
    for p in tpts:
        if p["label"] == "earlier":
            sel = next((q for q in tpts if q["key"] == p["key"] and q["label"] != "earlier"), None)
            if sel is not None:
                sentences.append(("test", sel, p))
    return {"valid": valid, "test": tpts, "widths": widths, "_sentences": sentences, "n_valid": n,
            "n_test": test["n_test"], "bar": table["bar"]}


def _gf(x) -> str:
    return f"{x / 1e9:.3f}"


def _fl(p) -> str:
    lo, hi = p["flops_mean"]
    return _gf(lo) if abs(hi - lo) <= 1e-9 * max(1.0, lo) else f"[{_gf(lo)}, {_gf(hi)}]"


def sentence(split: str, sel: dict, early: dict) -> str:
    """Row (c) (4)'s form: where the two points of one arm sit, the differences named as gaps."""
    da = early["accuracy"] - sel["accuracy"]
    de = early["emitted_mean"] - sel["emitted_mean"]
    f_e, f_s = (sum(early["flops_mean"]) / 2, sum(sel["flops_mean"]) / 2)
    return (f"{split}: the earlier-stage checkpoint of {early['run']} (epoch {early['epoch']}, n latents "
            f"{early['n_latent']}) scores {early['accuracy']:.1f} at {early['emitted_mean']:.2f} tokens and {_fl(early)} "
            f"GFLOPs; its selection (epoch {sel['epoch']}, n latents {sel['n_latent']}) {sel['accuracy']:.1f} at "
            f"{sel['emitted_mean']:.2f} and {_fl(sel)} (gaps: accuracy {da:+.1f} points, tokens {de:+.2f}, FLOPs "
            f"{(f_e - f_s) / 1e9:+.3f} GFLOPs{' at the range midpoints' if early['flops_mean'][0] != early['flops_mean'][1] else ''})")


def pareto_markdown(t: dict) -> str:
    L = [f"# {t['title']}", "", "Critic entry " + t["critic_row"] + ". " + " ".join(t["notes"][:2]), "",
         "Labels of the points (row (c) (3), verbatim): " + "; ".join(f'"{x}"' for x in t["point_labels"].values()) + ".",
         ""]
    for task in TASK_ORDER:
        b = t["tasks"][task]
        L += [f"## {TITLE[task]}", "", f"Valid of record: {b['table']['file']} at `{b['table']['commit'][:8]}` (valid "
              f"{b['n_valid']}); test: {t['test_read']['file']} at `{t['test_read']['commit'][:8]}` (test {b['n_test']}; "
              f"{t['test_label']})."]
        for split in ("valid", "test"):
            pts_ = b[split]
            for pid, title, ax in PANELS:
                L += ["", f"### {TITLE[task]}, {split}: accuracy against {title}", ""]
                hdr = "| point | run | epoch | n latents | accuracy | band (accuracy) | " + {
                    "tokens": "E' mean | band (E')", "flops": "FLOPs mean | band (FLOPs) | reference-cost FLOPs mean",
                    "passes": "passes mean | band (passes)"}[pid] + " |"
                L += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
                for p in sorted(pts_, key=lambda q: (sum(q[ax]) / 2 if isinstance(q[ax], list) else q[ax], q["id"])):
                    bd = p["band"]
                    lab = t["point_labels"]["earlier"] if p["label"] == "earlier" else p["label"]
                    lab = {L_SEL: "final selection", L_SWEEP: f"step sweep n = {p['n_latent']}"
                           + (" (OOD)" if p.get("ood") else ""), L_BAR: "tuned bar", L_NOCOT: "no_cot",
                           L_PAUSE: "pause (control)", L_PD: "plain depth 355M"}.get(lab, "earlier-stage checkpoint")
                    acc_b = f"[{bd['accuracy'][0]:.1f}, {bd['accuracy'][1]:.1f}]" if bd else L_NOBAND
                    cells = [lab + (" (from means)" if L_MEANS in p["labels"] else ""), p["run"], str(p["epoch"]),
                             str(p["n_latent"]), f"{p['accuracy']:.1f}", acc_b]
                    if pid == "tokens":
                        cells += [f"{p['emitted_mean']:.3f}", f"[{bd['emitted_mean'][0]:.3f}, {bd['emitted_mean'][1]:.3f}]"
                                  if bd else L_NOBAND]
                    elif pid == "flops":
                        parts = p.get("flops_parts")
                        fl = (_fl(p) if not parts or len(parts) == 1 else
                              "; ".join(f"{(_gf(v[0]) if v[0] == v[1] else '[' + _gf(v[0]) + ', ' + _gf(v[1]) + ']')} "
                                        f"{k}" for k, v in parts.items()))
                        if p.get("layer_fraction", 1.0) != 1.0:
                            fl += f" (the read's layer fraction {p['layer_fraction']:.4f}, as the file records)"
                        rf = p["reference_flops_mean"]
                        blf = p.get("band_layer_fraction", 1.0)
                        cells += [fl, (f"[{_gf(bd['flops_mean'][0])}, {_gf(bd['flops_mean'][1])}]"
                                       + (f" (the plain read's layer fraction {blf:.4f})" if blf != 1.0 else ""))
                                  if bd else L_NOBAND,
                                  _gf(rf[0]) if rf[0] == rf[1] else f"[{_gf(rf[0])}, {_gf(rf[1])}]"]
                    else:
                        cells += [f"{p['passes_mean']:.3f}", f"[{bd['passes_mean'][0]:.3f}, {bd['passes_mean'][1]:.3f}]"
                                  if bd else L_NOBAND]
                    L.append("| " + " | ".join(cells) + " |")
        L += ["", f"### {TITLE[task]}: where the earlier-stage points sit (descriptive; decides nothing)", ""]
        L += [f"- {s}" for s in b["sentences"]]
        w = max((x["width_frac"] for x in b["widths"]), default=0.0)
        L += [f"- The valid earlier-stage FLOPs are ranges over the unrecorded per-example E' (widest {100 * w:.3f}% of "
              f"the point's FLOPs" + ("; under 1%: no valid eval of ckpt_best.pt is needed by row (c) (2))" if w <= 0.01
                                       else "; above 1%: row (c) (2) lets a later row admit that eval)"), ""]
    L += ["## Not read here", ""] + [f"- {x}" for x in UNTESTED_73] + ["", "## Notes", ""] + \
        [f"- {x}" for x in t["notes"]]
    return "\n".join(L) + "\n"


def pareto_table(a, inp: RS.Inputs) -> tuple[dict, str]:
    log_text = inp.read_text(a.critic_log, "rules (critic log)")
    tr_path = a.test_read
    trc = committed_clean(tr_path, "the test read")
    test = inp.read_json(tr_path, "the test read (results/screens/test_read.json)")
    tasks, cites = {}, {}
    for task in TASK_ORDER:
        path = getattr(a, f"{task}_table") or os.path.join(EXP, TABLES[task][0])
        st = RS.git_path_state(path)
        if not st["in_repo"] or st["dirty"] or not st["commit"]:
            raise ReadError(f"{path}: {TABLES[task][1]} is not committed clean (it is cited by commit)")
        have = inp.sha256(path, f"{TABLES[task][1]} (valid of record)")
        if RS.committed_sha256(st["commit"], path) != have:
            raise ReadError(f"{path} differs from its content at {st['commit'][:12]} (write-once)")
        table = inp.read_json(path, f"{TABLES[task][1]} (valid of record)")
        tb = test["tasks"][task]
        if tb.get("valid_table", {}).get("sha256") != have:
            raise ReadError(f"{TITLE[task]}: the test read cites another valid table than {path}")
        sc = RS.Screen(task, ".", [])
        tasks[task] = pareto_task(inp, task, table, tb, sc)
        for f in sc.inp.listing():                      # the valid split, tokenizer and accounting code the costs read
            if f["root"] in ("repo", "other"):
                inp.sha256(os.path.join(REPO, f["path"]) if f["root"] == "repo" else f["path"], f["role"])
        cites[task] = {"file": os.path.relpath(os.path.abspath(path), REPO), "commit": st["commit"], "sha256": have}
    # the earlier-stage label's E' ranges (row (c) (3)): the test read's secondaries, rounded
    rng_ = {}
    for task in TASK_ORDER:
        es = [p["emitted_mean"] for p in tasks[task]["test"] if p["label"] == "earlier"]
        if not es:
            raise ReadError(f"{TITLE[task]}: no secondary test read: the earlier-stage label cannot be built")
        rng_[task] = (round(min(es)), round(max(es)))
    early = L_EARLY.format(g0=rng_["gsm50k"][0], g1=rng_["gsm50k"][1], p0=rng_["prosqa_hard"][0],
                           p1=rng_["prosqa_hard"][1])
    labels = {"selection": L_SEL, "earlier": early, "sweep": L_SWEEP, "bar": L_BAR, "no_cot": L_NOCOT, "pause": L_PAUSE,
              "plain_depth": L_PD, "means": L_MEANS, "no_band": L_NOBAND}
    for k, v in labels.items():
        row_has(log_text, ROW, v, f"row (c) (3)'s label ({k})")
    for task in TASK_ORDER:
        b = tasks[task]
        b["sentences"] = [sentence(split, s, e) for split, s, e in b.pop("_sentences")]
        b["table"] = cites[task]
    test_label = test["labels"][0]
    t = {"title": "The Pareto read (7.3: secondary, reported, decides nothing): accuracy against cost per task, valid and "
                  "test apart", "critic_row": f"{ROW} (c)", "point_labels": labels, "test_label": test_label,
         "test_read": {"file": trc["path"], "commit": trc["commit"]}, "tasks": tasks,
         "untested": UNTESTED_73,
         "notes": [
             "A compilation from the committed records (row (c) (2)): no VM, no new read of any checkpoint; it decides "
             "nothing and changes no line of any table.",
             "Valid and test are separate panels and never share an axis; every test point carries the test read's "
             f"label: {test_label}.",
             "Valid final selections: accuracy of record (the training log's selection, the screen table's); costs the "
             "table's cost row (exact means from the plain read's per-example file); the band from that plain read's "
             "per-example file. Valid earlier-stage points: the training log's epoch row (valid accuracy, "
             "val_emitted_mean), FLOPs by read_screen.Screen.costs on the logged mean (the exact range over every "
             "per-example E' with the logged total), passes exact (1 + latent steps + E'); no band.",
             "The bar, no_cot and plain depth 355M on valid: the tables' ranges (no per-example read of them on valid); "
             "on test every point is per-example (the test read's files). C1b FLOPs: on valid at the fractions the table "
             "charges (nominal, measured, at 1); on test at the read's own layer fraction, as the file records.",
             f"Bands: example bootstrap, {N_BOOT:,} draws, numpy default_rng({BOOT_SEED}), the 95% percentile interval of "
             "each mean on the same draws per point. FLOPs are 5.1's (the head charged as 5.4 says); the reference-cost "
             "FLOPs (5.5 (c)) beside them; the passes panel is 5.4 (b)'s matched-passes side.",
             "Not said here (row (c) (4)): no comparison of a test point with a valid point, no K3 beyond the tables' "
             "lines, nothing about what the latents carry (A11.3), no multiplier, no claim about the written steps of an "
             "earlier-stage checkpoint."],
         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return t, pareto_markdown(t)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("det2", help="row (a): the same-seed pair table and sigma_task's two readings")
    d.add_argument("--det1-dir", default=os.path.join(EXP, "results", "evals", "det1-eval-main"))
    d.add_argument("--s0-dir", default=os.path.join(EXP, "results", "evals", "k1-eval-main"))
    d.add_argument("--det2-dir", default=None, help="det2's eval job (det1-eval-main's form); absent: a dry run")
    d.add_argument("--det2-read", default=None, help="the critic entry registering det2's read (names its sha256)")
    d.add_argument("--run-dir", action="append", default=[], help="RUN=DIR holding RUN_DONE.json and log.jsonl")
    d.add_argument("--k1-table", default=os.path.join(EXP, "results", "k1", "k1_table_final.json"))
    d.add_argument("--out", default=os.path.join(EXP, "results", "k1"))
    d.add_argument("--name", default="prosqa_cot_same_seed_pair")
    p = sub.add_parser("pareto", help="row (c): the Pareto read, a compilation")
    for task in TASK_ORDER:
        p.add_argument(f"--{task.replace('_', '-')}-table", dest=f"{task}_table", default=None,
                       help=f"{TABLES[task][1]} (default {TABLES[task][0]})")
    p.add_argument("--test-read", default=os.path.join(EXP, "results", "screens", "test_read.json"))
    p.add_argument("--out", default=os.path.join(EXP, "results", "screens"))
    p.add_argument("--name", default="pareto_read")
    for q in (d, p):
        q.add_argument("--critic-log", default=os.path.join(EXP, "governance", "critic_log.md"))
        q.add_argument("--manifest", default=os.path.join(EXP, "configs", "data_manifest.json"))
    a = ap.parse_args(argv)
    if not RS.TABLE_NAME_RE.fullmatch(a.name):
        ap.error(f"--name {a.name!r} must match {RS.TABLE_NAME_RE.pattern}")
    inp = RS.Inputs(None, None)
    for path, role in ((os.path.abspath(__file__), "code (this tool)"),
                       (os.path.join(HERE, "read_screen.py"), "code (read_screen: costs, bootstrap, parsing)")):
        inp.sha256(path, role)
    try:
        if a.cmd == "det2":
            if bool(a.det2_dir) != bool(a.det2_read):
                ap.error("--det2-dir and --det2-read go together (det2's registered read)")
            if not a.det2_dir and os.path.abspath(a.out).startswith(EXP + os.sep):
                raise SystemExit("REFUSED: det2's read is absent: the pair table of record is written once with both "
                                 "members; give --out outside the experiment for a dry run")
            table, md = det2_table(a, inp)
        else:
            table, md = pareto_table(a, inp)
    except (ReadError, RS.RecordError, KeyError, ValueError) as e:
        raise SystemExit(f"REFUSED: {e}; nothing written") from None
    write_once(a.out, a.name, table, md, {"tool": "read_closing.py", "cmd": a.cmd, "argv": list(argv or sys.argv[1:])},
               inp, pareto=a.cmd == "pareto")
    print(md)
    return table


if __name__ == "__main__":
    main()
