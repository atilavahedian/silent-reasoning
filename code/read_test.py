"""The screen round's end-of-round test read (A3.4), written once: reporting only, no decision attached.

    <venv>/bin/python code/read_test.py \
        [--gsm50k-dir results/evals/screen-test-gsm50k-acctc] [--prosqa-hard-dir results/evals/screen-test-prosqa-hard-acctb] \
        [--jobs-dir configs/jobs] [--gsm50k-table results/screens/gsm50k/gsm50k_screen_with_hinge.json] \
        [--prosqa-hard-table results/screens/prosqa_hard/prosqa_hard_screen_with_plain_depth.json] \
        [--gsm50k-table-commit SHA] [--prosqa-hard-table-commit SHA] [--out results/screens]

A3.4: "Screen decisions use valid only ...; the screen checkpoints' test read happens once at the end of the screen
round, for reporting, with no decision attached". The two jobs `make_jobs --screen-test` writes (screen-test-gsm50k-acctc,
screen-test-prosqa-hard-acctb) score every screen checkpoint once on test (4.5) plus 3.3 item 3's secondary reads of
the earlier-stage `ckpt_best.pt` files. This tool reads both jobs' fetched results (JOB_RECEIPT.json and, per entry,
summary.json, examples.csv and eval.log, every file hashed), checks each entry against

* its pin: the committed job spec configs/jobs/<job>.json (the receipt's spec must equal it), the entry's
  checkpoint_sha256 (= the receipt's and the summary's), n_latent, config, max_new_tokens and precision;
* the registered row: the summary's registered_read is the job's, and that critic_log.md row names the checkpoint's
  hash in the generator's forms (make_jobs.test_row_hashes) and the test split's manifest hash;
* the read itself: split test, split_sha256 = the manifest's, intervention none, no audit, A10.5's flag set (TF32 on,
  "high") under the config's bf16 autocast, examples.csv = its summary hash with one row per test item in file order and
  the summary's accuracy as its count; one entry per checkpoint across both jobs (4.5);

and writes results/screens/test_read.json, test_read.md and test_read_provenance.json once (refused if any exists).
Per task and run: the valid accuracy of record (the fourth GSM-50k table, the third ProsQA-hard table; each cited by
the commit it is committed clean at) beside the test accuracy with its 95% interval in 6.2's form (binomial SE,
n = 1,319 / 500, +- 1.96 SE), the test emitted-token mean, the secondary reads labelled, and the test minus valid
gap printed as descriptive. Nothing here computes or changes a verdict, a kill line, a selection or a screen outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "launch"))

import read_screen as RS  # noqa: E402
from silent_launch import make_jobs as MJ  # noqa: E402

TASK_ORDER = ("gsm50k", "prosqa_hard")
TASK_TITLE = {"gsm50k": "GSM-50k", "prosqa_hard": "ProsQA-hard"}
TABLE_WHAT = {"gsm50k": "the fourth GSM-50k table", "prosqa_hard": "the third ProsQA-hard table"}
LABELS = [MJ.SCREEN_TEST_LABEL, "valid of record: the screen table's selection (A3.4: screen decisions use valid only)",
          "test minus valid: descriptive, decides nothing"]
NUMERICS = {"cuda.matmul.allow_tf32": True, "cudnn.allow_tf32": True, "float32_matmul_precision": "high"}   # A10.5
OK_STATUS = ("done", "already_read")
Z = 1.96


class ReadError(Exception):
    pass


def interval(k: int, n: int) -> dict:
    """6.2's form for one accuracy: p +- 1.96 SE with the binomial SE sqrt(p (1 - p) / n), in points."""
    p = k / n
    se = 100 * math.sqrt(p * (1 - p) / n)
    return {"k": k, "n": n, "points": 100 * p, "se_points": se, "ci95": [100 * p - Z * se, 100 * p + Z * se]}


def table_citation(inp: RS.Inputs, path: str, commit: str | None, what: str) -> dict:
    """The table must be committed clean and hash as committed at `commit` (default: the last commit touching it)."""
    st = RS.git_path_state(path)
    if not st["in_repo"] or st["dirty"] or not st["commit"]:
        raise ReadError(f"{path}: {what} is not committed clean in this repository (it is cited by commit)")
    full = RS.resolve_commit(commit or st["commit"])
    if full is None:
        raise ReadError(f"{what}: commit {commit} is not a commit of this repository")
    have = inp.sha256(path, f"{what} (valid of record)")
    if RS.committed_sha256(full, path) != have:
        raise ReadError(f"{path} differs from its content at {full[:12]} ({what} is write-once)")
    return {"file": os.path.relpath(os.path.abspath(path), REPO), "commit": full, "sha256": have}


def _final_latents(cfg: dict) -> int:
    return 0 if cfg.get("arm") in MJ.ALL_EPOCH_ARMS else int(cfg["max_latent_stage"]) * int(cfg["c_thought"])


def read_task(inp: RS.Inputs, task: str, res_dir: str, job_path: str, table: dict, critic_log: str,
              manifest: dict) -> dict:
    """One job's entries, checked (see the module docstring); ReadError on the first problem."""
    spec = MJ.SCREEN_TEST_TASKS[task]
    job = inp.read_json(job_path, f"{task} job spec (the pins)")
    rc_path = os.path.join(res_dir, "JOB_RECEIPT.json")
    if not os.path.isfile(rc_path):
        raise ReadError(f"{rc_path}: missing (fetch the job's results first)")
    rc = inp.read_json(rc_path, f"{task} receipt")
    if {k: v for k, v in (rc.get("spec") or {}).items() if k != "_launch"} != job:
        raise ReadError(f"{rc_path}: the receipt's spec is not {job_path} (the job read is not the committed one)")
    if rc.get("status") != "complete":
        raise ReadError(f"{rc_path}: status {rc.get('status')!r}, not complete")
    test_sha = next((e["sha256"] for e in MJ.jobs.manifest_entries(manifest, [task]) if e.get("path") == spec["test"]),
                    None)
    n_test = spec["n_test"]
    entries, by_name = job["evals"], {e["name"]: e for e in rc.get("evals") or []}
    ts = {e.get("registered_read") for e in entries}
    if len(ts) != 1:
        raise ReadError(f"{job_path}: the entries name {sorted(map(str, ts))}, not one registered read")
    ts = ts.pop()
    runs = sorted({e["run"] for e in entries})
    row = MJ.test_row_hashes(ts, runs, runs, critic_log)
    if row is None:
        raise ReadError(f"{ts} is not a row of {critic_log} (the registered read of {job['name']})")
    if row["ambiguous"]:
        raise ReadError(f"row {ts} names two hashes for {row['ambiguous']}")
    if test_sha not in row["hashes"]:
        raise ReadError(f"row {ts} does not name {spec['test']} by its manifest hash {test_sha}")
    trows = {r["run"]: r for r in table.get("runs") or []}
    n_valid = int(table["n_valid"])
    out = []
    for e in entries:
        name, run = e["name"], e["run"]
        secondary = name.endswith("_test_best")
        if not (secondary or name.endswith("_test_none")):
            raise ReadError(f"{name}: not a primary (_test_none) or secondary (_test_best) read")
        r = by_name.get(name)
        if r is None or r.get("status") not in OK_STATUS:
            raise ReadError(f"{name}: receipt status {None if r is None else r.get('status')!r}, not done")
        if r.get("checkpoint_sha256") != e["checkpoint_sha256"]:
            raise ReadError(f"{name}: the receipt's checkpoint_sha256 {r.get('checkpoint_sha256')} is not the pin")
        reg = (row["secondary"] if secondary else row["primary"]).get(run)
        if reg != e["checkpoint_sha256"]:
            raise ReadError(f"{name}: row {ts} names {reg} for {run}{'_best' if secondary else ''}, the pin is "
                            f"{e['checkpoint_sha256']}")
        d = os.path.join(res_dir, name)
        try:
            s = inp.read_json(os.path.join(d, "summary.json"), f"{name} summary")
            csv_text = inp.read_text(os.path.join(d, "examples.csv"), f"{name} per-example file")
            inp.sha256(os.path.join(d, "eval.log"), f"{name} eval log")
        except OSError as x:
            raise ReadError(f"{name}: {x}") from None
        cfg = inp.read_json(os.path.join(EXP, e["config"]), f"{run} config")
        n_lat = e["n_latent"] if e["n_latent"] is not None else _final_latents(cfg)
        want = {"run": run, "seed": 0, "split": "test", "split_file": spec["test"], "split_sha256": test_sha,
                "checkpoint_sha256": e["checkpoint_sha256"], "registered_read": ts, "audit_rescore": False,
                "intervention": "none", "n_latent": n_lat, "max_new_tokens": cfg.get("max_new_tokens"),
                "precision": cfg.get("precision"), "examples_csv_sha256": inp.sha256(os.path.join(d, "examples.csv"),
                                                                                     f"{name} per-example file")}
        bad = [f"{k} {s.get(k)!r} (want {v!r})" for k, v in want.items() if s.get(k) != v]
        bad += [f"numerics {k} {(s.get('numerics') or {}).get(k)!r}" for k, v in NUMERICS.items()
                if (s.get("numerics") or {}).get(k) != v]
        if bad:
            raise ReadError(f"{name}: summary.json " + "; ".join(bad))
        rows = RS.parse_examples_csv(csv_text)
        if [x["idx"] for x in rows] != list(range(n_test)):
            raise ReadError(f"{name}: examples.csv has {len(rows)} rows, not idx 0..{n_test - 1} in order")
        k = sum(x["correct"] for x in rows)
        acc = (s.get("summary") or {}).get("accuracy")
        if not isinstance(acc, (int, float)) or abs(acc - k / n_test) > 1e-9:
            raise ReadError(f"{name}: summary accuracy {acc!r} != {k}/{n_test} from examples.csv")
        em = sum(x["emitted_tokens"] for x in rows) / n_test
        tr = trows.get(run) or {}
        if secondary:
            v = tr.get("best_all") or {}
            if v.get("n_latent") != n_lat:
                raise ReadError(f"{name}: n_latent {n_lat} is not the table's best_all n_latent {v.get('n_latent')}")
            v_em = next((c.get("val_emitted_mean") for c in tr.get("curve") or [] if c.get("epoch") == v.get("epoch")),
                        None)
        else:
            v = tr.get("selected") or {}
            v_em = v.get("val_emitted_mean")
        if v.get("val_acc") is None:
            raise ReadError(f"{name}: the table has no {'best_all' if secondary else 'selected'} cell for {run}")
        kv = RS.count_of(v["val_acc"], n_valid, f"{run} valid accuracy")
        valid = interval(kv, n_valid)
        test = interval(k, n_test)
        out.append({"run": run, "role": tr.get("role"), "read": "secondary" if secondary else "primary",
                    "label": MJ.SCREEN_TEST_SECONDARY_LABEL if secondary else "primary read: the selected checkpoint",
                    "entry": name, "file": os.path.basename(e["checkpoint"]), "epoch": v.get("epoch"),
                    "stage": v.get("stage"), "n_latent": n_lat, "checkpoint_sha256": e["checkpoint_sha256"],
                    "valid": {"k": kv, "n": n_valid, "points": valid["points"], "emitted_mean": v_em},
                    "test": test, "test_emitted_mean": em, "gap_points_descriptive": test["points"] - valid["points"],
                    "gpu_type": s.get("gpu_type"), "commit": s.get("commit"), "utc": s.get("utc"),
                    "summary_sha256": inp.sha256(os.path.join(d, "summary.json"), f"{name} summary"),
                    "examples_csv_sha256": want["examples_csv_sha256"]})
    return {"job": job["name"], "project": job.get("project_alias"), "attempt": rc.get("attempt"),
            "registered_read": ts, "test_file": spec["test"], "test_sha256": test_sha, "n_test": n_test,
            "n_valid": n_valid, "rows": out}


def _p(x: float) -> str:
    return f"{x:.1f}"


def to_markdown(t: dict) -> str:
    L = [f"# {t['title']}", "", "Labels on every number: " + "; ".join(t["labels"]) + ".", ""]
    for task in TASK_ORDER:
        b = t["tasks"][task]
        c = b["valid_table"]
        L += [f"## {TASK_TITLE[task]}", "",
              f"Job `{b['job']}` ({b['project']}, attempt {b['attempt']}), registered read {b['registered_read']}; test "
              f"`{b['test_file']}` ({b['n_test']} items, sha256 {b['test_sha256'][:12]}...); valid of record: "
              f"{TABLE_WHAT[task]} `{c['file']}` at {c['commit'][:12]} (valid {b['n_valid']} items). Interval: 6.2's "
              f"form, +-1.96 binomial SE.", "",
              "| run | read | checkpoint | valid of record | test (95% interval) | test emitted tokens (mean) | "
              "test minus valid (descriptive) |", "|---|---|---|---|---|---|---|"]
        for r in b["rows"]:
            ts_, v = r["test"], r["valid"]
            L.append(f"| {r['run']} | {r['read']} | {r['file']} epoch {r['epoch']}, n_latent {r['n_latent']}, "
                     f"{r['checkpoint_sha256'][:12]} | {_p(v['points'])} ({v['k']}/{v['n']}) | {_p(ts_['points'])} "
                     f"[{_p(ts_['ci95'][0])}, {_p(ts_['ci95'][1])}] ({ts_['k']}/{ts_['n']}) | "
                     f"{r['test_emitted_mean']:.2f} | {r['gap_points_descriptive']:+.1f} |")
        sec = [r for r in b["rows"] if r["read"] == "secondary"]
        if sec:
            L += ["", f"Secondary reads ({len(sec)}): {MJ.SCREEN_TEST_SECONDARY_LABEL}. Each scores the run's all-epoch "
                      f"best `ckpt_best.pt` at its own stage's latent count once; it never replaces the primary row."]
        L.append("")
    L += ["## Notes", ""] + [f"- {n}" for n in t["notes"]] + [""]
    return "\n".join(L)


def _git_head() -> dict:
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", os.path.relpath(__file__, REPO)],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"head": head, "tool_uncommitted": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "tool_uncommitted": None}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for task in TASK_ORDER:
        opt = task.replace("_", "-")
        spec = MJ.SCREEN_TEST_TASKS[task]
        ap.add_argument(f"--{opt}-dir", default=os.path.join(EXP, "results", "evals", spec["job"]),
                        help=f"{spec['job']}'s fetched results (JOB_RECEIPT.json, <entry>/...)")
        ap.add_argument(f"--{opt}-table", default=str(spec["table"]), help=f"{TABLE_WHAT[task]} (valid of record)")
        ap.add_argument(f"--{opt}-table-commit", default=None,
                        help="the commit that wrote the table (default: the last commit touching it)")
    ap.add_argument("--jobs-dir", default=os.path.join(EXP, "configs", "jobs"))
    ap.add_argument("--critic-log", default=os.path.join(EXP, "governance", "critic_log.md"))
    ap.add_argument("--manifest", default=os.path.join(EXP, "configs", "data_manifest.json"))
    ap.add_argument("--out", default=os.path.join(EXP, "results", "screens"))
    a = ap.parse_args(argv)
    paths = {k: os.path.join(a.out, f"test_read{suf}") for k, suf in
             (("json", ".json"), ("md", ".md"), ("prov", "_provenance.json"))}
    if any(os.path.exists(p) for p in paths.values()):
        raise SystemExit(f"REFUSED: {' / '.join(p for p in paths.values() if os.path.exists(p))} exists; the test read "
                         f"is written once (A3.4; 4.5)")
    inp = RS.Inputs(None, None)
    manifest = inp.read_json(a.manifest, "data manifest (test split hashes)")
    inp.sha256(a.critic_log, "critic log (registered read rows)")
    tasks = {}
    try:
        for task in TASK_ORDER:
            opt = task.replace("-", "_")
            res_dir = getattr(a, f"{opt}_dir")
            st = RS.git_path_state(res_dir)
            if st["in_repo"] and (st["dirty"] or not st["commit"]):
                raise ReadError(f"{res_dir} is not committed clean (write-once eval results, A10.2)")
            tpath = getattr(a, f"{opt}_table")
            cite = table_citation(inp, tpath, getattr(a, f"{opt}_table_commit"), TABLE_WHAT[task])
            table = inp.read_json(tpath, f"{TABLE_WHAT[task]} (valid of record)")
            job_path = os.path.join(a.jobs_dir, f"{MJ.SCREEN_TEST_TASKS[task]['job']}.json")
            tasks[task] = read_task(inp, task, res_dir, job_path, table, a.critic_log, manifest)
            tasks[task]["valid_table"] = cite
            tasks[task]["results_dir"] = {"dir": os.path.relpath(os.path.abspath(res_dir), REPO) if st["in_repo"]
                                          else os.path.abspath(res_dir), "commit": st["commit"]}
        shas = [r["checkpoint_sha256"] for b in tasks.values() for r in b["rows"]]
        if len(shas) != len(set(shas)):
            raise ReadError("a checkpoint is read twice across the two jobs (4.5: one registered read per checkpoint)")
    except (ReadError, RS.RecordError, KeyError, ValueError) as e:
        raise SystemExit(f"REFUSED: {e}; nothing written") from None
    n_sec = sum(r["read"] == "secondary" for b in tasks.values() for r in b["rows"])
    table = {"title": "Screen round: the end-of-round test read (A3.4)", "labels": LABELS,
             "tasks": tasks,
             "notes": ["Reporting only (A3.4): no verdict, kill line, selection or screen outcome is computed or changed "
                       "here; every screen decision stays on valid, in the screen tables cited above.",
                       "1 seed (seed 0); each checkpoint scored once on test under its registered read (4.5); the "
                       "primary rows are each run's selected checkpoint (3.3 item 1); among the bar's three cot trials "
                       "only the bar's selected trial was read.",
                       f"{n_sec} secondary reads (3.3 item 3): the all-epoch best of a run whose earlier stage scored "
                       f"higher on valid, at that stage's latent count; labelled, never a primary cell.",
                       "The interval is 6.2's form for one accuracy (binomial SE); the test minus valid gap is "
                       "descriptive: valid is the selection split and test the scoring split (1.3)."],
             "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = to_markdown(table)
    import platform
    prov = {"table": "test_read", "generated_utc": table["generated_utc"], "argv": list(argv or sys.argv[1:]),
            "git": _git_head(), "versions": {"python": platform.python_version()}, "inputs": inp.listing(),
            "outputs": {"json": os.path.basename(paths["json"]), "md": os.path.basename(paths["md"])}}
    js = json.dumps(table, indent=1)
    pj = json.dumps(prov, indent=1)
    RS.check_words(md, js, pj)
    os.makedirs(a.out, exist_ok=True)
    for key, text in (("json", js + "\n"), ("md", md)):
        with open(paths[key], "x") as f:
            f.write(text)
    prov["outputs_sha256"] = {k: hashlib.sha256(open(paths[k], "rb").read()).hexdigest() for k in ("json", "md")}
    with open(paths["prov"], "x") as f:
        f.write(json.dumps(prov, indent=1) + "\n")
    print(md)
    return table


if __name__ == "__main__":
    main()
