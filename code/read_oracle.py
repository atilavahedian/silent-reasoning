"""The K1 oracle table (registration 4.1-4.3, 13.1 and 13.10; window scope A3.3) from run_eval.py summary JSONs.

    python3 code/read_oracle.py \
        --published governance/oracle_published.json \
        --summaries 'results/*/test_none_n*/summary.json' [GLOB ...] \
        [--out DIR] [--table-name k1_table]

The final K1 table (critic entry 2026-10-19T15:40Z condition (h); A11.1, A11.4) reads both jobs' plain reads and writes a
new path; the provisional results/k1/k1_table.{json,md} is write-once (4.3) and an existing table is never rewritten:
    --summaries '<evals>/k1-eval-main/*_test_none/summary.json' '<evals>/k1-eval-s12-main/*_test_none/summary.json' \
    --table-name k1_table_final

Published targets come from a JSON the critic fills from the Coconut PDF (registration 4.2 and the PDF row of A2 /
A3.1): {"source": "pdf", "pdf_sha256": ..., "critic_log_row": ..., "cells": [{"task", "arm", "value" | "relative_to",
"spread", "gates", "pdf_table", "pdf_row"}, ...]}. `governance/oracle_published.template.json` holds the 4.2 snippet
values with source "snippet". **The script refuses to run on snippet values** (no --published, source != "pdf", no
pdf_sha256 or critic_log_row, or a cell without pdf_table) unless --allow-snippet is given, and then every cell and
every verdict sentence carries the label "snippet".

Per cell (task, arm) over the test summaries with intervention "none" of plain (non-Phase-2) runs: our seeds in points,
mean, SD (ddof 1), sigma_cell = max(1.0, sqrt of the pooled variance of our seeds (df n - 1) and the published spread
(df 2) when reported), tolerance = max(2.0, 2 sigma_cell), matched iff |mean - published| <= tolerance (4.1); a
relative cell ("about No-CoT") uses our own mean of the named arm as its target. Cells with fewer than 3 seeds carry
"single seed" (PLAN.md A1 item 1; A3.3). K1 passes on a task iff every gating cell is present and matched (4.3); a
missing gating cell makes the task UNTESTED. With fewer than 3 seeds of cot or coconut on a task the verdict is
"provisional" (A3.3). The registered sentence for Phase 1 ("ProsQA and ProntoQA") is printed beside the window's.

The window's GSM8k-Aug clause (critic entry 2026-10-20T12:57Z conditions (ii) and (3); PLAN.md A1 item 1 reads GSM cot and
coconut at seed 0 only, so the registered 4.3 sentence stays "UNTESTED (no_cot not read)" and is printed first, unchanged).
When a GSM cot or coconut test read is among the summaries, two sentences are printed beside it and the window sentence
names GSM8k-Aug; no cell, cell verdict or registered sentence changes. The rule is the row's, frozen at the read (A7.2,
single seed): sigma_cell cot 1.0, coconut 1.5; cot |x - 42.9| <= 2.0, coconut |x - 34.1| <= 3.0; pass iff both match;
a cell beyond 2 x tolerance (4.0 / 6.0) carries "gross miss: round 1 may start" (4.4 item 5). Each read must be the pinned
checkpoint of the row on the 1,319-item test split, else the clause is UNTESTED. The final window table (condition (iii)):
    --summaries '<evals>/k1-eval-main/*_test_none/summary.json' '<evals>/k1-eval-s12-main/*_test_none/summary.json' \
                '<evals>/k1-eval-gsm-main/*_test_none/summary.json' --table-name k1_table_window
refuses to write unless every ProsQA cell serialises byte-equal to <out>/k1_table_final.json's.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
TASK_NAMES = {"prosqa": "ProsQA", "prontoqa": "ProntoQA", "gsm": "GSM8k-Aug"}
TABLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

# critic entry 2026-10-20T12:57Z: the GSM8k-Aug K1 registered read (window clause, A1 item 1: cot and coconut, seed 0)
GSM_ROW = "2026-10-20T12:57Z"
GSM_N = 1319
GSM_TEST_SHA256 = "46ce5b0c44fde0bfc8cf1e7492aa640b96111d9dcb80a54340d9911141a9caa9"
# arm: (published, sigma_cell frozen at the read, tolerance, the pinned ckpt_best_final.pt sha256 of the row's run)
GSM_WINDOW = {"cot": (42.9, 1.0, 2.0, "6d798e8a3c2f27b3fcccf988b740a00c4f377aa3c94de984e21b64dbaa900200"),
              "coconut": (34.1, 1.5, 3.0, "0c18e12384f1706b855c3cb7987e6b863c1067583ace2055587751034e1c9f35")}
GPU_SHORT = {"NVIDIA A100-SXM4-40GB": "A100-40GB", "NVIDIA A100-SXM4-80GB": "A100-80GB"}
WINDOW_TABLE, FINAL_TABLE = "k1_table_window", "k1_table_final"

# registration 4.2, every value "(snippet; verify from PDF before the read)"
SNIPPET_DEFAULTS = [
    {"task": "prosqa", "arm": "coconut", "value": 97.0, "spread": 0.3, "gates": True},
    {"task": "prosqa", "arm": "pause", "value": 96.6, "spread": 0.8, "gates": True},
    {"task": "prosqa", "arm": "cot", "value": 77.5, "spread": None, "gates": True},
    {"task": "prosqa", "arm": "no_cot", "value": 76.7, "spread": None, "gates": True},
    {"task": "prosqa", "arm": "coconut_no_curriculum", "relative_to": "no_cot", "spread": None, "gates": True},
    {"task": "prontoqa", "arm": "coconut", "value": 99.8, "spread": None, "gates": True},
    {"task": "prontoqa", "arm": "cot", "value": 98.8, "spread": None, "gates": True},
    {"task": "prontoqa", "arm": "no_cot", "value": 93.8, "spread": None, "gates": True},
    {"task": "gsm", "arm": "cot", "value": 42.9, "spread": None, "gates": True},
    {"task": "gsm", "arm": "coconut", "value": 34.1, "spread": None, "gates": True},
    {"task": "gsm", "arm": "no_cot", "value": 16.5, "spread": None, "gates": True},
    {"task": "gsm", "arm": "pause", "value": 24.1, "spread": 0.7, "gates": False},   # "pause as thought" (PDF Table 1;
                                                                                      # the snippet's 16.4 was the Pause Token baseline)
]


def snippet_template() -> dict:
    return {"source": "snippet", "pdf_sha256": None, "pdf_url": None, "critic_log_row": None,
            "_note": "registration 4.2 snippet values; the critic replaces source with 'pdf', fills pdf_sha256, "
                     "critic_log_row and per cell pdf_table / pdf_row (and corrects value / spread) from the PDF",
            "cells": [{**c, "pdf_table": None, "pdf_row": None} for c in SNIPPET_DEFAULTS]}


def published_status(pub: dict | None) -> tuple[bool, list[str]]:
    """(is_pdf, reasons it is not)."""
    if pub is None:
        return False, ["no --published file: the 4.2 snippet defaults"]
    why = []
    if pub.get("source") != "pdf":
        why.append(f"source is {pub.get('source')!r}, not 'pdf'")
    for k in ("pdf_sha256", "critic_log_row"):
        if not pub.get(k):
            why.append(f"{k} missing")
    for c in pub.get("cells", []):
        if not c.get("pdf_table"):
            why.append(f"{c.get('task')}/{c.get('arm')}: no pdf_table")
        if c.get("value") is None and not c.get("relative_to"):
            why.append(f"{c.get('task')}/{c.get('arm')}: no value")
    if not pub.get("cells"):
        why.append("no cells")
    return not why, why


def load_summaries(patterns) -> list[dict]:
    out = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            with open(p) as f:
                s = json.load(f)
            s["_path"] = p
            out.append(s)
    return out


def _plain(s: dict) -> bool:
    v = s.get("variant") or {}
    return not (v.get("latent_band") or v.get("state_sup") or v.get("dep_loss"))


def collect(summaries) -> dict:
    """{(task, arm): {seed: (accuracy points, gpu_type, path)}} from plain test reads without intervention."""
    cells = {}
    for s in summaries:
        if s.get("split") != "test" or s.get("intervention") != "none" or not _plain(s):
            continue
        key = (s["task"], s["arm"])
        seed = s.get("seed")
        if seed in cells.setdefault(key, {}):
            raise ValueError(f"two test reads of {key} seed {seed}: {cells[key][seed][2]} and {s['_path']}")
        cells[key][seed] = (100.0 * s["summary"]["accuracy"], s.get("gpu_type"), s["_path"])
    return cells


def cell_verdict(values: list[float], target: float, spread: float | None) -> dict:
    n = len(values)
    mean = st.mean(values)
    sd = st.stdev(values) if n >= 2 else None
    num, df = 0.0, 0
    if n >= 2:
        num, df = num + (n - 1) * sd ** 2, df + (n - 1)
    if spread is not None:
        num, df = num + 2 * spread ** 2, df + 2
    sigma = max(1.0, math.sqrt(num / df)) if df else 1.0
    tol = max(2.0, 2 * sigma)
    return {"n": n, "mean": mean, "sd": sd, "sigma_cell": sigma, "tolerance": tol, "published": target,
            "diff": mean - target, "matched": abs(mean - target) <= tol}


def build_table(pub_cells: list, ours: dict, snippet: bool) -> dict:
    rows, tasks = [], {}
    for c in pub_cells:
        key = (c["task"], c["arm"])
        seeds = ours.get(key, {})
        row = {"task": c["task"], "arm": c["arm"], "gates": bool(c.get("gates", True)),
               "seeds": {str(k): v[0] for k, v in sorted(seeds.items())},
               "gpu_types": {str(k): v[1] for k, v in sorted(seeds.items())}, "labels": []}
        if snippet:
            row["labels"].append("snippet")
        target = c.get("value")
        if c.get("relative_to"):
            ref = ours.get((c["task"], c["relative_to"]), {})
            target = st.mean(v[0] for v in ref.values()) if ref else None
            row["labels"].append(f"relative to our {c['relative_to']}")
        if not seeds or target is None:
            row.update(verdict="UNTESTED")
        else:
            row.update(cell_verdict([v[0] for v in seeds.values()], target, c.get("spread")))
            row["verdict"] = "matched" if row["matched"] else "missed"
            if len(seeds) < 3:
                row["labels"].append("single seed" if len(seeds) == 1 else f"single seed ({len(seeds)} seeds)")
        rows.append(row)
        t = tasks.setdefault(c["task"], {"gating": [], "missed": [], "untested": [], "provisional": False})
        if row["gates"]:
            t["gating"].append(c["arm"])
            if row["verdict"] == "missed":
                t["missed"].append(c["arm"])
            elif row["verdict"] == "UNTESTED":
                t["untested"].append(c["arm"])
        if c["arm"] in ("cot", "coconut") and len(seeds) < 3:
            t["provisional"] = True
    sentences = {}
    for task, t in tasks.items():
        name = TASK_NAMES.get(task, task)
        if t["missed"]:
            v = f"fail ({', '.join(t['missed'])})"
        elif t["untested"]:
            v = f"UNTESTED ({', '.join(t['untested'])} not read)"
        else:
            v = f"pass ({', '.join(t['gating'])})"
        labels = (["provisional"] if t["provisional"] and not t["untested"] else []) + (["snippet"] if snippet else [])
        if task == "gsm":
            labels.append("single seed")
        t["verdict"] = v.split(" ")[0]
        sentences[task] = f"K1 {name}: {v}" + (f" [{'; '.join(labels)}]" if labels else "")
    gsm = gsm_window(rows, ours, snippet)
    if gsm:
        sentences["gsm_window"] = gsm["clause"]
        sentences["gsm_window_read"] = gsm["sentence"]
    reg = [tasks.get(k, {}).get("verdict", "UNTESTED") for k in ("prosqa", "prontoqa")]
    sentences["phase1_registered"] = ("K1 (registered, ProsQA and ProntoQA): "
                                      + ("pass" if reg == ["pass", "pass"] else
                                         "fail" if "fail" in reg else "UNTESTED")
                                      + ("" if tasks.get("prontoqa") and reg[1] != "UNTESTED"
                                         else "; ProntoQA UNTESTED (A3.3)"))
    sentences["window"] = f"K1 for the window (A3.3) = K1 on ProsQA: {tasks.get('prosqa', {}).get('verdict', 'UNTESTED')}"
    if gsm:
        sentences["window"] += f"; GSM8k-Aug: {gsm['verdict']} [1 seed]"
    out = {"rows": rows, "tasks": tasks, "sentences": sentences}
    if gsm:
        out["gsm_window"] = gsm
    return out


def _gpu(t) -> str:
    return GPU_SHORT.get(t, str(t))


def gsm_window(rows: list, ours: dict, snippet: bool) -> dict | None:
    """The window's GSM8k-Aug clause (critic entry 2026-10-20T12:57Z (ii) and (3)) from the cot and coconut cells as
    build_table computed them; None when neither arm was read (the table then prints as before). Reads nothing but the
    cells and the two summaries they came from."""
    if not any(ours.get(("gsm", a)) for a in GSM_WINDOW):
        return None
    cells = {(r["task"], r["arm"]): r for r in rows}
    arms, untested = {}, []
    for arm, (pub, sigma, tol, pin) in GSM_WINDOW.items():
        seeds = ours.get(("gsm", arm), {})
        if not seeds:
            untested.append(f"{arm} not read")
            continue
        if len(seeds) != 1:
            raise ValueError(f"GSM8k-Aug {arm}: {len(seeds)} seeds; the {GSM_ROW} row registers one (seed 0)")
        row = cells.get(("gsm", arm))
        if not snippet and row is not None and row.get("n") is not None and (
                abs(row["published"] - pub) > 1e-9 or abs(row["sigma_cell"] - sigma) > 1e-9
                or abs(row["tolerance"] - tol) > 1e-9):
            raise ValueError(f"GSM8k-Aug {arm}: the published targets give {row['published']} / sigma_cell "
                             f"{row['sigma_cell']} / tolerance {row['tolerance']}, not the {GSM_ROW} row's frozen "
                             f"{pub} / {sigma} / {tol}")
        (seed, (x, gpu, path)), = seeds.items()
        with open(path) as f:
            s = json.load(f)
        n = (s.get("summary") or {}).get("n")
        why = []
        if seed != 0:
            why.append(f"seed {seed}, not 0")
        if n != GSM_N:
            why.append(f"n {n}, not {GSM_N:,}")
        if s.get("split_sha256") != GSM_TEST_SHA256:
            why.append("split_sha256 is not the registered GSM8K test file's")
        if s.get("checkpoint_sha256") != pin:
            why.append(f"checkpoint is not the {GSM_ROW} row's pin {pin[:8]}")
        if why:
            untested.append(f"{arm}: {', '.join(why)}")
            continue
        diff = x - pub
        arms[arm] = {"value": x, "count": round(s["summary"]["accuracy"] * n), "n": n, "published": pub,
                     "sigma_cell": sigma, "tolerance": tol, "diff": diff, "matched": abs(diff) <= tol,
                     "gross_miss": abs(diff) > 2 * tol, "gpu_type": gpu, "precision": s.get("precision"),
                     "checkpoint_sha256": s.get("checkpoint_sha256"), "summary": path}
    if untested:
        verdict = "UNTESTED"
        parts = "; ".join(untested)
    else:
        verdict = "pass" if all(a["matched"] for a in arms.values()) else "fail"
        parts = "; ".join(
            f"{arm} {a['value']:.2f}, {a['count']:,} of {a['n']:,}, "
            + (f"matched: |{a['value']:.2f} - {a['published']}| = {abs(a['diff']):.2f} <= {a['tolerance']}"
               if a["matched"] else
               f"missed: |{a['value']:.2f} - {a['published']}| = {abs(a['diff']):.2f} > {a['tolerance']}")
            + ("; gross miss: round 1 may start" if a["gross_miss"] else "")
            for arm, a in arms.items())
    snip = "; snippet" if snippet else ""
    clause = (f"K1 GSM8k-Aug (window, A1 item 1: cot and coconut): {verdict} ({parts}) "
              f"[single seed; no_cot UNTESTED{snip}]")
    if verdict == "UNTESTED":
        sentence = f"K1 on GSM8k-Aug, 1 seed: UNTESTED ({parts}) [single seed; no_cot UNTESTED{snip}]"
    else:
        cot, coco = arms["cot"], arms["coconut"]
        prec = {a["precision"] for a in arms.values()}
        prec = prec.pop() if len(prec) == 1 else " / ".join(f"{k} {a['precision']}" for k, a in arms.items())
        cot_read, coco_read = _gpu(cot["gpu_type"]), _gpu(coco["gpu_type"])
        labels = ["single seed", "no_cot UNTESTED", "KNOWN (reproduction)",
                  f"{prec} (footnote: the reference's probable strict fp32, A4; d_prec 0.0 on ProsQA does not "
                  f"transfer to GSM and is not claimed for it)",
                  "cot trained and read on A100-40GB" if cot_read == "A100-40GB" else
                  f"cot trained on A100-40GB, read on {cot_read}",
                  f"coconut: cot stage on A100-40GB, coconut stage on A100-80GB, read on {coco_read} "
                  f"(A3.2 (i) to (iii) on the eval VM)",
                  "images v20261016 / v20261019", "zero resumes", "audit label per A10.3 / (v): pending"]
        if snippet:
            labels.append("snippet")
        sentence = (f"K1 on GSM8k-Aug, 1 seed: {verdict} (cot {cot['value']:.2f}, coconut {coco['value']:.2f}) "
                    f"against {cot['published']} / {coco['published']}, tolerances {cot['tolerance']} / "
                    f"{coco['tolerance']} [{'; '.join(labels)}]")
    return {"verdict": verdict, "critic_log_row": GSM_ROW, "arms": arms, "untested": untested,
            "clause": clause, "sentence": sentence}


def to_markdown(table: dict) -> str:
    lines = ["| task | arm | gates | seeds (points) | mean | SD | sigma_cell | tolerance | published | verdict | labels |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "" if x is None else f"{x:.2f}"
    for r in table["rows"]:
        seeds = ", ".join(f"s{k} {v:.1f} ({r['gpu_types'][k]})" for k, v in r["seeds"].items())
        lines.append(f"| {r['task']} | {r['arm']} | {'yes' if r['gates'] else 'no'} | {seeds} | {f(r.get('mean'))} | "
                     f"{f(r.get('sd'))} | {f(r.get('sigma_cell'))} | {f(r.get('tolerance'))} | "
                     f"{f(r.get('published'))} | {r['verdict']} | {'; '.join(r['labels'])} |")
    lines += ["", *[f"- {s}" for s in table["sentences"].values()]]
    return "\n".join(lines) + "\n"


def check_window_prosqa(table: dict, final_path: str) -> None:
    """Critic entry 2026-10-20T12:57Z (iii): every ProsQA cell of the window table equals k1_table_final.json's byte for
    byte (each row as json.dumps(indent=2) serialises it); refuses before anything is written."""
    if not os.path.exists(final_path):
        raise SystemExit(f"REFUSED: {final_path} does not exist; {WINDOW_TABLE} is checked against it ({GSM_ROW} (iii))")
    with open(final_path) as f:
        final = json.load(f)

    def prosqa(t):
        return [json.dumps(r, indent=2) for r in t["rows"] if r["task"] == "prosqa"]
    ours, theirs = prosqa(table), prosqa(final)
    if ours != theirs:
        bad = [json.loads(r)["arm"] for r, q in zip(ours, theirs) if r != q] or ["row count"]
        raise SystemExit(f"REFUSED: the ProsQA cells differ from {final_path} ({', '.join(bad)}); {GSM_ROW} (iii) "
                         f"requires them byte-equal. Nothing written.")


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--published", default=None, help="the critic's PDF-verified targets JSON")
    ap.add_argument("--summaries", nargs="+", required=True,
                    help="one or more summary.json files or globs (run_eval.py); the final table: both jobs' reads")
    ap.add_argument("--out", default=os.path.join(EXP, "results", "k1"))
    ap.add_argument("--table-name", default="k1_table",
                    help="writes <out>/<name>.{json,md}; the final K1 table is k1_table_final (15:40Z (h)); "
                         "an existing table is never rewritten (4.3, A11.4)")
    ap.add_argument("--allow-snippet", action="store_true",
                    help="run on snippet values anyway; every cell and sentence is labelled 'snippet'")
    ap.add_argument("--write-template", default=None, help="write the snippet template JSON here and exit")
    a = ap.parse_args(argv)
    if a.write_template:
        with open(a.write_template, "w") as f:
            json.dump(snippet_template(), f, indent=2)
            f.write("\n")
        return {}
    if not TABLE_NAME_RE.fullmatch(a.table_name):
        ap.error(f"--table-name {a.table_name!r} must match {TABLE_NAME_RE.pattern} (a file stem, no path)")
    paths = [os.path.join(a.out, f"{a.table_name}.{ext}") for ext in ("json", "md")]
    if any(os.path.exists(p) for p in paths):
        raise SystemExit(f"REFUSED: {' / '.join(p for p in paths if os.path.exists(p))} exists; the K1 table is "
                         f"write-once (4.3, A11.4): the final read writes --table-name k1_table_final (15:40Z (h))")
    pub = None
    if a.published:
        with open(a.published) as f:
            pub = json.load(f)
    is_pdf, why = published_status(pub)
    if not is_pdf and not a.allow_snippet:
        raise SystemExit("REFUSED: the published targets are not PDF-verified (" + "; ".join(why) + "). Registration "
                         "4.2 / A2: the snippet values do not become targets by default. Pass --allow-snippet to "
                         "print a table labelled 'snippet'.")
    cells = (pub or {}).get("cells") or SNIPPET_DEFAULTS
    table = build_table(cells, collect(load_summaries(a.summaries)), snippet=not is_pdf)
    if a.table_name == WINDOW_TABLE:
        check_window_prosqa(table, os.path.join(a.out, f"{FINAL_TABLE}.json"))
    table.update(published_source="pdf" if is_pdf else "snippet", published_problems=why,
                 published_file=a.published, summaries=a.summaries,
                 utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    os.makedirs(a.out, exist_ok=True)
    with open(paths[0], "w") as f:
        json.dump(table, f, indent=2)
    with open(paths[1], "w") as f:
        f.write(to_markdown(table))
    print(to_markdown(table))
    return table


if __name__ == "__main__":
    main()
