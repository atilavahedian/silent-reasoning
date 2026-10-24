"""Score one checkpoint with latent.eval.evaluate: per-example CSV and summary JSON (registration 2.2, 4.5, 13.3; A3.4).

    PYTHONPATH=common python3 code/run_eval.py \
        --config configs/screen/prosqa_hard_c1b_k8.json \
        --checkpoint runs/prosqa_hard_c1b_k8/ckpt_best_final.pt [--out DIR] \
        [--split valid|test] [--intervention none|scrambled|zero_latent] [--n_latent K] [--set key=value ...] \
        [--registered-read "<critic_log Logged (UTC) value>"] [--tf32 off|on] [--no-time-steps] [--commit SHA]

Reconciles the GCP plumbing agent's thin entry point (same flags; the launcher calls it with --config, --checkpoint,
--out, --split, --n-latent, --registered-read) with the mechanism agent's brief:
* The split file is the config's val_path with `_valid.json` replaced by `_<split>.json` (ProsQA-hard:
  prosqa_hard_test.json; GSM-50k: gsm_test.json).
* **The test split is scored only with --registered-read naming an existing critic_log.md row** (registration
  section 0, 1.3, 4.5: one test read per checkpoint and intervention); a test summary that already exists in the
  output directory is never overwritten (a second read happens only under the 7.10 audit, with --audit-rescore, into
  a separate directory). Valid is the default split.
* Latent count: --n_latent K (the step sweep {0, 1, 2, 4, 8}) if given; otherwise the final regime of the config
  (c_thought x max_latent_stage latents for the latent arms; registration 3.3 item 4). zero_latent forces 0.
  scrambled: the registered derangement over the whole evaluation set within latent-count classes, seed 0; it works
  on C1b band models (the k = 8 scrambled kill line, A3.4).
* A C1b config (latent_band) builds the band model; its FLOPs carry layer_fraction; --time-steps (default on for the
  latent-feedback arms) records the synchronised wall-clock per latent step and per emitted token (A3.4).
* Writes <out>/examples.csv (idx, correct, cot_match and every cost axis; **no example text**, section 1.3) and
  <out>/summary.json (latent.eval.summarize: accuracy with the 10,000-draw bootstrap 95% CI, means / medians / CIs of
  emitted tokens, latent steps, sequential passes and FLOPs; the paper-style token count; timing; checkpoint, data
  and CSV sha256; config, versions, device, numerics, commit). Default --out: results/<run>/<split>_<intervention>_n<K>/.
* `commit` (A11.4): --commit SHA (the launcher's code commit, the receipt's `code_commit`; the VM's tarball has no
  .git) when given, else `git rev-parse HEAD`.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, HERE)

from run_train import load_config, set_tf32  # noqa: E402

CRITIC_LOG = os.path.join(EXP, "governance", "critic_log.md")
CSV_FIELDS = ("idx", "correct", "cot_match", "prefill_tokens", "latent_steps", "latent_positions", "emitted_tokens",
              "sequential_passes", "flops", "reference_flops")       # reference_flops: 5.5 (c), A6.4, secondary
LATENT_FEEDBACK_ARMS = ("coconut", "coconut_no_curriculum")


def critic_row_exists(stamp: str, path: str | None = None) -> bool:
    """True if critic_log.md has a table row whose first cell is exactly `stamp`."""
    path = path or CRITIC_LOG
    if not stamp or not os.path.exists(path):
        return False
    with open(path) as f:
        return any(ln.startswith(f"| {stamp} |") for ln in f)


def split_path(val_path: str, split: str) -> str:
    if split == "valid":
        return val_path
    if not val_path.endswith("_valid.json"):
        raise ValueError(f"cannot derive the {split} file from val_path {val_path!r}")
    return val_path[: -len("_valid.json")] + f"_{split}.json"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def final_latents(cfg) -> int:
    """Registration 3.3 item 4: the final regime's latent count (pause: its pause-token count); 0 for cot / no_cot."""
    if cfg.arm in ("cot", "no_cot"):
        return 0
    return cfg.c_thought * cfg.max_latent_stage


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:  # pragma: no cover - environment dependent
        return None


def aggregate_timing(timing: list) -> dict | None:
    """Mean wall-clock per latent step and per emitted token over the batches (each batch a LatentLM timing dict)."""
    timing = [t for t in timing if t]
    if not timing:
        return None
    out = {}
    for kind in ("prefill", "latent", "token"):
        n = sum(t[f"{kind}_passes"] for t in timing)
        ms = sum(t[f"{kind}_ms_total"] for t in timing)
        out[f"{kind}_passes"] = n
        out[f"{kind}_ms_per_pass"] = (ms / n) if n else None
    if out["latent_ms_per_pass"] and out["token_ms_per_pass"]:
        out["latent_over_token_ms"] = out["latent_ms_per_pass"] / out["token_ms_per_pass"]
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", choices=("valid", "test"), default="valid")
    ap.add_argument("--intervention", choices=("none", "scrambled", "zero_latent"), default="none")
    ap.add_argument("--n-latent", "--n_latent", dest="n_latent", type=int, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--registered-read", default=None)
    ap.add_argument("--audit-rescore", action="store_true",
                    help="registration 7.10 item 2: a second test read, written beside the first, never over it")
    ap.add_argument("--tf32", choices=("off", "on"), default="off")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--time-steps", dest="time_steps", action="store_true", default=None)
    ap.add_argument("--no-time-steps", dest="time_steps", action="store_false")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--commit", default=None,
                    help="A11.4: the code commit written into summary.json (default: git rev-parse HEAD)")
    a = ap.parse_args(argv)
    if a.split == "test" and not critic_row_exists(a.registered_read or ""):
        raise SystemExit("REFUSED: the test split is scored only at a registered read: pass --registered-read with the "
                         "Logged (UTC) value of an existing critic_log.md row (registration section 0, 4.5)")
    numerics = set_tf32(a.tf32 == "on")

    import torch
    import transformers

    from latent.data import load_examples
    from latent.eval import evaluate, summarize
    from latent.train import autocast_ctx, build_model, resolve_device

    cfg = load_config(a.config, a.set)
    n_latent = final_latents(cfg) if a.n_latent is None else int(a.n_latent)
    if a.intervention == "zero_latent":
        n_latent = 0
    tag = f"{a.split}_{a.intervention}_n{n_latent}" + ("_audit" if a.audit_rescore else "")
    out_dir = a.out or os.path.join(EXP, "results", cfg.name, tag)
    if a.split == "test" and os.path.exists(os.path.join(out_dir, "summary.json")):
        raise SystemExit(f"REFUSED: {out_dir}/summary.json exists; a test read is write-once (registration 1.3); "
                         f"an audit rescoring uses --audit-rescore")
    device = resolve_device(cfg.device)
    lm, tokenizer, _ = build_model(cfg, load_path=a.checkpoint)
    lm.to(device)
    path = split_path(cfg.val_path, a.split)
    examples = load_examples(path, None)
    timed = cfg.arm in LATENT_FEEDBACK_ARMS if a.time_steps is None else a.time_steps
    timing = [] if timed else None
    t0 = time.time()
    with autocast_ctx(cfg.precision, device):
        res = evaluate(lm, tokenizer, examples, arm=cfg.arm, n_latent=n_latent,
                       batch_size=a.batch_size or cfg.eval_batch_size, device=device,
                       intervention=None if a.intervention == "none" else a.intervention,
                       max_new_tokens=cfg.max_new_tokens, **({"timing_out": timing} if timed else {}))
    seconds = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "examples.csv")
    lat_pos = 0 if cfg.arm in ("cot", "no_cot") else n_latent
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in res:
            row = {k: getattr(r, k) for k in CSV_FIELDS if k != "latent_positions"}
            row["latent_positions"] = lat_pos
            w.writerow(row)
    summ = summarize(res, n_boot=a.n_boot)
    summ["paper_style_tokens_mean"] = summ.get("emitted_tokens_mean", 0.0) + summ.get("latent_steps_mean", 0.0)
    task = os.path.basename(cfg.train_path)[: -len("_train.json")]
    out = {"summary": summ, "task": task, "run": cfg.name, "arm": cfg.arm, "seed": cfg.seed,
           "variant": {"latent_band": cfg.latent_band, "layer_fraction": lm.layer_fraction,
                       "state_sup": cfg.state_sup, "state_null": cfg.state_null, "dep_loss": cfg.dep_loss,
                       "dep_null": cfg.dep_null},
           "config": os.path.relpath(os.path.abspath(a.config), REPO), "overrides": a.set,
           "checkpoint": a.checkpoint, "checkpoint_sha256": sha256_file(a.checkpoint), "split": a.split,
           "split_file": os.path.relpath(path, REPO), "split_sha256": sha256_file(path),
           "intervention": a.intervention, "n_latent": n_latent, "scramble_seed": 0,
           "max_new_tokens": cfg.max_new_tokens, "precision": cfg.precision, "device": device,
           "gpu_type": torch.cuda.get_device_name() if device.startswith("cuda") else device,
           "torch": torch.__version__, "transformers": transformers.__version__, "commit": a.commit or git_commit(),
           "registered_read": a.registered_read, "audit_rescore": a.audit_rescore, "numerics": numerics,
           "eval_seconds": seconds, "timing": aggregate_timing(timing) if timing is not None else None,
           "examples_csv_sha256": sha256_file(csv_path),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in summ.items() if not k.endswith("_ci95")}))
    return out


if __name__ == "__main__":
    main()
