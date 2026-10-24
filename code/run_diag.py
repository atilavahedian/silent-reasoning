"""GPU diagnostics on the valid split for one checkpoint, per numerics mode (registration 7.10 items 4 and 5; A11.3 (1);
A12.3; critic entries 2026-10-19T15:35Z and 2026-10-19T16:59Z). Spends no test read: any split but valid is refused before
the model loads.

    PYTHONPATH=common python3 code/run_diag.py \
        --config configs/prosqa_cot.json \
        --checkpoint runs/prosqa_cot_s1/ckpt_best_final.pt --mode cot_batch|latent_path|band_logits|donor_answer \
        [--tf32 off|on|both] [--out DIR] [--batch-size N] [--n-boot 0] [--perm-seed 0] [--set key=value ...] \
        [--commit SHA] [--split valid] [--extra-idx 273,497] [--dtype float32|float64] [--strict-only]
        [--base-config DIR] [--tol X] [--critic-row TS] [--checks 1,1b,2,3,4] [--identify-dir DIR ...]

* --mode cot_batch (A12.3, the explanation of a 7.10 item 2 audit above the line): latent.diag.cot_batch: prompt
  logits B = 1 against B = batch_size in two companion sets plus the same-shape swap; greedy generation B = 1 against
  B = batch_size (evaluate, the config's max_new_tokens), the first divergent position and top-2 margins; correctness.
* --mode latent_path (A11.3 (1); latent-feedback arms only): latent.diag.latent_path: the scrambled pass's fed vectors
  against the own latents per rank (zero distance = BUG), the first-answer-token logits own against scrambled, and
  the isolation and generation comparisons on the latent path.
* --mode band_logits (A6.2 rule 3's strict-fp32 logits check; critic entry 2026-10-20T12:40Z, the registered
  diagnostic; A14.1): latent.diag.band_logits on a full-depth latent-feedback checkpoint (the f_meas denominator), the
  HF path against band [0, L] (L = n_layer; [0, 12] at 124M) over the same weights: (1) at strict fp32, one batch (the
  split's first --batch-size examples in file order, the read's first batch): max |first-answer-token logits| HF vs
  band, line 1e-4 (else BUG); (2) the same at the read's registered numerics with the diverging greedy generations;
  (3) band [0, L] at B = 1 against B = batch size on that batch plus --extra-idx (default 273, 497: deviations.md
  2026-10-20 12:05 UTC), each extra in its own file-order batch: diverging generations, top-2 margins at the first
  answer token and at the first divergent token; (4) the same-shape companion swap on band [0, L] at B = batch size,
  line 1e-4 at each numerics. Numerics are set by this tool per check, not by the job: a diag entry may not set
  `precision` (the launcher's FORBIDDEN_SET / DIAG_FORBIDDEN), so (1), (3), (4) run under set_tf32(False) asserted as
  REGISTERED_FLAGS["fp32"] with no autocast, then (2), (3) beside and (4) under the config's registered set
  (REGISTERED_FLAGS[precision] with autocast_ctx(precision): bf16 autocast over TF32 on for the screens, A10.5);
  needs --tf32 both. Refuses a latent_band config, a non-feedback arm, --tf32 off / on, a critic_log.md without the
  registering row 2026-10-20T12:40Z, an --extra-idx outside the split and an existing <out>/summary.json (write-once).
  The record names the row in `critic_row` and keeps `registered_read` null (a diagnostic, not a 4.5 read).
  Off the registered check (each needs --out, so such a record never lands in results/ by default; the critic decides
  whether it enters the record): --strict-only runs the strict set alone ((1), (3), (4); (2) not run); --dtype float64
  (needs --strict-only; e.g. with --set device="cpu") casts the model to float64 after loading, so the HF path vs
  band [0, L] comparison carries kernel-order noise near 1e-12 instead of fp32's (a path difference shows, noise does
  not); --base-config DIR builds the base from DIR's config.json and tokenizer files (each must match the
  sha256 pinned for the config's model_id in configs/data_manifest.json) with no pretrained weights, for a container
  without the hub: refused unless the checkpoint supplies every tensor of the model's state dict, bit for bit after
  loading, so no weight comes from anywhere but the checkpoint. --strict-only also runs check (1b) (the HF path at
  B = 1 against band [0, L] at B = 1 on (3)'s targets) and records the floating dtypes of the weights and of every
  module output during the checks (refused unless all are --dtype). --tol sets the line of (1), (1b), (3)'s logits
  and (4) (default 1e-4; 1e-8 with --dtype float64: critic entry 2026-10-20T14:54Z reads the float64 run against 1e-8).
  --critic-row TS names the critic_log.md row that registers this record's job (refused unless it is a row; default
  2026-10-20T12:40Z, the mode's row): diag-band-prosqa-hard-main's entries carry 2026-10-20T21:33Z, which accepts
  float64 on the A100 (--set device="cuda"); the record keeps the mode's row in `mode_critic_row`. Every band_logits
  record also names the attention implementation the loader resolved (`attn_implementation`, with torch's SDPA backend
  switches on cuda) and the peak memory (`peak_memory`: torch.cuda.max_memory_allocated / max_memory_reserved over the
  checks on cuda, and the process's peak RSS), critic entry 2026-10-20T21:33Z (2).
  --checks (a comma list from 1, 1b, 2, 3, 4; default: every check the settings make, the record as before) runs the
  listed checks only; a check not listed is recorded with verdict "not run (--checks)", no line and no flag line, and
  the record carries `checks_requested`, `checks_run` and `checks_not_run` ({check: --not-run-reason TEXT, default
  "not run (--checks)"}) (a subset is off the registered check: needs --out; 1b needs --strict-only, 2 is refused with
  it). After each check the record's form with the checks finished so far and `complete` false is written to
  <out>/partial.json and <out>/partial_examples.csv, so a failure in a later check (e.g. out of memory) leaves the
  earlier numbers on disk; a partial file is never a record (critic entry 2026-10-20T22:36Z (1)): summary.json and
  examples.csv are written once at the end with `complete` true, and `partial_files` names the partial files (with
  their sha256; `files` stays the record's own). Between checks
  the finished check's rows are dropped once consumed and the cuda allocator's cache is emptied (latent.diag._release;
  no number changes).
* --mode donor_answer (A6.1 rule 4 (B)'s directional read on GSM-50k, the donor-answer read; design note
  governance/designs/a6_1_directional_read.md, rev230; registered by critic entry 2026-10-20T17:36Z = DONOR_ROW, whose
  eleven rulings this mode follows): latent.diag.donor_answer = latent.diag.scramble unchanged (evaluate(None) and
  evaluate("scrambled", scramble_seed 0) under the Recorder's fed-vector checks: whole set, one class of 500 at 6
  latents, the donor's latents bit for bit at every rank) plus the pure scorer over the two passes' ExampleResults: the
  class (the donor's step min(n_steps, 3) result equals its answer: 295 of 500 on the valid file of record), the 6
  ambiguous pairs excluded (289 scored), R_dir / R_own / Delta_dir with 6.2's interval (10,000 paired draws,
  default_rng(0), percentile beside, no sigma term), the garbage readings (G_chg > R_dir is rule 5's of record; the
  gold-literal and excess readings beside), complement and secondary diagnostics. Numerics are set by this tool (an
  entry may not set precision): the record of record is the config's registered set (bf16 autocast over TF32 on,
  A10.5), the strict set (TF32 off, no autocast, fp32 weights) runs beside it under --tf32 both as a diagnostic;
  --tf32 off alone is refused. Identification (ruling 9): the registered set's correct_own / correct_scr against the
  `<run>_valid_none` and `<run>_valid_scrambled` reads of the same checkpoint found under --identify-dir (default: the
  directory holding --out, i.e. the job's other entries on the VM), 0 of N expected on each, any difference holds the
  record; "not run" when the reads are absent (read_screen.py does it at table time). Refuses before the model loads:
  a split other than valid, a non-feedback arm, a latent_band config, --tf32 off, a batch size other than the
  config's eval batch, --n-boot other than 10,000, a non-GSM record, a final regime whose last supervised step is not
  3, the GSM-50k config on a valid file other than the one of record, the class, ambiguous set, scored count or first
  ten pairs other than the pinned ones (295, idx 15 124 125 175 470 477, 289; a different count is a refusal, ruling
  1), a critic_log.md without DONOR_ROW and an existing <out>/summary.json (write-once). A dep_loss false config is
  read and labelled "reference arm (no hinge); diagnostic; decides nothing" (ruling 8). Writes examples.csv (a row per
  numerics set and example; ids, booleans and numbers only, no pred, gold or text string), fed_vectors.csv (a row per
  set, example and rank) and summary.json (the common block, per set `donor_answer` and `scramble`, the strict set's
  `strict_vs_registered`, the outcome word, the pass/fail line, the labels, the predictions, registered_read null and
  critic_row DONOR_ROW).
* --tf32 both (default) runs off (strict fp32) then on (the reads' numerics) in one process; run_train.set_tf32 sets
  and reads back the flags per mode, recorded as run_eval.py records them.
* Writes <out>/examples.csv (a row per numerics mode and example; ids and numbers only, no example text; band_logits:
  a row per check, numerics set and example), <out>/fed_vectors.csv (latent_path: a row per numerics mode, example and
  latent rank) and <out>/summary.json (per-mode results, numerics, checkpoint / split / file sha256, commit, flags;
  "bug" true when a fed-vector check fails; band_logits: per check its quantities, maxima, counts, numerics and
  verdict, the row, the torch build and device). Default --out: results/<run>/valid_diag_<mode>/. `commit`: --commit
  SHA (the launcher's code commit; A11.4), else git HEAD.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common"))
sys.path.insert(0, HERE)

from run_eval import LATENT_FEEDBACK_ARMS, critic_row_exists, final_latents, git_commit, sha256_file  # noqa: E402
from run_train import load_config, set_tf32  # noqa: E402

MODES = ("cot_batch", "latent_path", "band_logits", "donor_answer")
ANSWERS = {"cot_batch": "A12.3 (critic entry 2026-10-19T16:59Z)",
           "latent_path": "A11.3 (1) (critic entries 2026-10-19T15:35Z and 2026-10-19T16:59Z)",
           "band_logits": "A6.2 rule 3's strict-fp32 logits check (critic entry 2026-10-20T12:40Z, the registered "
                          "diagnostic; A14.1; the 10:51Z row part (3))",
           "donor_answer": "A6.1 rule 4 (B), the donor-answer read on GSM-50k (critic entries 2026-10-20T17:12Z part "
                           "(5) and 2026-10-20T17:36Z)"}
CRITIC_LOG = os.path.join(EXP, "governance", "critic_log.md")
MANIFEST = os.path.join(EXP, "configs", "data_manifest.json")
BAND_ROW = "2026-10-20T12:40Z"          # the critic entry that registers --mode band_logits (its four checks)
BAND_EXTRA_IDX = (273, 497)             # deviations.md 2026-10-20 12:05 UTC: the band [0, 12] read's batch-1 flips
PARTIAL_JSON, PARTIAL_CSV = "partial.json", "partial_examples.csv"   # band_logits: after each check; never a record
BAND_TOL64 = 1e-8                       # critic entry 2026-10-20T14:54Z: the float64 CPU run is read against 1e-8
BAND_RULES = {
    "1": "A6.2 rule 3: expected <= 1e-4; larger is a BUG until explained, and no saving label may be written on "
         "either C1b arm while it stands (critic entry 2026-10-20T12:40Z (1))",
    "2": "printed beside (1) at the read's registered numerics, with the diverging greedy generations; no line "
         "(critic entry 2026-10-20T12:40Z (2))",
    "3": "A14.1: a count of 0 diverging greedy generations at strict fp32 is the shape pass; a diverging generation "
         "at strict fp32 is A12.3's outcome B condition (critic entry 2026-10-20T12:40Z (3))",
    "4": "A14.1: a swap difference above 1e-4 at either numerics is A12.3's outcome B condition (critic entry "
         "2026-10-20T12:40Z (4))"}
# --mode donor_answer (critic entry 2026-10-20T17:36Z, rulings 1 to 11 and parts (2), (3), (7))
DONOR_ROW = "2026-10-20T17:36Z"         # the critic entry that registers --mode donor_answer (its `critic_row`)
DONOR_N_BOOT, DONOR_BOOT_SEED, DONOR_SCRAMBLE_SEED = 10_000, 0, 0     # ruling 4; section 8's EVAL_SCRAMBLE_SEED
DONOR_IDENT_MAX_DIFF = 0                # ruling 9: 0 of N expected on each read; any difference holds the record
DONOR_REFERENCE_LABEL = "reference arm (no hinge); diagnostic; decides nothing"     # ruling 8, verbatim
DONOR_VALID_SHA256 = "1833a3479186e4e6c0183aad32fb7ee343479c958f2a1b57631ec22f5ee3aa7f"   # data/gsm_valid.json
DONOR_PIN = {"n": 500, "n_class": 295, "ambiguous_idx": [15, 124, 125, 175, 470, 477], "n_scored": 289,
             "first_ten": [[0, 431], [2, 459], [3, 201], [6, 419], [8, 366], [9, 108], [10, 196], [11, 326],
                           [12, 118], [13, 153]]}           # ruling 1 / T8 on the valid file of record, seed 0
DONOR_TRAINING_FIGURE = ("0.770: A5 item 5's figure for the training subset at 3 latent steps; it does not transfer "
                         "(valid has 173 records with 4 or more steps)")
DONOR_NO_TEST = ("No test read of this checkpoint exists under this read and none is registered by critic entry "
                 "2026-10-20T17:36Z; the test split is never scored under the donor-answer read.")
DONOR_PREDICTIONS = {      # critic entry 2026-10-20T17:36Z (3) (the registered prediction) and the note's 2.5 (the operator's)
    "critic (2026-10-20T17:36Z (3), the registered prediction)": {
        "all": "R_own 0.010 [0.003, 0.025] on every arm",
        "c3": "Delta_dir +0.03 [-0.01, +0.12], P(present) 0.35; G_chg 0.30 [0.10, 0.70]",
        "c2c3": "Delta_dir +0.05 [-0.01, +0.20], P(present) 0.45; G_chg 0.25 [0.08, 0.60]",
        "c3_foreign": "Delta_dir +0.01 [-0.01, +0.05], P(present) 0.15; G_chg 0.12 [0.03, 0.40]",
        "coconut": "Delta_dir +0.02 [-0.01, +0.08], P(present) 0.30; G_chg 0.10 [0.03, 0.30]"},
    "lead (design note 2.5; the operator's, not the registered prediction)": {
        "all": "R_own 0.010 [0.003, 0.025] on every arm",
        "c3": "Delta_dir +0.02 [-0.01, +0.12], P(present) 0.25; excess garbage +0.15 [0, +0.5]",
        "c2c3": "Delta_dir +0.04 [-0.01, +0.20], P(present) 0.35",
        "c3_foreign": "Delta_dir +0.00 [-0.01, +0.04], P(present) 0.10",
        "coconut": None}}


def write_csv(path: str, rows: list[dict]) -> str:
    """Rows to CSV (columns: the union of keys in first-seen order; None as empty); returns the file's sha256."""
    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return sha256_file(path)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", "--ckpt", dest="checkpoint", required=True)
    ap.add_argument("--mode", required=True, choices=MODES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="valid", help="valid only (anything else is refused: no test read is spent)")
    ap.add_argument("--tf32", choices=("off", "on", "both"), default="both")
    ap.add_argument("--batch-size", type=int, default=None, help="default: the config's eval_batch_size (the reads')")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="bootstrap draws for the accuracy CIs (default 0: none); donor_answer: 10,000, the registered "
                         "draws, and nothing else")
    ap.add_argument("--perm-seed", type=int, default=0, help="seed of the second companion set's permutation")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--commit", default=None, help="A11.4: the code commit written into summary.json")
    ap.add_argument("--extra-idx", default=None,
                    help="band_logits: comma list of valid idx compared at B = 1 against their own batch in check (3) "
                         f"(default {','.join(map(str, BAND_EXTRA_IDX))})")
    ap.add_argument("--dtype", choices=("float32", "float64"), default="float32",
                    help="band_logits: the strict set's model dtype (float64 needs --strict-only and --out)")
    ap.add_argument("--strict-only", action="store_true",
                    help="band_logits: the strict set alone (checks (1), (1b), (3), (4); (2) not run); needs --out")
    ap.add_argument("--tol", type=float, default=None,
                    help=f"band_logits: the checks' line (default 1e-4; {BAND_TOL64} with --dtype float64); needs "
                         "--out when not the default")
    ap.add_argument("--base-config", default=None, metavar="DIR",
                    help="band_logits: base from DIR's pinned config.json and tokenizer, no pretrained weights; the "
                         "checkpoint must supply every tensor (needs --out)")
    ap.add_argument("--checks", default=None, metavar="LIST",
                    help="band_logits: comma list of the checks to run, from 1, 1b, 2, 3, 4 (default: every check the "
                         "settings make); a check not listed is recorded \"not run (--checks)\"; a subset needs --out")
    ap.add_argument("--not-run-reason", default=None, metavar="TEXT",
                    help="band_logits with a --checks subset: why the checks not listed are not run (the record's "
                         "checks_not_run; default \"not run (--checks)\")")
    ap.add_argument("--critic-row", default=None, metavar="TS",
                    help=f"band_logits: the critic_log.md row that registers this record's job (default {BAND_ROW})")
    ap.add_argument("--identify-dir", action="append", default=None, metavar="DIR",
                    help="donor_answer: directories holding <run>_valid_none/ and <run>_valid_scrambled/ (the "
                         "identification check, ruling 9; default: the directory holding --out)")
    a = ap.parse_args(argv)
    if a.split != "valid":
        raise SystemExit(f"REFUSED: run_diag.py reads the valid split only and spends no test read (A12.3, A11.3 (1)); "
                         f"got --split {a.split!r}")
    cfg = load_config(a.config, a.set)
    path = cfg.val_path
    if not os.path.basename(path).endswith("_valid.json"):
        raise SystemExit(f"REFUSED: the config's val_path {path!r} is not a *_valid.json file")
    if a.mode == "band_logits":
        return band_main(a, cfg, path)
    if a.extra_idx is not None:
        raise SystemExit("REFUSED: --extra-idx belongs to --mode band_logits")
    if (a.dtype != "float32" or a.strict_only or a.base_config or a.tol is not None or a.critic_row is not None
            or a.checks is not None or a.not_run_reason is not None):
        raise SystemExit("REFUSED: --checks, --not-run-reason, --dtype, --strict-only, --base-config, --tol and --critic-row belong to "
                         "--mode band_logits")
    if a.mode == "donor_answer":
        return donor_main(a, cfg, path)
    if a.identify_dir:
        raise SystemExit("REFUSED: --identify-dir belongs to --mode donor_answer")
    if cfg.latent_band:
        raise SystemExit("REFUSED: latent.diag follows the HF path; a latent_band (C1b) model is not covered")
    if a.mode == "latent_path" and cfg.arm not in LATENT_FEEDBACK_ARMS:
        raise SystemExit(f"REFUSED: --mode latent_path needs a latent-feedback arm {LATENT_FEEDBACK_ARMS}, "
                         f"not {cfg.arm}")
    modes = ("off", "on") if a.tf32 == "both" else (a.tf32,)

    import torch
    import transformers

    from latent import diag
    from latent.data import load_examples
    from latent.train import autocast_ctx, build_model, resolve_device

    n_latent = final_latents(cfg)
    out_dir = a.out or os.path.join(EXP, "results", cfg.name, f"valid_diag_{a.mode}")
    device = resolve_device(cfg.device)
    lm, tokenizer, _ = build_model(cfg, load_path=a.checkpoint)
    lm.to(device)
    lm.eval()
    examples = load_examples(path, None)
    bs = a.batch_size or cfg.eval_batch_size
    n_boot = 0 if a.n_boot is None else a.n_boot
    kw = dict(arm=cfg.arm, n_latent=n_latent, batch_size=bs, device=device, max_new_tokens=cfg.max_new_tokens,
              perm_seed=a.perm_seed, n_boot=n_boot)
    results, numerics, seconds, rows, ranks = {}, {}, {}, [], []
    for m in modes:
        numerics[m] = set_tf32(m == "on")
        t0 = time.time()
        with autocast_ctx(cfg.precision, device):
            if a.mode == "cot_batch":
                r, s = diag.cot_batch(lm, tokenizer, examples, **kw)
                rr = []
            else:
                r, rr, s = diag.latent_path(lm, tokenizer, examples, scramble_seed=0, **kw)
        seconds[m] = time.time() - t0
        results[m] = s
        rows += [{"tf32": m, **x} for x in r]
        ranks += [{"tf32": m, **x} for x in rr]
        print(json.dumps({"tf32": m, "seconds": round(seconds[m], 1), "bug": s.get("bug", []),
                          "diverging": {k: s["generation"][k]["diverging"] for k in diag.SETS}}))
    os.makedirs(out_dir, exist_ok=True)
    files = {"examples.csv": write_csv(os.path.join(out_dir, "examples.csv"), rows)}
    if a.mode == "latent_path":
        files["fed_vectors.csv"] = write_csv(os.path.join(out_dir, "fed_vectors.csv"), ranks)
    task = os.path.basename(cfg.train_path)[: -len("_train.json")]
    bug = any(results[m].get("bug") for m in modes)
    out = {"tool": "run_diag.py", "mode": a.mode, "answers": ANSWERS[a.mode], "task": task, "run": cfg.name,
           "arm": cfg.arm, "seed": cfg.seed, "config": os.path.relpath(os.path.abspath(a.config), REPO),
           "overrides": a.set, "checkpoint": a.checkpoint, "checkpoint_sha256": sha256_file(a.checkpoint),
           "split": "valid", "split_file": os.path.relpath(path, REPO), "split_sha256": sha256_file(path),
           "n": len(examples), "batch_size": bs, "max_new_tokens": cfg.max_new_tokens, "n_latent": n_latent,
           "perm_seed": a.perm_seed, "scramble_seed": 0 if a.mode == "latent_path" else None, "tol": diag.TOL,
           "n_boot": n_boot, "tf32_modes": list(modes), "numerics": numerics, "precision": cfg.precision,
           "device": device, "gpu_type": torch.cuda.get_device_name() if device.startswith("cuda") else device,
           "torch": torch.__version__, "transformers": transformers.__version__, "commit": a.commit or git_commit(),
           "registered_read": None, "bug": bug, "flags": diag.flags(results, bs), "results": results,
           "files": files, "examples_csv_sha256": files["examples.csv"], "seconds": seconds,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"bug": bug, "flags": out["flags"], "files": files}))
    return out


def numerics_sets(precision: str, device: str, dtype: str = "float32") -> dict:
    """band_logits' two numerics sets (the entry may not set `precision`, so the tool sets them per check): "strict"
    = set_tf32(False) asserted as latent.train.REGISTERED_FLAGS["fp32"] (TF32 off, "highest") with no autocast;
    "registered" = set_tf32 to REGISTERED_FLAGS[precision]'s TF32 flag, asserted as that set, inside
    autocast_ctx(precision) (bf16: autocast over TF32 on, A10.5). Each yields the read-back state. dtype "float64"
    only relabels the strict set (the same flags; the caller casts the model to float64, which TF32 does not touch)."""
    import contextlib

    import torch

    from latent.train import REGISTERED_FLAGS, autocast_ctx, numerics_state
    dt = device.split(":")[0]

    @contextlib.contextmanager
    def strict():
        st = set_tf32(False)
        numerics_state("fp32")
        if torch.is_autocast_enabled(dt):
            raise RuntimeError("strict fp32 entered with autocast on")
        if dtype == "float64":
            yield {"set": f"strict float64: model weights and activations float64 on {device}, no autocast (TF32 off "
                          "and float32 matmul precision highest as for strict fp32; float64 matmuls do not use TF32)",
                   "registered_flags": "fp32", "autocast": None, "dtype": "float64", **st}
            return
        yield {"set": "strict fp32: TF32 off (matmul and cuDNN), float32 matmul precision highest, no autocast",
               "registered_flags": "fp32", "autocast": None, **st}

    @contextlib.contextmanager
    def registered():
        st = set_tf32(REGISTERED_FLAGS[precision][0])
        numerics_state(precision)
        with autocast_ctx(precision, device):
            auto = precision if precision in ("bf16", "fp16") else None
            if bool(torch.is_autocast_enabled(dt)) != (auto is not None):
                raise RuntimeError(f"autocast state does not match precision {precision!r}")
            yield {"set": f"the read's registered numerics: precision {precision} (autocast {auto or 'off'}) over "
                          f"REGISTERED_FLAGS[{precision!r}] = {REGISTERED_FLAGS[precision]} (A10.1, A10.5)",
                   "registered_flags": precision, "autocast": auto, **st}
    return {"strict": strict, "registered": registered}


def pinned_base_files(model_id: str, d: str, manifest: str = None) -> dict:
    """--base-config: DIR's config.json and tokenizer files against the sha256 the manifest pins for model_id (every
    file present in DIR that the pin lists must match; config.json and vocab.json must be present). Returns
    {file: sha256}; SystemExit on a mismatch or no pin."""
    with open(manifest or MANIFEST) as f:
        pins = [m for m in json.load(f).get("models", []) if m.get("id") == model_id]
    if not pins or not pins[0].get("files_sha256"):
        raise SystemExit(f"REFUSED: --base-config: no files_sha256 pinned for {model_id!r} in the manifest")
    want, got = pins[0]["files_sha256"], {}
    for name in sorted(os.listdir(d)):
        if name in want:
            got[name] = sha256_file(os.path.join(d, name))
            if got[name] != want[name]:
                raise SystemExit(f"REFUSED: --base-config: {name} sha256 {got[name]} is not the pinned {want[name]} "
                                 f"({model_id} revision {pins[0].get('revision')})")
    for need in ("config.json", "vocab.json"):
        if need not in got:
            raise SystemExit(f"REFUSED: --base-config: {d} has no pinned {need}")
    return {"files_sha256": got, "revision": pins[0].get("revision")}


def base_from_config(d: str):
    """--base-config: the base model built from DIR's config.json (random init, float32: every tensor is then
    replaced from the checkpoint, checked by checkpoint_covers) and DIR's tokenizer, pad = eos as load_base sets it."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(d))
    tokenizer = AutoTokenizer.from_pretrained(d)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def checkpoint_covers(base_model, path: str) -> int:
    """Every tensor of base_model's state dict is in the checkpoint (a plain causal-LM state dict or a LatentLM one
    with "base_causallm." keys) with the same shape and equal values bit for bit (the model as loaded); returns the
    count. SystemExit otherwise: a model whose weights do not all come from the checkpoint is refused."""
    import torch
    sd = torch.load(path, map_location="cpu")
    if any(k.startswith("base_causallm.") for k in sd):
        sd = {k[len("base_causallm."):]: v for k, v in sd.items() if k.startswith("base_causallm.")}
    msd = base_model.state_dict()
    missing = [k for k in msd if k not in sd]
    differ = [k for k in msd if k in sd and (sd[k].shape != msd[k].shape
                                             or not torch.equal(sd[k].to(msd[k].dtype), msd[k].detach().cpu()))]
    if missing or differ:
        raise SystemExit(f"REFUSED: --base-config: the checkpoint does not supply every tensor of the model (missing "
                         f"{missing[:5]}, differing {differ[:5]}); weights would come from the random init")
    return len(msd)


def peak_memory(device: str) -> dict:
    """band_logits' peak memory over the checks: torch.cuda.max_memory_allocated / max_memory_reserved on cuda (reset
    before the checks; None elsewhere) and the process's peak resident set (ru_maxrss, KiB on Linux) in bytes."""
    import resource

    import torch
    cuda = device.startswith("cuda")
    return {"device_max_allocated_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None,
            "device_max_reserved_bytes": int(torch.cuda.max_memory_reserved()) if cuda else None,
            "host_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024}


def attention_record(model, device: str) -> dict:
    """The attention implementation the loader resolved (`config._attn_implementation`) and, on cuda, torch's SDPA
    backend switches (flash, memory-efficient, math, cuDNN) as torch reports them (at float64 only the math kernel
    takes the inputs; critic entry 2026-10-20T21:33Z (1))."""
    import torch
    cfg = getattr(model, "config", None)
    out = {"attn_implementation": getattr(cfg, "_attn_implementation", None) or getattr(cfg, "attn_implementation",
                                                                                         None)}
    if device.startswith("cuda"):
        b = torch.backends.cuda
        out["sdpa_backends_enabled"] = {k: bool(f()) for k, f in (
            ("flash", getattr(b, "flash_sdp_enabled", None)), ("mem_efficient", getattr(b, "mem_efficient_sdp_enabled", None)),
            ("math", getattr(b, "math_sdp_enabled", None)), ("cudnn", getattr(b, "cudnn_sdp_enabled", None))) if f}
    return out


def band_main(a, cfg, path: str) -> dict:
    """--mode band_logits (see the module docstring): refusals before the model loads, then latent.diag.band_logits
    under numerics_sets, then examples.csv and summary.json (write-once in --out)."""
    from latent.train import REGISTERED_FLAGS
    if cfg.latent_band:
        raise SystemExit("REFUSED: --mode band_logits takes the full-depth checkpoint's config (latent_band null); it "
                         "builds band [0, L] over the same weights itself (A6.2 rule 1)")
    if cfg.arm not in LATENT_FEEDBACK_ARMS:
        raise SystemExit(f"REFUSED: --mode band_logits needs a latent-feedback arm {LATENT_FEEDBACK_ARMS}, "
                         f"not {cfg.arm}")
    if a.tf32 != "both":
        raise SystemExit("REFUSED: --mode band_logits sets the numerics per check itself (strict fp32 for (1), (3), "
                         "(4); the registered set for (2)): run it with --tf32 both")
    if cfg.precision not in REGISTERED_FLAGS:
        raise SystemExit(f"REFUSED: precision {cfg.precision!r} has no registered flag set {sorted(REGISTERED_FLAGS)}")
    if a.n_boot:
        raise SystemExit("REFUSED: --n-boot belongs to cot_batch / latent_path")
    if a.dtype == "float64" and not a.strict_only:
        raise SystemExit("REFUSED: --dtype float64 runs the strict set alone (the registered set is autocast over "
                         "fp32): add --strict-only")
    from latent import diag
    tol = a.tol if a.tol is not None else (BAND_TOL64 if a.dtype == "float64" else diag.TOL)
    if not tol > 0:
        raise SystemExit(f"REFUSED: --tol {a.tol}: a positive line")
    try:
        made = diag.band_checks(None, strict_only=a.strict_only, paths_b1=a.strict_only)
        checks = made if a.checks is None else diag.band_checks(
            [c for c in a.checks.split(",") if c.strip()], strict_only=a.strict_only, paths_b1=a.strict_only)
    except ValueError as e:
        raise SystemExit(f"REFUSED: --checks {a.checks!r}: {e}")
    if a.not_run_reason is not None and (checks == made or not a.not_run_reason.strip()):
        raise SystemExit("REFUSED: --not-run-reason names why a --checks subset leaves checks out: give a subset and a "
                         "reason")
    reason = a.not_run_reason or diag.BAND_NOT_RUN
    off = a.dtype != "float32" or a.strict_only or a.base_config or tol != diag.TOL or checks != made
    if off and not a.out:
        raise SystemExit("REFUSED: --dtype float64, --strict-only, --base-config and a --checks subset are off the "
                         "registered check: give --out (the record does not land in results/ by default)")
    if not critic_row_exists(BAND_ROW, CRITIC_LOG):
        raise SystemExit(f"REFUSED: --mode band_logits is registered by critic entry {BAND_ROW}, which is not in "
                         f"{CRITIC_LOG}")
    row = a.critic_row or BAND_ROW
    if not critic_row_exists(row, CRITIC_LOG):
        raise SystemExit(f"REFUSED: --critic-row {row!r} is not a row of {CRITIC_LOG}")
    try:
        extra = (tuple(BAND_EXTRA_IDX) if a.extra_idx is None
                 else tuple(int(x) for x in a.extra_idx.split(",") if x.strip()))
    except ValueError:
        raise SystemExit(f"REFUSED: --extra-idx {a.extra_idx!r}: a comma list of integers")
    out_dir = a.out or os.path.join(EXP, "results", cfg.name, f"valid_diag_{a.mode}")
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        raise SystemExit(f"REFUSED: {out_dir}/summary.json exists; a band_logits record is write-once")

    from latent.data import load_examples
    examples = load_examples(path, None)
    bad = [i for i in extra if not 0 <= i < len(examples)]
    if bad:
        raise SystemExit(f"REFUSED: --extra-idx {bad} outside the split's idx 0 to {len(examples) - 1}")
    bs = a.batch_size or cfg.eval_batch_size

    import torch
    import transformers

    from latent import diag
    from latent.train import build_model, resolve_device

    n_latent = final_latents(cfg)
    device = resolve_device(cfg.device)
    base = None
    if a.base_config:
        base = {"from": "config (no pretrained weights; every tensor from the checkpoint)",
                "dir": a.base_config, "model_id": cfg.model_id, **pinned_base_files(cfg.model_id, a.base_config)}
        model, tok = base_from_config(a.base_config)
        lm, tokenizer, _ = build_model(cfg, model=model, tokenizer=tok, load_path=a.checkpoint)
        base["checkpoint_covers"] = checkpoint_covers(lm.base_causallm, a.checkpoint)
    else:
        lm, tokenizer, _ = build_model(cfg, load_path=a.checkpoint)
    dtype = getattr(torch, a.dtype)
    lm.to(device=device, dtype=dtype)
    lm.eval()
    sets = numerics_sets(cfg.precision, device, a.dtype)
    if a.strict_only:
        sets = {"strict": sets["strict"]}
    cuda = device.startswith("cuda")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    attn = attention_record(lm.base_causallm, device)
    task = os.path.basename(cfg.train_path)[: -len("_train.json")]
    rules = dict(BAND_RULES)
    if tol != diag.TOL:
        rules["1"] = (f"read against {tol:g} ({a.dtype}, off the registered check; critic entry 2026-10-20T14:54Z for "
                      f"the float64 run); A6.2 rule 3's registered line is 1e-4 at strict fp32")
    rules["1b"] = f"beside (1): the HF path vs band [0, L] at B = 1, line {tol:g} (critic entry 2026-10-20T14:54Z)"
    not_run = ("not run (strict set alone)", diag.BAND_NOT_RUN)
    per_check = ({"1": "strict", "1b": "strict", "2": "not run (--strict-only)", "3": "strict (the line)",
                  "4": "strict (the line)"} if a.strict_only else
                 {"1": "strict", "2": "registered", "3": "strict (the line); registered beside",
                  "4": "strict and registered (the line at each)"})
    per_check = {k: (v if k in checks or v.startswith("not run") else diag.BAND_NOT_RUN) for k, v in per_check.items()}
    t0 = time.time()

    def ruled(s):
        """Each check's rule (none for a check not run); check (1) over its line heads the bugs."""
        for k, c in s["checks"].items():
            c["rule"] = rules[k] if c["verdict"] not in not_run else None
        if s["checks"].get("1", {}).get("verdict", "pass") not in ("pass", diag.BAND_NOT_RUN):
            s["bug"].insert(0, f"check (1) is over its line: BUG until explained; {rules['1']}")
        return s

    def record(rows, s, complete):
        """The record's dict, written once at the end as examples.csv and summary.json (`complete` true); after each
        check the same form goes to PARTIAL_CSV and PARTIAL_JSON (`complete` false): a partial file survives a crash
        and is never a record (critic entry 2026-10-20T22:36Z (1)); the final record names the partial files the run
        wrote, with their sha256, in `partial_files` (`files` stays the record's own files)."""
        os.makedirs(out_dir, exist_ok=True)
        csv_name, json_name = ("examples.csv", "summary.json") if complete else (PARTIAL_CSV, PARTIAL_JSON)
        files = {"examples.csv" if complete else PARTIAL_CSV: write_csv(os.path.join(out_dir, csv_name), rows)}
        parts = ({f: sha256_file(os.path.join(out_dir, f)) for f in (PARTIAL_JSON, PARTIAL_CSV)
                  if os.path.isfile(os.path.join(out_dir, f))} if complete else None)
        out = {"tool": "run_diag.py", "mode": a.mode, "answers": ANSWERS[a.mode], "critic_row": row,
               "critic_row_in_log": True, "mode_critic_row": BAND_ROW, "task": task, "run": cfg.name, "arm": cfg.arm,
               "seed": cfg.seed, "config": os.path.relpath(os.path.abspath(a.config), REPO), "overrides": a.set,
               "checkpoint": a.checkpoint, "checkpoint_sha256": sha256_file(a.checkpoint), "split": "valid",
               "split_file": os.path.relpath(path, REPO), "split_sha256": sha256_file(path), "n": len(examples),
               "batch_size": bs, "batch": s["batch"], "extra_idx": list(extra), "band": s["band"],
               "n_layer": s["n_layer"], "max_new_tokens": cfg.max_new_tokens, "n_latent": n_latent,
               "perm_seed": a.perm_seed, "tol": tol, "precision": cfg.precision, "tf32_modes": s["sets"],
               "dtype": s["dtype"], "strict_only": a.strict_only, "base": base, "dtypes": s.get("dtypes"),
               "numerics": s["numerics"], "numerics_per_check": per_check,
               "numerics_note": "set by run_diag.py --mode band_logits itself, per check: a diag entry may not set "
                                "precision (the launcher's FORBIDDEN_SET and DIAG_FORBIDDEN), and the job's tf32 field "
                                "does not reach a diag entry (--tf32 both)",
               "device": device, "gpu_type": torch.cuda.get_device_name() if cuda else device,
               "torch": torch.__version__, "cuda": torch.version.cuda,
               "cudnn": torch.backends.cudnn.version() if cuda else None, "transformers": transformers.__version__,
               **attn, "peak_memory": peak_memory(device),
               "commit": a.commit or git_commit(), "registered_read": None, "bug": bool(s["bug"]),
               "verdicts": {k: c["verdict"] for k, c in s["checks"].items()}, "flags": s["flags"] + s["bug"],
               "complete": complete, "checks_requested": s["checks_requested"], "checks_run": s["checks_run"],
               "checks_not_run": {k: reason for k in made if k not in checks},
               "results": {"checks": s["checks"], "bug": s["bug"], "swap_pool": s["swap_pool"]},
               "files": files, "examples_csv_sha256": files["examples.csv" if complete else PARTIAL_CSV],
               "partial_files": parts, "seconds": time.time() - t0,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        with open(os.path.join(out_dir, json_name), "w") as f:
            json.dump(out, f, indent=2)
        return out

    def partial(rows, s):
        out = record(rows, ruled(s), False)
        print(json.dumps({"partial": PARTIAL_JSON, "checks_run": out["checks_run"], "verdicts": out["verdicts"]}))

    rows, s = diag.band_logits(lm, tokenizer, examples, arm=cfg.arm, n_latent=n_latent, batch_size=bs, device=device,
                               max_new_tokens=cfg.max_new_tokens, numerics=sets, extra_idx=extra,
                               perm_seed=a.perm_seed, dtype=dtype, strict_only=a.strict_only, tol=tol,
                               paths_b1=a.strict_only, checks=checks, on_check=partial)
    seconds = time.time() - t0
    ruled(s)
    print(json.dumps({"seconds": round(seconds, 1), "bug": s["bug"], "flags": s["flags"]}))
    out = record(rows, s, True)
    files = out["files"]
    print(json.dumps({"bug": out["bug"], "verdicts": out["verdicts"], "files": files}))
    return out


# ------------------------------------------------------------------------------------------------- donor_answer
def donor_preview(examples, n_latent: int, n_latent_steps: int) -> dict:
    """The class before any model loads (a pure function of the valid records and the registered derangement): the
    donor map scramble_donors({idx: n_latent}, 0), the class, the ambiguous pairs in it, the scored count and the
    first ten (idx, donor) pairs of the class."""
    from latent import diag
    from latent.data import answer_normalize
    from latent.eval import scramble_donors
    by = {ex.idx: ex for ex in examples}
    donors = scramble_donors({ex.idx: n_latent for ex in examples}, seed=DONOR_SCRAMBLE_SEED)
    cls = {i: diag.donor_in_class(by[i], n_latent_steps) for i in by}
    C = [ex.idx for ex in examples if cls[donors[ex.idx]]]
    amb = [i for i in C if answer_normalize(by[i].answer) == answer_normalize(by[donors[i]].answer)]
    return {"n": len(examples), "n_class": len(C), "ambiguous_idx": amb, "n_scored": len(C) - len(amb),
            "first_ten": [[i, donors[i]] for i in C[:10]]}


def _bool_cell(v: str, where: str) -> bool:
    if v not in ("True", "False"):
        raise ValueError(f"{where}: {v!r} is not True/False")
    return v == "True"


def identification(rows: list[dict], run: str, checkpoint_sha256: str, split_sha256: str, n_latent: int,
                   dirs: list[str]) -> dict:
    """Ruling 9: the registered set's correct_own / correct_scr against the `correct` column of the `none` and
    `scrambled` reads of the same checkpoint (<dir>/<run>_valid_none/, <dir>/<run>_valid_scrambled/: summary.json and
    examples.csv, the summary's checkpoint, split hash, intervention, n_latent, scramble seed and examples.csv hash
    checked), example by example; status "passed" (0 of N on each), "held" (any difference: the record lifts nothing
    until a critic entry explains it) or "not run" (no such reads here: read_screen.py performs it at table time)."""
    tried = []
    by = {r["idx"]: r for r in rows}
    for d in dirs:
        ent = {k: os.path.join(d, f"{run}_valid_{k}") for k in ("none", "scrambled")}
        if not all(os.path.isfile(os.path.join(x, f)) for x in ent.values() for f in ("summary.json", "examples.csv")):
            tried.append(f"{d}: no {run}_valid_none and {run}_valid_scrambled reads")
            continue
        reads, bad = {}, []
        for k, x in ent.items():
            with open(os.path.join(x, "summary.json")) as f:
                sm = json.load(f)
            csv_sha = sha256_file(os.path.join(x, "examples.csv"))
            want = {"run": run, "split": "valid", "checkpoint_sha256": checkpoint_sha256, "split_sha256": split_sha256,
                    "intervention": k, "n_latent": n_latent, "examples_csv_sha256": csv_sha}
            if k == "scrambled":
                want["scramble_seed"] = DONOR_SCRAMBLE_SEED
            bad += [f"{os.path.basename(x)} {f_} {sm.get(f_)!r} != {v!r}" for f_, v in want.items() if sm.get(f_) != v]
            with open(os.path.join(x, "examples.csv"), newline="") as f:
                try:
                    reads[k] = {int(r["idx"]): _bool_cell(r["correct"], os.path.basename(x))
                                for r in csv.DictReader(f)}
                except (KeyError, ValueError) as e:
                    bad.append(f"{os.path.basename(x)} examples.csv: {e}")
            reads[f"{k}_sha256"] = csv_sha
        if not bad and any(set(reads[k]) != set(by) for k in ent):
            bad.append("the reads do not list the record's examples")
        if bad:
            tried.append(f"{d}: " + "; ".join(bad))
            continue
        nd = [i for i in sorted(by) if by[i]["correct_own"] != reads["none"][i]]
        sd = [i for i in sorted(by) if by[i]["correct_scr"] != reads["scrambled"][i]]
        held = max(len(nd), len(sd)) > DONOR_IDENT_MAX_DIFF
        return {"status": "held" if held else "passed", "none_diff": len(nd), "scrambled_diff": len(sd),
                "none_diff_idx": nd, "scrambled_diff_idx": sd, "n": len(by), "dir": d,
                "entries": [os.path.basename(ent["none"]), os.path.basename(ent["scrambled"])],
                "examples_csv_sha256": {"none": reads["none_sha256"], "scrambled": reads["scrambled_sha256"]},
                "rule": "ruling 9: 0 of N expected on each; any difference holds the record until a critic entry"}
    return {"status": "not run", "none_diff": None, "scrambled_diff": None, "entries": [],
            "why": "; ".join(tried) or "no directory given",
            "rule": "ruling 9: read_screen.py performs it from the committed files at table time"}


def _x(v, fmt: str) -> str:
    return "n/a" if v is None else format(v, fmt)


def donor_line(task: str, run: str, sha: str, s: dict, sc: dict, ident: dict, outcome: str, numerics: str,
               gpu: str, bug: list) -> str:
    """The pass/fail line of the note's 2.4 for one checkpoint (the registered set), every quantity with its count."""
    from latent.diag import DONOR_PASS_LINE
    head = f"directional read (A6.1 rule 4 (B), donor-answer, {task} valid, seed 0, {run} {sha[:8]}): "
    if bug:
        return head + "BUG: " + "; ".join(bug) + "; no statistic is a result (the numbers stay in the file, bug true)"
    n, fv = s["n_scored"], sc["fed_vectors"]
    thin = f" ({s['label_thin']})" if s.get("label_thin") else ""
    idt = (f"identification {ident['none_diff']}/{ident['scrambled_diff']} of {s['n']} vs the none / scrambled reads "
           f"({ident['status']})" if ident["status"] in ("passed", "held") else
           "identification not run here (no none / scrambled read of this checkpoint beside the record; read_screen.py "
           "performs it at table time)")
    return (head + f"R_dir {s['k_dir']} of {n} = {_x(s['r_dir'], '.3f')} vs R_own {s['k_own']} of {n} = "
            f"{_x(s['r_own'], '.3f')}; Delta_dir {_x(s['delta_dir'], '+.3f')} ({s['k_net']:+d} of {n}) "
            f"[{_x(s['lo'], '+.3f')}, {_x(s['hi'], '+.3f')}] (6.2 form, {s['n_boot']:,} paired draws, default_rng("
            f"{s['boot_seed']}); percentile [{_x(s['pct_lo'], '+.3f')}, {_x(s['pct_hi'], '+.3f')}]): {outcome}{thin} "
            f"({DONOR_PASS_LINE}); garbage G_chg {s['k_g_chg']} of {n} = {_x(s['g_chg'], '.3f')} vs R_dir "
            f"{_x(s['r_dir'], '.3f')}: {'exceeded' if s['garbage_exceeds'] else 'not exceeded'} (rule 5's reading of "
            f"record; G_chg - R_dir {_x(s['chg_minus_dir'], '+.3f')} [{_x(s['lo_chg'], '+.3f')}, "
            f"{_x(s['hi_chg'], '+.3f')}], decides nothing); beside: G_scr {s['k_g_scr']} of {n} = "
            f"{_x(s['g_scr'], '.3f')} (gold-literal reading, "
            f"{'exceeds' if s['garbage_literal_exceeds'] else 'not above'} R_dir; decides nothing), Delta_g "
            f"{_x(s['delta_g'], '+.3f')} [{_x(s['lo_g'], '+.3f')}, "
            f"{_x(s['hi_g'], '+.3f')}] (excess reading = drop_S {_x(s['drop_S'], '+.3f')} minus Delta_dir, "
            f"{'exceeds' if s['garbage_excess_exceeds'] else 'not above'} Delta_dir; decides nothing); class "
            f"{s['n_class']} of {s['n']}, ambiguous {s['n_ambiguous']} excluded, scored {n}; {idt}; fed vectors: "
            f"{fv['zero_distance']} zero distances of {fv['pairs']}, "
            f"{fv['donor_is_self']} self donors, {fv['fed_equals_donor']} of {s['n']} fed the donor's latents bit for "
            f"bit, every prompt column fed once: {'yes' if fv['covered_once'] else 'NO'}; {numerics}; read on {gpu}; "
            f"screen number, 1 seed, valid, no test read; ProsQA-hard UNTESTED (A2.1)")


def donor_main(a, cfg, path: str) -> dict:
    """--mode donor_answer (see the module docstring and critic entry 2026-10-20T17:36Z): the refusals before the model
    loads, then latent.diag.donor_answer under the registered set (and the strict set beside it with --tf32 both),
    the identification check, then examples.csv, fed_vectors.csv and summary.json, write-once in --out."""
    from latent.train import REGISTERED_FLAGS
    if cfg.latent_band:
        raise SystemExit("REFUSED: --mode donor_answer follows the HF path (latent.diag.scramble's Recorder); a "
                         "latent_band (C1b) config is not covered")
    if cfg.arm not in LATENT_FEEDBACK_ARMS:
        raise SystemExit(f"REFUSED: --mode donor_answer needs a latent-feedback arm {LATENT_FEEDBACK_ARMS} (the "
                         f"scrambled pass feeds a donor's latents), not {cfg.arm}")
    if a.tf32 == "off":
        raise SystemExit("REFUSED: --mode donor_answer's record of record is the registered set (bf16 autocast over "
                         "TF32 on, A10.5); the strict set runs beside it under --tf32 both: --tf32 off alone is "
                         "refused (critic entry 2026-10-20T17:36Z ruling 6)")
    if cfg.precision not in REGISTERED_FLAGS:
        raise SystemExit(f"REFUSED: precision {cfg.precision!r} has no registered flag set {sorted(REGISTERED_FLAGS)}")
    if a.n_boot not in (None, DONOR_N_BOOT):
        raise SystemExit(f"REFUSED: --n-boot {a.n_boot}: the donor-answer interval is registered at {DONOR_N_BOOT:,} "
                         f"paired draws (ruling 4)")
    if a.batch_size is not None and a.batch_size != cfg.eval_batch_size:
        raise SystemExit(f"REFUSED: --batch-size {a.batch_size}: the donor-answer pass runs at the reads' eval batch "
                         f"{cfg.eval_batch_size} in file order (the identification check expects the reads' shapes)")
    if not critic_row_exists(DONOR_ROW, CRITIC_LOG):
        raise SystemExit(f"REFUSED: --mode donor_answer is registered by critic entry {DONOR_ROW}, which is not in "
                         f"{CRITIC_LOG}")
    out_dir = a.out or os.path.join(EXP, "results", cfg.name, f"valid_diag_{a.mode}")
    if os.path.exists(os.path.join(out_dir, "summary.json")):
        raise SystemExit(f"REFUSED: {out_dir}/summary.json exists; a donor_answer record is write-once")

    from latent import diag
    from latent.data import load_examples
    from latent.state_targets import task_kind
    examples = load_examples(path, None)
    kinds = sorted({task_kind(ex) for ex in examples})
    if kinds != ["gsm"]:
        raise SystemExit(f"REFUSED: the donor-answer read is A6.1 rule 4 (B)'s GSM read; the split's records are "
                         f"{kinds} (ProsQA-hard's directional read, the F1 patch, is UNTESTED under A2.1)")
    n_latent = final_latents(cfg)
    steps = n_latent // max(1, int(cfg.c_thought))
    if steps != diag.DONOR_CLASS_STEPS:
        raise SystemExit(f"REFUSED: the class rule is registered at step min(n_steps, {diag.DONOR_CLASS_STEPS}) (ruling "
                         f"1); this config's final regime supervises {steps} steps ({n_latent} latents at c_thought "
                         f"{cfg.c_thought})")
    split_sha = sha256_file(path)
    task = os.path.basename(cfg.train_path)[: -len("_train.json")]
    pre = donor_preview(examples, n_latent, steps)
    if task == "gsm50k" and split_sha != DONOR_VALID_SHA256:
        raise SystemExit(f"REFUSED: {path} sha256 {split_sha} is not the GSM-50k valid file of record "
                         f"{DONOR_VALID_SHA256}")
    if split_sha == DONOR_VALID_SHA256:
        diff = [f"{k} {pre[k]!r} != {v!r}" for k, v in DONOR_PIN.items() if pre[k] != v]
        if diff:
            raise SystemExit("REFUSED: the class on the valid file of record is not the registered one (ruling 1: a "
                             "different count is a refusal; the file or the derangement changed): " + "; ".join(diff))
    data_pin = ("checked: " + ", ".join(f"{k} {v}" for k, v in DONOR_PIN.items() if k != "first_ten")
                + ", first ten pairs" if split_sha == DONOR_VALID_SHA256 else
                "not the valid file of record: no pin applies")

    import torch
    import transformers

    from latent.train import build_model, resolve_device
    bs = cfg.eval_batch_size
    device = resolve_device(cfg.device)
    lm, tokenizer, _ = build_model(cfg, load_path=a.checkpoint)
    lm.to(device)
    lm.eval()
    if next(lm.parameters()).dtype != torch.float32:
        raise SystemExit(f"REFUSED: the model's weights are {next(lm.parameters()).dtype}; both numerics sets read "
                         f"fp32 weights (the registered set's autocast is the config's precision)")
    sets = numerics_sets(cfg.precision, device)
    order = ("registered", "strict") if a.tf32 == "both" else ("registered",)
    results, numerics, seconds, rows, ranks = {}, {}, {}, [], []
    for name in order:
        with sets[name]() as st:
            start, t0 = diag._flags_now(device), time.time()
            r, rr, s = diag.donor_answer(lm, tokenizer, examples, arm=cfg.arm, n_latent=n_latent, batch_size=bs,
                                         device=device, max_new_tokens=cfg.max_new_tokens,
                                         scramble_seed=DONOR_SCRAMBLE_SEED, n_boot=DONOR_N_BOOT,
                                         boot_seed=DONOR_BOOT_SEED, n_latent_steps=steps)
            end = diag._flags_now(device)
            if end != start:
                raise RuntimeError(f"numerics {name!r} changed during the read: {start} -> {end}")
        seconds[name] = time.time() - t0
        numerics[name] = {**dict(st), "torch_at_start": start, "torch_at_end": end}
        got = {"n": s["donor_answer"]["n"], "n_class": s["donor_answer"]["n_class"],
               "ambiguous_idx": s["donor_answer"]["ambiguous_idx"], "n_scored": s["donor_answer"]["n_scored"]}
        if any(pre[k] != v for k, v in got.items()):
            s["bug"].append(f"the read's class {got} is not the one computed before the model loaded "
                            f"{ {k: pre[k] for k in got} }")
        results[name] = s
        rows += [{"numerics": name, **x} for x in r]
        ranks += [{"numerics": name, **x} for x in rr]
        print(json.dumps({"numerics": name, "seconds": round(seconds[name], 1), "bug": s["bug"]}))
    reg_rows = [x for x in rows if x["numerics"] == "registered"]
    ck_sha = sha256_file(a.checkpoint)
    dirs = list(a.identify_dir) if a.identify_dir else [os.path.dirname(os.path.abspath(out_dir))]
    ident = identification(reg_rows, cfg.name, ck_sha, split_sha, n_latent, dirs)
    flags = []
    for name in order:
        da = results[name]["donor_answer"]
        if name == "registered":
            da["identification"] = ident
            da["outcome"] = diag.donor_answer_outcome(da, results[name]["bug"], ident["status"])
        else:
            da["identification"] = {"status": "not applicable (the strict set is a diagnostic; the reads ran at the "
                                              "registered numerics)"}
            da["outcome"] = diag.donor_answer_outcome(da, results[name]["bug"], None)
        da["label_thin"] = (f"present (thin: k_dir minus k_own = {da['k_net']})"
                            if da["outcome"] == diag.DONOR_OUTCOMES[2] and da["thin"] else None)
        flags += [f"{name}: BUG: {b}" for b in results[name]["bug"]]
    reg = results["registered"]["donor_answer"]
    if ident["status"] == "held":
        flags.append(f"held: identification check failed: {ident['none_diff']} (none) and {ident['scrambled_diff']} "
                     f"(scrambled) of {ident['n']} examples differ from the reads (ruling 9; a critic entry explains it)")
    if "strict" in results:
        st_ = results["strict"]["donor_answer"]
        srows = {x["idx"]: x for x in rows if x["numerics"] == "strict"}
        dd = [x["idx"] for x in reg_rows if x["d_scr"] != srows[x["idx"]]["d_scr"]]
        svr = {"outcome_registered": reg["outcome"], "outcome_strict": st_["outcome"],
               "outcome_word_disagrees": reg["outcome"] != st_["outcome"],
               "delta_dir_registered": reg["delta_dir"], "delta_dir_strict": st_["delta_dir"],
               "examples_whose_d_scr_differs": len(dd), "idx_d_scr_differs": dd,
               "examples_whose_correct_own_differs": sum(x["correct_own"] != srows[x["idx"]]["correct_own"]
                                                         for x in reg_rows),
               "examples_whose_correct_scr_differs": sum(x["correct_scr"] != srows[x["idx"]]["correct_scr"]
                                                         for x in reg_rows),
               "rule": "ruling 6: the strict set decides nothing; a disagreement of the outcome word goes to the "
                       "critic by a row, never resolved by the tool"}
        results["strict"]["strict_vs_registered"] = svr
        if svr["outcome_word_disagrees"]:
            flags.append(f"the strict set's outcome word ({st_['outcome']}) differs from the registered set's "
                         f"({reg['outcome']}): for the critic (ruling 6)")
    if reg["label_thin"]:
        flags.append(reg["label_thin"])
    os.makedirs(out_dir, exist_ok=True)
    files = {"examples.csv": write_csv(os.path.join(out_dir, "examples.csv"), rows),
             "fed_vectors.csv": write_csv(os.path.join(out_dir, "fed_vectors.csv"), ranks)}
    cuda = device.startswith("cuda")
    gpu = torch.cuda.get_device_name() if cuda else device
    numlab = (("bf16 autocast over TF32 on" if cfg.precision == "bf16" else
               f"{cfg.precision} over REGISTERED_FLAGS[{cfg.precision!r}]")
              + (" (strict fp32 beside, diagnostic)" if "strict" in results else " (strict fp32 not run)"))
    reg_bug = results["registered"]["bug"]
    line = donor_line(task, cfg.name, ck_sha, reg, results["registered"]["scramble"], ident, reg["outcome"], numlab,
                      gpu, reg_bug)
    arm_key = cfg.name[len(task) + 1:].rsplit("_s", 1)[0] if cfg.name.startswith(task + "_") else cfg.name
    out = {"tool": "run_diag.py", "mode": a.mode, "answers": ANSWERS[a.mode], "critic_row": DONOR_ROW,
           "critic_row_in_log": True, "task": task, "run": cfg.name, "arm": cfg.arm, "seed": cfg.seed,
           "dep_loss": bool(cfg.dep_loss), "reference": None if cfg.dep_loss else DONOR_REFERENCE_LABEL,
           "config": os.path.relpath(os.path.abspath(a.config), REPO), "overrides": a.set,
           "checkpoint": a.checkpoint, "checkpoint_sha256": ck_sha, "split": "valid",
           "split_file": os.path.relpath(path, REPO), "split_sha256": split_sha, "n": len(examples),
           "batch_size": bs, "max_new_tokens": cfg.max_new_tokens, "n_latent": n_latent, "n_latent_steps": steps,
           "scramble_seed": DONOR_SCRAMBLE_SEED, "tol": diag.TOL, "n_boot": DONOR_N_BOOT, "boot_seed": DONOR_BOOT_SEED,
           "precision": cfg.precision, "tf32_modes": list(order), "numerics": numerics,
           "numerics_note": "set by run_diag.py --mode donor_answer itself (an entry may not set precision): the "
                            "registered set (REGISTERED_FLAGS[precision] inside autocast_ctx(precision)) is the record "
                            "of record; the strict set (TF32 off, 'highest', no autocast) is beside it, deciding "
                            "nothing (critic entry 2026-10-20T17:36Z ruling 6)",
           "device": device, "gpu_type": gpu, "torch": torch.__version__, "cuda": torch.version.cuda,
           "transformers": transformers.__version__, "commit": a.commit or git_commit(), "registered_read": None,
           "data_pin": data_pin, "class_count_of_record": f"{DONOR_PIN['n_class']} of {DONOR_PIN['n']} (seed 0)",
           "training_figure": DONOR_TRAINING_FIGURE, "pass_line": diag.DONOR_PASS_LINE,
           "bug": any(results[k]["bug"] for k in order), "outcome": reg["outcome"], "line": line,
           "labels": [f"screen number: 1 seed, valid {len(examples)} (scored {reg['n_scored']}); bar LR-tuned 3 trials; "
                      f"arm 1 trial; diagnostic entry, no 4.5 read spent; {numlab}; read on {gpu}, trained on "
                      f"A100-80GB; no test read; ProsQA-hard UNTESTED (A2.1)"]
           + ([DONOR_REFERENCE_LABEL] if not cfg.dep_loss else []),
           "no_test_read": DONOR_NO_TEST,
           "predictions": {who: {"all": p["all"], "arm": p.get(arm_key)} for who, p in DONOR_PREDICTIONS.items()},
           "flags": flags, "results": results, "files": files, "examples_csv_sha256": files["examples.csv"],
           "seconds": seconds, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"bug": out["bug"], "outcome": out["outcome"], "line": line, "files": files}))
    return out


if __name__ == "__main__":
    main()
