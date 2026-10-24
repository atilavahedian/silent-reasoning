"""The screen table of one Phase 2 screen task (registration 7.11, A3.4, A3.4.5) from the run records and the registered
cost accounting: valid split only, screen numbers (1 seed), never a verdict.

    <torch venv>/bin/python code/read_screen.py --task gsm50k \
        --records DIR --source-uri store://<bucket>/silent \
        --receipts 'DIR/receipts/*/*/JOB_RECEIPT.json' [GLOB ...] \
        [--out results/screens/gsm50k] [--table-name gsm50k_screen]

DIR mirrors the bucket's `silent/` prefix for the small record files only: `runs/<run>/{config.json, log.jsonl,
RUN_DONE.json, ckpt_state.json, SYNC_MANIFEST.json, runs_index_row.csv}` (no checkpoint is ever read). The tool writes
<out>/<name>.json, <out>/<name>.md and <out>/<name>_provenance.json (every input file with its sha256, the globs and the
arguments), **write-once**: it refuses if any of the three exists. It never opens a test split: the valid file is
checked against the manifest's valid entry for the task.

What it reads, per registered run of the task (TASKS below):
* the rows of record of `log.jsonl` (A8.2: per epoch the row with the largest `resumes`); the selection recomputed as
  registered (3.3 item 1, A2 item 3 / D16: GSM-50k latent arms epoch_index 12-24, ProsQA-hard 21-29; `cot` / `no_cot`
  every epoch; argmax of valid accuracy, ties to the earliest epoch) and compared with RUN_DONE.json and ckpt_state.json
  (a mismatch is a BUG trigger, 7.10 / A8.2, and the run enters no cell); the "earlier stage better on valid" flag
  (3.3 item 3);
* record checks: RUN_DONE done / rc 0; the record files equal SYNC_MANIFEST.json's hashes; the effective config equals
  the registered config file in every recipe key; the base-model revision is the manifest's pin; one row of record per
  epoch; bf16 rows carry A10.5's flag set (else "precision-mismatched"); finite train_loss (A6.6); valid accuracy and
  the emitted-token mean are counts over the full valid split;
* A6.3 rule 4 (GSM-50k latent arms): `config.json` `load_model_path` / `load_model_sha256` / `load_model_epoch` and
  RUN_DONE's `init` against the critic's start row, the source run's own RUN_DONE, and the selected epoch inside the
  pool; a mismatch is "speculative, unused" and the run enters no cell;
* costs by `latent.accounting.inference_record` and `reference_cost_flops` (5.1, 5.3, 5.5 (c), A6.4) per valid example:
  |q| is exact (the hash-checked valid file through the pinned tokenizer, `latent.data.tokenize_examples`), L follows
  the selected epoch's stage (generation latents = min(stage, max_latent_stage) x c_thought), and E' is known only as
  the logged mean (`val_emitted_mean`; the per-example counts were not recorded by training). Mean emitted tokens and
  mean sequential passes are therefore exact; the FLOP means are printed as the exact range over every per-example E'
  assignment in [0, max_new_tokens] with the logged total (`sum_bounds`); a cost comparison is decided only when the
  ranges decide it. C1b arms (A3.4, A6.2): without the timed read their per-step saving is "nominal"; FLOPs are printed
  at the nominal (b - a) / L and at 1 (the largest value min(1, f_meas) can take), and a leg holds only if it holds at 1;
* sigma_screen (A3.4) = max(1.0, 100 sqrt(p (1 - p) / n_valid) at the bar's valid accuracy p, the ProsQA `cot` 3-seed SD
  once it exists), the reading of record; the lines are recomputed under the two narrower readings and every line whose
  outcome changes is listed (section 4 of the table).

Lines written (each carries "screen numbers (1 seed)" and A3.3's "bar LR-tuned 3 trials; arm 1 trial"): the bar (3.4 /
A3.3 pick), the screen rule of 6.3 for every card arm, the card lines of A3.4 (C1b, C2, C3, C2+C3; the parts this task
can decide), K2 (pending until a scrambled-at-inference valid read exists; "K2 not independent" for dep_loss arms,
A6.1), K3 (7.5 with 7.7's A_pd; A13.2: the 124M `no_cot` point alone and "nearest size" until both plain-depth runs have
a selection; once they do, "plain depth 355M, 1 seed" and the micro-batch label when it is not 32 x 4), the pause
comparison (7.5), the too-good triggers (7.10, A3.4), the 0.5 B / B fade instrument (7.6), the 7.2 cost legs with the
"billing claim only" label and the 5.5 accounting-sensitive check, the C2 labels (A5 item 5, A6.7 items 4-5), the launch
condition of A5 item 8, and UNTESTED for every registered run without records. A11.3's words are refused in any output.

The second table, "with the eval reads" (critic entry 2026-10-20T10:51Z part (6); A6.4's reissue pattern):

    ... read_screen.py --task gsm50k --records DIR --source-uri ... --receipts ... \
        --eval-dir results/evals/screen-eval-gsm50k-main --first-commit rev10

writes <out>/<task>_screen_with_eval.{json,md} and _provenance.json, write-once, citing the first table (its three files
must hash as committed at --first-commit; its bar, sigma_screen and selections must equal the records read now) and the
eval job (receipt, attempt, registered row, the commit of the results directory, which must be committed clean). Every
entry of the job is checked (receipt row; summary / examples.csv / eval.log hashed; the valid split's hash; A10.5's
numerics, else precision-mismatched; the registered row; the run's selected checkpoint sha256 and epoch; every row's
5.1 / 5.3 cost recomputed by accounting.inference_record) and enters no cell when a check fails. It fills: K2 per
feedback arm (none vs scrambled paired per example, Delta_scr, 6.2's 10,000-draw example bootstrap from
default_rng(0); fires iff Delta_scr < sigma_screen or the interval includes 0; the zero-latent drop and the pause
comparison beside it; pause has no K2 read), 6.3's scrambled leg, the C1b and C2 card lines with K2 decided, the 7.10
null triggers, the step sweep (reported; n = 0 against zero-latent example by example, A10.4), f_meas per C1b arm
(A6.2: the band read over the coconut checkpoint's band [0, L] read, batch 1 of record and the eval batch beside;
rule 4's line; "rule 3 logits check pending" and no saving label), rule 3's like-for-like count, 5.5 (b) at batch 1,
the plain read against the training-log selection, and exact per-example cost means in place of the ranges. Every line
carries EVAL_LABEL. Without --eval-dir the tool writes the first table exactly as before.

The third table, "with the plain-depth 355M control" (critic entries 2026-10-20T15:53Z, two; 15:26Z row 3; 14:54Z row 2;
A6.4's reissue pattern; 7.7; A13.2):

    ... read_screen.py --task gsm50k --records DIR --source-uri ... --receipts ... \
        --eval-dir results/evals/screen-eval-gsm50k-main --first-commit rev10 \
        --plain-depth-dir PD --second-commit rev11

PD mirrors the bucket's `silent/` prefix for the control alone: runs/<run>/{config.json, log.jsonl, RUN_DONE.json,
ckpt_state.json, SYNC_MANIFEST.json, runs_index_row.csv} of both plain-depth runs (A13.2: the 355M point enters A_pd
only when both are done with a selection) and receipts/<job>/<attempt>/{JOB_RECEIPT.json, memory_probe.json} of the
plain-depth job. The control is read as every run is (record checks, the registered config, the pin, A8.2), plus A13.2's
config check against the 124M no_cot config and the shared tokenizer pin; it is selected by the no_cot rule (3.3 item 1:
every epoch, argmax of valid accuracy, ties to the earliest), priced at its own dims (5.1; 5.3's no_cot row; V 50,257,
A11.2) on the hash-checked valid |q| with its logged E' total as the exact range over the unrecorded per-example E' (no
per-example read of it exists), and enters 7.7's A_pd(F) beside the 124M no_cot point (`a_pd`, `a_pd_range`: linear in
ln F between the two points; the nearest point outside, labelled "nearest size"; every corner of the arm's and the
points' FLOP ranges; C1b spanned over its nominal, measured and full fractions). Its label (PD_LABEL, the micro-batch
and the memory probe, derived from its config and the job's receipts) must appear verbatim in critic entry
2026-10-20T15:53Z and is printed on its section 1 and 2 rows and on every K3 line. The f_meas labels (15:26Z row 3), the
logits check's label (15:26Z row 1) and the batch-1 labels (14:54Z row 2) are built from the eval reads and the diag
records under results/diag/ (their commits from git) and must appear verbatim in those rows. Rule 3's logits gate
(_diag; critic entries 2026-10-20T21:03Z row 3 (3) and (4), 21:33Z (1) and (2)): the task's float64 record is required
and must meet the 1e-8 line on checks (1), (1b) and (3) (and cover the like-for-like idx), with its dtypes exactly
float64 and the strict set read back off at the start and the end; it runs on a CPU, or on the read's GPU type when the
CPU cannot hold the shapes (ProsQA-hard: diag-band-prosqa-hard-main on the A100), and every label built from it names
its device from the record ("on CPU" only for device cpu; "on A100-40GB (device cuda)" for cuda); the strict-fp32 GPU
record is optional and printed beside with whatever verdict it carries (DIAG_RECORDS per task; --diag-float64 /
--diag-gpu replace a default; ProsQA-hard's default GPU record is read when it exists, 'not run on this task' else).
ProsQA-hard's labels take 21:03Z row 3 (2) and (5)'s forms ('strict-fp32 GPU figure: not run on this task' without a
GPU record) and must appear verbatim in a critic entry logged at or after 21:03Z (THIRD_LABELS). The table cites the first
(--first-commit) and the second (--second-commit) tables, whose six files must hash as committed, and is refused unless
every json cell outside the plain-depth cells, the six K3 lines, the receipts, those labels and the two notes they
touch equals the second table's (compare_second_table).

The directional read of A6.1 rule 4 (B) on GSM-50k (the donor-answer read; critic entry 2026-10-20T17:36Z, DONOR_ROW):
an eval job's `"tool": "diag"`, `"mode": "donor_answer"` entries (run_diag.py records, write-once beside the reads)
are recognised (never as a read), their files hashed into the provenance and checked (_load_donor: the registering
row; the run's selected checkpoint; the valid split's hash; the registered set's numerics; the class pinned by ruling
1, else a refusal; the registered set's statistic recomputed from examples.csv by donor_recompute, the 6.2-form
interval on the same draws, equal to the summary's), and identified against the job's `none` / `scrambled` reads of
the same checkpoint (ruling 9: any difference holds). For each dep_loss arm (C3, C3-foreign, C2+C3) the K2 cell and
the card label then follow part (5) verbatim (a61_state): rule 6's "drop leg not independent; directional read
pending" (with the BUG / held / not-run / excluded reason) and no K2 sentence while no usable record exists; else
exactly one of "fired on the drop leg", "not fired" with the lifted "directional read present" label, or "counts as
fired (rule 5): drop without direction (fingerprint-consistent)" (a non-fired drop whose direction is absent or whose
G_chg exceeds R_dir); rule 2's C3 minus C3-foreign drop difference and the Delta_dir difference beside (diagnostic);
the C3 card lines (A3.4) with ProsQA-hard UNTESTED; the coconut reference record in the notes only. Every label must
be verbatim in DONOR_ROW (a refusal otherwise).

The fourth table, "with the hinge arms" (critic entries 2026-10-20T17:12Z parts (5), (6) and (9), 2026-10-20T17:36Z part
(5); A6.4's reissue pattern; A6.1; A2.1):

    ... read_screen.py --task gsm50k <the third table's arguments> --third-commit rev12 \
        --hinge-records H --hinge-source-uri store://<AcctA's bucket>/silent \
        --hinge-eval-dir results/evals/screen-eval-gsm50k-hinge-main \
        --registered-read <the critic's read row of the hinge eval job>

H mirrors AcctA's bucket for the three hinge runs (runs/<run>/{RUN_DONE.json, log.jsonl, config.json,
ckpt_state.json, SYNC_MANIFEST.json, runs_index_row.csv}) and the hinge job's receipts
(receipts/screen-gsm50k-accta/<attempt>/JOB_RECEIPT.json); every file is hashed with H's own source URI. The three runs
are read as the six arms are (the selection over the rows of record across resumes, A8.2; the start pin, A6.3 rule 4;
plus the receipt of the attempt RUN_DONE names, which must list the run done with RUN_DONE's checkpoints); their eval
reads are the hinge eval job's (make_jobs --screen-eval --task gsm50k --arms-set hinge: each arm's none, scrambled,
zero_latent and n in 0, 1, 2, 4, 8, then its donor_answer record, the coconut reference record last), registered by
the critic entry --registered-read (the entries must name it; the row must exist and name each checkpoint). The table
carries the six arms' lines for the three arms (section 1 and 2 rows; K2 under A6.1 with the labels of 17:36Z (5);
rule 2's C3 minus C3-foreign drop difference; the zero-latent and n = 0 reads as rule 3's diagnostics; the C3 and C2+C3
card lines with the accuracy leg of rule 4 (A) and ProsQA-hard UNTESTED; the 7.10 null triggers; the 7.2 legs; 5.5 (b);
K3 against 7.7's A_pd at the arms' FLOPs; the receipts), the directional reads (section 3.6) and the coconut reference
record in the notes only. A label whose form carries numbers (the lifted "directional read present" label, the K2 line
"K2 not independent (A6.1): Delta_scr ...") is built from the files and must appear quoted in a critic entry logged at or
after DONOR_ROW (_row_has, later), as the third table's labels do. It cites the first three tables (their nine files
must hash as committed at --first-commit, --second-commit, --third-commit) and both eval jobs (committed clean), and is
refused unless every json cell outside the hinge arms' cells equals the third table's (compare_third_table); an absent
--hinge-eval-dir is a refusal before anything is read.

The no-curriculum reissue, "with the no-curriculum control" (PLAN.md A2.2 item 1 (b); critic entry 2026-10-21T20:21Z (b)
parts (1) to (8), CLOSING_ROW; A6.4's reissue pattern): the fifth GSM-50k table, the fourth table's reissue (`rev9`),
and the fourth ProsQA-hard table, the third table's reissue (`rev18`), each over the arm `coconut_no_curriculum`:

    ... read_screen.py --task gsm50k <the fourth table's arguments> \
        --extra-arm coconut_no_curriculum --extra-records X --extra-source-uri store://<AcctC's bucket>/silent \
        --extra-eval-dir results/evals/screen-eval-gsm50k-nocur-acctc \
        --extra-registered-read <the critic's read row of that eval job> --prior-commit rev9
    ... read_screen.py --task prosqa_hard <the third table's arguments> --extra-arm coconut_no_curriculum ... \
        --prior-commit rev18

X mirrors the training project's bucket for the arm (runs/<run>/{RUN_DONE.json, log.jsonl, config.json,
ckpt_state.json, SYNC_MANIFEST.json, runs_index_row.csv} and receipts/<its train job>/<attempt>/JOB_RECEIPT.json); every
file is hashed with X's own source URI. The arm (in _task_runs; read only in this mode, so tables 1 to 4 of GSM-50k and 1
to 3 of ProsQA-hard regenerate byte for byte) is read as the coconut control is, with its own registered rules (row (b)
(1), (2)): the selection over every trained epoch (argmax of valid accuracy, strict >, ties to the earliest; every row of
record carries final_stage true), the latent count of the pinned stage max_latent_stage + 1 at every epoch (6 latents on
GSM-50k, 7 on ProsQA-hard; the log's `stage` column is the schedule's arithmetic and is not read as the stage), every
row's n_latent_eval equal to it, the GSM start pin (A6.3 rule 4) without a pool, the receipt of the attempt that
completed it. Its eval reads are its own eval job's (make_jobs --closing-eval: none timed, scrambled, zero_latent, n in
0, 1, 2, 4, 8, band [0, 12] timed at batch 1 and 32), registered by --extra-registered-read (the entries must name it;
the row must exist and name the checkpoint). The table carries the coconut control's lines for the arm (section 1 and 2
rows; the plain read; K2 with the zero-latent drop and the pause comparison beside it; the pause comparison with 7.5's
CURRICULUM-ONLY label; the step sweep; K3 against 7.7's A_pd; the 7.2 legs; the fade instrument; 5.5 (b) beside its
band reads; the 7.10 triggers; no 6.3 screen line) and row (b) (5)'s Curriculum line per task (zones of (6) on the
valid numbers of record against sigma_screen of record; CURRICULUM): its form and the zone sentence are verbatim in
CLOSING_ROW, the filled line must appear quoted in a critic entry logged at or after it (_row_has, later), and A_nocot,
A_coc and the pause value read now must equal the values row (b) (5) names. It cites the prior table (--prior-commit:
its three files must hash as committed there) besides every table the prior mode cites, and both eval jobs of the arm's
task (committed clean), and is refused unless every json cell outside the arm's run row, cost row, its lines, the
receipts line, the notes those touch, generated_utc, the citations and the vs / diff fields equals the prior table's
(compare_prior_table).
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import heapq
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(EXP, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "common"))

TABLE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
FORBIDDEN = ("inert", "mechanism", "reproduces", "k2 would fire")          # A11.3, A12.4
SCREEN_LABEL = "screen numbers (1 seed)"
BAR_LABEL = "bar LR-tuned 3 trials; arm 1 trial"                           # A3.3: every screen sentence
BAR_VALUE_LABEL = "bar: single seed, LR-tuned (3 trials)"                  # A3.3 / critic entry 2026-10-19T20:31Z
BF16_FLAGS = {"cuda.matmul.allow_tf32": True, "cudnn.allow_tf32": True, "float32_matmul_precision": "high"}  # A10.5
RECIPE_KEYS = ("arm", "uniform_prob", "lr", "weight_decay", "batch_size", "grad_accum", "seed", "resume", "precision",
               "max_new_tokens", "eval_batch_size", "max_train_examples", "max_val_examples", "c_thought",
               "epochs_per_stage", "max_latent_stage", "pad_latent_to_max", "num_epochs", "reset_optimizer",
               "save_only_improve", "optimizer_reset", "latent_band", "state_sup", "state_lambda", "state_null",
               "dep_loss", "dep_margin", "dep_beta", "dep_null")
MODEL_CONFIGS = {"openai-community/gpt2": {}, "openai-community/gpt2-medium": {"n_embd": 1024, "n_layer": 24,
                                                                                "n_head": 16}}
V_BASE, V_LATENT = 50257, 50260        # A11.2: V as the arm's head (cot / no_cot unresized, latent arms resized, D5)
# The second table ("with the eval reads", critic entry 2026-10-20T10:51Z part (6)): every line carries this label.
EVAL_LABEL = "screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial"
SWEEP = (0, 1, 2, 4, 8)                # section 8, 7.3, A3.4: the step sweep; n > the trained count is labelled OOD
N_BOOT, BOOT_SEED, BOOT_CHUNK = 10_000, 0, 1000   # 6.2: 10,000 paired example draws, numpy default_rng(0) (eval.py)
F_MEAS_LINES = {(8, 12): 0.77, (4, 12): 0.43}     # A6.2 rule 4 as printed: k = 8 and k = 4 of L = 12
RULE3_MAX_DIFF = 1                     # A6.2 rule 3: more than 1 example in 500 goes to deviations.md
CSV_INT = ("idx", "prefill_tokens", "latent_steps", "latent_positions", "emitted_tokens", "sequential_passes")
CSV_FLOAT = ("flops", "reference_flops")
CSV_BOOL = ("correct", "cot_match")
# The third table ("with the plain-depth 355M control"): the control's label and the critic entries whose labels it prints.
PD_LABEL = "plain depth 355M, 1 seed"                     # A13.2; critic entry 2026-10-20T15:53Z
ROW_PD, ROW_FMEAS, ROW_BATCH1 = "2026-10-20T15:53Z", "2026-10-20T15:26Z", "2026-10-20T14:54Z"
# A6.2 rule 3's logits check per task (critic entries 2026-10-20T15:26Z rows 1 and 4, 2026-10-20T21:03Z row 3 (3) and
# (4), 2026-10-20T21:33Z (2)): the float64 record is required (the line: 1e-8 at float64; GSM-50k's on CPU,
# ProsQA-hard's on the A100, diag-band-prosqa-hard-main entry (i)); the strict-fp32 GPU record is optional (printed
# beside, no line; None or, for a task in DIAG_GPU_IF_PRESENT, an absent default file: "not run on this task"). Each
# record is keyed to the task's coconut checkpoint and valid split.
DIAG_RECORDS = {
    "gsm50k": {
        "gpu": "results/diag/diag-band-gsm50k-main/gsm50k_coconut_s0_valid_band_logits/summary.json",
        "float64": "results/diag/diag-band-gsm50k-cpu/gsm50k_coconut_s0_valid_band_logits_cpu_float64/summary.json"},
    "prosqa_hard": {
        "gpu": "results/diag/diag-band-prosqa-hard-main/prosqa_hard_coconut_s0_valid_band_logits/summary.json",
        "float64": "results/diag/diag-band-prosqa-hard-main/prosqa_hard_coconut_s0_valid_band_logits_cuda_float64/"
                   "summary.json"}}
DIAG_GPU_IF_PRESENT = ("prosqa_hard",)    # the default GPU record is optional there: read when it exists (21:33Z (2))
DIAG64_DTYPES = {"weights": ["float64"], "activations": ["float64"]}     # the DtypeWatch's sets, exactly (21:33Z (2))
DIAG_TOL_GPU, DIAG_TOL64 = 1e-4, 1e-8     # the strict-fp32 figure's printed line; the float64 line (15:26Z row 4)
# The third table's labels per task: the label form, the critic entry(s) the built text must appear in verbatim (with
# `later`, that row or any later critic entry: the form was fixed before the numbers existed, so the critic writes the
# numbers in a later row; lead rows never count) and how the lines cite them.
THIRD_LABELS = {
    "gsm50k": {"form": "15:26Z", "rows": {"fmeas": ROW_FMEAS, "batch1": ROW_BATCH1}, "later": False,
               "fmeas": "critic entry 2026-10-20T15:26Z row 3", "logits": "critic entry 2026-10-20T15:26Z row 1",
               "batch1": "critic entry 2026-10-20T14:54Z row 2", "deviation": "12:05Z",
               "logits_check": "rule 3's strict-fp32 first-token logits check", "logits_status": "explained",
               "md_rows": f"{ROW_FMEAS} row 3 and {ROW_BATCH1} row 2",
               "fmeas_note": "The labels are critic entry 2026-10-20T15:26Z row 3's, verbatim (rule 3's logits check "
                             "explained by the float64 record, row 1).",
               "pending_both_task": "the ProsQA-hard side of every both-task clause; "},
    "prosqa_hard": {"form": "21:03Z", "rows": {"fmeas": "2026-10-20T21:03Z", "batch1": "2026-10-20T21:03Z"},
                    "later": True, "fmeas": "critic entry 2026-10-20T21:03Z row 3 (5)",
                    "logits": "critic entry 2026-10-20T21:03Z row 3 (3) and (5)",
                    "batch1": "critic entry 2026-10-20T21:03Z rows 3 (2) and 4 (4)", "deviation": "21:10Z",
                    "logits_check": "rule 3's first-token logits check (float64 on {dev64}, the line; strict-fp32 "
                                    "GPU figure beside)", "logits_status": "met at float64 on {dev64}",
                    "md_rows": "2026-10-20T21:03Z rows 3 and 4",
                    "fmeas_note": "The labels take critic entry 2026-10-20T21:03Z row 3 (5)'s form and appear verbatim in "
                                  "a critic entry (rule 3's logits check met by this task's own float64 record; the "
                                  "strict-fp32 GPU figure printed beside when measured, else 'not run on this task').",
                    "pending_both_task": ""}}
TOKENIZER_FILES = ("merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json")
# A6.1 rule 4 (B)'s directional read on GSM-50k, the donor-answer read (design note
# governance/designs/a6_1_directional_read.md, rev230; registered by critic entry 2026-10-20T17:36Z: rulings 1 to 11, the
# lines of part (5)): run_diag.py --mode donor_answer records, one per hinge arm (and the coconut reference), entered
# from the eval job's directory as "tool": "diag" / "mode": "donor_answer" entries.
DONOR_ROW = "2026-10-20T17:36Z"
DONOR_PIN = {"gsm50k": {"n": 500, "n_class": 295, "ambiguous_idx": [15, 124, 125, 175, 470, 477], "n_scored": 289}}
DONOR_N_BOOT, DONOR_BOOT_SEED, DONOR_THIN = 10_000, 0, 10
A61 = {   # the labels of record, verbatim in critic entry 2026-10-20T17:36Z part (5); the table refuses without them
    "pending": "drop leg not independent; directional read pending",
    "present": "drop leg not independent (A6.1 rule 1); directional read present",
    "fingerprint": "drop without direction (fingerprint-consistent)",
    "k2_rule5": "K2 not independent (A6.1); counts as fired (rule 5): drop without direction (fingerprint-consistent)",
    "k2_drop": "K2 not independent; fired on the drop leg (7.5; A3.4)",
    "k2_form": "K2 not independent (A6.1): Delta_scr",
    "rule2": "dependence on own latents (content or fingerprint)",
    "reference": "reference arm (no hinge); diagnostic; decides nothing",
    "held": "(held: identification", "bug": "(BUG: "}
DONOR_CSV_BOOL = ("donor_is_self", "fed_equals_donor", "fed_equals_collected", "in_class", "ambiguous", "scored",
                  "correct_own", "correct_scr", "d_own", "d_scr", "changed", "g_chg", "g_own", "g_scr",
                  "scr_eq_any_donor_step", "scr_eq_any_own_step", "donor_gold_in_own_steps")
DONOR_CSV_INT = ("idx", "donor", "own_n_steps", "donor_n_steps", "donor_last_sup_step", "first_argmax_own",
                 "first_argmax_scr", "stream_div_own_scr", "emitted_own", "emitted_scr")
DONOR_CSV_FLOAT = ("fed_min_l2", "fed_max_cos", "first_maxabs_own_scr")
PD_FREE_KEYS = ("_note", "model_id", "name", "save_dir", "batch_size", "grad_accum")   # A13.2 (and its (b) fallback)
# The fourth table ("with the hinge arms"; critic entries 2026-10-20T17:12Z parts (5), (6) and (9), 2026-10-20T17:36Z part
# (5)): the third table's reissue over the three hinge arms of the one hinge job A2.1 admits (GSM-50k only; the
# ProsQA-hard hinge job stays unlaunched, 17:12Z (8)). Their records live under their own root (AcctA's bucket,
# --hinge-records, with its own --hinge-source-uri), their eval reads in their own job (--hinge-eval-dir, registered by
# the critic's read row --registered-read); the coconut reference record of that job enters the notes only.
HINGE = {"gsm50k": {"job": "screen-gsm50k-accta", "eval_job": "screen-eval-gsm50k-hinge-main",
                    "keys": ("c3", "c3_foreign", "c2c3"), "reference": "coconut"}}
ROW_HINGE = "2026-10-20T17:12Z"           # the registered form of the hinge arms' screen read (parts (5), (6), (9))
# 17:36Z (2)'s label block on every number of the directional read (its <gpu_type> slot is the record's read GPU; the
# hinge eval's read row 2026-10-21T11:07Z fills it as "read on A100-80GB, trained on A100-80GB"), and 11:07Z (3)'s label
# of the coconut reference record's identification across GPU types (k: the examples that differ).
DONOR_LABEL_FORM = ("screen number: 1 seed, valid {n} (scored {s}); bar LR-tuned 3 trials; arm 1 trial; diagnostic entry, "
                    "no 4.5 read spent; bf16 autocast over TF32 on (strict fp32 beside, diagnostic); read on <gpu_type>, "
                    "trained on {t}; no test read; ProsQA-hard UNTESTED (A2.1)")
XGPU_FORM = "identification across GPU types: k of {n} ({a} vs {b})"
CRITIC_TS_RE = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\dZ$")
# The closing plan's no-curriculum control (PLAN.md A2.2 item 1 (b); critic entry 2026-10-21T20:21Z (b) parts (1) to (8)):
# the arm read only by the reissue that carries it (--extra-arm), its train and eval jobs per task (row (b) (3), (4)),
# the table each task's reissue cites as its prior (the GSM-50k fourth table rev9, the ProsQA-hard third rev18),
# and row (b) (5)'s values of record (A_nocot, A_coc, pause: the tables of record's selections).
CLOSING_ROW = "2026-10-21T20:21Z"
CLOSING = {"coconut_no_curriculum": {
    "role": "control: coconut without the curriculum (3.1)",
    "jobs": {"gsm50k": "screen-gsm50k-no-curriculum-acctc", "prosqa_hard": "screen-prosqa-hard-no-curriculum-acctb"},
    "eval_jobs": {"gsm50k": "screen-eval-gsm50k-nocur-acctc", "prosqa_hard": "screen-eval-prosqa-hard-nocur-acctb"},
    "prior": {"gsm50k": ("gsm50k_screen_with_hinge", "fourth"), "prosqa_hard": ("prosqa_hard_screen_with_plain_depth",
                                                                             "third")},
    "ordinal": {"gsm50k": "fifth", "prosqa_hard": "fourth"},
    "values_of_record": {"gsm50k": ("6.2", "11.0", "9.6"), "prosqa_hard": ("57.7", "73.3", "74.7")}}}
CLOSING_NAME = "{task}_screen_with_no_curriculum"
# Row (b) (5)'s line and (6)'s zones and sentences, verbatim (<task> is the task's title; the BOTH sentence's gap and its
# ratio to sigma_screen are filled from the values of record).
CURRICULUM_FORM = ("Curriculum (3.1's control; A2.2 (b)), <task>: no-curriculum A_nc vs no_cot A_nocot (A_nc \u2212 A_nocot) "
                   "and vs full-depth coconut A_coc (A_nc \u2212 A_coc), sigma_screen 3.60: <zone>; <sentence>")
CURRICULUM_SENTENCES = {
    "NEAR NO_COT": "On <task> at this read the same latent layout trained without the staged curriculum scores with "
                   "no_cot: the gain of the latent arms over no_cot does not appear without the curriculum; the "
                   "curriculum is necessary for it at this read (1 seed)",
    "NEAR COCONUT": "On <task> at this read the gain over no_cot appears without the curriculum: the curriculum is not "
                    "necessary for it, and the staged chain supervision is not the candidate the controls point to; a "
                    "pause arm without the curriculum was not run, so the token-budget reading is untested",
    "BETWEEN": "between no_cot and coconut at this read (1 seed): unresolved",
    "BOTH": "within sigma_screen of both no_cot and coconut; the coconut-vs-no_cot gap on this task ({gap}) is itself "
            "{ratio} sigma_screen: unresolved at 1 seed"}
# Critic entry 2026-10-22T03:18Z (3) (ii): BETWEEN is signed too (BETWEEN iff A_nc - A_nocot > sigma and A_coc - A_nc >
# sigma), so exactly one zone holds for every A_nc; below A_nocot - sigma the zone is NEAR NO_COT and above A_coc + sigma
# NEAR COCONUT, each sentence with its suffix, verbatim (<d>: the distance in points, on counts).
CURRICULUM_SIGNED_ROW = "2026-10-22T03:18Z"
CURRICULUM_OUTSIDE = {
    "NEAR NO_COT": " (below no_cot by <d> points, more than sigma_screen: without the curriculum the latent layout costs "
                   "accuracy against no_cot at this read)",
    "NEAR COCONUT": " (above coconut by <d> points, more than sigma_screen)"}
CLOSING_LINE_KINDS = ("cost", "k2", "k3", "pause", "fade", "eval_none", "sweep", "wallclock", "curriculum", "untested")
CLOSING_NOTES = ("The no-curriculum control (critic entry",)


def _run(key, run, role, config, card=None):
    return {"key": key, "run": run, "role": role, "config": config, "card": card}


def _task_runs(t: str) -> list[dict]:
    c = f"configs/screen/{t}_"
    return [
        _run("cot_lr3e-5", f"{t}_cot_lr3e-5_s0", "bar trial: cot, lr 3e-5", c + "cot_lr3e-5.json"),
        _run("cot_lr1e-4", f"{t}_cot_lr1e-4_s0", "bar trial: cot, lr 1e-4", c + "cot_lr1e-4.json"),
        _run("cot_lr3e-4", f"{t}_cot_lr3e-4_s0", "bar trial: cot, lr 3e-4", c + "cot_lr3e-4.json"),
        _run("no_cot", f"{t}_no_cot_s0", "floor; the 124M plain-depth point (7.7)", c + "no_cot.json"),
        _run("plain_depth_355m", f"{t}_plain_depth_355m_s0", "plain depth: 355M no_cot (7.7, A13.2)",
             c + "plain_depth_355m.json"),
        _run("pause", f"{t}_pause_s0", "control: pause-same-curriculum", c + "pause.json"),
        _run("coconut", f"{t}_coconut_s0", "control: full-depth coconut (C1b null; C2 / C3 base)", c + "coconut.json"),
        _run("c1b_k8", f"{t}_c1b_k8_s0", "card C1b, k = 8 (band [2, 10))", c + "c1b_k8.json", "C1b"),
        _run("c1b_k4", f"{t}_c1b_k4_s0", "card C1b, k = 4 (band [4, 8))", c + "c1b_k4.json", "C1b"),
        _run("c2", f"{t}_c2_s0", "card C2", c + "c2.json", "C2"),
        _run("c2_shuffled", f"{t}_c2_shuffled_s0", "null: C2-shuffled", c + "c2_shuffled.json"),
        _run("c3", f"{t}_c3_s0", "card C3", c + "c3.json", "C3"),
        _run("c3_foreign", f"{t}_c3_foreign_s0", "null: C3-foreign", c + "c3_foreign.json"),
        _run("c2c3", f"{t}_c2c3_s0", "card C2+C3", c + "c2c3.json", "C2+C3"),
        # row (b) (5): read only by the reissue that carries it (Screen(extra_arm=...)); every earlier table leaves it out
        _run("coconut_no_curriculum", f"{t}_coconut_no_curriculum_s0", CLOSING["coconut_no_curriculum"]["role"],
             c + "coconut_no_curriculum.json"),
    ]


TASKS = {
    "gsm50k": {
        "title": "GSM-50k",
        "data_sentence": "GSM-50k: training on the first 50,000 records of a seed-0 permutation of the GSM8k-Aug train "
                         "set (A3.4.5); valid = the full GSM8k-Aug valid set",
        "n_valid": 500, "max_new_tokens": 64, "epochs": 25,
        "latent_pool": (12, 24),          # epoch_index, inclusive (A3.4.5; A2 item 3 / D16)
        "too_good": {"over_bar": 10.0},   # A3.4 / 7.10: more than 10 points over the bar at 124M on GSM-50k
        "start_pin": {"source_run": "gsm50k_cot_lr1e-4_s0",
                      "sha256": "11500b741cca007a561a6e04906354caeb36a2e7ee8238a70dc9f3fa3e626c2f",
                      "epoch": 25, "critic_rows": ["2026-10-19T20:31Z", "2026-10-19T20:32Z"]},
        "plain_depth_all": ["gsm50k_plain_depth_355m_s0", "prosqa_hard_plain_depth_355m_s0"],   # A13.2: both runs
        "c2_sentences": {
            "c2": "A5 item 5: the last supervised target contains the answer in 0.770 of the GSM-50k records at 3 "
                  "latent steps; printed beside every C2 number",
            "c2_shuffled": "A6.7 item 4: on GSM-50k the C2-shuffled donor shares the answer 430 of 50,000 = 0.86% "
                           "(chance)"},
        "notes": ["A2 item 4 (c): the GSM valid set shares a 13-gram with the full GSM8k-Aug train on 7 of 500 items "
                  "(the 50k subset at most that); a known bias, valid slightly optimistic, equally for every arm."],
    },
    "prosqa_hard": {
        "title": "ProsQA-hard",
        "data_sentence": "ProsQA-hard: depth 5-7, 1.7x edges, 300-word questions (released: depth 3-6, 36 edges, "
                         "180 words)",
        "n_valid": 300, "max_new_tokens": 128, "epochs": 30,
        "latent_pool": (21, 29),
        "too_good": {"above": 99.0},      # A3.4 / 7.10: any arm above 99.0% on ProsQA-hard
        "start_pin": None,                # latent arms start from the pinned base model (no load_model_path)
        "plain_depth_all": ["gsm50k_plain_depth_355m_s0", "prosqa_hard_plain_depth_355m_s0"],
        "c2_sentences": {
            "c2": "A5 item 5: the last supervised target contains the answer in 1.00 of the ProsQA-hard records; "
                  "printed beside every C2 number",
            "c2_shuffled": "A6.7 item 4: C2-shuffled's donor frontier contains the record's target name 34.8% of the "
                           "time, symmetric between target and negative target, so the null has no answer signal"},
        "notes": ["A6.7 item 2: ProsQA-hard 13-gram overlap vs its own train 37.0% valid, 36.0% test (template "
                  "sentences over a tiny vocabulary); duplicate questions 0 and duplicate graphs 0 across splits; "
                  "never read as leakage."],
    },
}
for _t in TASKS:
    TASKS[_t]["runs"] = _task_runs(_t)


class RecordError(Exception):
    pass


# ----------------------------------------------------------------------------------------------------------- inputs
class Inputs:
    """Every file the table reads, with its sha256 (the provenance file is this list)."""

    def __init__(self, records_root: str | None, source_uri: str | None, extra_roots: dict | None = None):
        self.records_root = os.path.abspath(records_root) if records_root else None
        self.source_uri = source_uri.rstrip("/") if source_uri else None
        self.extra_roots, self.extra_uris = {}, {}       # more bucket mirrors; a (path, uri) value names its own bucket
        for k, v in (extra_roots or {}).items():
            if isinstance(v, tuple):
                v, uri = v
                self.extra_uris[k] = uri.rstrip("/") if uri else None
            self.extra_roots[k] = os.path.abspath(v)
        self.files: dict[str, dict] = {}

    def _entry(self, path: str, role: str) -> dict:
        ap = os.path.abspath(path)
        if ap in self.files:
            return self.files[ap]
        with open(ap, "rb") as f:
            data = f.read()
        e = {"role": role, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        if self.records_root and ap.startswith(self.records_root + os.sep):
            rel = os.path.relpath(ap, self.records_root)
            e.update(root="records", path=rel)
            if self.source_uri:
                e["source_uri"] = f"{self.source_uri}/{rel}"
        elif any(ap.startswith(r + os.sep) for r in self.extra_roots.values()):
            name, root = next((k, r) for k, r in self.extra_roots.items() if ap.startswith(r + os.sep))
            rel = os.path.relpath(ap, root)
            e.update(root=name, path=rel)
            uri = self.extra_uris[name] if name in self.extra_uris else self.source_uri
            if uri:
                e["source_uri"] = f"{uri}/{rel}"
        elif ap.startswith(REPO + os.sep):
            e.update(root="repo", path=os.path.relpath(ap, REPO))
        else:
            e.update(root="other", path=ap)
        e["_data"] = data
        self.files[ap] = e
        return e

    def read_bytes(self, path: str, role: str) -> bytes:
        return self._entry(path, role)["_data"]

    def read_text(self, path: str, role: str) -> str:
        return self.read_bytes(path, role).decode("utf-8")

    def read_json(self, path: str, role: str):
        return json.loads(self.read_text(path, role))

    def sha256(self, path: str, role: str) -> str:
        return self._entry(path, role)["sha256"]

    def listing(self) -> list[dict]:
        out = [{k: v for k, v in e.items() if k != "_data"} for e in self.files.values()]
        return sorted(out, key=lambda e: (e["root"], e["path"]))


# ------------------------------------------------------------------------------------------------ pure helpers
def count_of(frac: float, n: int, what: str) -> int:
    """A logged fraction or mean over n examples as its integer total (accuracy k of n; emitted tokens M)."""
    k = round(frac * n)
    if abs(frac * n - k) > 1e-6:
        raise RecordError(f"{what} {frac!r} is not a count over {n} examples")
    return int(k)


def pts(k: int, n: int) -> Fraction:
    return Fraction(100 * k, n)


def f1(x) -> str:
    return f"{float(x):.1f}"


def f2(x) -> str:
    return f"{float(x):.2f}"


def rows_of_record(rows: list[dict]) -> tuple[list[dict], int]:
    """A8.2: for each epoch the row with the largest `resumes`; returns (rows in epoch order, superseded count)."""
    by = {}
    for r in rows:
        e, res = int(r["epoch"]), int(r.get("resumes", 0))
        if e in by and by[e][0] == res:
            raise RecordError(f"epoch {e}: two log rows with resumes {res} (A8.2 cannot pick the row of record)")
        if e not in by or res > by[e][0]:
            by[e] = (res, r)
    return [by[e][1] for e in sorted(by)], len(rows) - len(by)


def select(rows: list[dict], n: int, pool: tuple[int, int] | None) -> dict | None:
    """3.3 item 1 / A2 item 3: the highest valid accuracy among the rows of record whose epoch_index is in the pool
    (None = every epoch), ties to the earliest epoch (strict >)."""
    best, best_k = None, -1
    for r in rows:
        if pool is not None and not pool[0] <= int(r["epoch_index"]) <= pool[1]:
            continue
        k = count_of(r["val_acc"], n, "val_acc")
        if k > best_k:
            best, best_k = r, k
    return best


def gen_latents(cfg: dict, stage: int) -> int:
    """Latent (or pause) positions of a generation prompt at `stage` (latent.data.build_stage_dataset with
    for_generation=True and pad_latent_to_max): min(max_latent_stage, stage) x c_thought; 0 for cot / no_cot."""
    if cfg["arm"] in ("cot", "no_cot"):
        return 0
    if not cfg.get("pad_latent_to_max", True):
        raise RecordError("pad_latent_to_max False: per-example latent counts vary; not covered")
    return min(int(cfg["max_latent_stage"]), int(stage)) * int(cfg["c_thought"])


def sum_bounds(costs: list[list[float]], total: int) -> tuple[float, float]:
    """Exact minimum and maximum of sum_i costs[i][E_i] over integers 0 <= E_i <= U (U = len(row) - 1) with
    sum_i E_i = total, for rows convex in E (asserted). Minimum: greedy marginal allocation (exact for separable convex
    costs under one sum constraint). Maximum: a maximiser has at most one E_i strictly inside (0, U) (moving units
    between two interior coordinates along a convex line ends at a bound), so it is k = total // U rows at U, one row at
    total % U and the rest at 0, the k rows the largest gains c_i(U) - c_i(0) given the remainder row."""
    n = len(costs)
    if n == 0:
        raise ValueError("no examples")
    U = len(costs[0]) - 1
    if not 0 <= total <= n * U:
        raise ValueError(f"total {total} outside [0, {n * U}]")
    for row in costs:
        if len(row) != U + 1:
            raise ValueError("ragged cost table")
        for e in range(1, U):
            if row[e + 1] - 2 * row[e] + row[e - 1] < -1e-9 * max(1.0, abs(row[e])):
                raise ValueError("cost not convex in E'")
    base = sum(row[0] for row in costs)
    heap = [(row[1] - row[0], i, 0) for i, row in enumerate(costs)] if U else []
    heapq.heapify(heap)
    lo = base
    for _ in range(total):
        d, i, e = heapq.heappop(heap)
        lo += d
        if e + 1 < U:
            heapq.heappush(heap, (costs[i][e + 2] - costs[i][e + 1], i, e + 1))
    k, r = divmod(total, U) if U else (0, 0)
    gain = sorted(((row[U] - row[0], i) for i, row in enumerate(costs)), reverse=True)
    top = sum(g for g, _ in gain[:k])
    if r == 0:
        hi = base + top
    else:
        rank = {i: j for j, (_, i) in enumerate(gain)}
        nxt = gain[k][0] if k < n else 0.0
        hi = base + max((costs[i][r] - costs[i][0]) + (top - gain[rank[i]][0] + nxt if rank[i] < k else top)
                        for i in range(n))
    return lo, hi


def interval_ratio(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] / b[1], a[1] / b[0]


def decide_le(a: tuple[float, float], b: tuple[float, float]) -> str:
    """'met' if every value in a is <= every value in b; 'not met' if every value in a exceeds b; else undetermined."""
    if a[1] <= b[0]:
        return "met"
    if a[0] > b[1]:
        return "not met"
    return "undetermined"


def a_pd(F: float, points: list[tuple[float, float]]) -> tuple[float, bool]:
    """7.7: no_cot accuracy at inference FLOPs F, linear in log F between the sizes' points (F, acc), sorted by F; the
    nearest point outside the range (returns (acc, nearest_size))."""
    pts_ = sorted(points)
    if len(pts_) == 1 or F <= pts_[0][0] or F >= pts_[-1][0]:
        near = pts_[0] if (len(pts_) == 1 or F <= pts_[0][0]) else pts_[-1]
        return near[1], True
    for (f0, a0), (f1_, a1) in zip(pts_, pts_[1:]):
        if f0 <= F <= f1_:
            w = (math.log(F) - math.log(f0)) / (math.log(f1_) - math.log(f0))
            return a0 + w * (a1 - a0), False
    raise AssertionError("unreachable")


def a_pd_range(F: tuple[float, float], points: list[tuple[tuple[float, float], float]]) -> tuple[float, float, bool]:
    """A_pd over every corner of the FLOP ranges (the arm's and each size point's); returns (min, max, nearest)."""
    vals, near = [], False
    corners = [[]]
    for (lo, hi), acc in points:
        corners = [c + [(x, acc)] for c in corners for x in (lo, hi)]
    for c in corners:
        for f in F:
            v, nr = a_pd(f, c)
            vals.append(v)
            near = near or nr
    return min(vals), max(vals), near


# ------------------------------------------------------------------------------- pure helpers for the eval reads
def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = BOOT_SEED, chunk: int = BOOT_CHUNK) -> dict:
    """6.2 on one checkpoint's two reads, paired per example (a, b: 0/1 correctness in the same example order).
    Delta = 100 mean(a - b) points, exact (a Fraction). Var_ex = the variance (ddof 1) of Delta over n_boot paired
    example-bootstrap draws: numpy default_rng(seed), examples resampled with replacement, drawn exactly as
    latent.eval._boot_ci draws them for `summarize` (chunks of `chunk` rows of rng.integers(0, n, size=(k, n))), so
    the same draw is applied to every read. The interval is 6.2's Delta +- 1.96 sqrt(Var_ex); 6.2's seed term
    sigma_task^2 (1/n_A + 1/n_B) has no place here (both reads score one checkpoint). The percentile interval of the
    same draws (summarize's form) is returned beside it."""
    import numpy as np
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) < 2:
        raise ValueError("paired reads of equal length >= 2 needed")
    if not (np.isin(a, (0.0, 1.0)).all() and np.isin(b, (0.0, 1.0)).all()):
        raise ValueError("correctness must be 0/1 per example")
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s0)
        means[s0:s0 + k] = d[rng.integers(0, n, size=(k, n))].mean(1)
    delta = Fraction(100 * (int(a.sum()) - int(b.sum())), n)
    se = 100.0 * float(np.sqrt(means.var(ddof=1)))
    plo, phi = (100.0 * float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {"delta": delta, "n": n, "k_a": int(a.sum()), "k_b": int(b.sum()), "only_a": int((d > 0).sum()),
            "only_b": int((d < 0).sum()), "se": se, "lo": float(delta) - 1.96 * se, "hi": float(delta) + 1.96 * se,
            "pct_lo": plo, "pct_hi": phi, "n_boot": n_boot, "seed": seed}


def k2_outcome(delta, lo: float, hi: float, sigma: float) -> tuple[str, list[str]]:
    """7.5 / A3.4 at the screen: K2 fires iff Delta_scr < sigma_screen or its 95% interval includes 0 (inclusive)."""
    why = []
    if Fraction(delta) < Fraction(sigma):
        why.append("Delta_scr < sigma_screen")
    if lo <= 0.0 <= hi:
        why.append("the interval includes 0")
    return ("fired" if why else "not fired"), why


def f_meas_reading(f: float, band, n_L: int) -> dict:
    """A6.2 rule 4 on a measured f_meas for band [a, b) of n_L layers. The registered numbers (0.77 for k = 8, 0.43 for
    k = 4 of 12) are the line of record; the formula (b - a) / L + 0.10 is evaluated beside it because the printed 0.43
    sits below the formula's 0.4333 (a value between the two reads differently). The saving shows iff f <= line; else
    "nominal" with layer_fraction = min(1, f) in K3 and 7.2; f > 1 forbids any saving label. No saving label is written
    while A6.2 rule 3's strict-fp32 logits check is pending (critic entry 2026-10-20T10:51Z part (3))."""
    a, b = (int(x) for x in band)
    nominal = Fraction(b - a, n_L)
    formula = float(nominal) + 0.10
    line = F_MEAS_LINES.get((b - a, n_L), formula)
    met = f <= line
    return {"f_meas": f, "band": [a, b], "nominal_fraction": float(nominal), "line": line, "line_formula": formula,
            "threshold_met": met, "formula_met": f <= formula, "readings_disagree": met != (f <= formula),
            "above_one": f > 1.0, "outcome": "threshold met" if met else "nominal",
            "layer_fraction_k3_7_2": float(nominal) if met else min(1.0, f), "measured_fraction": min(1.0, f),
            "saving_label": None, "rule3": "rule 3 logits check pending"}


def null_trigger_hits(null_pts, bar_pts, arm_pts: list, sigma: float) -> list[str]:
    """7.10 (A3.4): a null above the tuned bar by 2 sigma, or above the arm it is a null of by 1 sigma ("above":
    strictly). arm_pts: every value of the arm read (the selection of record and the plain eval read)."""
    S = Fraction(sigma)
    hits = []
    if Fraction(null_pts) > Fraction(bar_pts) + 2 * S:
        hits.append("above the bar + 2 sigma_screen")
    if any(Fraction(null_pts) > Fraction(a) + S for a in arm_pts):
        hits.append("above the arm + sigma_screen")
    return hits


def parse_examples_csv(text: str) -> list[dict]:
    """run_eval.py's examples.csv (one row per example) with typed columns; refuses an unknown bool spelling."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        r = {}
        for k, v in row.items():
            if k in CSV_INT:
                r[k] = int(v)
            elif k in CSV_FLOAT:
                r[k] = float(v)
            elif k in CSV_BOOL:
                if v not in ("True", "False"):
                    raise RecordError(f"examples.csv {k} {v!r} is not True/False")
                r[k] = v == "True"
            else:
                r[k] = v
        out.append(r)
    return out


def rows_differ(a: list[dict], b: list[dict], fields=None) -> list[tuple[int, str]]:
    """(idx, field) for every per-example field that differs between two reads in the same example order."""
    if [r["idx"] for r in a] != [r["idx"] for r in b]:
        raise RecordError("the two reads do not list the same examples in the same order")
    if not a:
        return []
    fs = list(fields) if fields else [k for k in a[0] if k in b[0]]
    return [(x["idx"], f) for x, y in zip(a, b) for f in fs if x[f] != y[f]]


def classify_eval_entry(e: dict, n_L: int) -> str:
    """The kind of one screen-eval entry from its spec fields (10:51Z part (2), (3)): none, scrambled, zero_latent,
    n<k> (the sweep), timed_bs<B> (a plain read at a batch other than the config's, timed), band_bs<B> (latent_band
    [0, L] on the coconut checkpoint, timed; B = 'cfg' for the config's batch); donor_answer (a run_diag.py
    --mode donor_answer diag entry, A6.1 rule 4 (B), critic entry 2026-10-20T17:36Z); 'unregistered' otherwise (any
    other diag entry: a diagnostic is never one of the reads)."""
    if e.get("tool", "eval") != "eval":
        return "donor_answer" if (e.get("tool"), e.get("mode")) == ("diag", "donor_answer") else "unregistered"
    iv = e.get("intervention", "none")
    n = e.get("n_latent")
    st = e.get("set") or {}
    bs = e.get("batch_size")
    ts = bool(e.get("time_steps"))
    btag = "cfg" if bs is None else str(int(bs))
    if set(st) - {"latent_band"}:
        return "unregistered"
    if "latent_band" in st:
        return f"band_bs{btag}" if list(st["latent_band"]) == [0, n_L] and iv == "none" and n is None and ts \
            else "unregistered"
    if bs is not None:
        return f"timed_bs{btag}" if iv == "none" and n is None and ts else "unregistered"
    if iv in ("scrambled", "zero_latent") and n is None:
        return iv
    if iv == "none":
        return "none" if n is None else f"n{int(n)}"
    return "unregistered"


# ---------------------------------------------------- pure helpers for A6.1 rule 4 (B)'s donor-answer read (GSM-50k)
def parse_donor_csv(text: str) -> list[dict]:
    """run_diag.py --mode donor_answer's examples.csv (a row per numerics set and example) with typed columns; refuses
    an unknown column (the file holds ids, booleans and numbers only) and a bool other than True / False."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        r = {}
        for k, v in row.items():
            if k == "numerics":
                r[k] = v
            elif k in DONOR_CSV_BOOL:
                if v not in ("True", "False"):
                    raise RecordError(f"donor_answer examples.csv {k} {v!r} is not True/False")
                r[k] = v == "True"
            elif k in DONOR_CSV_INT:
                r[k] = None if v == "" else int(v)
            elif k in DONOR_CSV_FLOAT:
                r[k] = None if v == "" else float(v)
            else:
                raise RecordError(f"donor_answer examples.csv has an unregistered column {k!r}")
        out.append(r)
    return out


def values_bootstrap(d, n_boot: int = N_BOOT, seed: int = BOOT_SEED, chunk: int = BOOT_CHUNK) -> dict:
    """6.2's form on per-example values (any reals, in example order), drawn exactly as paired_bootstrap draws (numpy
    default_rng(seed), chunks of `chunk` rows of rng.integers(0, n, size=(k, n))): the mean, se = sqrt(Var_ex) (ddof
    1 over the draw means), mean +- 1.96 se and the percentile interval."""
    import numpy as np
    d = np.asarray(d, dtype=float)
    n = len(d)
    if d.ndim != 1 or n < 2:
        raise ValueError("per-example values of length >= 2 needed")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for s0 in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s0)
        means[s0:s0 + k] = d[rng.integers(0, n, size=(k, n))].mean(1)
    se = float(np.sqrt(means.var(ddof=1)))
    m = float(d.mean())
    plo, phi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {"mean": m, "n": n, "se": se, "lo": m - 1.96 * se, "hi": m + 1.96 * se, "pct_lo": plo, "pct_hi": phi}


def donor_recompute(rows: list[dict], n_boot: int = DONOR_N_BOOT, seed: int = DONOR_BOOT_SEED) -> dict:
    """The registered set's statistic recomputed from the record's examples.csv rows by this tool (no diag code):
    the class, the ambiguous pairs, the scored set S (idx order), the counts, R_dir, R_own, Delta_dir (exact) and its
    6.2-form interval on the same draws, G_chg (17:36Z ruling 3) and the three garbage comparisons."""
    rows = sorted(rows, key=lambda r: r["idx"])
    C = [r for r in rows if r["in_class"]]
    S = [r for r in rows if r["scored"]]
    m = len(S)
    k = {c: sum(r[c] for r in S) for c in ("d_scr", "d_own", "changed", "g_chg", "g_scr", "g_own", "correct_own",
                                           "correct_scr")}
    out = {"n": len(rows), "n_class": len(C), "ambiguous_idx": [r["idx"] for r in C if r["ambiguous"]],
           "n_scored": m, "k_dir": k["d_scr"], "k_own": k["d_own"], "k_changed": k["changed"], "k_g_chg": k["g_chg"],
           "k_g_scr": k["g_scr"], "k_g_own": k["g_own"], "k_net": k["d_scr"] - k["d_own"],
           "k_acc_own": sum(r["correct_own"] for r in rows), "k_acc_scr": sum(r["correct_scr"] for r in rows),
           "garbage_exceeds": k["g_chg"] > k["d_scr"], "garbage_literal_exceeds": k["g_scr"] > k["d_scr"],
           "garbage_excess_exceeds": k["g_scr"] - k["g_own"] > k["d_scr"] - k["d_own"]}
    if m < 2:
        return {**out, "delta_dir": None, "r_dir": None, "r_own": None, "g_chg": None, "lo": None, "hi": None}
    b = values_bootstrap([float(r["d_scr"]) - float(r["d_own"]) for r in S], n_boot, seed)
    c = values_bootstrap([float(r["g_chg"]) - float(r["d_scr"]) for r in S], n_boot, seed)
    return {**out, "delta_dir": Fraction(k["d_scr"] - k["d_own"], m), "r_dir": Fraction(k["d_scr"], m),
            "r_own": Fraction(k["d_own"], m), "g_chg": Fraction(k["g_chg"], m), "lo": b["lo"], "hi": b["hi"],
            "pct_lo": b["pct_lo"], "pct_hi": b["pct_hi"], "se": b["se"], "lo_chg": c["lo"], "hi_chg": c["hi"],
            "present": k["d_scr"] > k["d_own"] and b["lo"] > 0}


def donor_identification(rec_rows: list[dict], none_rows: list[dict], scr_rows: list[dict]) -> dict:
    """17:36Z ruling 9 at table time: the record's registered-set correct_own / correct_scr against the committed
    `none` / `scrambled` reads' correct columns, example by example; any difference holds the record."""
    by = {r["idx"]: r for r in rec_rows}
    for name, rows in (("none", none_rows), ("scrambled", scr_rows)):
        if sorted(x["idx"] for x in rows) != sorted(by):
            raise RecordError(f"the {name} read does not list the record's examples")
    nd = [x["idx"] for x in sorted(none_rows, key=lambda x: x["idx"]) if x["correct"] != by[x["idx"]]["correct_own"]]
    sd = [x["idx"] for x in sorted(scr_rows, key=lambda x: x["idx"]) if x["correct"] != by[x["idx"]]["correct_scr"]]
    return {"none_diff": len(nd), "scrambled_diff": len(sd), "none_diff_idx": nd, "scrambled_diff_idx": sd,
            "n": len(by), "status": "held" if nd or sd else "passed"}


def a61_state(k2: dict, sigma: float, rd: dict | None) -> dict:
    """The K2 cell of a dep_loss arm (A6.1 rules 1, 5, 6; critic entry 2026-10-20T17:36Z parts (2) and (5)). rd: the
    arm's directional reading (None: no record). Without a usable directional read (none, excluded, BUG, held or the
    identification not run) the cell is rule 6's pending label and no K2 sentence; with one, exactly one of three
    states: fired on the drop leg (7.5's K2 fires: Delta_scr < sigma_screen or the interval includes 0); not fired
    with a present direction and the garbage of record not exceeded; or counts as fired under rule 5 (a non-fired
    drop, the "large scrambled drop" of 17:36Z (2), whose direction is absent or whose G_chg exceeds R_dir)."""
    o, why = k2_outcome(k2["delta"], k2["lo"], k2["hi"], sigma)
    st = (rd or {}).get("status", "pending")
    if st not in ("present", "absent"):
        return {"state": "pending", "k2": "pending (A6.1 rule 6)", "drop_k2": o, "why": why}
    direction = st == "present" and not rd["garbage_exceeds"]
    if o == "fired":
        return {"state": "fired", "k2": "fired", "drop_k2": o, "why": why, "direction": direction}
    if direction:
        return {"state": "present", "k2": "not fired", "drop_k2": o, "why": why, "direction": True}
    return {"state": "rule 5", "k2": "counts as fired (rule 5)", "drop_k2": o, "why": why, "direction": False}


def directional_sentence(rd: dict) -> str:
    """The directional read's numbers as the fourth table prints them (every quantity with its count)."""
    n = rd["n_scored"]
    word = "present" if rd["status"] == "present" else "absent (interval includes 0)"
    thin = f" (present (thin: k_dir minus k_own = {rd['k_net']}))" if rd["status"] == "present" and rd["thin"] else ""
    return (f"directional read {word}{thin}: donor-answer rate {rd['k_dir']} of {n} = {float(rd['r_dir']):.3f} vs "
            f"{rd['k_own']} of {n} = {float(rd['r_own']):.3f} under own latents, Delta_dir "
            f"{float(rd['delta_dir']):+.3f} [{rd['lo']:+.3f}, {rd['hi']:+.3f}] (6.2 form, {DONOR_N_BOOT:,} paired "
            f"draws, default_rng({DONOR_BOOT_SEED}); percentile [{rd['pct_lo']:+.3f}, {rd['pct_hi']:+.3f}]); garbage "
            f"G_chg {rd['k_g_chg']} of {n} = {float(rd['g_chg']):.3f} vs R_dir {float(rd['r_dir']):.3f}: "
            f"{'exceeded' if rd['garbage_exceeds'] else 'not exceeded'} (rule 5's reading of record, critic entry "
            f"{DONOR_ROW} ruling 3; the gold-literal and excess readings decide nothing) (1 seed, valid; "
            f"{rd['entry']})")


def present_label(rd: dict) -> str:
    """17:36Z (5)'s lifted label, verbatim in form: present and the garbage of record not exceeded."""
    n = rd["n_scored"]
    thin = f"; present (thin: k_dir minus k_own = {rd['k_net']})" if rd["thin"] else ""
    return (f"{A61['present']}: donor-answer rate {rd['k_dir']} of {n} = {float(rd['r_dir']):.3f} vs {rd['k_own']} of "
            f"{n} = {float(rd['r_own']):.3f} under own latents, Delta_dir {float(rd['delta_dir']):+.3f} "
            f"[{rd['lo']:+.3f}, {rd['hi']:+.3f}]; garbage G_chg {float(rd['g_chg']):.3f} vs R_dir "
            f"{float(rd['r_dir']):.3f}: not exceeded (1 seed, valid){thin}")


def pending_label(rd: dict | None) -> str:
    """Rule 6's label with its reason (17:36Z (5): "(BUG: <reason>)" or "(held: identification <k> of 500, A12.3)")."""
    st = (rd or {}).get("status", "pending")
    if st == "BUG":
        return f"{A61['pending']} {A61['bug']}{rd['why']})"
    if st == "held":
        i = rd["identification"]
        return (f"{A61['pending']} {A61['held']} {i['none_diff']} (none) and {i['scrambled_diff']} (scrambled) of "
                f"{i['n']}, A12.3)")
    if st == "not run":
        return f"{A61['pending']} (identification not run: {rd['why']})"
    if st == "excluded":
        return f"{A61['pending']} (the record enters no cell: {rd['why']})"
    return f"{A61['pending']} (no donor-answer record)"


# --------------------------------------------------------------------------------------------------- the reader
class Screen:
    def __init__(self, task: str, records: str, receipts: list[str], *, source_uri: str | None = None,
                 manifest: str | None = None, critic_log: str | None = None, k1_table: str | None = None,
                 valid_file: str | None = None, tokenizer: str | None = None, q_lens: list[int] | None = None,
                 eval_dir: str | None = None, plain_depth_dir: str | None = None, diag: dict | None = None,
                 hinge_records: str | None = None, hinge_source_uri: str | None = None,
                 hinge_eval_dir: str | None = None, hinge_read: str | None = None, extra_arm: str | None = None,
                 extra_records: str | None = None, extra_source_uri: str | None = None,
                 extra_eval_dir: str | None = None, extra_read: str | None = None):
        if task not in TASKS:
            raise SystemExit(f"unknown task {task!r}; registered: {sorted(TASKS)}")
        if extra_arm is not None and extra_arm not in CLOSING:
            raise SystemExit(f"REFUSED: --extra-arm {extra_arm!r}: registered {sorted(CLOSING)}")
        self.task = task
        # the closing arms enter only the reissue that carries them (row (b) (5)); every earlier table is unchanged
        self.spec = {**TASKS[task], "runs": [s for s in TASKS[task]["runs"]
                                             if s["key"] not in CLOSING or s["key"] == extra_arm]}
        self.n = self.spec["n_valid"]
        self.records = records
        self.pd_dir = os.path.abspath(plain_depth_dir) if plain_depth_dir else None
        self.third = bool(eval_dir and self.pd_dir)       # the third table: the eval reads and the 355M control
        self.hinge = HINGE.get(task) if (hinge_records or hinge_eval_dir) else None
        if (hinge_records or hinge_eval_dir) and self.hinge is None:
            raise SystemExit(f"REFUSED: no hinge job on {task} (A2.1: one hinge job, GSM-50k; critic entry "
                             f"{ROW_HINGE} (8): C3 and C2+C3 on ProsQA-hard close UNTESTED)")
        self.h_dir = os.path.abspath(hinge_records) if hinge_records else None
        self.h_runs = ({s["run"]: s["key"] for s in self.spec["runs"] if s["key"] in self.hinge["keys"]}
                       if self.hinge else {})
        self.h_eval_dir = hinge_eval_dir
        self.h_read = hinge_read
        self.fourth = bool(self.third and hinge_eval_dir)  # the fourth table: the third's reissue over the hinge arms
        if self.fourth and not (self.h_dir and hinge_read):
            raise SystemExit("REFUSED: the fourth table needs the hinge runs' records (--hinge-records) and the hinge "
                             "eval job's registered read row (--registered-read)")
        self.ev_h = None                                  # the hinge eval job (load_evals with hinge=True)
        self.extra = CLOSING[extra_arm] if extra_arm else None   # the no-curriculum reissue (critic entry 20:21Z (b))
        self.x_keys = (extra_arm,) if extra_arm else ()
        self.x_dir = os.path.abspath(extra_records) if extra_records else None
        self.x_runs = {s["run"]: s["key"] for s in self.spec["runs"] if s["key"] in self.x_keys}
        self.x_eval_dir, self.x_read = extra_eval_dir, extra_read
        self.x_job = self.extra["jobs"][task] if self.extra else None
        self.ev_x = None                                  # the arm's eval job (load_evals with closing=True)
        self._x_verify: list = []                         # the arm's labels (role, Curriculum line) to verify
        if self.extra and not (self.third and self.x_dir and extra_eval_dir and extra_read
                               and (self.fourth or task not in HINGE)):
            raise SystemExit("REFUSED: the no-curriculum reissue needs the prior table's inputs (the fourth table's on "
                             "GSM-50k, the third's on ProsQA-hard), the arm's records (--extra-records), its eval job "
                             "(--extra-eval-dir) and that job's registered read row (--extra-registered-read)")
        self.ref_directional = None                       # the coconut reference record's reading (notes only)
        self._verify: dict = {}                           # per sigma reading: the number-carrying labels to verify
        self._bar_key = None
        self.diag_paths = {k: os.path.join(EXP, v) if v else None for k, v in DIAG_RECORDS.get(task, {}).items()}
        self.diag_paths.update(diag or {})                # --diag-gpu / --diag-float64 replace the task's defaults
        self.diag_named = set(diag or {})                 # named on the command line: must exist
        self.diag_where = None                            # the float64 record's device clause (third_labels)
        self.diag64_meets = False                         # the float64 line met on this task (third_labels)
        self.pd_probe = None                              # the control's memory-probe label (its job's receipts)
        self.labels3 = None                               # the third table's labels (critic entries), verified
        extra = {"plain_depth_records": self.pd_dir} if self.pd_dir else {}
        if self.h_dir:                                    # AcctA's bucket: its own source URI, never the main one
            extra["hinge_records"] = (self.h_dir, hinge_source_uri)
        if self.x_dir:                                    # the arm's training project's bucket: its own source URI
            extra["extra_records"] = (self.x_dir, extra_source_uri)
        self.inp = Inputs(records, source_uri, extra or None)
        self.receipt_globs = list(receipts)
        self.manifest = self.inp.read_json(manifest or os.path.join(EXP, "configs", "data_manifest.json"),
                                           "data manifest (pins, valid hash)")
        self.critic_log = critic_log or os.path.join(EXP, "governance", "critic_log.md")
        self.k1_table = k1_table or os.path.join(EXP, "results", "k1", "k1_table_final.json")
        self.valid_file = valid_file
        self.tokenizer = tokenizer or os.path.join(REPO, "common", "babylm", "recipes", "sprint1_port",
                                                   "gpt2_tokenizer")
        self._q_lens = q_lens
        self.eval_dir = eval_dir
        self.ev = None                     # the eval reads (load_evals), for the second table only
        self.lab_inner = EVAL_LABEL if eval_dir else f"{SCREEN_LABEL}; {BAR_LABEL}"
        for p, role in ((os.path.join(EXP, "governance", "registration.md"), "rules (registration)"),
                        (os.path.abspath(__file__), "code (this tool)"),
                        (os.path.join(REPO, "common", "latent", "accounting.py"), "code (5.1 / 5.3 / 5.5 (c))"),
                        (os.path.join(REPO, "common", "latent", "data.py"), "code (tokenize_examples)")):
            self.inp.sha256(p, role)

    # ---------------------------------------------------------------------------------------------- loading
    def _path(self, run: str, name: str) -> str:
        if self.pd_dir and run in self.spec["plain_depth_all"]:          # the control's own mirror (--plain-depth-dir)
            return os.path.join(self.pd_dir, "runs", run, name)
        if self.h_dir and run in self.h_runs:                            # the hinge runs' mirror (--hinge-records)
            return os.path.join(self.h_dir, "runs", run, name)
        if self.x_dir and run in self.x_runs:                            # the closing arm's mirror (--extra-records)
            return os.path.join(self.x_dir, "runs", run, name)
        return os.path.join(self.records, "runs", run, name)

    def load_receipts(self) -> list[dict]:
        out = []
        for pat in self.receipt_globs:
            for p in sorted(glob.glob(pat)):
                r = self.inp.read_json(p, "job receipt")
                r["_path"] = p
                out.append(r)
        return out

    def load_run(self, spec: dict) -> dict:
        run = spec["run"]
        rec = {"run": run, "key": spec["key"], "role": spec["role"], "card": spec["card"], "status": "UNTESTED",
               "checks": [], "problems": [], "labels": []}
        if not os.path.isfile(self._path(run, "RUN_DONE.json")):
            rec["why"] = "no run record (RUN_DONE.json absent)"
            return rec
        rd = self.inp.read_json(self._path(run, "RUN_DONE.json"), "run record")
        cfg = self.inp.read_json(self._path(run, "config.json"), "run record")
        rows_all = [json.loads(ln) for ln in self.inp.read_text(self._path(run, "log.jsonl"), "run record")
                    .splitlines() if ln.strip()]
        state = self.inp.read_json(self._path(run, "ckpt_state.json"), "run record")
        rec.update(run_done=rd, config=cfg, state=state)
        prob = rec["problems"]

        def check(name: str, ok: bool, detail: str = "", fatal: str | None = None):
            rec["checks"].append({"check": name, "ok": bool(ok), "detail": detail})
            if not ok:
                prob.append(f"{name}: {detail}" if detail else name)
                if fatal and rec.get("exclusion") is None:
                    rec["exclusion"] = fatal

        check("RUN_DONE done, rc 0", rd.get("status") == "done" and rd.get("rc") == 0,
              f"status {rd.get('status')}, rc {rd.get('rc')}", "incomplete")
        check("RUN_DONE names the registered config", rd.get("config") == spec["config"],
              f"{rd.get('config')} vs {spec['config']}", "config mismatch")
        sm_path = self._path(run, "SYNC_MANIFEST.json")
        if os.path.isfile(sm_path):
            sm = self.inp.read_json(sm_path, "run record")["files"]
            bad = [f for f in ("config.json", "log.jsonl", "ckpt_state.json")
                   if sm.get(f, {}).get("sha256") != self.inp.sha256(self._path(run, f), "run record")]
            ck = (sm.get("ckpt_best_final.pt") or {}).get("sha256")
            if ck != (rd.get("checkpoints") or {}).get("ckpt_best_final.pt"):
                bad.append("ckpt_best_final.pt (RUN_DONE vs SYNC_MANIFEST)")
            check("record files equal SYNC_MANIFEST.json", not bad, ", ".join(bad), "record integrity")
        else:
            check("record files equal SYNC_MANIFEST.json", False, "SYNC_MANIFEST.json absent", "record integrity")
        if self.h_dir and run in self.h_runs:
            self._hinge_receipt_check(run, rd, check)
        if self.x_dir and run in self.x_runs:
            self._hinge_receipt_check(run, rd, check, root=self.x_dir, job=self.x_job)
        idx_path = self._path(run, "runs_index_row.csv")
        if os.path.isfile(idx_path):
            row = next(csv.DictReader(io.StringIO(self.inp.read_text(idx_path, "run record"))))
            rec["gpu_type"] = row.get("gpu_type")
            rec["fragmented"] = row.get("fragmented")
        reg_path = os.path.join(EXP, spec["config"])
        if os.path.isfile(reg_path):
            reg = self.inp.read_json(reg_path, "registered config")
            diff = [k for k in RECIPE_KEYS if reg.get(k) != cfg.get(k)]
            diff += [k for k in ("train_path", "val_path")
                     if os.path.basename(str(reg.get(k))) != os.path.basename(str(cfg.get(k)))]
            check("effective config = registered config (recipe keys)", not diff, ", ".join(diff), "config mismatch")
        else:
            check("registered config file exists", False, spec["config"], "config mismatch")
        rec["model_id"], rev = self._model_id(cfg.get("model_id", ""))
        pin = next((m for m in self.manifest.get("models", []) if m.get("id") == rec["model_id"]), None)
        check("base model revision = manifest pin", pin is not None and rev == pin.get("revision"),
              f"{rec['model_id']}@{rev}", "model not pinned")
        val_name = os.path.basename(str(cfg.get("val_path")))
        check("validation file is the task's valid split", val_name == os.path.basename(self.valid_entry()["path"]),
              val_name, "wrong split")
        if spec["key"] == "plain_depth_355m":
            self._plain_depth_checks(cfg, check)

        try:
            rows, superseded = rows_of_record(rows_all)
        except RecordError as e:
            check("rows of record (A8.2)", False, str(e), "BUG trigger (A8.2)")
            return self._finish(rec)
        rec.update(rows=rows, superseded=superseded)
        first = int(cfg.get("resume", 0)) + 1
        want = list(range(first, int(cfg["num_epochs"]) + 1))
        check("one row of record per epoch", [int(r["epoch"]) for r in rows] == want,
              f"epochs {[int(r['epoch']) for r in rows]} vs {first}..{cfg['num_epochs']}", "incomplete")
        n_train = int(self.split_entry("train")["records"])
        bad = [int(r["epoch"]) for r in rows if int(r["max_new_tokens"]) != self.spec["max_new_tokens"]
               or int(r["n_train"]) != n_train]
        check("rows: max_new_tokens and n_train", not bad, f"epochs {bad} (want {self.spec['max_new_tokens']}, "
              f"{n_train})", "recipe mismatch")
        if cfg.get("precision") == "bf16":
            bad = [int(r["epoch"]) for r in rows if r.get("precision") != "bf16" or r.get("numerics") != BF16_FLAGS]
            check("A10.5 flag set on every row (bf16)", not bad, f"epochs {bad}", "precision-mismatched")
        nonfinite = [int(r["epoch"]) for r in rows if not math.isfinite(float(r["train_loss"]))]
        check("finite train_loss (A6.6)", not nonfinite, f"epochs {nonfinite}", "numerics-failed")
        try:
            for r in rows:
                count_of(r["val_acc"], self.n, "val_acc")
                count_of(r["val_emitted_mean"], self.n, "val_emitted_mean")
        except RecordError as e:
            check("valid counts over the full split", False, str(e), "not a full-split read")
            return self._finish(rec)

        latent = cfg["arm"] not in ("cot", "no_cot")
        nocur = cfg["arm"] == "coconut_no_curriculum"     # row (b) (1), (2): every trained epoch is selectable
        pool = tuple(self.spec["latent_pool"]) if latent and not nocur else None
        rec["pool"] = pool
        if latent and not nocur:
            bad = [int(r["epoch"]) for r in rows if bool(r.get("final_stage")) != (pool[0] <= int(r["epoch_index"])
                                                                                 <= pool[1])]
        else:
            bad = [int(r["epoch"]) for r in rows if not r.get("final_stage")]
        if nocur:
            check("final_stage true on every epoch (the pool: every trained epoch; critic entry 2026-10-21T20:21Z (b) "
                  "(2))", not bad, f"epochs {bad}", "BUG trigger (A8.2)")
        else:
            check("final_stage flags = the registered pool (D16)", not bad, f"epochs {bad}", "BUG trigger (A8.2)")
        sel = select(rows, self.n, pool)
        if sel is None:
            check("a selectable epoch in the pool", False, "", "incomplete")
            return self._finish(rec)
        k = count_of(sel["val_acc"], self.n, "val_acc")
        best_rd = (rd.get("best") or {}).get("final")
        best_st = (state.get("best") or {}).get("final")
        for name, b in (("RUN_DONE", best_rd), ("ckpt_state", best_st)):
            ok = bool(b) and count_of(b[0], self.n, "best") == k and int(b[1]) == int(sel["epoch"])
            check(f"selection = {name} best.final (A8.2)", ok, f"{b} vs [{sel['val_acc']}, {sel['epoch']}]",
                  "BUG trigger (A8.2)")
        L = gen_latents(cfg, int(sel["stage"]))
        if nocur:                                     # pinned to stage max_latent_stage + 1 at every epoch (data.py)
            ps = int(cfg["max_latent_stage"]) + 1
            L = gen_latents(cfg, ps)
            bad = [int(r["epoch"]) for r in rows if int(r.get("n_latent_eval", -1)) != L]
            check("logged n_latent_eval = the pinned stage's latents on every epoch", not bad,
                  f"epochs {bad} (want {L})", "BUG trigger (A8.2)")
            rec["closing"] = {"pinned_stage": ps, "n_latent": L, "pool": "every trained epoch",
                              "logged_stage": "the schedule's arithmetic (epoch // epochs_per_stage), not the stage"}
        elif cfg["arm"] in ("coconut", "coconut_no_curriculum"):
            check("logged n_latent_eval = the stage's latents", int(sel["n_latent_eval"]) == L,
                  f"{sel['n_latent_eval']} vs {L}", "BUG trigger (A8.2)")
        best_all = select(rows, self.n, None)
        rec.update(selected=sel, k=k, L=L, M=count_of(sel["val_emitted_mean"], self.n, "val_emitted_mean"),
                   latent=latent, feedback=cfg["arm"] in ("coconut", "coconut_no_curriculum"),
                   best_all=best_all, k_all=count_of(best_all["val_acc"], self.n, "val_acc"))
        rec["earlier_stage_better"] = latent and rec["k_all"] > k
        rec["resumes"] = int(rd.get("resumes", 0))
        rec["attempts"] = sorted({rd.get("attempt")} | {str(a).split("@")[0] for a in rd.get("resume_list", [])}
                                 - {None})
        rec["train_flops_own"] = float(sum(float(r["flops"]) for r in rows))
        if self.spec["start_pin"] and latent:
            self._start_pin(rec, check)
        elif latent:
            check("starts from the pinned base model (no load_model_path)", cfg.get("load_model_path") is None,
                  str(cfg.get("load_model_path")), "speculative, unused")
        return self._finish(rec)

    def _plain_depth_checks(self, cfg: dict, check) -> None:
        """A13.2 and 7.7 on the control's record: its registered config equals the 124M no_cot screen config in every
        key but model_id, name, save_dir, the note and the micro-batch (32 x 4, or A13.2 (b)'s 16 x 8 fallback, the
        effective batch unchanged); its model is gpt2-medium; and the 355M pin's tokenizer files are the 124M pin's, so
        the |q| of the hash-checked valid file (pinned 124M tokenizer) is the control's |q| too."""
        t = self.task
        reg_pd = self.inp.read_json(os.path.join(EXP, "configs", "screen", f"{t}_plain_depth_355m.json"),
                                    "registered config")
        reg_nc = self.inp.read_json(os.path.join(EXP, "configs", "screen", f"{t}_no_cot.json"), "registered config")
        diff = sorted(k for k in set(reg_pd) | set(reg_nc) if k not in PD_FREE_KEYS and reg_pd.get(k) != reg_nc.get(k))
        mb = (int(cfg["batch_size"]), int(cfg["grad_accum"]))
        eff = int(reg_nc["batch_size"]) * int(reg_nc["grad_accum"])
        ok = (not diff and mb in ((32, 4), (16, 8)) and mb[0] * mb[1] == eff
              and reg_pd.get("model_id") == "openai-community/gpt2-medium")
        check("A13.2: registered config = the 124M no_cot config but model_id, name, save_dir, note, micro-batch",
              ok, f"other keys differing {diff}; micro-batch {mb[0]} x {mb[1]} (effective {mb[0] * mb[1]} vs {eff}); "
              f"model {reg_pd.get('model_id')}", "config mismatch")
        pins = {m.get("id"): m.get("files_sha256") or {} for m in self.manifest.get("models", [])}
        a, b = pins.get("openai-community/gpt2", {}), pins.get("openai-community/gpt2-medium", {})
        same = all(a.get(f) and a.get(f) == b.get(f) for f in TOKENIZER_FILES)
        check("tokenizer files of the 355M pin = the 124M pin's (|q| shared)", same,
              ", ".join(f for f in TOKENIZER_FILES if not (a.get(f) and a.get(f) == b.get(f))), "model not pinned")

    def pd_receipts(self, runs: dict) -> list[dict]:
        """The plain-depth job's receipts under PD/receipts/<job>/<attempt>/ (A13.2 (b); deviations 11:05Z): the
        receipt of the attempt that ran the control joins the receipts read (A3.2 (i)); every other attempt of the job
        is read only as memory-probe evidence. Sets self.pd_probe = {"label", ...}: "probe at <b> not measured" when the
        run's attempt skipped its probe (or "measured" with its peak), then "OOM at <B> measured (<attempt tail>)" for
        each earlier attempt whose probe ran out of memory in every precision (its step and its memory_probe.json)."""
        if not self.pd_dir:
            return []
        pd = runs.get("plain_depth_355m") or {}
        att = (pd.get("run_done") or {}).get("attempt")
        found = sorted(glob.glob(os.path.join(self.pd_dir, "receipts", "*", "*", "JOB_RECEIPT.json")))
        mine = [p for p in found if os.path.basename(os.path.dirname(p)) == att]
        if len(mine) != 1:
            self.pd_probe = {"label": None, "why": f"no single receipt of the control's attempt {att} under "
                                                   f"{self.pd_dir}/receipts"}
            return []
        rr = self.inp.read_json(mine[0], "job receipt")
        rr["_path"] = mine[0]
        why = []
        if rr.get("status") != "complete" or rr.get("attempt") != att:
            why.append(f"receipt status {rr.get('status')}, attempt {rr.get('attempt')}")
        for name in self.spec["plain_depth_all"]:
            x = next((x for x in rr.get("runs", []) if x.get("run") == name), None)
            rd = self.inp.read_json(self._path(name, "RUN_DONE.json"), "run record") if \
                os.path.isfile(self._path(name, "RUN_DONE.json")) else {}
            if not x or x.get("status") != "done" or x.get("checkpoints") != rd.get("checkpoints"):
                why.append(f"the receipt's {name} entry is not done with RUN_DONE's checkpoints")
        b = ((rr.get("spec") or {}).get("memory_probe") or {}).get("batch")
        st = next((s for s in rr.get("steps", []) if s.get("step") == "memory_probe"), None)
        if st is None or b is None:
            first = "memory probe: no step in the run's receipt"
            why.append(first)
        elif st.get("skipped"):
            first = f"probe at {b} not measured"
        elif st.get("ok") and st.get("peak_mem_mb"):
            first = f"probe at {b} measured (peak {max(st['peak_mem_mb'].values())} MB)"
        else:
            first = f"probe at {b} failed"
        parts, evidence = [], []
        for p in found:
            a = os.path.basename(os.path.dirname(p))
            if p in mine:
                continue
            rc = self.inp.read_json(p, "memory-probe evidence (an earlier attempt of the plain-depth job)")
            s2 = next((s for s in rc.get("steps", []) if s.get("step") == "memory_probe"), None)
            if rc.get("job") != rr.get("job") or s2 is None or s2.get("skipped") or s2.get("ok") is not False:
                continue
            mp = os.path.join(os.path.dirname(p), "memory_probe.json")
            bench = (self.inp.read_json(mp, "memory-probe evidence (an earlier attempt of the plain-depth job)")
                     .get("bench") or []) if os.path.isfile(mp) else []
            oom = bool(bench) and all("OutOfMemoryError" in str(x.get("error", "")) for x in bench) and \
                all("OutOfMemoryError" in str(e) for e in s2.get("errors", [])) and bool(s2.get("errors"))
            if oom:
                parts.append(f"OOM at {s2.get('batch')} measured ({a[-4:]})")
                evidence.append({"attempt": a, "batch": s2.get("batch"), "seq": s2.get("seq"),
                                 "precisions": [x.get("precision") for x in bench]})
        self.pd_probe = {"label": ", ".join([first] + parts), "run_attempt": att, "evidence": evidence,
                         "problems": why}
        return [rr]

    def _hinge_receipt_check(self, run: str, rd: dict, check, root: str | None = None, job: str | None = None) -> None:
        """A hinge run's record against the receipt of the attempt that completed it (RUN_DONE's `attempt`; critic entry
        2026-10-20T17:36Z (7): "the receipt of the attempt that writes RUN_DONE.json for the three runs, whichever
        attempt id that is"): the hinge job's receipt of that attempt lists the run done with RUN_DONE's checkpoints.
        With root and job: the same check for a closing arm under --extra-records (its own train job)."""
        job, att = job or self.hinge["job"], rd.get("attempt")
        p = os.path.join(root or self.h_dir, "receipts", job, str(att), "JOB_RECEIPT.json")
        ok, detail = False, f"no receipt of RUN_DONE's attempt {att} under receipts/{job}/"
        if att and os.path.isfile(p):
            rr = self.inp.read_json(p, "job receipt")
            x = next((x for x in rr.get("runs") or [] if x.get("run") == run), None)
            ok = (rr.get("job") == job and rr.get("attempt") == att and x is not None and x.get("status") == "done"
                  and x.get("rc") == 0 and x.get("checkpoints") == rd.get("checkpoints"))
            same = bool(x) and x.get("checkpoints") == rd.get("checkpoints")
            detail = (f"{att}: receipt job {rr.get('job')}, status {rr.get('status')}; the run's entry "
                      f"{(x or {}).get('status')}, checkpoints {'=' if same else '!='} RUN_DONE's")
        check("the completing attempt's receipt lists the run done with RUN_DONE's checkpoints", ok, detail,
              "record integrity")

    def hinge_receipts(self) -> list[dict]:
        """Every receipt of the hinge job under --hinge-records (receipts/<job>/<attempt>/JOB_RECEIPT.json): they join
        the receipts read (env_reference, admissions, the attempts the runs name; A3.2 (i)). A receipt of another job
        there is a refusal."""
        if not self.h_dir:
            return []
        out = []
        for p in sorted(glob.glob(os.path.join(self.h_dir, "receipts", self.hinge["job"], "*", "JOB_RECEIPT.json"))):
            r = self.inp.read_json(p, "job receipt")
            if r.get("job") != self.hinge["job"] or r.get("attempt") != os.path.basename(os.path.dirname(p)):
                raise SystemExit(f"REFUSED: {p}: job {r.get('job')!r}, attempt {r.get('attempt')!r}: not a receipt of "
                                 f"{self.hinge['job']} at its own attempt")
            r["_path"] = p
            out.append(r)
        if not out:
            raise SystemExit(f"REFUSED: no receipt of {self.hinge['job']} under {self.h_dir}/receipts")
        return out

    def extra_receipts(self) -> list[dict]:
        """Every receipt of the closing arm's train job under --extra-records (receipts/<job>/<attempt>/
        JOB_RECEIPT.json), as hinge_receipts reads the hinge job's: they join the receipts read; a receipt of another
        job there is a refusal."""
        if not self.x_dir:
            return []
        out = []
        for p in sorted(glob.glob(os.path.join(self.x_dir, "receipts", self.x_job, "*", "JOB_RECEIPT.json"))):
            r = self.inp.read_json(p, "job receipt")
            if r.get("job") != self.x_job or r.get("attempt") != os.path.basename(os.path.dirname(p)):
                raise SystemExit(f"REFUSED: {p}: job {r.get('job')!r}, attempt {r.get('attempt')!r}: not a receipt of "
                                 f"{self.x_job} at its own attempt")
            r["_path"] = p
            out.append(r)
        if not out:
            raise SystemExit(f"REFUSED: no receipt of {self.x_job} under {self.x_dir}/receipts")
        return out

    def _finish(self, rec: dict) -> dict:
        rec["status"] = "excluded" if rec.get("exclusion") else ("read" if "k" in rec else "excluded")
        if rec["status"] == "excluded" and not rec.get("exclusion"):
            rec["exclusion"] = "incomplete"
        return rec

    def _start_pin(self, rec: dict, check) -> None:
        """A6.3 rule 4 against the critic's start row (and rules 2-3 as the records show them)."""
        sp, cfg, rd = self.spec["start_pin"], rec["config"], rec["run_done"]
        want_path = f"runs/{sp['source_run']}/ckpt_best_final.pt"
        got = (cfg.get("load_model_path"), cfg.get("load_model_sha256"), cfg.get("load_model_epoch"))
        init = rd.get("init") or {}
        ok_cfg = (str(got[0]).endswith(want_path) and got[1] == sp["sha256"] and got[2] == sp["epoch"])
        ok_init = (str(init.get("load_model_path", "")).endswith(want_path) and init.get("load_model_sha256") ==
                   sp["sha256"] and init.get("load_model_epoch") == sp["epoch"])
        pool = rec.get("pool")                         # None: every trained epoch (coconut_no_curriculum, row (b) (2))
        in_pool = pool is None or pool[0] <= int(rec["selected"]["epoch_index"]) <= pool[1]
        src = self._path(sp["source_run"], "RUN_DONE.json")
        ok_src, src_detail = False, "source RUN_DONE.json absent"
        if os.path.isfile(src):
            srd = self.inp.read_json(src, "run record")
            src_sha = (srd.get("checkpoints") or {}).get("ckpt_best_final.pt")
            src_ep = ((srd.get("best") or {}).get("final") or [None, None])[1]
            ok_src = src_sha == sp["sha256"] and src_ep == sp["epoch"]
            src_detail = f"source {src_sha[:12] if src_sha else None}..., epoch {src_ep}"
        rows_ok, row_detail = self._critic_rows_cite(sp)
        detail = (f"config ({got[0]}, {str(got[1])[:12]}..., {got[2]}); RUN_DONE init "
                  f"{'=' if ok_init else '!='}; {src_detail}; selected epoch_index {rec['selected']['epoch_index']} "
                  + (f"{'in' if in_pool else 'outside'} {pool[0]}-{pool[1]}" if pool is not None else
                     "(selection over every trained epoch: no pool)") + f"; {row_detail}")
        ok = ok_cfg and ok_init and in_pool and ok_src and rows_ok
        rec["start_pin"] = {"ok": ok, "load_model_path": got[0], "load_model_sha256": got[1],
                            "load_model_epoch": got[2], "detail": detail}
        check("A6.3 rule 4 start pin", ok, detail, "speculative, unused")

    def _critic_rows_cite(self, sp: dict) -> tuple[bool, str]:
        text = self.inp.read_text(self.critic_log, "rules (critic log)")
        first = sp["critic_rows"][0]
        row = next((ln for ln in text.splitlines() if ln.startswith(f"| {first} |")), None)
        missing = [s for s in sp["critic_rows"] if not any(ln.startswith(f"| {s} |") for ln in text.splitlines())]
        if row is None or missing:
            return False, f"critic entries missing: {missing or [first]}"
        ok = sp["sha256"] in row and f"epoch {sp['epoch']}" in row and sp["source_run"] in row
        return ok, f"critic entry {first} {'names' if ok else 'does not name'} the sha256, run and epoch"

    @staticmethod
    def _model_id(path: str) -> tuple[str, str | None]:
        m = re.search(r"models--([^/]+?)--([^/]+)/snapshots/([0-9a-f]{40})", path)
        if m:
            return f"{m.group(1)}/{m.group(2)}", m.group(3)
        return path, None

    def split_entry(self, split: str) -> dict:
        """The manifest's screen-block entry for the task's `split` file (train or valid; the test entry is never
        used)."""
        if split not in ("train", "valid"):
            raise ValueError("the screen reads the train record count and the valid split only")
        blk = (self.manifest.get("screen") or {}).get(self.task) or {}
        ents = blk.get("files") or [blk[k] for k in ("train", "valid", "test") if isinstance(blk.get(k), dict)]
        hit = [e for e in ents if str(e.get("path", "")).endswith(f"_{split}.json")]
        if len(hit) != 1:
            raise SystemExit(f"the manifest has no single {split} entry for {self.task}")
        return hit[0]

    def valid_entry(self) -> dict:
        return self.split_entry("valid")

    # ---------------------------------------------------------------------------------------------- costs
    def q_lens(self) -> list[int]:
        if self._q_lens is not None:
            return list(self._q_lens)
        ent = self.valid_entry()
        path = self.valid_file or os.path.join(EXP, ent["path"])
        if "test" in os.path.basename(path):
            raise SystemExit("REFUSED: the screen reads the valid split only (A3.4); no test file is opened")
        if self.inp.sha256(path, "valid split") != ent["sha256"]:
            raise SystemExit(f"REFUSED: {path} does not hash to the manifest's valid entry {ent['sha256']}")
        pinned = next(m for m in self.manifest["models"] if m["id"] == "openai-community/gpt2")["files_sha256"]
        for f in sorted(os.listdir(self.tokenizer)):
            p = os.path.join(self.tokenizer, f)
            if f not in pinned or self.inp.sha256(p, "tokenizer (pinned files)") != pinned[f]:
                raise SystemExit(f"REFUSED: tokenizer file {f} is not the pinned snapshot's (manifest models[0])")
        from transformers import AutoTokenizer
        from latent.data import load_examples, tokenize_examples
        tok = AutoTokenizer.from_pretrained(self.tokenizer)
        ex = load_examples(path)
        if len(ex) != self.n:
            raise SystemExit(f"valid file has {len(ex)} records, registered {self.n}")
        self._q_lens = [len(t["question_tokenized"]) for t in tokenize_examples(ex, tok)]
        return list(self._q_lens)

    _dims_cache: dict = {}

    def dims(self, model_id: str, V: int):
        key = (model_id, V)
        if key not in self._dims_cache:
            from transformers import GPT2Config
            from latent.accounting import model_dims
            d = model_dims(GPT2Config(vocab_size=V, **MODEL_CONFIGS[model_id]))
            pin = next(m for m in self.manifest["models"] if m["id"] == model_id)
            if d.N_ne != pin["n_ne"]:
                raise SystemExit(f"N_ne {d.N_ne} != the manifest's {pin['n_ne']} for {model_id}")
            self._dims_cache[key] = d
        return self._dims_cache[key]

    def costs(self, rec: dict, layer_fraction: float = 1.0) -> dict:
        """Per-run cost summary on the valid questions (5.1, 5.3, 5.5 (c)): exact mean E' and passes; FLOP and
        reference-FLOP means as exact ranges over the unrecorded per-example E' (sum_bounds)."""
        from latent.accounting import inference_record
        cfg = rec["config"]
        arm = cfg["arm"]
        acc_arm = arm if arm in ("cot", "no_cot", "pause") else "coconut"
        V = V_BASE if arm in ("cot", "no_cot") else V_LATENT
        d = self.dims(rec["model_id"], V)
        U, L, M = self.spec["max_new_tokens"], rec["L"], rec["M"]
        q = self.q_lens()
        cache, fl, rf = {}, [], []
        for qi in q:
            if qi not in cache:
                fr, rr = [], []
                for e in range(U + 1):
                    kw = {"layer_fraction": layer_fraction} if layer_fraction != 1.0 else {}
                    ir = inference_record(d, acc_arm, qi, L, e, **kw)
                    fr.append(ir.flops)
                    rr.append(ir.reference_flops)
                cache[qi] = (fr, rr)
            fl.append(cache[qi][0])
            rf.append(cache[qi][1])
        lo, hi = sum_bounds(fl, M)
        rlo, rhi = sum_bounds(rf, M)
        n = len(q)
        lat_steps = L if rec["feedback"] else 0
        return {"layer_fraction": layer_fraction, "V": V, "N_ne": d.N_ne, "L": L, "emitted_total": M,
                "emitted_mean": M / n, "passes_mean": 1 + lat_steps + M / n, "flops_mean": (lo / n, hi / n),
                "reference_flops_mean": (rlo / n, rhi / n), "q_mean": sum(q) / n, "n": n}

    # ------------------------------------------------------------------------ the eval reads (the second table)
    def load_evals(self, runs: dict, eval_dir: str | None = None, *, hinge: bool = False,
                   closing: bool = False) -> dict:
        """The screen eval job's reads (critic entry 2026-10-20T10:51Z): JOB_RECEIPT.json and per entry
        <name>/{summary.json, examples.csv, eval.log}, every file hashed into the provenance. Each entry is checked
        against the receipt, the registered row, the run's record (the selected checkpoint's sha256 and epoch), the
        valid split's hash, A10.5's numerics and the 5.1 / 5.3 accounting of its own per-example file; an entry that
        fails a check enters no cell. Structural problems (no receipt, an entry of an unknown run, an unregistered
        kind, two reads of one tuple) refuse the table. With hinge (the fourth table): the hinge eval job's directory
        (critic entries 2026-10-20T17:12Z (6), 17:36Z (6)): reads of the three hinge arms only, their donor_answer records
        and the coconut reference record; its entries' registered row must be --registered-read; the reference
        record's identification uses the first job's committed none / scrambled reads of coconut (17:36Z ruling 9).
        With closing (the no-curriculum reissue, critic entry 2026-10-21T20:21Z (b) (4)): the arm's own eval job (its
        registered job name), reads of the arm only, no diag entry, its registered row --extra-registered-read; the
        earlier jobs hold no read of the arm."""
        d = os.path.abspath(eval_dir or self.eval_dir)
        rp = os.path.join(d, "JOB_RECEIPT.json")
        if not os.path.isfile(rp):
            raise SystemExit(f"REFUSED: {rp} absent: the eval reads need their job receipt")
        rc = self.inp.read_json(rp, "eval job receipt")
        spec = rc.get("spec") or {}
        sents = spec.get("evals") or []
        done = {e.get("name"): e for e in rc.get("evals") or []}
        run2key = {s_["run"]: s_["key"] for s_ in self.spec["runs"]}
        rows_reg = sorted({str(e.get("registered_read")) for e in sents if e.get("tool", "eval") == "eval"})
        if len(rows_reg) != 1:
            raise SystemExit(f"REFUSED: the eval entries name {rows_reg} as their registered read; one row expected")
        row_ts = rows_reg[0]
        if hinge and row_ts != self.h_read:
            raise SystemExit(f"REFUSED: the hinge eval entries name {row_ts} as their registered read, not "
                             f"--registered-read {self.h_read}")
        if closing and row_ts != self.x_read:
            raise SystemExit(f"REFUSED: the no-curriculum eval entries name {row_ts} as their registered read, not "
                             f"--extra-registered-read {self.x_read}")
        if closing and rc.get("job") != self.extra["eval_jobs"][self.task]:
            raise SystemExit(f"REFUSED: {rp}: job {rc.get('job')!r}, not the registered no-curriculum eval job "
                             f"{self.extra['eval_jobs'][self.task]!r} (critic entry {CLOSING_ROW} (b) (4))")
        log = self.inp.read_text(self.critic_log, "rules (critic log)")
        row = next((ln for ln in log.splitlines() if ln.startswith(f"| {row_ts} |")), None)
        if row is None:
            raise SystemExit(f"REFUSED: no critic_log.md row {row_ts}: the eval reads are not a registered read")
        env_ok = next((bool(s_.get("ok")) and not s_.get("mismatches") for s_ in rc.get("steps", [])
                       if s_.get("step") == "env_reference"), None)
        job = {"job": rc.get("job"), "attempt": rc.get("attempt"), "status": rc.get("status"),
               "registered_read": row_ts, "started": rc.get("started"), "ended": rc.get("ended"),
               "code_commit": (spec.get("_launch") or {}).get("code_commit"), "gpu": spec.get("gpu"),
               "env_reference": env_ok, "receipt": os.path.relpath(rp, REPO) if rp.startswith(REPO + os.sep) else rp,
               "receipt_sha256": self.inp.sha256(rp, "eval job receipt"), "n_entries": len(sents), "problems": []}
        if rc.get("status") != "complete":
            job["problems"].append(f"receipt status {rc.get('status')!r}, not complete")
        if env_ok is not True:
            job["problems"].append("env_reference step not ok")
        names = [e.get("name") for e in sents]
        extra = sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)) and x not in names)
        if extra:
            job["problems"].append(f"directories without a spec entry: {extra}")
        valid = self.valid_entry()
        n_L = self.dims("openai-community/gpt2", V_LATENT).n_L
        entries, by_run, donor_specs = [], {}, []
        for e in sents:
            name = e.get("name")
            if e.get("run") not in run2key:
                raise SystemExit(f"REFUSED: eval entry {name} names run {e.get('run')!r}, not a run of {self.task}")
            key = run2key[e["run"]]
            r = runs[key]
            cfg = r.get("config") or {}
            kind = classify_eval_entry(e, n_L)
            if kind == "unregistered":
                raise SystemExit(f"REFUSED: eval entry {name} is not a read the registered row defines")
            if hinge and not (key in self.hinge["keys"] or (kind == "donor_answer" and key == self.hinge["reference"])):
                raise SystemExit(f"REFUSED: hinge eval entry {name} ({e['run']}, {kind}): the hinge job holds the three "
                                 f"hinge arms' reads and donor_answer records and the coconut reference record only "
                                 f"(critic entries {ROW_HINGE} (6), {DONOR_ROW} (6))")
            if not hinge and self.fourth and key in self.hinge["keys"] and not closing:
                raise SystemExit(f"REFUSED: eval entry {name} of a hinge arm in {d}: the hinge arms' reads are the hinge "
                                 f"eval job's (--hinge-eval-dir)")
            if closing and (key not in self.x_keys or kind == "donor_answer"):
                raise SystemExit(f"REFUSED: no-curriculum eval entry {name} ({e['run']}, {kind}): that job holds the "
                                 f"arm's reads only (critic entry {CLOSING_ROW} (b) (4))")
            if not closing and key in self.x_keys:
                raise SystemExit(f"REFUSED: eval entry {name} of {e['run']} in {d}: the arm's reads are its own eval "
                                 f"job's (--extra-eval-dir)")
            if kind == "donor_answer":                 # A6.1 rule 4 (B): checked after the reads (identification)
                if any(x["run"] == e["run"] for x in donor_specs):
                    raise SystemExit(f"REFUSED: two donor_answer records of {e['run']}: one per checkpoint (A10.2)")
                donor_specs.append(e)
                continue
            if kind.endswith("_bscfg"):
                kind = kind[:-3] + str(int(cfg.get("eval_batch_size", 0)))
            ent = {"name": name, "key": key, "run": e["run"], "kind": kind, "checks": [], "problems": [],
                   "batch": int(e["batch_size"]) if e.get("batch_size") is not None
                   else int(cfg.get("eval_batch_size", 0))}

            def check(cname: str, ok: bool, detail: str = "", ent=ent):
                ent["checks"].append({"check": cname, "ok": bool(ok), "detail": detail})
                if not ok:
                    ent["problems"].append(f"{cname}: {detail}" if detail else cname)

            if kind in by_run.get(key, {}):
                raise SystemExit(f"REFUSED: two reads of ({e['run']}, {kind}) ({by_run[key][kind]['name']}, {name}): "
                                 f"one read per tuple (10:51Z part (6), A10.2)")
            by_run.setdefault(key, {})[kind] = ent
            entries.append(ent)
            got = done.get(name) or {}
            check("receipt: entry done, rc 0", got.get("status") == "done" and got.get("rc") == 0,
                  f"status {got.get('status')}, rc {got.get('rc')}")
            check("the run has a readable record", r["status"] == "read", r.get("exclusion") or r.get("why") or "")
            ed = os.path.join(d, name)
            paths = {f: os.path.join(ed, f) for f in ("summary.json", "examples.csv", "eval.log")}
            miss = [f for f, pth in paths.items() if not os.path.isfile(pth)]
            check("summary.json, examples.csv, eval.log present", not miss, ", ".join(miss))
            if miss or r["status"] != "read":
                continue
            sm = self.inp.read_json(paths["summary.json"], "eval read (summary)")
            self.inp.sha256(paths["eval.log"], "eval read (log)")
            csv_sha = self.inp.sha256(paths["examples.csv"], "eval read (per-example file)")
            try:
                rows = parse_examples_csv(self.inp.read_text(paths["examples.csv"], "eval read (per-example file)"))
            except (RecordError, ValueError) as ex_:
                check("examples.csv parses", False, str(ex_))
                continue
            ent.update(summary=sm, rows=rows, csv_sha256=csv_sha)
            check("examples.csv = summary examples_csv_sha256", sm.get("examples_csv_sha256") == csv_sha,
                  f"{str(sm.get('examples_csv_sha256'))[:12]}... vs {csv_sha[:12]}...")
            check("valid split (hash = the manifest's valid entry)", sm.get("split") == "valid" and
                  sm.get("split_sha256") == valid["sha256"] and os.path.basename(str(sm.get("split_file"))) ==
                  os.path.basename(valid["path"]), f"{sm.get('split')}, {str(sm.get('split_sha256'))[:12]}...")
            check("one row per valid example, in file order", [x["idx"] for x in rows] == list(range(self.n)) and
                  int((sm.get("summary") or {}).get("n", -1)) == self.n, f"{len(rows)} rows")
            if len(rows) != self.n:
                continue
            k = sum(1 for x in rows if x["correct"])
            ent["k"] = k
            ss = sm.get("summary") or {}
            try:
                acc_ok = count_of(float(ss.get("accuracy")), self.n, "accuracy") == k
            except (RecordError, TypeError, ValueError):
                acc_ok = False
            check("summary accuracy = the file's count", acc_ok, f"{ss.get('accuracy')} vs {k} of {self.n}")
            means = {ax: Fraction(sum(x[col] for x in rows), self.n) for ax, col in
                     (("emitted_tokens", "emitted_tokens"), ("latent_steps", "latent_steps"),
                      ("sequential_passes", "sequential_passes"))}
            bad = [ax for ax, m in means.items() if abs(float(ss.get(f"{ax}_mean", -1)) - float(m)) > 1e-9]
            check("summary means = the file's means", not bad, ", ".join(bad))
            nm = sm.get("numerics") or {}
            prec_ok = (sm.get("precision") == "bf16" == cfg.get("precision") and nm.get("tf32_requested") is True
                       and all(nm.get(f) == v for f, v in BF16_FLAGS.items()))
            check("precision bf16 and A10.5's flags (A10.1)", prec_ok,
                  f"precision {sm.get('precision')}, numerics " + ", ".join(f"{f} {nm.get(f)}" for f in BF16_FLAGS))
            if not prec_ok:
                ent["exclusion"] = "precision-mismatched (A10.1: enters no cell; the read is spent)"
            check("registered read = the row", sm.get("registered_read") == row_ts == e.get("registered_read"),
                  f"{sm.get('registered_read')} / {e.get('registered_read')} vs {row_ts}")
            ck = ((r.get("run_done") or {}).get("checkpoints") or {}).get("ckpt_best_final.pt")
            check("checkpoint = the run's selected file (RUN_DONE), named by the row",
                  sm.get("checkpoint_sha256") == e.get("checkpoint_sha256") == ck and bool(ck) and ck in row,
                  f"{str(sm.get('checkpoint_sha256'))[:12]}... / {str(e.get('checkpoint_sha256'))[:12]}... vs "
                  f"{str(ck)[:12]}...")
            m_ = re.search(r"epoch (\d+) \(epoch_index (\d+)", str(e.get("_note", "")))
            sel = r["selected"]
            check("entry names the selected epoch", bool(m_) and int(m_.group(1)) == int(sel["epoch"]) and
                  int(m_.group(2)) == int(sel["epoch_index"]),
                  f"note {m_.group(0) if m_ else None} vs epoch {sel['epoch']} (epoch_index {sel['epoch_index']})")
            want_n = (0 if kind == "zero_latent" else int(kind[1:]) if re.fullmatch(r"n\d+", kind) else r["L"])
            check("summary run, seed, config, intervention, n_latent = the entry",
                  sm.get("run") == e["run"] and int(sm.get("seed", -1)) == 0 and
                  str(sm.get("config", "")).endswith(str(e.get("config"))) and
                  sm.get("intervention") == e.get("intervention", "none") and int(sm.get("n_latent", -1)) == want_n,
                  f"{sm.get('run')}, seed {sm.get('seed')}, {sm.get('intervention')}, n_latent {sm.get('n_latent')}")
            band = (e.get("set") or {}).get("latent_band", cfg.get("latent_band"))
            v = sm.get("variant") or {}
            lf_want = 1.0 if band is None else (band[1] - band[0]) / n_L
            check("variant = the config's band (or the entry's set)",
                  v.get("latent_band") == (None if band is None else list(band)) and
                  abs(float(v.get("layer_fraction", -1)) - lf_want) < 1e-12, f"{v.get('latent_band')} vs {band}")
            if e.get("time_steps"):
                check("timed entry carries a timing block", bool(sm.get("timing")), "")
            if kind == "none" or kind.startswith(("timed_bs", "band_bs")):
                feedback = cfg.get("arm") in ("coconut", "coconut_no_curriculum")
                want = [(r["L"] if feedback else 0, r["L"] if cfg.get("arm") not in ("cot", "no_cot") else 0)]
                got_ = sorted({(x["latent_steps"], x["latent_positions"]) for x in rows})
                check("every example carries the trained count (latent steps, positions)", got_ == want,
                      f"{got_} vs {want}")
            ent["layer_fraction"] = lf_want
            ent["cost_check"] = self._cost_check(r, rows, lf_want)
            check("per-example costs = 5.1 / 5.3 (accounting.inference_record on the hashed valid |q|)",
                  ent["cost_check"]["mismatches"] == 0, f"{ent['cost_check']['mismatches']} of {self.n} rows differ")
        for ent in entries:
            ent["ok"] = not ent["problems"]
        def reads_of(key):                     # the reference record's reads are the first job's (17:36Z ruling 9)
            if hinge and key == self.hinge["reference"]:
                return ((self.ev or {}).get("by_run") or {}).get(key) or {}
            return by_run.get(key) or {}
        directional = {run2key[e["run"]]: self._load_donor(e, d, done, runs[run2key[e["run"]]], row, valid,
                                                           reads_of(run2key[e["run"]])) for e in donor_specs}
        return {"job": job, "entries": entries, "by_run": by_run, "eval_dir": d, "directional": directional}

    def _load_donor(self, e: dict, d: str, done: dict, r: dict, read_row: str, valid: dict, reads: dict) -> dict:
        """One run_diag.py --mode donor_answer record (A6.1 rule 4 (B); critic entry 2026-10-20T17:36Z): its files hashed
        into the provenance and checked (the receipt row; the registering row; the run's selected checkpoint, named by
        the read row or the registering row; the valid split's hash; the registered set's numerics, A10.5; the seeds,
        draws, count and batch; the class pinned by ruling 1, else a refusal; the registered set's statistic recomputed
        from examples.csv and equal to the summary's, outcome word included); then the identification check against
        the job's none / scrambled reads of the run when they enter cells (ruling 9). An entry failing a check enters no
        cell (its arm keeps rule 6's pending label)."""
        name = e["name"]
        cfg = r.get("config") or {}
        ent = {"name": name, "key": r["key"], "run": e["run"], "kind": "donor_answer", "checks": [], "problems": []}

        def check(cname: str, ok: bool, detail: str = ""):
            ent["checks"].append({"check": cname, "ok": bool(ok), "detail": detail})
            if not ok:
                ent["problems"].append(f"{cname}: {detail}" if detail else cname)

        got = done.get(name) or {}
        check("receipt: entry done, rc 0", got.get("status") == "done" and got.get("rc") == 0,
              f"status {got.get('status')}, rc {got.get('rc')}")
        check("the run has a readable record", r["status"] == "read", r.get("exclusion") or r.get("why") or "")
        ed = os.path.join(d, name)
        paths = {f: os.path.join(ed, f) for f in ("summary.json", "examples.csv", "fed_vectors.csv", "eval.log")}
        miss = [f for f, pth in paths.items() if not os.path.isfile(pth)]
        check("summary.json, examples.csv, fed_vectors.csv, eval.log present", not miss, ", ".join(miss))
        if miss or r["status"] != "read":
            ent["ok"] = False
            return ent
        sm = self.inp.read_json(paths["summary.json"], "directional read (donor_answer record)")
        self.inp.sha256(paths["eval.log"], "directional read (log)")
        shas = {f: self.inp.sha256(paths[f], "directional read (per-example files)")
                for f in ("examples.csv", "fed_vectors.csv")}
        log = self.inp.read_text(self.critic_log, "rules (critic log)")
        drow = [ln for ln in log.splitlines() if ln.startswith(f"| {DONOR_ROW} |")]
        check("tool run_diag.py, mode donor_answer, critic_row = the registering row (in critic_log.md), "
              "registered_read null", sm.get("tool") == "run_diag.py" and sm.get("mode") == "donor_answer"
              and sm.get("critic_row") == DONOR_ROW == e.get("critic_row") and bool(drow)
              and sm.get("registered_read") is None,
              f"{sm.get('tool')}, {sm.get('mode')}, critic_row {sm.get('critic_row')} / {e.get('critic_row')}")
        check("files = the summary's hashes", sm.get("files") == shas and sm.get("examples_csv_sha256") ==
              shas["examples.csv"], "")
        check("valid split (hash = the manifest's valid entry)", sm.get("split") == "valid" and
              sm.get("split_sha256") == valid["sha256"] and os.path.basename(str(sm.get("split_file"))) ==
              os.path.basename(valid["path"]) and sm.get("n") == self.n, f"{str(sm.get('split_sha256'))[:12]}...")
        ck = ((r.get("run_done") or {}).get("checkpoints") or {}).get("ckpt_best_final.pt")
        check("checkpoint = the run's selected file (RUN_DONE), named by the read row or the registering row",
              sm.get("checkpoint_sha256") == e.get("checkpoint_sha256") == ck and bool(ck)
              and (ck in read_row or any(ck in ln for ln in drow)),
              f"{str(sm.get('checkpoint_sha256'))[:12]}... vs {str(ck)[:12]}...")
        check("run, seed, config = the entry", sm.get("run") == e["run"] and int(sm.get("seed", -1)) == 0 and
              str(sm.get("config", "")).endswith(str(e.get("config"))), f"{sm.get('run')}, seed {sm.get('seed')}")
        check("scramble seed 0, 10,000 draws from default_rng(0), the trained n_latent and step 3, the eval batch",
              (sm.get("scramble_seed"), sm.get("n_boot"), sm.get("boot_seed"), sm.get("n_latent"),
               sm.get("n_latent_steps"), sm.get("batch_size")) ==
              (0, DONOR_N_BOOT, DONOR_BOOT_SEED, r["L"], 3, int(cfg.get("eval_batch_size", -1))),
              f"{sm.get('scramble_seed')}, {sm.get('n_boot')}, {sm.get('boot_seed')}, n_latent {sm.get('n_latent')}")
        reg = ((sm.get("results") or {}).get("registered") or {})
        nm = (sm.get("numerics") or {}).get("registered") or {}
        prec_ok = (sm.get("precision") == "bf16" == cfg.get("precision") and nm.get("autocast") == "bf16"
                   and nm.get("tf32_requested") is True and all(nm.get(f) == v for f, v in BF16_FLAGS.items()))
        check("the registered set: bf16 autocast over A10.5's flags (the record of record, ruling 6)", prec_ok,
              f"precision {sm.get('precision')}, autocast {nm.get('autocast')}")
        try:
            rows = parse_donor_csv(self.inp.read_text(paths["examples.csv"], "directional read (per-example files)"))
        except (RecordError, ValueError) as ex_:
            check("examples.csv parses", False, str(ex_))
            ent["ok"] = False
            return ent
        rrows = [x for x in rows if x.get("numerics") == "registered"]
        check("one registered row per valid example, in file order", [x["idx"] for x in rrows] == list(range(self.n)),
              f"{len(rrows)} rows")
        if [x["idx"] for x in rrows] != list(range(self.n)):
            ent["ok"] = False
            return ent
        rc_ = donor_recompute(rrows)
        pin = DONOR_PIN.get(self.task)
        if pin is not None and any(rc_[k] != v for k, v in pin.items()):
            raise SystemExit(f"REFUSED: {name}: the class is not the registered one (critic entry {DONOR_ROW} ruling 1: "
                             f"a different count is a refusal): " + ", ".join(f"{k} {rc_[k]} vs {v}"
                                                                              for k, v in pin.items()))
        from latent.eval import scramble_donors
        want = scramble_donors({i: r["L"] for i in range(self.n)}, seed=0)
        check("donors = scramble_donors(seed 0) over the whole set, fed bit for bit, no self donor",
              all(x["donor"] == want[x["idx"]] and x["fed_equals_donor"] and not x["donor_is_self"] for x in rrows),
              "")
        da = reg.get("donor_answer") or {}
        keys = ("n_class", "ambiguous_idx", "n_scored", "k_dir", "k_own", "k_changed", "k_g_chg", "k_g_scr", "k_g_own",
                "k_acc_own", "k_acc_scr", "garbage_exceeds", "garbage_literal_exceeds", "garbage_excess_exceeds")
        diff = [k for k in keys if da.get(k) != rc_[k]]
        if rc_["lo"] is not None:
            diff += [k for k in ("lo", "hi", "pct_lo", "pct_hi", "lo_chg", "hi_chg")
                     if da.get(k) is None or abs(float(da[k]) - rc_[k]) > 1e-9]
            diff += [k for k in ("delta_dir", "r_dir", "r_own", "g_chg") if abs(float(da.get(k, 9)) - float(rc_[k])) >
                     1e-12]
        bug = bool(sm.get("bug"))
        if not bug and rc_["lo"] is not None:
            word = "directional read present" if rc_["present"] else "directional read absent (interval includes 0)"
            if da.get("outcome") not in (word, "held: identification check failed"):
                diff.append(f"outcome {da.get('outcome')!r} vs {word!r}")
        check("the registered set's statistic recomputed from examples.csv = the summary's", not diff, ", ".join(diff))
        ent.update(summary=sm, rows=rrows, recompute=rc_, csv_sha256=shas["examples.csv"], bug=bug,
                   bug_reasons=[b for k_ in ("registered", "strict") for b in ((sm.get("results") or {}).get(k_) or {})
                                .get("bug", [])],
                   tool_identification=da.get("identification") or {"status": "not run"},
                   reference=sm.get("reference"))
        nr, sr = reads.get("none"), reads.get("scrambled")
        if nr is not None and sr is not None and nr.get("ok") and sr.get("ok"):
            try:
                ent["table_identification"] = {**donor_identification(rrows, nr["rows"], sr["rows"]),
                                                "entries": [nr["name"], sr["name"]],
                                                "by": "read_screen.py from the job's committed reads"}
            except RecordError as ex_:
                check("identification: the reads list the record's examples", False, str(ex_))
        ent["ok"] = not ent["problems"]
        return ent

    def _cost_check(self, rec: dict, rows: list[dict], lf: float) -> dict:
        """Recompute every row's prefill tokens, passes, FLOPs and reference FLOPs (5.1, 5.3, 5.5 (c)) from the hashed
        valid file's |q| and the row's own L and E'; count rows that differ from the file."""
        from latent.accounting import inference_record
        arm = rec["config"]["arm"]
        acc_arm = arm if arm in ("cot", "no_cot", "pause") else "coconut"
        d = self.dims(rec["model_id"], V_BASE if arm in ("cot", "no_cot") else V_LATENT)
        q = self.q_lens()
        bad = 0
        for i, x in enumerate(rows):
            if arm == "pause":
                ir = inference_record(d, "pause", q[i], x["latent_positions"], x["emitted_tokens"])
            else:
                kw = {"layer_fraction": lf} if (lf != 1.0 and x["latent_steps"]) else {}
                ir = inference_record(d, acc_arm, q[i], x["latent_steps"], x["emitted_tokens"], **kw)
            if (ir.prefill_tokens != x["prefill_tokens"] or ir.sequential_passes != x["sequential_passes"] or
                    abs(ir.flops - x["flops"]) > 1e-6 * max(1.0, ir.flops) or
                    abs(ir.reference_flops - x["reference_flops"]) > 1e-6 * max(1.0, ir.reference_flops)):
                bad += 1
        return {"mismatches": bad, "n": len(rows)}

    def costs_exact(self, rec: dict, ent: dict, layer_fraction: float = 1.0) -> dict:
        """Per-run cost summary from a read's per-example file (A6.4, 5.3): exact means of E', passes, FLOPs and
        reference FLOPs over the valid examples, the FLOPs recomputed by inference_record at `layer_fraction`
        (C1b: 1, the nominal (b - a) / L, or the measured min(1, f_meas)). Same keys as costs(); ranges have lo = hi."""
        from latent.accounting import inference_record
        cfg = rec["config"]
        arm = cfg["arm"]
        acc_arm = arm if arm in ("cot", "no_cot", "pause") else "coconut"
        V = V_BASE if arm in ("cot", "no_cot") else V_LATENT
        d = self.dims(rec["model_id"], V)
        q = self.q_lens()
        fl, rf, used = [], [], set()
        for i, x in enumerate(ent["rows"]):
            if arm == "pause":
                ir = inference_record(d, "pause", q[i], x["latent_positions"], x["emitted_tokens"])
            else:
                kw = {"layer_fraction": layer_fraction} if (layer_fraction != 1.0 and x["latent_steps"]) else {}
                ir = inference_record(d, acc_arm, q[i], x["latent_steps"], x["emitted_tokens"], **kw)
            fl.append(ir.flops)
            rf.append(ir.reference_flops)
            used.add(x["latent_steps"])
        n = len(fl)
        M = sum(x["emitted_tokens"] for x in ent["rows"])
        lat = sum(x["latent_steps"] for x in ent["rows"])
        f_m, r_m = math.fsum(fl) / n, math.fsum(rf) / n
        eff = float(Fraction(layer_fraction).limit_denominator(1000)) if layer_fraction != 1.0 else 1.0
        return {"layer_fraction": layer_fraction, "layer_fraction_charged": eff, "V": V, "N_ne": d.N_ne,
                "L": rec["L"], "emitted_total": M, "emitted_mean": M / n, "passes_mean": 1 + lat / n + M / n,
                "flops_mean": (f_m, f_m), "reference_flops_mean": (r_m, r_m), "q_mean": sum(q) / n, "n": n,
                "exact": True, "entry": ent["name"]}

    def eval_stats(self, R: dict) -> None:
        """Sigma-free statistics of the eval reads, per run: the 6.2 paired bootstrap for K2 (none vs scrambled) and
        the zero-latent drop (none vs zero_latent); the sweep; n = 0 vs zero_latent example by example; A6.2's timed
        reads and f_meas; rule 3's like-for-like counts."""
        n = self.n
        n_L = self.dims("openai-community/gpt2", V_LATENT).n_L
        for key, r in R.items():
            ents = {k: e for k, e in (self.ev["by_run"].get(key) or {}).items() if e.get("ok")}
            st = {"reads": {k: {"entry": e["name"], "k": e["k"], "points": float(pts(e["k"], n))}
                            for k, e in ents.items()}}
            r["eval_stats"] = st
            r["eval_ents"] = ents
            bad = {k: e["problems"] for k, e in (self.ev["by_run"].get(key) or {}).items() if not e.get("ok")}
            if bad:
                st["excluded"] = bad
            if "none" in ents:
                nr = ents["none"]
                st["none_vs_log"] = {"k_eval": nr["k"], "k_log": r["k"], "M_eval": sum(x["emitted_tokens"]
                                                                                         for x in nr["rows"]),
                                     "M_log": r["M"], "agree": nr["k"] == r["k"]}
            for kind, tag in (("scrambled", "k2"), ("zero_latent", "zero_drop")):
                if "none" in ents and kind in ents:
                    a = [x["correct"] for x in ents["none"]["rows"]]
                    b = [x["correct"] for x in ents[kind]["rows"]]
                    st[tag] = paired_bootstrap(a, b)
                    st[tag]["entries"] = [ents["none"]["name"], ents[kind]["name"]]
            if "n0" in ents and "zero_latent" in ents:
                diff = rows_differ(ents["n0"]["rows"], ents["zero_latent"]["rows"])
                st["n0_zero"] = {"entries": [ents["n0"]["name"], ents["zero_latent"]["name"]],
                                 "examples_differing": len({i for i, _ in diff}), "fields": sorted({f for _, f in diff}),
                                 "agree": not diff}
            sweep = {}
            for nn in sorted(set(SWEEP) | {r["L"]}):
                kind = "none" if nn == r["L"] else f"n{nn}"
                if kind in ents and nn in SWEEP + (r["L"],):
                    e = ents[kind]
                    sweep[nn] = {"entry": e["name"], "k": e["k"], "points": float(pts(e["k"], n)),
                                 "plain_read": kind == "none", "ood": nn > r["L"],
                                 "emitted_mean": sum(x["emitted_tokens"] for x in e["rows"]) / n,
                                 "passes_mean": sum(x["sequential_passes"] for x in e["rows"]) / n,
                                 "flops_mean": math.fsum(x["flops"] for x in e["rows"]) / n}
            if sweep:
                st["sweep"] = sweep
            timing = {}
            for kind, e in ents.items():
                t = (e["summary"] or {}).get("timing")
                if not t:
                    continue
                batches = int(t["prefill_passes"])
                timing[kind] = {"entry": e["name"], "batch": e["batch"], "batches": batches,
                                "batches_expected": -(-n // e["batch"]), "prefill_ms": t.get("prefill_ms_per_pass"),
                                "latent_ms": t.get("latent_ms_per_pass"), "latent_passes": int(t["latent_passes"]),
                                "latent_per_batch": int(t["latent_passes"]) / batches if batches else None,
                                "token_ms": t.get("token_ms_per_pass"), "token_passes": int(t["token_passes"]),
                                "latent_last_ms": t.get("latent_last_ms_per_pass"),
                                "device": e["summary"].get("gpu_type"), "torch": e["summary"].get("torch"),
                                "eval_seconds": e["summary"].get("eval_seconds"),
                                "latent_band": (e["summary"].get("variant") or {}).get("latent_band")}
            if timing:
                st["timing"] = timing
        co = R.get("coconut")
        for key, r in R.items():
            band = r["config"].get("latent_band")
            if not band or co is None:
                continue
            fm = {}
            for tag, num_kind in (("bs1", "timed_bs1"), (f"bs{r['config']['eval_batch_size']}", "none")):
                num = (r.get("eval_stats") or {}).get("timing", {}).get(num_kind)
                den = (co.get("eval_stats") or {}).get("timing", {}).get(f"band_{tag}")
                why = []
                if num is None:
                    why.append(f"no timed {num_kind} read of {r['run']}")
                if den is None:
                    why.append(f"no timed band [0, {n_L}] read of {co['run']} at batch {tag[2:]}")
                if not why:
                    if den["latent_band"] != [0, n_L]:
                        why.append("the denominator is not the band [0, L] path (A6.2 rule 1: never the HF path)")
                    if num["latent_band"] != list(band):
                        why.append("the numerator is not the arm's own band")
                    if num["batch"] != den["batch"] or num["batch"] != int(tag[2:]):
                        why.append("batch sizes differ")
                    if (num["device"], num["torch"]) != (den["device"], den["torch"]):
                        why.append("device or torch build differs")
                    for t_ in (num, den):
                        if t_["batches"] != t_["batches_expected"] or t_["latent_passes"] % t_["batches"]:
                            why.append(f"{t_['entry']}: timed batches or latent passes per batch inconsistent")
                if why:
                    fm[tag] = {"untested": "; ".join(why)}
                    continue
                f = num["latent_ms"] / den["latent_ms"]
                fm[tag] = {**f_meas_reading(f, band, n_L), "num": num, "den": den, "batch": num["batch"]}
            r["eval_stats"]["f_meas"] = fm
        for key, r in R.items():                   # A6.1 rule 4 (B): the donor-answer record's reading (17:36Z)
            rd = self.directional_reading(key)
            if rd is not None and self.fourth and key not in self.hinge["keys"]:
                self.ref_directional = {**rd, "run": r["run"]}       # 17:36Z (5): the reference, in the notes only
            elif rd is not None:
                r["eval_stats"]["directional"] = rd
        if co is not None and "none" in co.get("eval_ents", {}):
            hf = co["eval_ents"]["none"]
            r3 = {}
            for kind, e in co["eval_ents"].items():
                if not kind.startswith("band_bs"):
                    continue
                dc = [i for i, f in rows_differ(hf["rows"], e["rows"], ["correct"])]
                de = [i for i, f in rows_differ(hf["rows"], e["rows"], ["emitted_tokens"])]
                r3[kind] = {"entry": e["name"], "hf_entry": hf["name"], "batch": e["batch"], "hf_batch": hf["batch"],
                            "k": e["k"], "k_hf": hf["k"], "correct_differ": dc, "emitted_differ": len(de),
                            "net": e["k"] - hf["k"], "over_line": len(dc) > RULE3_MAX_DIFF}
            co["eval_stats"]["rule3"] = r3

    # ------------------------------------------------------------- A6.1 rule 4 (B): the donor-answer read (GSM-50k)
    def directional_reading(self, key: str) -> dict | None:
        """The run's donor-answer record read as the fourth table reads it (critic entry 2026-10-20T17:36Z): None without
        a record; status "excluded" (a check failed), "BUG" (the record's bug list), "held" (identification: any
        difference, by the tool or by this tool from the job's reads), "not run" (no identification anywhere), else
        "present" / "absent" by the pass line on this tool's recomputation, with every count and interval."""
        ent = ((self.ev or {}).get("directional") or {}).get(key)
        if ent is None:
            return None
        base = {"entry": ent["name"], "reference": ent.get("reference"),
                "gpu_type": (ent.get("summary") or {}).get("gpu_type")}
        if not ent.get("ok"):
            return {**base, "status": "excluded", "why": "; ".join(ent["problems"])}
        if ent["bug"]:
            return {**base, "status": "BUG", "why": "; ".join(ent["bug_reasons"]) or "bug true in the record"}
        ti, tl = ent.get("table_identification"), ent["tool_identification"]
        held = [i for i in (ti, tl) if i and i.get("status") == "held"]
        ident = held[0] if held else (ti or tl)
        if held:
            return {**base, "status": "held", "identification": ident}
        if (ident or {}).get("status") != "passed":
            return {**base, "status": "not run", "identification": ident,
                    "why": "no none / scrambled read of the checkpoint entered the table and the tool did not run it"}
        rc = ent["recompute"]
        return {**base, "status": "present" if rc["present"] else "absent", "identification": ident,
                "thin": rc["k_net"] < DONOR_THIN, **rc}

    def _drop_diff(self, a: dict, b: dict) -> dict | None:
        """Rule 2's C3 minus C3-foreign scrambled-drop difference in points, per example ((none - scrambled) of a minus
        that of b), with 6.2's interval on the K2 reads' own draws (n 500, default_rng(0)); None without the reads."""
        ea, eb = a.get("eval_ents") or {}, b.get("eval_ents") or {}
        if not all(k in e for e in (ea, eb) for k in ("none", "scrambled")):
            return None
        v = [100.0 * ((x["correct"] - y["correct"]) - (u["correct"] - w["correct"])) for x, y, u, w in
             zip(ea["none"]["rows"], ea["scrambled"]["rows"], eb["none"]["rows"], eb["scrambled"]["rows"])]
        return values_bootstrap(v)

    def _dir_diff(self, a_key: str, b_key: str) -> dict | None:
        """The C3 minus C3-foreign Delta_dir difference over the common scored set, from the same draws (17:36Z (5);
        diagnostic, decides nothing); None unless both records read present or absent on one scored set."""
        ra = ((self.ev or {}).get("directional") or {}).get(a_key)
        rb = ((self.ev or {}).get("directional") or {}).get(b_key)
        if not (ra and rb and ra.get("ok") and rb.get("ok")):
            return None
        sa = [x for x in ra["rows"] if x["scored"]]
        sb = {x["idx"]: x for x in rb["rows"] if x["scored"]}
        if [x["idx"] for x in sa] != sorted(sb) or len(sa) < 2:
            return None
        return values_bootstrap([(x["d_scr"] - x["d_own"]) - (sb[x["idx"]]["d_scr"] - sb[x["idx"]]["d_own"])
                                 for x in sa], DONOR_N_BOOT, DONOR_BOOT_SEED)

    def _a61(self, r: dict, R: dict, k2: dict, sigma: float) -> dict:
        """A dep_loss arm's K2 line and card label (A6.1 rules 1, 2, 5, 6; critic entry 2026-10-20T17:36Z (5))."""
        n = self.n
        rd = (r.get("eval_stats") or {}).get("directional")
        st = a61_state(k2, sigma, rd)
        drop = (f"Delta_scr = {f1(pts(k2['k_a'], n))} - {f1(pts(k2['k_b'], n))} = {float(k2['delta']):+.1f} "
                f"({k2['k_a']} - {k2['k_b']} of {n}; {k2['entries'][0]} vs {k2['entries'][1]}) "
                f"[{k2['lo']:+.2f}, {k2['hi']:+.2f}] (6.2: Delta +- 1.96 sqrt(Var_ex), {k2['n_boot']:,} paired example "
                f"draws, default_rng({k2['seed']}); percentile [{k2['pct_lo']:+.1f}, {k2['pct_hi']:+.1f}])")
        head = f"K2 (7.5; A3.4; A6.1 rules 1, 5 and 6), {r['run']}: "
        r2 = ""
        fo = R.get("c3_foreign")
        if r["key"] != "c3_foreign" and fo is not None:
            dd = self._drop_diff(r, fo)
            if dd is not None:
                r2 = (f"; {A61['rule2']} (A6.1 rule 2): {r['run']} - {fo['run']} scrambled-drop difference "
                      f"{dd['mean']:+.1f} [{dd['lo']:+.2f}, {dd['hi']:+.2f}] (the same {N_BOOT:,} draws)")
        form = (f"{A61['k2_form']} {float(k2['delta']):+.1f} [{k2['lo']:+.2f}, {k2['hi']:+.2f}] against sigma_screen "
                f"{f2(sigma)}: ")
        killed = (f"; the arm's latent claim is killed at the screen on {self.spec['title']} (A3.4's every-latent-arm "
                  f"rule read with A6.1 rules 1, 5 and 6); it continues only as a labelled effect and cannot be A1's "
                  f"positive read on this task")
        zd = (r.get("eval_stats") or {}).get("zero_drop")
        r3 = ("" if zd is None else
              f"; beside it, a diagnostic that decides nothing for this arm (A6.1 rule 3; section 10): zero-latent drop "
              f"{f1(pts(zd['k_a'], n))} - {f1(pts(zd['k_b'], n))} = {float(zd['delta']):+.1f} [{zd['lo']:+.2f}, "
              f"{zd['hi']:+.2f}] ({zd['entries'][1]}; the sweep's n = 0 read is the same construction; no pause-mode read "
              f"of this checkpoint exists; a collapse is reported as such, not killed: A3.4's degenerate-solution note)")
        if st["state"] == "pending":
            label = pending_label(rd)
            return {**st, "label": label, "outcome": label, "verify": [],
                    "text": head + label + f": the scrambled drop {drop} is printed and decides nothing; no K2 "
                            f"sentence and no verdict naming the arm until the directional read exists on this "
                            f"checkpoint (A6.1 rule 6 (a) to (c); the card's pass and kill lines apply as registered)"
                            + r2 + r3}
        sent = directional_sentence(rd)
        blk = []
        if self.fourth:                            # 17:36Z (2): the label block on every number of the read
            dl, blk = self._dlabel(r, rd)
            sent += f" [{dl}]"
            rd["label_block"] = dl
        what = f"the {r['run']} label of critic entry {DONOR_ROW} (5)'s form"
        if st["state"] == "present":
            return {**st, "label": present_label(rd), "outcome": f"not fired; {A61['present']}",
                    "verify": blk + [(present_label(rd), what),
                                     (form + "not fired", f"the {r['run']} K2 line (17:36Z (5))")],
                    "text": head + form + f"not fired ({drop}; Delta_scr >= sigma_screen and the interval excludes "
                            f"0)" + r2 + f"; the non-circular evidence of rule 4 (B): {sent}; still no verdict (6.3; "
                            f"K2 stays not independent, rule 1; rule 7's positive control is still required for any "
                            f"readout label)" + r3}
        if st["state"] == "fired":
            if st["direction"]:
                return {**st, "label": present_label(rd), "outcome": f"fired; {A61['present']}",
                        "verify": blk + [(present_label(rd), what),
                                         (form + "fired", f"the {r['run']} K2 line (17:36Z (5))")],
                        "text": head + form + f"fired ({'; '.join(st['why'])}; {drop}): a fired K2 still kills (A6.1 "
                                f"rule 1)" + killed + r2 + f"; beside it: {sent}" + r3}
            return {**st, "label": A61["k2_drop"], "outcome": A61["k2_drop"], "verify": blk,
                    "text": head + A61["k2_drop"] + f": fired ({'; '.join(st['why'])}; {drop})" + killed + r2
                            + f"; beside it: {sent}" + r3}
        return {**st, "label": A61["fingerprint"], "outcome": A61["k2_rule5"], "verify": blk,
                "text": head + A61["k2_rule5"] + f"; the drop {drop} does not fire 7.5's K2 (the large scrambled drop "
                        f"of rule 5, critic entry {DONOR_ROW} (2)) and its " + sent[0].lower() + sent[1:] + killed + r2
                        + r3}

    def _dlabel(self, r: dict, rd: dict) -> tuple[str, list]:
        """17:36Z (2)'s label block for one directional record (fourth table): the form with the record's counts and the
        run's training GPU must be in the registering row verbatim (its <gpu_type> slot unfilled); the filled slot
        ("read on <record GPU>, trained on <training GPU>") must be quoted in a critic entry logged at or after it (the
        hinge read row 11:07Z fills it). Returns (the filled label, its verification items)."""
        g_read, g_train = _gpu_short(rd.get("gpu_type")), _gpu_short(r.get("gpu_type"))
        form = DONOR_LABEL_FORM.format(n=rd.get("n"), s=rd.get("n_scored"), t=g_train)
        frag = f"read on {g_read}, trained on {g_train}"
        return form.replace("<gpu_type>", g_read), [
            (form, f"17:36Z (2)'s label block of {r['run']}'s directional read", DONOR_ROW, False),
            (frag, f"the read GPU of {r['run']}'s directional read (17:36Z (2)'s <gpu_type>)", DONOR_ROW, True)]

    def _c3_card_eval(self, key: str, r: dict, R: dict, sigma: float, lab: str, other: str) -> dict:
        """A3.4's C3 lines (C2+C3 the same, plus its C2-side line against C2-shuffled) with the eval reads, under A6.1:
        the label of 17:36Z (5), the pass parts (one task) and this task's kill parts (each needs both tasks)."""
        n, S = self.n, Fraction(sigma)
        name = "C3" if key == "c3" else "C2+C3"
        coco, fo, sh = R.get("coconut"), R.get("c3_foreign"), R.get("c2_shuffled")
        k2 = (r.get("eval_stats") or {}).get("k2")
        a = r.get("_a61")
        if k2 is None or a is None or coco is None:
            why = "no none / scrambled read entered the table" if k2 is None else "no full-depth coconut record"
            return {"id": f"card:{key}", "kind": "card", "run": r["run"], "outcome": "UNTESTED",
                    "text": f"{name} (card line, A3.4), {r['run']}: UNTESTED: {why}; {A61['pending']} {lab}"}
        drop = Fraction(k2["delta"])
        x = pts(r["k"], n) - pts(coco["k"], n)
        fk2 = (fo or {}).get("eval_stats", {}).get("k2") if fo else None
        fdrop = Fraction(fk2["delta"]) if fk2 else None

        def m(v):
            return "pending (no C3-foreign K2 read)" if v is None else ("met" if v else "not met")
        p_drop, p_acc = drop > 10, x >= -S
        p_fo = None if fdrop is None else drop - fdrop > S
        k_drop, k_acc = drop < 5, x < -2 * S
        k_fo = None if fdrop is None else abs(drop - fdrop) <= S
        parts_ok = p_drop and p_acc and p_fo is True
        if not parts_ok:
            po = "not met on this task" if (p_fo is not None or not (p_drop and p_acc)) else "pending (C3-foreign)"
        elif a["state"] == "present":
            po = "met on this task (no verdict: a screen filters, 6.3; K2 stays not independent, A6.1 rule 1)"
        elif a["state"] == "pending":
            po = "pending the directional read (A6.1 rule 6)"
        elif a["state"] == "fired":
            po = "not met: K2 fired on the drop leg"
        else:
            po = "not met: K2 counts as fired (rule 5)"
        c2 = ""
        if key == "c2c3" and sh is not None:
            y = pts(r["k"], n) - pts(sh["k"], n)
            c2 = (f"; C2-side line (A3.4): {r['run']} - {sh['run']} = {float(y):+.1f}: above C2-shuffled by more than "
                  f"sigma_screen {'met' if y > S else 'not met'} (the gap includes learnable vs unlearnable auxiliary "
                  f"target)" + (f" ({self.spec['c2_sentences']['c2'].split(';')[0]})" if self.fourth else ""))
        bar = R.get(self._bar_key) if self._bar_key else None
        leg = ""
        if self.fourth and bar is not None:     # A6.1 rule 4 (A): the accuracy leg, as the card's lines require
            leg = (f"; accuracy leg (A6.1 rule 4 (A)): {f1(pts(r['k'], n))} vs full-depth coconut {f1(pts(coco['k'], n))} "
                   f"({float(x):+.1f})" + (f", vs C3-foreign {f1(pts(fo['k'], n))} "
                                          f"({float(pts(r['k'], n) - pts(fo['k'], n)):+.1f})" if fo is not None else "")
                   + f", vs the bar {f1(pts(bar['k'], n))} ({float(pts(r['k'], n) - pts(bar['k'], n)):+.1f}) at "
                     f"sigma_screen {f2(sigma)}")
        dd = self._dir_diff(key, "c3_foreign") if key != "c3_foreign" else None
        diag_ = (f"; {r['run']} minus {fo['run']} Delta_dir {dd['mean']:+.3f} [{dd['lo']:+.3f}, {dd['hi']:+.3f}] over "
                 f"the common scored set (the same draws; diagnostic, decides nothing)") if dd and fo else ""
        killed = (f"; the arm's latent claim is killed at the screen on {self.spec['title']} by K2 ({a['k2']})"
                  if a["k2"] in ("fired", "counts as fired (rule 5)") else "")
        rdr = (r.get("eval_stats") or {}).get("directional") or {}
        blk = (f" [{self._dlabel(r, rdr)[0]}]" if self.fourth and rdr.get("status") in ("present", "absent") else "")
        if self.fourth and coco is not None and R.get("c2") is not None and key == "c2c3":
            c2 = c2 + (f"; attribution set (A3.4): {r['run']} - {R['c2']['run']} = "
                       f"{float(pts(r['k'], n) - pts(R['c2']['k'], n)):+.1f}" + (
                           f", {r['run']} - {R['c3']['run']} = {float(pts(r['k'], n) - pts(R['c3']['k'], n)):+.1f}"
                           if R.get("c3") is not None else "")
                       + f", {r['run']} - {coco['run']} = {float(x):+.1f}")
        return {"id": f"card:{key}", "kind": "card", "run": r["run"],
                "outcome": f"{a['label'] if a['state'] != 'present' else A61['present']}; pass: {po}; kill parts "
                           f"(both tasks needed): drop {m(k_drop)}, accuracy {m(k_acc)}, C3-foreign {m(k_fo)}",
                "text": f"{name} (card line, A3.4), {r['run']}: {a['label']}; scrambled drop {float(drop):+.1f} "
                        + (f"vs C3-foreign's {float(fdrop):+.1f}" if fdrop is not None else "(C3-foreign: no K2 read)")
                        + f"; accuracy vs full-depth coconut {float(x):+.1f}: pass needs (one task) a drop above 10 "
                        f"with accuracy within sigma_screen {f2(sigma)} of coconut or better and the drop exceeding "
                        f"C3-foreign's by more than sigma_screen: drop part {m(p_drop)}, accuracy part {m(p_acc)}, "
                        f"C3-foreign part {m(p_fo)}: pass {po}{killed}; kill needs both tasks (drop under 5; accuracy "
                        f"below coconut by more than 2 sigma_screen = {f2(2 * sigma)}; within sigma_screen of "
                        f"C3-foreign on the drop): this task's parts {m(k_drop)}, {m(k_acc)}, {m(k_fo)}; ProsQA-hard "
                        f"UNTESTED (A2.1): no both-tasks clause can be met as written (critic entry 2026-10-20T17:12Z (8))"
                        f"{leg}{c2}{diag_}{blk}; {other} {lab}"}

    def _a61_labels_check(self) -> None:
        """The table refuses a directional label not verbatim in the registering row (critic entry 2026-10-20T17:36Z
        (2): "labels of the table are verbatim from this row")."""
        log = self.inp.read_text(self.critic_log, "rules (critic log)")
        rows = [ln for ln in log.splitlines() if ln.startswith(f"| {DONOR_ROW} |")]
        if not rows:
            raise SystemExit(f"REFUSED: no critic_log.md row {DONOR_ROW} (the directional read's lines)")
        miss = [v for v in A61.values() if not any(v in ln for ln in rows)]
        if miss:
            raise SystemExit(f"REFUSED: the directional labels {miss} are not verbatim in critic entry {DONOR_ROW}")

    # ---------------------------------------------------------------------------------------------- the table
    def build(self) -> dict:
        if self.pd_dir:
            both = [x for x in self.spec["plain_depth_all"] if os.path.isdir(os.path.join(self.records, "runs", x))]
            if both:
                raise SystemExit(f"REFUSED: {both} also under --records: one source per record (--plain-depth-dir)")
        if self.h_dir:
            both = [x for x in self.h_runs if os.path.isdir(os.path.join(self.records, "runs", x))]
            if both:
                raise SystemExit(f"REFUSED: {both} also under --records: one source per record (--hinge-records)")
        if self.x_dir:
            both = [x for x in self.x_runs if os.path.isdir(os.path.join(self.records, "runs", x))
                    or (self.h_dir and os.path.isdir(os.path.join(self.h_dir, "runs", x)))]
            if both:
                raise SystemExit(f"REFUSED: {both} also under --records or --hinge-records: one source per record "
                                 f"(--extra-records)")
            miss = [x for x in self.x_runs if not os.path.isfile(self._path(x, "RUN_DONE.json"))]
            if miss:
                raise SystemExit(f"REFUSED: {miss}: no RUN_DONE.json under {self.x_dir}/runs: the reissue reads the arm "
                                 f"(critic entry {CLOSING_ROW} (b) (5)); nothing written")
        receipts = self.load_receipts()
        runs = {s["key"]: self.load_run(s) for s in self.spec["runs"]}
        receipts += self.pd_receipts(runs)
        receipts += self.hinge_receipts()
        receipts += self.extra_receipts()
        readable = {k: r for k, r in runs.items() if r["status"] == "read"}
        trials = [(k, r) for k, r in readable.items() if k.startswith("cot_lr")]
        if not trials:
            raise SystemExit("no bar trial has a readable record: no screen table")
        lr = {k: float(r["config"]["lr"]) for k, r in trials}
        bar_key = max(trials, key=lambda kr: (kr[1]["k"], -lr[kr[0]]))[0]          # 3.4: ties to the smaller LR
        if len(trials) != 3:
            raise SystemExit(f"{len(trials)} readable bar trials; A3.3's screen bar is 3 LR trials (trial parity)")
        bar = readable[bar_key]
        p = bar["k"] / self.n
        binom = 100 * math.sqrt(p * (1 - p) / self.n)
        sd = self._prosqa_cot_sd()
        readings = {"registered": max([1.0, binom] + ([sd] if sd is not None else [])),
                    "without the ProsQA cot SD term": max(1.0, binom), "floor only": 1.0}
        if self.eval_dir:
            self.ev = self.load_evals(runs)
            if self.fourth:                        # the hinge eval job's reads join the first job's per run
                self.ev_h = self.load_evals(runs, self.h_eval_dir, hinge=True)
                for k, v in self.ev_h["by_run"].items():
                    self.ev["by_run"][k] = v
                self.ev["directional"] = dict(self.ev_h["directional"])
            if self.extra:                         # the closing arm's own eval job joins per run
                self.ev_x = self.load_evals(runs, self.x_eval_dir, closing=True)
                for k, v in self.ev_x["by_run"].items():
                    self.ev["by_run"][k] = v
            self.eval_stats(readable)
        if self.third:
            self.labels3 = self.third_labels(readable)
        for k, r in readable.items():
            r["cost"] = self.costs(r)
            if r["config"].get("latent_band"):
                a, b = r["config"]["latent_band"]
                n_L = self.dims(r["model_id"], V_LATENT).n_L
                r["cost_nominal"] = self.costs(r, layer_fraction=(b - a) / n_L)
                r["n_L"] = n_L
            nr = (r.get("eval_ents") or {}).get("none")
            if nr is not None:                     # A6.4 / 5.3: the per-example file makes the means exact
                r["cost_range"], r["M_log"] = r["cost"], r["M"]
                r["M"] = sum(x["emitted_tokens"] for x in nr["rows"])
                r["cost"] = self.costs_exact(r, nr)
                if r["config"].get("latent_band"):
                    r["cost_nominal_range"] = r["cost_nominal"]
                    r["cost_nominal"] = self.costs_exact(r, nr, layer_fraction=(b - a) / n_L)
                    fm = (r["eval_stats"].get("f_meas") or {}).get("bs1") or {}
                    if "f_meas" in fm:             # A6.2 rule 4: K3 and 7.2 at min(1, f_meas) when nominal
                        r["cost_measured"] = self.costs_exact(r, nr, layer_fraction=fm["measured_fraction"])
        plain = self._plain_depth(runs)
        if self.third:
            if not plain["usable"] or "points" not in plain:
                raise SystemExit(f"REFUSED: the third table needs the 355M point (A13.2: both plain-depth runs done with "
                                 f"a selection, the GSM run read): {plain['runs']}, "
                                 f"{runs['plain_depth_355m'].get('exclusion') or runs['plain_depth_355m'].get('why')}")
            if not (self.pd_probe or {}).get("label") or (self.pd_probe or {}).get("problems"):
                raise SystemExit(f"REFUSED: the control's memory-probe label is not shown by its receipts: {self.pd_probe}")
            self._row_has(ROW_PD, plain["label"], "the control's label (A13.2 (b); deviations 11:05Z)")
        if plain.get("points"):
            runs["plain_depth_355m"]["label"] = plain["label"]
            plain["arms"] = {k: self._apd_eval(r, plain) for k, r in readable.items() if r.get("latent")}
        self._bar_key = bar_key
        self._sigma_reg = readings["registered"]
        lines = {name: self.lines(runs, bar_key, s, plain) for name, s in readings.items()}
        if self.fourth:                            # labels whose form carries numbers: verbatim in a later critic entry
            for item in self._verify.get(readings["registered"], []):
                text, what, ts, later = item if len(item) == 4 else (*item, DONOR_ROW, True)
                self._row_has(ts, text, what, later=later)
        for text, what, ts, later in self._x_verify:   # the closing arm's labels (critic entry 2026-10-21T20:21Z (b))
            self._row_has(ts, text, what, later=later)
        for k, a in (plain.get("arms") or {}).items():         # the registered reading's K3 outcome per arm
            a["k3"] = next((ln["outcome"] for ln in lines["registered"] if ln["id"] == f"k3:{k}"), None)
        if self.eval_dir:
            for ls in lines.values():              # 10:51Z part (6): every line of the second table carries it
                for ln in ls:
                    if EVAL_LABEL not in ln["text"]:
                        ln["text"] += f" [{EVAL_LABEL}]"
        if self.fourth:                            # 11:07Z (1)(f): the label on every line of the three arms
            hl = self._hinge_labels(runs)
            for ls in lines.values():
                for ln in ls:
                    if ln.get("run") in self.h_runs:
                        ln["text"] += f" [{hl}]"
        sens = self._sensitivity(lines)
        table = {
            "task": self.task, "title": self.spec["title"], "data_sentence": self.spec["data_sentence"],
            "labels": [SCREEN_LABEL, BAR_LABEL, BAR_VALUE_LABEL, "valid split only (A3.4); no test read",
                       "bf16 autocast over the TF32 flags (A10.5)"],
            "n_valid": self.n, "bar": {"run": bar["run"], "key": bar_key, "k": bar["k"],
                                       "points": float(pts(bar["k"], self.n)), "epoch": int(bar["selected"]["epoch"])},
            "sigma_screen": {"registered": readings["registered"], "terms": {"floor": 1.0, "binomial": binom,
                                                                            "prosqa_cot_3seed_sd": sd},
                             "alternatives": {k: v for k, v in readings.items() if k != "registered"}},
            "runs": [self._run_json(runs[s["key"]]) for s in self.spec["runs"]],
            "plain_depth": plain, "lines": lines["registered"], "sensitivity": sens,
            "receipts": self._receipt_checks(receipts, runs), "notes": self._notes(),
        }
        if self.ev is not None:
            dep_read = [k for k, r in readable.items() if r["config"].get("dep_loss") and r.get("_a61")]
            if dep_read or self.ev.get("directional"):
                self._a61_labels_check()
            ref = (self.ref_directional if self.fourth else
                   (readable.get("coconut") or {}).get("eval_stats", {}).get("directional"))
            ref_note = None
            if ref is not None and self.fourth:    # 17:36Z (2)'s block; 11:07Z (3)'s identification across GPU types
                ref = self.ref_directional = self._reference_labels(readable["coconut"], ref)
            if ref is not None:
                ref_note = (
                    f"{A61['reference']} (critic entry {DONOR_ROW} ruling 8; it enters no C3 or C2+C3 line and no K2 "
                    f"cell): {readable['coconut']['run']}: " + (directional_sentence(ref) if ref["status"] in
                                                                 ("present", "absent") else
                                                                 f"no reading ({ref['status']}: "
                                                                 f"{ref.get('why') or ref.get('identification')})")
                    + (f" [{ref['label_block']}]" if ref.get("label_block") else ""))
                if not self.fourth:                # the fourth table appends its notes after the third's
                    table["notes"].append(ref_note)
            table["labels"] = [EVAL_LABEL] + table["labels"]
            table["eval"] = self._eval_json(runs)
            if self.third:
                table["timed_read_labels"] = self.labels3["timed"]
            sig = readings["registered"]
            alt = []
            for s_ in self.spec["runs"]:
                k2 = (runs[s_["key"]].get("eval_stats") or {}).get("k2")
                if k2:
                    w = 1.96 * math.sqrt(k2["se"] ** 2 + 2 * sig ** 2)
                    alt.append(f"{runs[s_['key']]['run']} {float(k2['delta']):+.1f} +- {w:.2f} "
                               f"({'includes' if float(k2['delta']) - w <= 0 <= float(k2['delta']) + w else 'excludes'} 0)")
            if alt:
                table["notes"].append(
                    "Not a registered reading, printed for the critic: 7.5 names K2's interval an example-bootstrap "
                    "interval and the 10:51Z row the 10,000-draw example bootstrap, so 6.2's seed term "
                    "sigma^2 (1/n_A + 1/n_B) is not in it (both reads score one checkpoint); with that term at one seed "
                    f"(+ 2 sigma_screen^2) the intervals would be: {'; '.join(alt)}.")
            if self.fourth:
                table["notes"] += self._fourth_notes(ref_note)
                table["eval_hinge"] = self._eval_hinge_json()
            if self.extra:
                table["notes"] += self._extra_notes(runs)
                table["eval_extra"] = self._eval_extra_json()
        return table

    def _extra_notes(self, runs: dict) -> list[str]:
        """The note the no-curriculum reissue appends (critic entry 2026-10-21T20:21Z (b) (1), (2), (4) to (6))."""
        jx = self.ev_x["job"]
        parts = []
        for key in self.x_keys:
            r = runs[key]
            c = r.get("closing") or {}
            parts.append(f"{r['run']} ({self.x_job}): resumed {r.get('resumes', 0)} times"
                         + (f", pinned stage {c['pinned_stage']}, n latents {c['n_latent']} at every epoch" if c else ""))
        return [
            f"The no-curriculum control (critic entry {CLOSING_ROW} (b); PLAN.md A2.2 item 1 (b)): " + "; ".join(parts)
            + f". Read from its own records (the training project's bucket) by its registered rules: the selection over "
            f"every trained epoch (3.3 item 1; strict >, ties to the earliest; ckpt_best_final.pt = ckpt_best.pt, no "
            f"earlier stage, no secondary read), the latent count of the pinned stage max_latent_stage + 1 at every epoch "
            f"(the log's stage column is the schedule's arithmetic), the start pin of A6.3 rule 4 where the task has one "
            f"(no pool), the receipt of the attempt that completed it. Its eval reads are its own job {jx['job']} "
            f"(attempt {jx['attempt']}, registered row {jx['registered_read']}): the coconut control's ten entries; its "
            f"5.5 (b) line prints its band [0, 12] batch-1 read beside its plain read at the eval batch. It carries the "
            f"coconut control's lines (no 6.3 screen line: a control advances nowhere) and the Curriculum line of row "
            f"(b) (5) with (6)'s zones on the valid numbers of record against sigma_screen of record; no other arm's K2, "
            f"K3, card, pass or screen outcome changes; no test read of this arm (UNTESTED unless a later row registers "
            f"one); nothing here transfers between the two tasks."]

    def _eval_extra_json(self) -> dict:
        """The closing arm's eval job block (its job and read entries, as _eval_hinge_json lists the hinge job's)."""
        xj = self.ev_x
        ents = []
        for e in xj["entries"]:
            sm = e.get("summary") or {}
            ents.append({"name": e["name"], "run": e["run"], "kind": e["kind"], "batch": e["batch"], "ok": e["ok"],
                         "exclusion": e.get("exclusion"), "problems": e["problems"], "checks": e["checks"],
                         "k": e.get("k"), "points": float(pts(e["k"], self.n)) if "k" in e else None,
                         "examples_csv_sha256": e.get("csv_sha256"), "checkpoint_sha256": sm.get("checkpoint_sha256"),
                         "precision": sm.get("precision"), "gpu_type": sm.get("gpu_type"), "torch": sm.get("torch"),
                         "timing": sm.get("timing"), "cost_check": e.get("cost_check")})
        return {"job": xj["job"], "entries": ents}

    def _curriculum(self, R: dict, sigma: float) -> dict:
        """Row (b) (5)'s Curriculum line, (6)'s zones on the valid numbers of record (the selections: A_nc the arm's,
        A_nocot no_cot's, A_coc full-depth coconut's) against sigma_screen of record, on counts (d1 = A_nc - A_nocot,
        d2 = A_coc - A_nc): NEAR NO_COT iff d1 <= sigma and d2 > sigma; NEAR COCONUT iff d2 <= sigma and d1 > sigma;
        BETWEEN iff d1 > sigma and d2 > sigma (signed, critic entry 2026-10-22T03:18Z (3) (ii)); BOTH iff |d1| <= sigma and
        |d2| <= sigma. Exactly one zone holds for every A_nc while A_coc >= A_nocot (the values of record); the refusal of
        a zero or two-zone reading stays as a guard. Below A_nocot - sigma the NEAR NO_COT sentence carries the suffix
        " (below no_cot by <d> points, ...)", above A_coc + sigma the NEAR COCONUT sentence " (above coconut by <d>
        points, ...)" (CURRICULUM_OUTSIDE, verified in CURRICULUM_SIGNED_ROW). Returns the line and its verification items
        (the form and the sentence in CLOSING_ROW; the suffix in CURRICULUM_SIGNED_ROW; the filled line quoted in a critic
        row at or after CLOSING_ROW); the values of record must be row (b) (5)'s."""
        n, S, t = self.n, Fraction(sigma), self.spec["title"]
        out = []
        for key in self.x_keys:
            r, nc, co, pz = R.get(key), R.get("no_cot"), R.get("coconut"), R.get("pause")
            if r is None:                          # the arm enters no cell: lines() prints its UNTESTED line
                continue
            if nc is None or co is None or pz is None:
                miss = [k for k, v in (("no_cot", nc), ("coconut", co), ("pause", pz)) if v is None]
                raise SystemExit(f"REFUSED: the Curriculum line (critic entry {CLOSING_ROW} (b) (5)) needs the selections "
                                 f"of {miss}")
            want = self.extra["values_of_record"][self.task]
            got = (f1(pts(nc["k"], n)), f1(pts(co["k"], n)), f1(pts(pz["k"], n)))
            if got != want:
                raise SystemExit(f"REFUSED: the values of record read now (no_cot, coconut, pause: {got}) are not row "
                                 f"(b) (5)'s {want}")
            a_nc, a_no, a_co = pts(r["k"], n), pts(nc["k"], n), pts(co["k"], n)
            d1, d2 = a_nc - a_no, a_co - a_nc
            z = {"NEAR NO_COT": d1 <= S < d2, "NEAR COCONUT": d2 <= S < d1,
                 "BETWEEN": d1 > S and d2 > S, "BOTH": abs(d1) <= S and abs(d2) <= S}
            hold = [k for k, v in z.items() if v]
            if len(hold) != 1:                     # a guard: with A_coc >= A_nocot exactly one zone holds
                raise SystemExit(f"REFUSED: the Curriculum line, {r['run']}: A_nc {f1(a_nc)} against A_nocot {f1(a_no)} "
                                 f"and A_coc {f1(a_co)} at sigma_screen {f2(sigma)} falls in {hold or 'no zone'} of row "
                                 f"(b) (6) as signed by row {CURRICULUM_SIGNED_ROW} (3) (ii); a critic entry decides the "
                                 f"line")
            zone = hold[0]
            gap = a_co - a_no
            sent_form = CURRICULUM_SENTENCES[zone]
            sent = sent_form.format(gap=f1(gap), ratio=f"{float(gap) / sigma:.1f}") if zone == "BOTH" else sent_form
            out_d = -d1 if zone == "NEAR NO_COT" and d1 < -S else -d2 if zone == "NEAR COCONUT" and d2 < -S else None
            suffix = CURRICULUM_OUTSIDE[zone] if out_d is not None else None
            form = CURRICULUM_FORM
            text = (form.replace("<task>", t, 1).replace("A_nc vs", f"{f1(a_nc)} vs", 1)
                    .replace("no_cot A_nocot (A_nc \u2212 A_nocot)", f"no_cot {f1(a_no)} ({float(d1):+.1f})", 1)
                    .replace("coconut A_coc (A_nc \u2212 A_coc)", f"coconut {f1(a_co)} ({float(-d2):+.1f})", 1)
                    .replace("sigma_screen 3.60", f"sigma_screen {f2(sigma)}", 1)
                    .replace("<zone>; <sentence>", f"{zone}; " + sent.replace("<task>", t)
                             + (suffix.replace("<d>", f1(out_d)) if suffix else ""), 1))
            verify = [(form, f"row (b) (5)'s Curriculum line form", CLOSING_ROW, False),
                      (sent, f"row (b) (6)'s {zone} sentence", CLOSING_ROW, False)]
            if suffix:
                verify.append((suffix, f"row {CURRICULUM_SIGNED_ROW} (3) (ii)'s suffix outside [A_nocot - sigma, A_coc "
                                       f"+ sigma]", CURRICULUM_SIGNED_ROW, False))
            verify.append((text, f"the Curriculum line of {r['run']} as built from the records (row (b) (5)'s form "
                                 f"filled; quoted in a critic entry at or after {CLOSING_ROW})", CLOSING_ROW, True))
            out.append({"key": key, "run": r["run"], "zone": zone, "text": text, "verify": verify,
                        "values": {"A_nc": float(a_nc), "A_nocot": float(a_no), "A_coc": float(a_co),
                                   "A_nc_minus_A_nocot": float(d1), "A_nc_minus_A_coc": float(-d2), "sigma": sigma}})
        return out

    def _hinge_labels(self, runs: dict) -> str:
        """The label on every line of the three hinge arms (critic entry 2026-10-21T11:07Z (1)(f): "resumed 3 / 0 / 2
        times, exact boundary resumes, same GPU type"), built from the records and verified verbatim in the read row
        (or a later critic entry): the resume counts (RUN_DONE, in the HINGE order); "exact boundary resumes" iff every
        run's resumes are ckpt_state's resume entries, no row of record is superseded, the log's resumes column steps
        up exactly after each entry's epoch and RUN_DONE's last resume is ok and exact; "same GPU type" iff every resume
        host is the run's training GPU type. Also refuses a hinge read or record on another GPU type than the run was
        trained on (the read row registers no cross-type label, A7.6) and records each run's labels for section 1."""
        counts, exact, same = [], True, True
        for key in self.hinge["keys"]:
            r = runs[key]
            if "rows" not in r:
                raise SystemExit(f"REFUSED: {r['run']} has no rows of record: the hinge arms' label (critic entry "
                                 f"{self.h_read} (1)(f)) cannot be built")
            rd, res = r.get("run_done") or {}, (r.get("state") or {}).get("resumes") or []
            n, after = int(rd.get("resumes", 0)), sorted(int(x.get("after_epoch", -1)) for x in res)
            counts.append(str(n))
            exact = exact and len(res) == n and r.get("superseded") == 0 and all(
                int(x.get("resumes", 0)) == sum(1 for e in after if e < int(x["epoch"])) for x in r["rows"]) and (
                n == 0 or ((rd.get("resume") or {}).get("ok") is True and (rd.get("resume") or {}).get("exact") is True))
            same = same and all(x.get("gpu_type") == r.get("gpu_type") for x in res)
            g_train = _gpu_short(r.get("gpu_type"))
            ents = list((self.ev_h["by_run"].get(key) or {}).values()) + [
                e for k, e in (self.ev_h.get("directional") or {}).items() if k == key]
            reads = sorted({_gpu_short((e.get("summary") or {}).get("gpu_type")) for e in ents if e.get("summary")})
            if any(g != g_train for g in reads):
                raise SystemExit(f"REFUSED: {r['run']} was trained on {g_train} and read on {reads}: the read row "
                                 f"{self.h_read} registers no cross-type label (A7.6); a cross-type read needs its row")
            sp = r.get("start_pin") or {}
            r["hinge"] = {"trained_on": g_train, "read_on": reads, "start_pin": f"{str(sp.get('load_model_sha256'))[:8]}... "
                                                                             f"epoch {sp.get('load_model_epoch')}"}
        label = (f"resumed {' / '.join(counts)} times, "
                 + ("exact boundary resumes" if exact else "not all exact boundary resumes") + ", "
                 + ("same GPU type" if same else "across GPU types"))
        self._row_has(self.h_read, label, f"the hinge arms' label (critic entry {self.h_read} (1)(f))", later=True)
        for key in self.hinge["keys"]:
            runs[key].setdefault("hinge", {})["label"] = label
        return label

    def _reference_labels(self, co: dict, ref: dict) -> dict:
        """The coconut reference record's labels in the fourth table: 17:36Z (2)'s block beside its numbers (verified
        as _dlabel's), or, when the table's identification holds it across GPU types (its none / scrambled reads are
        the first job's), 11:07Z (3)'s label "identification across GPU types: k of 500 (<record GPU> vs <reads' GPU>)",
        its form verified in the read row (or a later critic entry)."""
        ref = dict(ref)
        if ref.get("status") in ("present", "absent"):
            lbl, items = self._dlabel(co, ref)
            for text, what, ts, later in items:
                self._row_has(ts, text, what, later=later)
            ref["label_block"] = lbl
        ident = ref.get("identification") or {}
        if ref.get("status") == "held" and ident.get("by"):
            g_rec = _gpu_short(ref.get("gpu_type"))
            g_reads = sorted({_gpu_short((e.get("summary") or {}).get("gpu_type")) for k_, e in
                              ((self.ev or {}).get("by_run", {}).get("coconut") or {}).items() if k_ in
                              ("none", "scrambled")})
            if len(g_reads) == 1 and g_reads[0] != g_rec:
                form = XGPU_FORM.format(n=ident.get("n"), a=g_rec, b=g_reads[0])
                self._row_has(self.h_read, form, "the reference record's identification label (11:07Z (3))",
                              later=True)
                k = len(set(ident.get("none_diff_idx") or []) | set(ident.get("scrambled_diff_idx") or []))
                ref["identification"] = form.replace("k of", f"{k} of", 1)
        return ref

    def _fourth_notes(self, ref_note: str | None) -> list[str]:
        """The notes the fourth table adds after the third's (critic entries 2026-10-20T17:12Z (5) to (9), 17:36Z)."""
        jh = self.ev_h["job"]
        out = [
            f"The hinge arms (critic entry {ROW_HINGE} (5)): gsm50k_c3_s0, gsm50k_c3_foreign_s0 and gsm50k_c2c3_s0 of "
            f"{self.hinge['job']} (A2.1: the one hinge job, GSM-50k only), read from their own records (AcctA's bucket) "
            f"by the six arms' rules: the selection over the rows of record across resumes (A8.2), the start pin (A6.3 "
            f"rule 4), the receipt of the attempt that completed each run; their eval reads are the hinge eval job "
            f"{jh['job']} (attempt {jh['attempt']}, registered row {jh['registered_read']}; no timed read, 17:12Z (6)). "
            f"Their K2 cells follow A6.1 rules 1, 5 and 6 with the labels of critic entry {DONOR_ROW} (5) verbatim; a "
            f"label whose form carries numbers (the lifted 'directional read present' label, the K2 line 'K2 not "
            f"independent (A6.1): Delta_scr ...') is built from the files and must appear quoted in a critic entry logged "
            f"at or after {DONOR_ROW}. Every line of the three arms carries the read row's label of their records "
            f"(critic entry {self.h_read} (1)(f): the resume counts, exact boundary resumes, same GPU type; built from "
            f"RUN_DONE, ckpt_state and the rows of record, quoted there), and every directional number 17:36Z (2)'s "
            f"label block, its <gpu_type> slot filled from the record (quoted there); a read on another GPU type than "
            f"the training one is refused (the read row registers no cross-type label, A7.6).",
            f"A6.1 rules 2 and 3 on the hinge arms: the C3 minus C3-foreign (and C2+C3 minus C3-foreign) scrambled-drop "
            f"difference is 'dependence on own latents (content or fingerprint)', with the same 10,000 draws; the "
            f"zero-latent and n = 0 reads of these checkpoints are diagnostics that decide nothing for them (no "
            f"pause-mode read exists); the comparison with the separately trained pause arm (7.5) is unaffected. "
            f"ProsQA-hard is UNTESTED for C3 and C2+C3 (A2.1; 17:12Z (8)): no both-tasks kill clause of A3.4 C3 can be "
            f"met as written, and the F1 patch directional read is UNTESTED for both arms.",
        ]
        return out + ([ref_note] if ref_note else [])

    def _eval_hinge_json(self) -> dict:
        """The hinge eval job's block of the fourth table: its job, its read entries (as _eval_json lists the first
        job's), its donor_answer records' checks and the coconut reference record's reading (notes only)."""
        hj = self.ev_h
        ents = []
        for e in hj["entries"]:
            sm = e.get("summary") or {}
            ents.append({"name": e["name"], "run": e["run"], "kind": e["kind"], "batch": e["batch"], "ok": e["ok"],
                         "exclusion": e.get("exclusion"), "problems": e["problems"], "checks": e["checks"],
                         "k": e.get("k"), "points": float(pts(e["k"], self.n)) if "k" in e else None,
                         "examples_csv_sha256": e.get("csv_sha256"), "checkpoint_sha256": sm.get("checkpoint_sha256"),
                         "precision": sm.get("precision"), "gpu_type": sm.get("gpu_type"), "torch": sm.get("torch"),
                         "timing": sm.get("timing"), "cost_check": e.get("cost_check")})
        dirs = [{"name": e["name"], "run": e["run"], "ok": e.get("ok"), "problems": e["problems"],
                 "checks": e["checks"], "examples_csv_sha256": e.get("csv_sha256")}
                for e in (hj.get("directional") or {}).values()]
        ref = json.loads(json.dumps(self.ref_directional, default=lambda o: float(o) if isinstance(o, Fraction)
                                    else str(o))) if self.ref_directional is not None else None
        return {"job": hj["job"], "entries": ents, "directional": dirs, "reference": ref}

    def _eval_json(self, runs: dict) -> dict:
        ev = self.ev
        ents = []
        for e in ev["entries"]:
            sm = e.get("summary") or {}
            ents.append({"name": e["name"], "run": e["run"], "kind": e["kind"], "batch": e["batch"], "ok": e["ok"],
                         "exclusion": e.get("exclusion"), "problems": e["problems"], "checks": e["checks"],
                         "k": e.get("k"), "points": float(pts(e["k"], self.n)) if "k" in e else None,
                         "examples_csv_sha256": e.get("csv_sha256"), "checkpoint_sha256": sm.get("checkpoint_sha256"),
                         "precision": sm.get("precision"), "gpu_type": sm.get("gpu_type"), "torch": sm.get("torch"),
                         "timing": sm.get("timing"), "cost_check": e.get("cost_check")})
        per_run = {}
        for s_ in self.spec["runs"]:
            r = runs[s_["key"]]
            if "eval_stats" in r:
                per_run[s_["key"]] = json.loads(json.dumps(r["eval_stats"], default=lambda o: float(o)
                                                           if isinstance(o, Fraction) else str(o)))
        dirs = [{"name": e["name"], "run": e["run"], "ok": e.get("ok"), "problems": e["problems"],
                 "checks": e["checks"], "examples_csv_sha256": e.get("csv_sha256")}
                for e in (ev.get("directional") or {}).values()]
        return {"job": ev["job"], "entries": ents, "per_run": per_run,
                **({"directional": dirs} if dirs and not self.fourth else {})}     # the fourth: eval_hinge holds them

    def _prosqa_cot_sd(self) -> float | None:
        if not os.path.isfile(self.k1_table):
            return None
        t = self.inp.read_json(self.k1_table, "ProsQA cot 3-seed SD (final K1 table)")
        row = next((r for r in t.get("rows", []) if (r.get("task"), r.get("arm")) == ("prosqa", "cot")), None)
        if not row or len(row.get("seeds") or {}) < 3 or row.get("sd") is None:
            return None
        return float(row["sd"])

    def _plain_depth(self, runs: dict) -> dict:
        """A13.2: the 355M point enters A_pd only when both plain-depth runs have a selection and a valid read."""
        have = []
        for name in self.spec["plain_depth_all"]:
            ok = os.path.isfile(self._path(name, "RUN_DONE.json"))
            if ok:
                rd = self.inp.read_json(self._path(name, "RUN_DONE.json"), "run record")
                ok = rd.get("status") == "done" and bool(((rd.get("best") or {}).get("final") or [None])[0] is not
                                                          None)
            have.append(ok)
        pd = runs.get("plain_depth_355m")
        usable = all(have) and pd is not None and pd["status"] == "read"
        out = {"usable": usable, "runs": dict(zip(self.spec["plain_depth_all"], have))}
        if usable:
            c = pd["config"]
            mb = (int(c["batch_size"]), int(c["grad_accum"]))
            out["micro_batch"] = None if mb == (32, 4) else f"micro-batch {mb[0]} x {mb[1]}"
            lab = [PD_LABEL] + ([out["micro_batch"]] if out["micro_batch"] else [])
            if self.pd_probe is not None:
                out["probe"] = self.pd_probe
                lab.append(self.pd_probe["label"] or "memory probe: no receipt of the control's attempt read")
            out["label"] = "; ".join(lab)
            nc = runs.get("no_cot")
            if nc is not None and nc["status"] == "read" and "cost" in nc and "cost" in pd:
                out["points"] = [self._apd_point(nc, "124M"), self._apd_point(pd, "355M")]
                f0, f1 = out["points"][0]["flops_mean"], out["points"][1]["flops_mean"]
                lr = [math.log(b / a) for a in f0 for b in f1]
                out["construction"] = (
                    "7.7: A_pd(F) = a_124M + (a_355M - a_124M) x ln(F / F_124M) / ln(F_355M / F_124M) for F_124M <= F <= "
                    "F_355M (linear in ln F between the no_cot points of the sizes run); below F_124M a_124M and above "
                    "F_355M a_355M, labelled 'nearest size' (no extrapolation); every F is a mean per-example "
                    "inference FLOPs on the valid questions (5.1, 5.3), and A_pd is evaluated at every corner of the "
                    "arm's and the two points' FLOP ranges (a_pd_range)")
                out["ln_ratio"] = [min(lr), max(lr)]
                out["ratio"] = [math.exp(min(lr)), math.exp(max(lr))]
        return out

    def _apd_point(self, r: dict, size: str) -> dict:
        """One no_cot point of 7.7's A_pd: the run, its selection (every epoch, 3.3 item 1), its valid accuracy and its
        mean per-example inference FLOPs at its own dims (an exact range: E' per example not recorded)."""
        c = r["cost"]
        d = self.dims(r["model_id"], c["V"])
        s = r["selected"]
        return {"size": size, "run": r["run"], "model_id": r["model_id"],
                "model_revision": self._model_id(r["config"].get("model_id", ""))[1],
                "checkpoint": ((r.get("run_done") or {}).get("checkpoints") or {}).get("ckpt_best_final.pt"),
                "selection": "every epoch (3.3 item 1: the no_cot rule), argmax of valid accuracy, ties to the earliest",
                "epoch": int(s["epoch"]), "epoch_index": int(s["epoch_index"]), "epochs": len(r["rows"]),
                "resumes": r.get("resumes", 0), "k": r["k"], "n": self.n, "points": float(pts(r["k"], self.n)),
                "emitted_total": c["emitted_total"], "emitted_mean": c["emitted_mean"], "q_mean": c["q_mean"],
                "flops_mean": list(c["flops_mean"]),
                "dims": {"N_ne": d.N_ne, "V": d.V, "d": d.d, "n_L": d.n_L, "d_a": d.d_a}}

    def _run_json(self, r: dict) -> dict:
        keep = ("run", "key", "role", "card", "status", "why", "exclusion", "problems", "checks", "gpu_type",
                "resumes", "attempts", "superseded", "k", "L", "M", "earlier_stage_better", "k_all", "start_pin",
                "train_flops_own", "model_id", "label", "hinge", "closing")
        out = {k: r[k] for k in keep if k in r}
        if r.get("label"):                                     # the control (A13.2; critic entry 2026-10-20T15:53Z)
            out["checkpoint"] = ((r.get("run_done") or {}).get("checkpoints") or {}).get("ckpt_best_final.pt")
            out["model_revision"] = self._model_id((r.get("config") or {}).get("model_id", ""))[1]
        if "selected" in r:
            s = r["selected"]
            out["selected"] = {"epoch": int(s["epoch"]), "epoch_index": int(s["epoch_index"]), "stage": int(s["stage"]),
                               "val_acc": s["val_acc"], "val_emitted_mean": s["val_emitted_mean"]}
            out["points"] = float(pts(r["k"], self.n))
            b = r["best_all"]
            ps = (r.get("closing") or {}).get("pinned_stage")   # the no-curriculum arm: the pinned stage, not the log's
            out["best_all"] = {"epoch": int(b["epoch"]), "stage": int(b["stage"]), "val_acc": b["val_acc"],
                               "n_latent": gen_latents(r["config"], ps if ps is not None else int(b["stage"]))}
            out["pool"] = list(r["pool"]) if r.get("pool") else None
            out["curve"] = [{"epoch": int(x["epoch"]), "stage": int(x["stage"]), "val_acc": x["val_acc"],
                             "val_emitted_mean": x["val_emitted_mean"],
                             **({"train_state_loss": x["train_state_loss"]} if "train_state_loss" in x else {})}
                            for x in r["rows"]]
        for k in ("cost", "cost_nominal", "cost_measured", "cost_range", "cost_nominal_range"):
            if k in r:
                out[k] = {kk: (list(v) if isinstance(v, tuple) else v) for kk, v in r[k].items()}
        if "M_log" in r:
            out["M_log"] = r["M_log"]
        st = (r.get("config") or {}).get("_state_targets")
        if st:
            out["state_targets"] = st
        return out

    # ---------------------------------------------------------------------------------------------- lines
    def lines(self, runs: dict, bar_key: str, sigma: float, plain: dict) -> list[dict]:
        n, out = self.n, []
        S = Fraction(sigma)
        self._verify[sigma] = []
        lab = f"[{self.lab_inner}]"
        ev = self.ev
        R = {k: r for k, r in runs.items() if r["status"] == "read"}
        bar = R[bar_key]
        kb = bar["k"]

        def P(r):
            return pts(r["k"], n)

        def add(lid, kind, outcome, text, run=None):
            out.append({"id": lid, "kind": kind, "run": run, "outcome": outcome, "text": text})

        add("bar", "bar", "read", f"Bar ({self.spec['title']}, valid): {bar['run']} {f1(P(bar))} ({kb} of {n}) at "
            f"epoch {bar['selected']['epoch']}; trials " + ", ".join(
                f"{r['run']} {f1(P(r))}" for k, r in R.items() if k.startswith("cot_lr"))
            + (f" [{BAR_VALUE_LABEL}; {SCREEN_LABEL}]" if ev is None else f" [{BAR_VALUE_LABEL}; {EVAL_LABEL}]"))
        add("sigma", "sigma", f"{sigma:.4f}", f"sigma_screen = {f2(sigma)} points (A3.4: max(1.0, 100 sqrt(p (1 - p) / "
            f"n_valid) at the bar's p = {kb}/{n}, the ProsQA cot 3-seed SD once it exists))")
        bar_cost = bar["cost"]
        latent_keys = [s["key"] for s in self.spec["runs"] if s["key"] in R and R[s["key"]]["latent"]]

        def flop_leg(r):
            res = [decide_le(r["cost"]["flops_mean"], bar_cost["flops_mean"])]
            for extra in ("cost_nominal", "cost_measured"):
                if extra in r:
                    res.append(decide_le(r[extra]["flops_mean"], bar_cost["flops_mean"]))
            return "met" if all(x == "met" for x in res) else ("not met" if res[0] == "not met" else
                                                                "undetermined")

        def legs_7_2(r):
            c = r["cost"]
            tok = Fraction(r["M"], 1) <= Fraction(bar["M"], 5)            # E'_arm <= 0.2 E'_bar (same n)
            lat = r["L"] if r["feedback"] else 0
            passes = Fraction(n * (1 + lat) + r["M"]) <= Fraction(n * 1 + bar["M"], 5)
            fl = flop_leg(r)
            ref = decide_le(c["reference_flops_mean"], bar_cost["reference_flops_mean"])
            pas_le = "met" if Fraction(n * (1 + lat) + r["M"]) <= Fraction(n + bar["M"]) else "not met"
            flips = [name for name, o in (("(a) sequential passes", pas_le), ("(c) reference-cost FLOPs", ref))
                     if fl != "undetermined" and o != "undetermined" and o != fl]
            return {"flop": fl, "token": "met" if tok else "not met", "passes": "met" if passes else "not met",
                    "reference": ref, "passes_le": pas_le, "accounting_sensitive": flips}

        def ratio_txt(a, b):
            lo, hi = interval_ratio(a, b)
            return f"[{lo:.3f}, {hi:.3f}]"

        # ---- cost legs per latent run (7.2 legs 2 and 3, the passes criterion, 5.5 check)
        for key in latent_keys:
            r = R[key]
            g = legs_7_2(r)
            r["_legs"] = g
            c = r["cost"]
            billing = g["token"] == "met" and g["passes"] == "not met"
            if ev is not None and c.get("exact"):
                add(f"cost:{key}", "cost", g["flop"] + ("; billing claim only" if billing else ""),
                    self._cost_text(r, bar, g, billing) + f" {lab}", r["run"])
                continue
            txt = (f"7.2 cost legs, {r['run']} vs the bar: FLOP leg (mean FLOPs <= bar's) {g['flop']} (ratio in "
                   f"{ratio_txt(c['flops_mean'], bar_cost['flops_mean'])}"
                   + (f"; nominal layer fraction: {ratio_txt(r['cost_nominal']['flops_mean'], bar_cost['flops_mean'])}"
                      if "cost_nominal" in r else "") + "); "
                   f"token leg (E' <= 0.2 x bar's) {g['token']} ({f2(c['emitted_mean'])} vs 0.2 x "
                   f"{f2(bar_cost['emitted_mean'])} = {f2(bar_cost['emitted_mean'] * 0.2)}); passes criterion "
                   f"(<= 0.2 x bar's) {g['passes']} ({f2(c['passes_mean'])} vs {f2(bar_cost['passes_mean'] * 0.2)})"
                   + ("; label: billing claim only (7.2)" if billing else "")
                   + (f"; accounting-sensitive ({', '.join(g['accounting_sensitive'])})" if g["accounting_sensitive"]
                      else "; 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED") + f" {lab}")
            add(f"cost:{key}", "cost", g["flop"] + ("; billing claim only" if billing else ""), txt, r["run"])

        # ---- K2 (pending: no scrambled-at-inference read on valid)
        for key in latent_keys:
            r = R[key]
            if r["config"]["arm"] == "pause":
                if ev is not None:
                    add(f"k2:{key}", "k2", "no K2 read defined",
                        f"K2 (7.5; A3.4), {r['run']}: no K2 read is defined for pause: section 8's scrambled "
                        f"construction replaces fed-back vectors and pause has none (5.3: latent steps 0, its "
                        f"{r['L']} positions are pause tokens); its content test is the 7.5 pause comparison itself "
                        f"(critic entry 2026-10-20T10:49Z (4)(a)) {lab}", r["run"])
                continue
            dep = bool(r["config"].get("dep_loss"))
            if ev is not None:
                k2 = (r.get("eval_stats") or {}).get("k2")
                if k2 is None:
                    why = self._eval_missing(r, ("none", "scrambled"))
                    add(f"k2:{key}", "k2", "UNTESTED", f"K2 (7.5; A3.4: every latent arm at the screen), {r['run']}: "
                        f"UNTESTED: {why}" + ("; K2 not independent (A6.1)" if dep else "") + f" {lab}", r["run"])
                    r["_k2"] = "UNTESTED"
                    continue
                if dep:                                  # A6.1 rules 1, 5, 6 (critic entry 2026-10-20T17:36Z (5))
                    a = self._a61(r, R, k2, sigma)
                    r["_k2"], r["_a61"] = a["k2"], a
                    self._verify[sigma] += a.get("verify") or []
                    add(f"k2:{key}", "k2", a["outcome"], a["text"] + f" {lab}", r["run"])
                    continue
                o, why = k2_outcome(k2["delta"], k2["lo"], k2["hi"], sigma)
                r["_k2"] = o
                add(f"k2:{key}", "k2", o + ("; K2 not independent (A6.1)" if dep else ""),
                    self._k2_text(r, R.get("pause"), k2, o, why, sigma) + ("; K2 not independent (A6.1)" if dep else
                                                                          "") + f" {lab}", r["run"])
                continue
            add(f"k2:{key}", "k2", "pending",
                f"K2 (7.5; A3.4: every latent arm at the screen), {r['run']}: pending: no scrambled-at-inference read "
                f"on valid exists for its selected checkpoint" + ("; K2 not independent (A6.1)" if dep else "")
                + f" {lab}", r["run"])
            r["_k2"] = "pending"

        # ---- K3 (7.5 with 7.7's A_pd; A13.2)
        nc = R.get("no_cot")
        for key in latent_keys:
            r = R[key]
            if nc is None:
                add(f"k3:{key}", "k3", "UNTESTED", f"K3, {r['run']}: UNTESTED: the 124M no_cot point has no record "
                    f"{lab}", r["run"])
                continue
            if plain.get("points"):                    # 7.7 with both sizes (A13.2; critic entry 2026-10-20T15:53Z)
                o, txt = self._k3_pd(r, plain, sigma)
                add(f"k3:{key}", "k3", o, txt, r["run"])
                continue
            pts_list = [(nc["cost"]["flops_mean"], float(P(nc)))]
            label = "nearest size; plain depth 355M UNTESTED (A13.2)"
            if plain["usable"]:
                pd = R["plain_depth_355m"]
                pts_list.append((pd["cost"]["flops_mean"], float(P(pd))))
                label = "plain depth 355M, 1 seed" + (f"; {plain['micro_batch']}" if plain.get("micro_batch") else "")
            F = tuple(r["cost"]["flops_mean"])
            for extra in ("cost_nominal", "cost_measured"):    # A3.4: K3 on measured cost; span every fraction
                if extra in r:
                    F = (min(F[0], r[extra]["flops_mean"][0]), max(F[1], r[extra]["flops_mean"][1]))
            lo, hi, near = a_pd_range(F, pts_list)
            if plain["usable"] and near:
                label += "; nearest size"
            acc = float(P(r))
            if acc < lo - sigma:
                o = "fired"
            elif acc >= hi - sigma:
                o = "not fired"
            else:
                o = "undetermined"
            within = abs(acc - lo) <= sigma and abs(acc - hi) <= sigma          # for every A_pd in the range
            maybe = not within and (acc - hi <= sigma) and (acc - lo >= -sigma)  # for some A_pd in the range
            apd = f2(lo) if abs(hi - lo) < 1e-12 else f"[{f2(lo)}, {f2(hi)}]"
            txt = (f"K3 (7.5 at the screen), {r['run']}: {f1(acc)} vs A_pd {apd} "
                   + (f"(the 124M no_cot point alone, {nc['run']} epoch {nc['selected']['epoch']}) " if not
                      plain["usable"] else "") + f"- sigma_screen {f2(sigma)}: {o}"
                   + ("; no better than plain depth (7.5)" if within and o != "fired" else "")
                   + ("; possibly no better than plain depth (A_pd range)" if maybe and o != "fired" else "")
                   + (f"; billing claim only (7.5)" if o == "fired" else "") + f" [{label}; {self.lab_inner}]")
            add(f"k3:{key}", "k3", o + ("; no better than plain depth" if within and o != "fired" else ""), txt,
                r["run"])

        # ---- pause comparison (7.5)
        pz = R.get("pause")
        for key in latent_keys:
            r = R[key]
            if key == "pause" or pz is None:
                continue
            d = P(r) - P(pz)
            within = abs(d) <= S
            add(f"pause:{key}", "pause", "CURRICULUM-ONLY" if within else "not within sigma",
                f"Pause comparison (7.5), {r['run']} - {pz['run']} = {d.numerator / d.denominator:+.1f}: "
                + (f"within sigma_screen {f2(sigma)}: CURRICULUM-ONLY" if within else
                   f"not within sigma_screen {f2(sigma)}") + f" {lab}", r["run"])

        # ---- the Curriculum line (critic entry 2026-10-21T20:21Z (b) (5), (6)): sigma_screen of record in every reading
        if self.extra:
            self._x_verify = [(self.extra["role"], "the arm's role (critic entry 2026-10-21T20:21Z (b) (5))",
                               CLOSING_ROW, False)]
            for key in self.x_keys:
                if key not in R:
                    r0 = runs[key]
                    why = r0.get("why") or f"{r0.get('exclusion')}: {'; '.join(r0['problems'])}"
                    add(f"curriculum:{key}", "curriculum", "UNTESTED",
                        f"Curriculum (3.1's control; A2.2 (b)), {self.spec['title']}: UNTESTED: {r0['run']} enters no "
                        f"cell ({why}) {lab}", r0["run"])
            for cl in self._curriculum(R, self._sigma_reg):
                self._x_verify += cl["verify"]
                add(f"curriculum:{cl['key']}", "curriculum", cl["zone"], cl["text"] + f" {lab}", cl["run"])

        # ---- screen rule 6.3 for card arms
        for key in latent_keys:
            r = R[key]
            if not r["card"]:
                continue
            legs = []
            ok_bar = P(r) >= pts(kb, n) - S
            legs.append(("bar", "met" if ok_bar else "not met",
                         f"{f1(P(r))} >= {f1(pts(kb, n))} - {f2(sigma)} = {f2(pts(kb, n) - S)}"))
            k2 = (r.get("eval_stats") or {}).get("k2") if ev is not None else None
            if k2 is not None:                           # 6.3: ahead of its scrambled-at-inference null by 1 sigma
                legs.append(("scrambled null + sigma", "met" if k2["delta"] >= S else "not met",
                             f"Delta_scr {float(k2['delta']):+.1f} vs {f2(sigma)}"
                             + ("; not independent, A6.1 rule 1" if r["config"].get("dep_loss") else "")))
            else:
                legs.append(("scrambled null + sigma", "pending", "no scrambled-at-inference valid read"))
            if pz is not None:
                d = P(r) - P(pz)
                legs.append(("pause + sigma", "met" if d >= S else "not met",
                             f"{d.numerator / d.denominator:+.1f} vs {f2(sigma)}"))
            else:
                legs.append(("pause + sigma", "pending", "pause has no record"))
            g = r["_legs"]
            legs.append(("FLOP leg", g["flop"], "7.2 leg 2"))
            legs.append(("token leg", g["token"], "7.2 leg 3"))
            states = [s for _, s, _ in legs]
            o = ("does not advance" if "not met" in states else
                 "pending" if ("pending" in states or "undetermined" in states) else "advances")
            add(f"screen:{key}", "screen", o,
                f"Screen rule (6.3), {r['run']} ({r['card']}): {o}. Legs: "
                + "; ".join(f"{name}: {s} ({why})" for name, s, why in legs) + f" {lab}", r["run"])

        # ---- card lines (A3.4; this task's parts)
        out.extend(self._card_lines(R, sigma, lab))

        # ---- too-good triggers (7.10; A3.4)
        tg = self.spec["too_good"]
        cand = [r for k, r in R.items() if not k.startswith("cot_lr")]
        if "over_bar" in tg:
            hit = [r["run"] for r in cand if P(r) - pts(kb, n) > Fraction(tg["over_bar"]).limit_denominator()]
            rule = f"more than {tg['over_bar']:g} points over the bar"
        else:
            hit = [r["run"] for r in cand if P(r) > Fraction(tg["above"]).limit_denominator()]
            rule = f"above {tg['above']:g}%"
        add("too_good", "too_good", "triggered" if hit else "not triggered",
            f"Too-good triggers (7.10; A3.4), any arm {rule}: " + (", ".join(hit) if hit else "none")
            + ("; the null triggers (scrambled-at-inference, zero-latent) are the next line (eval reads) "
               if ev is not None else "; the null triggers (scrambled-at-inference, zero-latent) are pending their "
                                      "valid reads ") + lab)
        if ev is not None:
            out.extend(self._null_trigger_lines(R, latent_keys, kb, sigma, lab))

        # ---- fade instrument (7.6)
        B = self.spec["epochs"]
        halves = sorted({B // 2, (B + 1) // 2})
        for key in latent_keys + (["no_cot"] if "no_cot" in R else []):
            r = R[key]
            leads = {}
            for e in halves + [B]:
                ra = next((x for x in r["rows"] if int(x["epoch"]) == e), None)
                rb = next((x for x in bar["rows"] if int(x["epoch"]) == e), None)
                if ra is not None and rb is not None:
                    leads[e] = pts(count_of(ra["val_acc"], n, "val_acc") - count_of(rb["val_acc"], n, "val_acc"), n)
            if B not in leads or not any(h in leads for h in halves):
                add(f"fade:{key}", "fade", "UNTESTED", f"Fade (7.6), {r['run']}: curve points missing {lab}", r["run"])
                continue
            flags = {h: leads[B] < leads[h] - S for h in halves if h in leads}
            o = "fading" if all(flags.values()) else ("not fading" if not any(flags.values()) else
                                                      "fading at one 0.5 B reading")
            add(f"fade:{key}", "fade", o,
                f"Fade instrument (7.6), {r['run']}: lead over the bar's curve at "
                + ", ".join(f"epoch {e} {float(v):+.1f}" for e, v in sorted(leads.items()))
                + f" (0.5 B = {B / 2:g}{' read at epochs ' + ' and '.join(map(str, halves)) if len(halves) > 1 else ''})"
                + f": {o} (sigma_screen {f2(sigma)}) {lab}", r["run"])

        # ---- C1b per-step cost (A3.4; A6.2 rules 1 and 4)
        for key in latent_keys:
            r = R[key]
            if not r["config"].get("latent_band"):
                continue
            a, b = r["config"]["latent_band"]
            lf = (b - a) / r["n_L"]
            if ev is not None:
                o, txt = self._f_meas_text(r, R.get("coconut"))
                add(f"c1b_cost:{key}", "c1b_cost", o, txt + f" {lab}", r["run"])
                continue
            add(f"c1b_cost:{key}", "c1b_cost", "nominal",
                f"C1b per-step cost, {r['run']}: nominal: f_meas UNTESTED (A6.2's timed read, band [0, 12] on the "
                f"full-depth coconut checkpoint, has not run; the training-time val_latent_step_ms rows are not "
                f"A6.2's fraction, rule 1); saving line f_meas <= {lf + 0.10:.2f}; FLOPs printed at the nominal "
                f"{b - a}/{r['n_L']} and at 1 (the largest min(1, f_meas)); the FLOP leg is read at 1 {lab}", r["run"])

        # ---- C2 labels (A5 item 5; A6.7 items 4, 5)
        for key in ("c2", "c2_shuffled"):
            if key in R:
                r = R[key]
                st = r["config"].get("_state_targets") or {}
                fr = {k: v for k, v in st.items() if k.startswith("answer_in")}
                sent = self.spec["c2_sentences"][key]
                if ev is not None and key == "c2_shuffled" and self.task == "gsm50k" and \
                        "answer_in_last_used_target_frac" in st:
                    n_tr = int(self.split_entry("train")["records"])
                    m = count_of(float(st["answer_in_last_used_target_frac"]), n_tr, "answer_in_last_used_target_frac")
                    sent = (f"A6.7 item 4 with the run's own value (critic entry 2026-10-20T10:49Z): on GSM-50k the "
                            f"C2-shuffled donor shares the answer {m} of {n_tr:,} = {100 * m / n_tr:.2f}% (chance; the "
                            f"review's replica at another derangement seed gave 430)")
                add(f"c2_label:{key}", "c2_label", "label",
                    f"{r['run']}: {sent}; config.json _state_targets "
                    + ", ".join(f"{k} {v}" for k, v in fr.items()) + f" {lab}", r["run"])
        if "c2" in R and "c2_shuffled" in R:
            def curve(r):
                return " ".join(f"{x['train_state_loss']:.3f}" for x in r["rows"] if "train_state_loss" in x)
            add("c2_state_loss", "c2_label", "label",
                f"train_state_loss by epoch {R['c2']['rows'][0]['epoch']}-{R['c2']['rows'][-1]['epoch']} (A6.7 item "
                f"5): C2 {curve(R['c2'])}; C2-shuffled {curve(R['c2_shuffled'])}; every C2 minus C2-shuffled "
                f"sentence carries: the gap includes learnable vs unlearnable auxiliary target {lab}")

        # ---- the eval reads' own lines (second table): the plain read against the selection, the sweep, rule 3,
        # ---- 5.5 (b)
        if ev is not None:
            out.extend(self._eval_lines(R, latent_keys, bar, lab))

        # ---- UNTESTED registered runs
        for s in self.spec["runs"]:
            r = runs[s["key"]]
            if r["status"] != "read":
                why = r.get("why") or f"{r.get('exclusion')}: {'; '.join(r['problems'])}"
                add(f"untested:{s['key']}", "untested", "UNTESTED" if r["status"] == "UNTESTED" else r["exclusion"],
                    f"{s['run']} ({s['role']}): {'UNTESTED' if r['status'] == 'UNTESTED' else r['exclusion']}: "
                    f"{why}; enters no cell", s["run"])
        return out

    # ------------------------------------------------------------------------------- K3 with the 355M point
    @staticmethod
    def _k3_F(r: dict) -> tuple[tuple[float, float], list]:
        """The arm's FLOPs for K3 (7.5): its cost range, and for C1b spanned over the nominal (b - a) / L, the measured
        min(1, f_meas) and 1 (A3.4: K3 on measured cost; critic entry 2026-10-20T15:53Z (2)); the parts are printed."""
        if "cost_nominal" in r:
            a, b = r["config"]["latent_band"]
            parts = [(f"nominal {b - a}/{r['n_L']}", tuple(r["cost_nominal"]["flops_mean"]))]
            if "cost_measured" in r:
                parts.append((f"measured {r['cost_measured']['layer_fraction']:.4f}",
                              tuple(r["cost_measured"]["flops_mean"])))
            parts.append(("at 1", tuple(r["cost"]["flops_mean"])))
        else:
            parts = [("", tuple(r["cost"]["flops_mean"]))]
        return (min(p[1][0] for p in parts), max(p[1][1] for p in parts)), parts

    def _apd_eval(self, r: dict, plain: dict) -> dict:
        """A_pd at the arm's FLOPs from the two points (a_pd_range over every corner), and w = ln(F / F_124M) /
        ln(F_355M / F_124M), the arm's place on the log-F segment, over the same corners."""
        F, parts = self._k3_F(r)
        p0, p1 = plain["points"]
        lo, hi, near = a_pd_range(F, [(tuple(p0["flops_mean"]), p0["points"]), (tuple(p1["flops_mean"]), p1["points"])])
        ws = [math.log(f / a) / math.log(b / a) for f in F for a in p0["flops_mean"] for b in p1["flops_mean"]]
        return {"run": r["run"], "points": float(pts(r["k"], self.n)), "F": list(F),
                "parts": [{"fraction": nm or "the arm's", "flops_mean": list(v)} for nm, v in parts],
                "entry": r["cost"].get("entry"), "exact": bool(r["cost"].get("exact")), "w": [min(ws), max(ws)],
                "a_pd": [lo, hi], "nearest_size": near}

    def _k3_pd(self, r: dict, plain: dict, sigma: float) -> tuple[str, str]:
        """7.5 at the screen with 7.7's A_pd from the 124M and 355M no_cot points: K3 fires iff acc < A_pd(F) - sigma
        for every A_pd in the range ('undetermined' when the range straddles the line); 'no better than plain depth'
        iff within sigma of A_pd for every A_pd in the range. The direct comparisons with the two points are printed
        beside (7.7: 'both are printed'); they decide nothing."""
        a = self._apd_eval(r, plain)
        lo, hi = a["a_pd"]
        acc = float(pts(r["k"], self.n))
        o = "fired" if acc < lo - sigma else ("not fired" if acc >= hi - sigma else "undetermined")
        within = abs(acc - lo) <= sigma and abs(acc - hi) <= sigma
        maybe = not within and (acc - hi <= sigma) and (acc - lo >= -sigma)
        p0, p1 = plain["points"]

        def rg(x, y, nd=2, sign=False):
            fx = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
            fy = f"{y:+.{nd}f}" if sign else f"{y:.{nd}f}"
            return fx if fx == fy else f"[{fx}, {fy}]"
        def apd(x, y):                            # the first table's form: a range unless the two ends are equal
            return f2(x) if abs(y - x) < 1e-12 else f"[{f2(x)}, {f2(y)}]"
        F = " / ".join(f"{_gfr(v)}" + (f" ({nm})" if nm else "") for nm, v in
                       ((p["fraction"] if p["fraction"] != "the arm's" else "", p["flops_mean"]) for p in a["parts"]))
        src = (f"exact mean from {a['entry']}'s per-example E'" if a["exact"] else
               "the exact range over the unrecorded per-example E'")
        label = plain["label"] + ("; nearest size" if a["nearest_size"] else "")
        txt = (f"K3 (7.5 at the screen; 7.7's A_pd from the two sizes), {r['run']}: {acc:.1f} at F = {F} GFLOPs ({src}; "
               f"w = {rg(*a['w'], nd=3)} of the ln F way from the 124M point to the 355M point) vs A_pd "
               f"{apd(lo, hi)} - sigma_screen {f2(sigma)} = {apd(lo - sigma, hi - sigma)}: {o}"
               + ("; no better than plain depth (7.5)" if within and o != "fired" else "")
               + ("; possibly no better than plain depth (A_pd range)" if maybe and o != "fired" else "")
               + ("; billing claim only (7.5)" if o == "fired" else "")
               + f" (acc - A_pd {rg(acc - hi, acc - lo, sign=True)}; A_pd linear in ln F between the 124M no_cot point "
               f"{p0['run']}, epoch {p0['epoch']}, {p0['points']:.1f} at {_rng(p0['flops_mean'])} GFLOPs, and the 355M "
               f"point {p1['run']}, epoch {p1['epoch']}, {p1['points']:.1f} at {_rng(p1['flops_mean'])} GFLOPs); beside, "
               f"deciding nothing (7.7: both are printed): {acc - p0['points']:+.1f} against the 124M no_cot point, "
               f"{acc - p1['points']:+.1f} against the 355M point at {a['F'][1] / p1['flops_mean'][0]:.2f} of its FLOPs or "
               f"less [{label}; {self.lab_inner}]")
        return o + ("; no better than plain depth" if within and o != "fired" else ""), txt

    # ------------------------------------------------------------------------ the third table's labels (verified)
    def _row_has(self, ts: str, text: str, what: str, later: bool = False) -> None:
        """Refuses unless `text` appears verbatim, as a quoted label ("..."), in a critic_log.md row with timestamp `ts`
        (the label is the critic's, and every number in it equals the files the tool read; the quotes make a truncated
        label, e.g. a probe clause without its OOM part, a refusal rather than a substring match). With `later`, a
        critic entry logged at `ts` or after it also counts (a form fixed before its numbers existed: the critic writes the
        numbers in a later row); a lead row ("**Lead: ...") never counts."""
        log = self.inp.read_text(self.critic_log, "rules (critic log)")
        rows = [ln for ln in log.splitlines() if ln.startswith(f"| {ts} |")]
        if not rows:
            raise SystemExit(f"REFUSED: no critic_log.md row {ts} ({what})")
        if later:
            rows = [ln for ln in log.splitlines() if re.match(r"\| \d{4}-\d\d-\d\dT\d\d:\d\dZ \| ", ln)
                    and ln[2:19] >= ts and not ln[22:].startswith("**Lead")]
        if not any(f'"{text}"' in ln for ln in rows):
            raise SystemExit(f"REFUSED: {what}: the text built from the files is not in critic entry {ts}"
                             f"{' or a later critic entry' if later else ''}: {text!r}")

    def _diag(self, co: dict) -> dict:
        """A6.2 rule 3's logits check on the f_meas denominator, as amended (critic entries 2026-10-20T15:26Z rows 1 and 4,
        2026-10-20T21:03Z row 3 (3) and (4), 2026-10-20T21:33Z (1) and (2)): the float64 record is required and must
        meet the line (dtype float64, strict set alone, tol 1e-8, check (1) pass and bug false; checks (1) and (1b)
        max and fed latents <= 1e-8, 0 first-argmax differences, 0 diverging; the shape comparator (3) likewise) and
        name its device (cpu, or cuda with its gpu_type), its dtypes exactly {weights: [float64], activations:
        [float64]} and the strict set read back off at the start and the end (TF32 off, "highest", no autocast); the
        strict-fp32 GPU record is optional (tol 1e-4; printed beside with whatever verdict it carries: no line; a
        task in DIAG_GPU_IF_PRESENT reads its default only when the file exists). Each on the task's coconut
        checkpoint of record and the valid split's hash, committed clean (its commit is cited), and complete
        (_diag_complete: `complete` true, or every check in `checks_requested` present; a partial form is refused).
        Returns {"float64":
        ..., "gpu": ... or None}; the float64 entry carries "device", "gpu_type" and "where" (the label's device
        clause: "CPU", or "A100-40GB (device cuda)")."""
        out = {}
        ck = ((co.get("run_done") or {}).get("checkpoints") or {}).get("ckpt_best_final.pt")
        vsha = self.valid_entry()["sha256"]
        for kind in ("float64", "gpu"):
            p = self.diag_paths.get(kind)
            if kind == "gpu" and (not p or (self.task in DIAG_GPU_IF_PRESENT and "gpu" not in self.diag_named
                                            and not os.path.isfile(p))):
                out[kind] = None                       # optional: "strict-fp32 GPU figure: not run on this task"
                continue
            if not p or not os.path.isfile(p):
                raise SystemExit(f"REFUSED: the {kind} diag record {p} is absent (rule 3's logits check"
                                 + ("; the float64 record is required)" if kind == "float64" else ")"))
            s = self.inp.read_json(p, f"diag record (A6.2 rule 3 logits check, {kind})")
            st = git_path_state(p)
            if not st.get("commit") or st.get("dirty"):
                raise SystemExit(f"REFUSED: {p} is not committed clean (write-once diag records)")
            ch = (s.get("results") or {}).get("checks") or {}
            c1 = ch.get("1") or {}
            fl = c1.get("first_answer_logits") or {}
            bad = []
            if s.get("checkpoint_sha256") != ck or s.get("split_sha256") != vsha or s.get("mode") != "band_logits":
                bad.append("checkpoint, split or mode")
            if kind == "gpu" and not (s.get("tol") == DIAG_TOL_GPU and s.get("dtype", "float32") == "float32"
                                      and (s.get("verdicts") or {}).get("1") in ("pass", "BUG")):
                bad.append("the GPU record's line or verdict")
            if not _diag_complete(s):
                bad.append("the record is not complete (complete not true and a requested check absent)")
            if kind == "float64" and not self._diag64_meets(s):
                bad.append("the float64 record's line or verdict")
            elif kind == "float64" and _diag64_where(s) is None:
                bad.append("the float64 record's device, dtypes or strict read-back")
            if bad:
                raise SystemExit(f"REFUSED: {p}: {', '.join(bad)}")
            out[kind] = {"path": os.path.relpath(p, REPO) if p.startswith(REPO + os.sep) else p, "commit": st["commit"],
                         "summary": s, "max": float(fl["max"]), "over": int(fl.get("over_tol", -1)),
                         "n": int(fl.get("n", -1)), "tol": float(s["tol"]), "gpu_type": s.get("gpu_type"),
                         "verdict": (s.get("verdicts") or {}).get("1")}
            if kind == "float64":
                out[kind].update(device=s.get("device"), where=_diag64_where(s))
        return out

    @staticmethod
    def _diag64_meets(s: dict) -> bool:
        """The float64 line (critic entry 2026-10-20T14:54Z row 1 part (5); 15:26Z row 4): the record's own settings and
        verdicts, then every quantity of checks (1), (1b) and (3, strict) re-read against 1e-8."""
        if not (s.get("dtype") == "float64" and s.get("strict_only") is True and s.get("tol") == DIAG_TOL64
                and (s.get("verdicts") or {}).get("1") == "pass" and s.get("bug") is False):
            return False
        ch = (s.get("results") or {}).get("checks") or {}
        for key, sub in (("1", None), ("1b", None), ("3", "strict")):
            c = ch.get(key) or {}
            c = (c.get(sub) or {}) if sub else c
            for q in ("first_answer_logits", "latents"):
                if not (c.get(q) or {}).get("n") or float(c[q].get("max", 1.0)) > DIAG_TOL64:
                    return False
            if c.get("first_argmax_differs") != 0 or c.get("diverging") != 0:
                return False
        return True

    @staticmethod
    def _logits_level_confirmed(idx: list, f64: dict, g: dict | None) -> str:
        """The deviation's logits-level clause once the float64 record meets the line (critic entry 2026-10-20T23:12Z
        row 2): the float64 record's figures on rule 3's like-for-like idx against 1e-8, the strict-fp32 record's check
        (3) divergence count on them, and the registered-numerics divergences' position and top-2 margins (the
        14:54Z row 2 class). Every number is read from the two records; nothing is assumed."""
        s64 = f64["summary"]
        ch = (s64.get("results") or {}).get("checks") or {}
        mx = 0.0
        for key, sub in (("1", None), ("1b", None), ("3", "strict")):
            c = ch.get(key) or {}
            c = (c.get(sub) or {}) if sub else c
            for i in idx:
                e = (c.get("extra") or {}).get(str(i))
                if e is not None:
                    mx = max(mx, float(e.get("first_maxabs") or 0.0), float(e.get("latents_maxabs") or 0.0))
                elif i in set((s64.get("batch") or {}).get("idx") or []):
                    mx = max(mx, float((c.get("first_answer_logits") or {}).get("max") or 0.0),
                             float((c.get("latents") or {}).get("max") or 0.0))
        if mx > DIAG_TOL64:
            raise SystemExit(f"REFUSED: the float64 record's figures on idx {idx} exceed {_sci(DIAG_TOL64)} ({mx})")
        ids = ", ".join(map(str, idx))
        s = f"confirmed at float64 (record {f64['commit'][:7]}; idx {ids} <= {_sci(DIAG_TOL64)})"
        c3 = ((g or {}).get("summary", {}).get("results") or {}).get("checks", {}).get("3") or {}
        if not c3:                                     # no GPU record, or one without check (3): the float64 clause alone
            return s
        st, rg = c3.get("strict") or {}, c3.get("registered") or {}
        n_st = sum(1 for i in idx if i in set(st.get("idx_diverging") or []))
        dvs = [d for d in (rg.get("divergences") or []) if d["idx"] in set(idx)]
        pos = sorted({int(d["div_pos"]) for d in dvs})
        mg = [m for d in dvs for m in (d["margin_b1"], d["margin_batch"])]
        ordinal = {0: "first", 1: "second", 2: "third", 3: "fourth", 4: "fifth", 5: "sixth", 6: "seventh",
                   7: "eighth", 8: "ninth", 9: "tenth"}
        k = len(idx)
        strict_s = ("neither diverges" if (k == 2 and n_st == 0) else "none diverges" if n_st == 0 else
                    f"{n_st} of {k} diverge")
        reg_s = ("both diverge" if (k == 2 and len(dvs) == 2) else "all diverge" if len(dvs) == k else
                 f"{len(dvs)} of {k} diverge")
        where = (f" at the {ordinal.get(pos[0], f'position-{pos[0]}')} emitted token" if len(pos) == 1 else
                 f" at emitted positions {', '.join(map(str, pos))}") if dvs else ""
        mg_s = f" on top-2 margins {min(mg):g} to {max(mg):g}" if mg else ""
        s += (f"; at strict fp32 {strict_s} (diag {g['commit'][:7]}); at the registered numerics {reg_s}{where}{mg_s}"
              + (", bf16 near-tie flips across the padded shape (the 14:54Z row 2 class)" if dvs else ""))
        return s

    def third_labels(self, R: dict) -> dict:
        """The labels the third table prints, each built from the files and verified verbatim against its critic entry
        (THIRD_LABELS[task]). GSM-50k (form "15:26Z"): the f_meas labels of 15:26Z row 3 (k = 8 'saving shows', k = 4
        'nominal'), the logits check's label of 15:26Z row 1 (it cites the GPU record: required for this form), the
        12:05Z deviation's label and the batch-1 timed reads' label of 14:54Z row 2. ProsQA-hard (form "21:03Z", critic
        row 2026-10-20T21:03Z row 3 (2) and (5)): the logits clause from this task's float64 record with the strict-fp32
        GPU figure beside ('not run on this task' without a GPU record), the f_meas labels around it, the deviation's
        label "shape, not path, by the same-path separation ...; logits-level: confirmed at float64 (...)" (the pending
        form only without a float64 record that meets the line), each in that row or
        a later critic entry. The f_meas dicts gain the line of record (12:40Z (a): the stricter of rule 4's printed and
        exact values) and the labels."""
        co = R.get("coconut")
        if co is None:
            raise SystemExit("REFUSED: the third table's labels need the coconut record (the f_meas denominator)")
        spec = THIRD_LABELS.get(self.task)
        if spec is None:
            raise SystemExit(f"REFUSED: no third-table label form for {self.task} (THIRD_LABELS)")
        rows, later = spec["rows"], spec["later"]
        dg = self._diag(co)
        g, f64 = dg["gpu"], dg["float64"]
        self.diag_where = f64["where"]                 # THIRD_LABELS' {dev64}: the record's device, never assumed
        c_f64 = f64["commit"][:7]
        r3 = ((co.get("eval_stats") or {}).get("rule3")) or {}
        B = int(co["config"]["eval_batch_size"])
        x1, xB = r3.get("band_bs1"), r3.get(f"band_bs{B}")
        if x1 is None or xB is None:
            raise SystemExit("REFUSED: rule 3's like-for-like reads (band [0, L] at batch 1 and at the eval batch) absent")
        s64 = f64["summary"]
        covered = set((s64.get("batch") or {}).get("idx") or []) | set(s64.get("extra_idx") or [])
        if s64.get("batch_size") not in (None, B) or (x1["over_line"] and not set(x1["correct_differ"]) <= covered):
            raise SystemExit(f"REFUSED: the float64 record {f64['path']} does not cover rule 3's like-for-like idx "
                             f"{x1['correct_differ']} at batch {B} (its batch and extra idx: {sorted(covered)})")
        if spec["form"] == "15:26Z":
            if g is None:
                raise SystemExit("REFUSED: the GSM-50k labels of record (critic entry 2026-10-20T15:26Z rows 1 and 3) cite "
                                 "the strict-fp32 GPU record: give it (--diag-gpu) or a label form without it")
            gdev = _gpu_short(g["gpu_type"]).split("-")[0]
            c_gpu = g["commit"][:7]
            logits = (f"{_sci(g['max'])} at strict fp32 on {gdev} ({g['over']} of {g['n']} over "
                      f"{_sci(g['tol'])}), <= {_sci(f64['max'], up=True)} at float64 on {f64['where']}: explained, fp32 "
                      f"kernel-order "
                      f"of the final pass (diag `{c_gpu}`; float64 record `{c_f64}`)")
            tail = (f"logits {_sci(g['max'])} at strict fp32 on {gdev}, <= {_sci(f64['max'], up=True)} at "
                    f"float64, explained; diag `{c_gpu}`, float64 record `{c_f64}`")
        else:
            gfig = ("not run on this task" if g is None else
                    f"{_sci(g['max'])} on {_gpu_short(g['gpu_type'])} ({g['over']} of {g['n']} over {_sci(g['tol'])}; "
                    f"diag {g['commit'][:7]})")
            logits = tail = (f"logits <= {_sci(f64['max'], up=True)} at float64 on {f64['where']} on this checkpoint, "
                             f"0 argmax "
                             f"differences; the fp32 kernel-order explanation as the GSM-50k records rev15 / rev16; "
                             f"float64 record {c_f64}; strict-fp32 GPU figure: {gfig}")
        self._row_has(rows["fmeas"], logits, f"the logits check's label of record ({spec['logits']})", later)
        n = self.n
        meets64 = self._diag64_meets(s64)              # the float64 line met: the confirmed forms (23:12Z row 2)
        out = {"logits": logits, "f_meas": {}, "timed": {}, "deviation": None, "diag": {
            k: {"path": v["path"], "commit": v["commit"], "max": v["max"]} for k, v in dg.items() if v}}
        rule3 = f"rule 3: {len(xB['correct_differ'])} of {n} differ at batch {B}"
        for key in ("c1b_k8", "c1b_k4"):
            r = R.get(key)
            fm = ((r or {}).get("eval_stats") or {}).get("f_meas") or {}
            b1 = fm.get("bs1") or {}
            if "f_meas" not in b1:
                raise SystemExit(f"REFUSED: no f_meas at batch 1 for {key}: the {spec['fmeas']} labels need it")
            f, printed, formula = b1["f_meas"], b1["line"], b1["line_formula"]
            rec = min(printed, formula)                                     # 12:40Z (a): the stricter reading
            rec_s = f"{printed:.2f}" if rec == printed else f"{formula:.4f}"
            dev = _gpu_short((b1.get("den") or {}).get("device") or (g or {}).get("gpu_type"))   # the timed reads'
            if spec["form"] == "15:26Z" and f <= rec:
                lab = (f"saving shows: f_meas {f:.4f} <= {rec_s} at batch 1 (A6.2; {dev}, 1 seed, valid; {rule3}; "
                       f"{tail})")
            elif spec["form"] == "15:26Z":
                lab = (f"nominal: f_meas {f:.4f} against the line {rec_s} (A6.2 rule 4 read at its stricter printed "
                       f"value, 12:40Z (a)); `layer_fraction = min(1, f_meas) = {min(1.0, f):.4f}` in K3 and 7.2; "
                       f"{dev}, 1 seed, valid; {rule3}; {tail}")
            elif f <= rec:
                lab = (f"saving shows: f_meas {f:.4f} <= {rec_s} at batch 1 (A6.2; {dev}, 1 seed, valid; {rule3}; "
                       f"{tail})")
            else:
                lab = (f"nominal: f_meas {f:.4f} against the line {rec_s} (12:40Z (a)); layer_fraction "
                       f"{min(1.0, f):.4f} in K3 and 7.2; {dev}, 1 seed, valid; {rule3}; {tail}")
            if f > 1.0:
                raise SystemExit(f"REFUSED: f_meas {f} > 1 for {key}: no saving label of any kind (rule 4)")
            self._row_has(rows["fmeas"], lab, f"the f_meas label of {key} ({spec['fmeas']})", later)
            for tag, d in fm.items():
                if "f_meas" not in d:
                    continue
                d["line_of_record"] = rec
                d["rule3"] = logits
                d["saving_label"] = lab if (tag == "bs1" and f <= rec) else None
                if tag == "bs1":
                    d["label"] = lab
            out["f_meas"][key] = {"label": lab, "line_of_record": rec, "saving_shows": f <= rec}
        idx = ", ".join(map(str, x1["correct_differ"]))
        if x1["over_line"] and spec["form"] == "15:26Z":
            d3 = ((g["summary"].get("results") or {}).get("checks") or {}).get("3") or {}
            d4 = ((g["summary"].get("results") or {}).get("checks") or {}).get("4") or {}
            mg = [m for dv in (d3.get("registered") or {}).get("divergences") or [] for m in (dv["margin_b1"],
                                                                                            dv["margin_batch"])]
            sw = max(float(((d4.get(nm) or {}).get(q) or {}).get("max", float("inf"))) for nm in ("strict", "registered")
                     for q in ("last_swap", "logits_swap", "latents_swap"))
            ds = d3.get("strict") or {}
            out["deviation"] = (
                f"{len(x1['correct_differ'])} of {n} at batch 1 (idx {idx}), {len(xB['correct_differ'])} of {n} at batch "
                f"{B}; shape, not path (12:40Z (e)); explained: bf16 near-tie flips across the padded shape at the second "
                f"token (margins {min(mg):g} to {max(mg):g} at bf16; {ds.get('diverging')} of {ds.get('n')} diverge at "
                f"strict fp32; same-shape swap {sw}; diag `{c_gpu}`)")
        elif x1["over_line"]:
            out["deviation"] = (
                f"{len(x1['correct_differ'])} of {n} at batch 1 (idx {idx}), {len(xB['correct_differ'])} of {n} at batch "
                f"{B}; shape, not path, by the same-path separation (critic entry 2026-10-20T21:03Z row 3); logits-level"
                + (": " + self._logits_level_confirmed(x1["correct_differ"], f64, g) if meets64
                   else " confirmation pending"))
        if out["deviation"]:
            self._row_has(rows["batch1"], out["deviation"],
                          f"the {spec['deviation']} deviation's label of record ({spec['batch1']})", later)
        explained = spec["form"] == "15:26Z"
        self.diag64_meets = meets64
        timed = (f"timing of record (A6.2 rule 2); accuracy {x1['k']} of {n} at batch 1 printed beside {xB['k']} at "
                 f"batch {B}, the batch-{B} read being the accuracy of record (rule 3's like-for-like: "
                 f"{len(xB['correct_differ'])} of {n} differ at batch {B})"
                 + (("; the batch-1 difference explained as above" if explained else
                     "; the batch-1 difference as above: shape, not path; logits-level "
                     + ((f"confirmed (float64 record {c_f64}"
                         + (f"; diag {g['commit'][:7]}" if g is not None else "") + ")") if meets64
                        else "confirmation pending"))
                    if x1["over_line"] else ""))
        self._row_has(rows["batch1"], timed, f"the batch-1 timed reads' label ({spec['batch1']})", later)
        out["timed"][x1["entry"]] = timed + (f" [the {spec['deviation']} deviation's label of record: "
                                             f"{out['deviation']}]" if out["deviation"] else "")
        for key in ("c1b_k8", "c1b_k4"):          # the C1b batch-1 reads: the same label with their own counts
            ents = (R[key].get("eval_ents") or {})
            e1, eB = ents.get("timed_bs1"), ents.get("none")
            if e1 is None or eB is None:
                continue
            dc = [i for i, _ in rows_differ(eB["rows"], e1["rows"], ["correct"])]
            de = len(rows_differ(eB["rows"], e1["rows"], ["emitted_tokens"]))
            out["timed"][e1["name"]] = (
                f"timing of record (A6.2 rule 2); accuracy {e1['k']} of {n} at batch 1 printed beside {eB['k']} at batch "
                f"{eB['batch']}, the batch-{eB['batch']} read being the accuracy of record ({len(dc)} of {n} differ in "
                f"correctness between the two batch sizes, {de} in E'; the 14:54Z row 2 label with this read's counts)")
        return out

    # ------------------------------------------------------------------------ texts of the eval-read lines
    def _eval_missing(self, r: dict, kinds) -> str:
        have = (self.ev["by_run"].get(r["key"]) or {}) if self.ev else {}
        why = []
        for k in kinds:
            e = have.get(k)
            if e is None:
                why.append(f"no {k} read of {r['run']} in the eval job")
            elif not e.get("ok"):
                why.append(f"{e['name']} enters no cell ({e.get('exclusion') or '; '.join(e['problems'])})")
        return "; ".join(why) or "eval read missing"

    def _cost_text(self, r: dict, bar: dict, g: dict, billing: bool) -> str:
        c, bc = r["cost"], bar["cost"]

        def rt(a):
            lo, hi = interval_ratio(a, bc["flops_mean"])
            return f"[{lo:.3f}, {hi:.3f}]"
        if "cost_nominal" in r:
            a, b = r["config"]["latent_band"]
            parts = [f"nominal {b - a}/{r['n_L']}: {_gf(r['cost_nominal']['flops_mean'][0])} GFLOPs, ratio "
                     f"{rt(r['cost_nominal']['flops_mean'])}"]
            if "cost_measured" in r:
                parts.append(f"measured min(1, f_meas) = {r['cost_measured']['layer_fraction']:.4f}: "
                             f"{_gf(r['cost_measured']['flops_mean'][0])} GFLOPs, ratio "
                             f"{rt(r['cost_measured']['flops_mean'])}")
            parts.append(f"at 1: {_gf(c['flops_mean'][0])} GFLOPs, ratio {rt(c['flops_mean'])}")
            fl = "; ".join(parts)
        else:
            fl = f"{_gf(c['flops_mean'][0])} GFLOPs, ratio {rt(c['flops_mean'])}"
        return (f"7.2 cost legs, {r['run']} vs the bar: FLOP leg (mean FLOPs <= bar's) {g['flop']} (exact mean from "
                f"{c['entry']}'s per-example E': {fl}; against the bar's range {_rng(bc['flops_mean'])} GFLOPs, whose "
                f"per-example E' was not recorded); token leg (E' <= 0.2 x bar's) {g['token']} "
                f"({c['emitted_mean']:.3f} vs 0.2 x {bc['emitted_mean']:.2f} = {bc['emitted_mean'] * 0.2:.2f}); passes "
                f"criterion (<= 0.2 x bar's) {g['passes']} ({c['passes_mean']:.3f} vs {bc['passes_mean'] * 0.2:.2f})"
                + ("; label: billing claim only (7.2)" if billing else "")
                + (f"; accounting-sensitive ({', '.join(g['accounting_sensitive'])})" if g["accounting_sensitive"]
                   else "; 5.5 (a), (c): no flip")
                + "; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED "
                  "(the arm's own batch-1 timing, where it exists, is its 5.5 (b) line)")

    def _k2_text(self, r: dict, pz: dict | None, k2: dict, o: str, why: list, sigma: float) -> str:
        n, S = self.n, Fraction(sigma)
        c2lab = f" ({self.spec['c2_sentences']['c2'].split(';')[0]})" if r["key"] == "c2" else ""
        txt = (f"K2 (7.5; A3.4: every latent arm at the screen), {r['run']}{c2lab}: Delta_scr = "
               f"{f1(pts(k2['k_a'], n))} - "
               f"{f1(pts(k2['k_b'], n))} = {float(k2['delta']):+.1f} ({k2['k_a']} - {k2['k_b']} of {n}, paired per "
               f"example: {k2['only_a']} correct only with own latents, {k2['only_b']} only with scrambled; "
               f"{k2['entries'][0]} vs {k2['entries'][1]}), 95% interval [{k2['lo']:+.2f}, {k2['hi']:+.2f}] (6.2: "
               f"Delta +- 1.96 sqrt(Var_ex), Var_ex over {k2['n_boot']:,} paired example draws, default_rng("
               f"{k2['seed']}); percentile [{k2['pct_lo']:+.1f}, {k2['pct_hi']:+.1f}]) against sigma_screen "
               f"{f2(sigma)}: ")
        if o == "fired":
            txt += (f"fired ({'; '.join(why)}): the arm's latent claim is killed at the screen on this task (A3.4); "
                    f"it continues only as a labelled curriculum or token-budget effect (7.5; its parenthetical label "
                    f"is withheld under A11.3)")
        else:
            txt += "not fired (Delta_scr >= sigma_screen and the interval excludes 0)"
        zd = (r.get("eval_stats") or {}).get("zero_drop")
        if zd is not None:
            txt += (f"; beside it: zero-latent drop {f1(pts(zd['k_a'], n))} - {f1(pts(zd['k_b'], n))} = "
                    f"{float(zd['delta']):+.1f} [{zd['lo']:+.2f}, {zd['hi']:+.2f}] ({zd['entries'][1]})")
        if pz is not None:
            d = pts(r["k"], n) - pts(pz["k"], n)
            within = abs(d) <= S
            txt += (f"; pause comparison {r['run']} - {pz['run']} = {float(d):+.1f}, "
                    + (f"within sigma_screen {f2(sigma)}: CURRICULUM-ONLY" if within else
                       f"not within sigma_screen {f2(sigma)}"))
        return txt

    def _null_trigger_lines(self, R: dict, latent_keys: list, kb: int, sigma: float, lab: str) -> list[dict]:
        n, S = self.n, Fraction(sigma)
        bar_p = pts(kb, n)
        items, hits, missing = [], [], []
        for key in latent_keys:
            r = R[key]
            if not r.get("feedback"):
                continue
            st = r.get("eval_stats") or {}
            reads = st.get("reads") or {}
            arm_vals = [pts(r["k"], n)] + ([pts(reads["none"]["k"], n)] if "none" in reads else [])
            got = []
            for kind, nm in (("scrambled", "scrambled"), ("zero_latent", "zero-latent")):
                if kind not in reads:
                    missing.append(f"{r['run']} {nm}")
                    continue
                v = pts(reads[kind]["k"], n)
                h = null_trigger_hits(v, bar_p, arm_vals, sigma)
                got.append(f"{nm} {f1(v)}" + (" (a diagnostic for this arm, A6.1 rule 3)" if kind == "zero_latent"
                                              and r["config"].get("dep_loss") else ""))
                hits += [f"{r['run']} {nm} {f1(v)}: {x}" for x in h]
            items.append(f"{r['run']} {', '.join(got) or 'no read'} (arm {f1(min(arm_vals))} + {f2(sigma)} = "
                         f"{f2(min(arm_vals) + S)})")
        if hits:
            o = "triggered"
        elif not items or len(missing) == 2 * len(items):
            o = "UNTESTED"
        else:
            o = "not triggered" + ("; partly UNTESTED" if missing else "")
        txt = (f"Null triggers (7.10; A3.4), the scrambled-at-inference and zero-latent reads on valid against the "
               f"bar + 2 sigma_screen = {f1(bar_p)} + {f2(2 * sigma)} = {f2(bar_p + 2 * S)} and against the arm + "
               f"sigma_screen: " + "; ".join(items) + ": "
               + ("TRIGGERED: " + "; ".join(hits) + ": every announcement and further run on this task waits for the "
                  "7.10 checklist, done and logged" if hits else "none above either line")
               + (f"; UNTESTED: {', '.join(missing)}" if missing else "") + f" {lab}")
        return [{"id": "null_trigger", "kind": "null_trigger", "run": None, "outcome": o, "text": txt}]

    def _f_meas_text(self, r: dict, co: dict | None) -> tuple[str, str]:
        a, b = r["config"]["latent_band"]
        nl = r["n_L"]
        fm = (r.get("eval_stats") or {}).get("f_meas") or {}
        b1 = fm.get("bs1") or {"untested": "no f_meas computed"}
        B = int(r["config"]["eval_batch_size"])
        bB = fm.get(f"bs{B}") or {"untested": "no f_meas computed"}
        if "untested" in b1:
            return ("nominal; f_meas UNTESTED; rule 3 logits check pending",
                    f"C1b per-step cost (A3.4; A6.2), {r['run']}: f_meas UNTESTED at batch 1 ({b1['untested']}): "
                    f"nominal; FLOPs printed at the nominal {b - a}/{nl} and at 1, the FLOP leg read at 1; rule 3 "
                    f"logits check pending")
        f, num, den = b1["f_meas"], b1["num"], b1["den"]
        txt = (f"C1b per-step cost (A3.4; A6.2), {r['run']}: f_meas = latent_ms_mean {num['latent_ms']:.3f} ms (band "
               f"[{a}, {b}), {num['entry']}) / {den['latent_ms']:.3f} ms (full-depth coconut through band [0, {nl}], "
               f"{den['entry']}) = {f:.4f} at batch 1, the fraction of record (latent passes {num['latent_passes']} / "
               f"{den['latent_passes']} over {num['batches']} / {den['batches']} timed batches; prefill "
               f"{num['prefill_ms']:.3f} / {den['prefill_ms']:.3f} ms, token {num['token_ms']:.3f} / "
               f"{den['token_ms']:.3f} ms; latent_last: not recorded (run_eval.py's summary keeps prefill, latent and "
               f"token only); {num['device']}, torch {num['torch']}, both reads)")
        if "untested" in bB:
            txt += f"; at the eval batch {B}: UNTESTED ({bB['untested']})"
        else:
            txt += (f"; at the eval batch {B} beside it: {bB['num']['latent_ms']:.3f} / {bB['den']['latent_ms']:.3f} = "
                    f"{bB['f_meas']:.4f} ({bB['num']['entry']} / {bB['den']['entry']}, {bB['num']['batches']} / "
                    f"{bB['den']['batches']} timed batches)")
        line, formula = b1["line"], b1["line_formula"]
        if b1["above_one"]:
            rd = "f_meas above 1: the band costs more than full depth: no saving label of any kind (rule 4)"
            o = "nominal (f_meas > 1: no saving label)"
        elif b1["threshold_met"]:
            rd = (f"saving line f_meas <= {line:.2f} (rule 4) met at batch 1"
                  + (f"; the formula (b - a) / L + 0.10 = {formula:.4f} would not be met (the readings of rule 4 "
                     f"differ at this value)" if b1["readings_disagree"] else ""))
            o = "threshold met"
        else:
            rd = (f"saving line f_meas <= {line:.2f} (rule 4) not met at batch 1: nominal, layer_fraction = "
                  f"min(1, f_meas) = {b1['measured_fraction']:.4f} in K3 and 7.2"
                  + (f"; the formula (b - a) / L + 0.10 = {formula:.4f} would be met: the two readings of rule 4 "
                     f"differ at this value (for the critic)" if b1["readings_disagree"] else ""))
            o = "nominal"
        if self.labels3 is not None and b1.get("label"):          # the third table (critic entry 2026-10-20T15:26Z)
            rec = b1["line_of_record"]
            met = b1["f_meas"] <= rec
            txt += (f"; rule 4's line under both readings: {line:.2f} printed, {formula:.4f} exact; the line of record is "
                    f"the stricter, {rec:.2f}" if rec == line else f"; rule 4's line under both readings: {line:.2f} "
                    f"printed, {formula:.4f} exact; the line of record is the stricter, {rec:.4f}")
            txt += (f" (critic entry 2026-10-20T12:40Z (a)): {'met' if met else 'not met'} at batch 1"
                    + ("" if met else f", so layer_fraction = min(1, f_meas) = {b1['measured_fraction']:.4f} in K3 and 7.2")
                    + f"; label ({THIRD_LABELS[self.task]['fmeas']}): \"{b1['label']}\"; FLOPs printed at the nominal "
                    f"{b - a}/{nl}, at the measured {b1['measured_fraction']:.4f} and at 1, and the FLOP leg holds only "
                    f"if it holds at all three")
            return ("saving shows" if met else "nominal"), txt
        txt += (f"; {rd}; no saving label is written: rule 3 logits check pending (critic entry 2026-10-20T10:51Z part "
                f"(3)); FLOPs printed at the nominal {b - a}/{nl}, at the measured {b1['measured_fraction']:.4f} and "
                f"at 1, and the FLOP leg holds only if it holds at all three")
        return o + "; rule 3 logits check pending", txt

    def _eval_lines(self, R: dict, latent_keys: list, bar: dict, lab: str) -> list[dict]:
        n, out, ev = self.n, [], self.ev
        job = ev["job"]
        n_ok = sum(1 for e in ev["entries"] if e["ok"])
        out.append({"id": "eval_job", "kind": "eval_check", "run": None,
                    "outcome": "ok" if not job["problems"] and n_ok == len(ev["entries"]) else "problems",
                    "text": f"Eval reads: job {job['job']}, attempt {job['attempt']}, registered row "
                            f"{job['registered_read']}, receipt {job['status']} ({job['started']} to {job['ended']}, "
                            f"code {str(job['code_commit'])[:7]}, env_reference "
                            f"{'ok' if job['env_reference'] else 'NOT ok'}): {n_ok} of {len(ev['entries'])} entries "
                            f"pass every check (receipt row, file hashes, valid split hash, A10.5 numerics, the "
                            f"selected checkpoint's sha256 and epoch, the 5.1 / 5.3 cost of every row)"
                            + (f"; problems: {'; '.join(job['problems'])}" if job["problems"] else "") + f" {lab}"})
        hj = self.ev_h
        if hj is not None:                         # the fourth table: the hinge eval job (critic entry 17:12Z (6))
            jh, dirs = hj["job"], list((hj.get("directional") or {}).values())
            h_ok = sum(1 for e in hj["entries"] if e["ok"])
            d_ok = sum(1 for e in dirs if e.get("ok"))
            out.append({"id": "eval_job:hinge", "kind": "eval_check", "run": None,
                        "outcome": "ok" if not jh["problems"] and h_ok == len(hj["entries"]) and d_ok == len(dirs)
                        else "problems",
                        "text": f"Eval reads of the hinge arms: job {jh['job']}, attempt {jh['attempt']}, registered row "
                                f"{jh['registered_read']}, receipt {jh['status']} ({jh['started']} to {jh['ended']}, code "
                                f"{str(jh['code_commit'])[:7]}, env_reference "
                                f"{'ok' if jh['env_reference'] else 'NOT ok'}): {h_ok} of {len(hj['entries'])} reads "
                                f"pass every check (as the first job's); {d_ok} of {len(dirs)} donor-answer records (A6.1 "
                                f"rule 4 (B), critic entry {DONOR_ROW}) pass every check (the registering row, the selected "
                                f"checkpoint, the valid split's hash, the registered set's numerics, the class pinned by "
                                f"ruling 1, the statistic recomputed from examples.csv); the coconut reference record "
                                f"decides nothing and is printed in the notes only (17:36Z (5))"
                                + (f"; problems: {'; '.join(jh['problems'])}" if jh["problems"] else "") + f" {lab}"})
        xj = self.ev_x
        if xj is not None:                         # the no-curriculum reissue: the arm's eval job (row (b) (4))
            jx = xj["job"]
            x_ok = sum(1 for e in xj["entries"] if e["ok"])
            out.append({"id": "eval_job:extra", "kind": "eval_check", "run": None,
                        "outcome": "ok" if not jx["problems"] and x_ok == len(xj["entries"]) else "problems",
                        "text": f"Eval reads of the no-curriculum control: job {jx['job']}, attempt {jx['attempt']}, "
                                f"registered row {jx['registered_read']}, receipt {jx['status']} ({jx['started']} to "
                                f"{jx['ended']}, code {str(jx['code_commit'])[:7]}, env_reference "
                                f"{'ok' if jx['env_reference'] else 'NOT ok'}): {x_ok} of {len(xj['entries'])} reads pass "
                                f"every check (as the first job's; critic entry {CLOSING_ROW} (b) (4))"
                                + (f"; problems: {'; '.join(jx['problems'])}" if jx["problems"] else "") + f" {lab}"})
        for e in ev["entries"] + (hj["entries"] if hj is not None else []) + (xj["entries"] if xj is not None else []):
            if not e["ok"]:
                out.append({"id": f"eval_excluded:{e['name']}", "kind": "eval_check", "run": e["run"],
                            "outcome": e.get("exclusion") or "excluded",
                            "text": f"Eval read {e['name']} enters no cell: {'; '.join(e['problems'])} {lab}"})
        for e in (ev.get("directional") or {}).values():
            if not e.get("ok"):
                out.append({"id": f"eval_excluded:{e['name']}", "kind": "eval_check", "run": e["run"],
                            "outcome": "excluded",
                            "text": f"Directional record {e['name']} (A6.1 rule 4 (B)) enters no cell: "
                                    f"{'; '.join(e['problems'])}; the arm keeps rule 6's label {lab}"})
        for key in latent_keys:
            r = R[key]
            st = r.get("eval_stats") or {}
            nv = st.get("none_vs_log")
            if nv is None:
                continue
            ent = r["eval_ents"]["none"]
            agree = nv["agree"]
            out.append({"id": f"eval_none:{key}", "kind": "eval_check", "run": r["run"],
                        "outcome": "agree" if agree else "disagree; the training-log value stays the selection",
                        "text": f"Plain read against the selection, {r['run']}: {ent['name']} {nv['k_eval']} of {n} "
                                f"({f1(pts(nv['k_eval'], n))}) vs the training log's selection {nv['k_log']} of {n} "
                                f"({f1(pts(nv['k_log'], n))}, epoch {r['selected']['epoch']}; same checkpoint, same "
                                f"numerics): " + ("agree" if agree else "DISAGREE: reported; the training-log value "
                                                  "stays the selection (3.3)")
                                + f"; emitted tokens {nv['M_eval']} vs the log's {nv['M_log']} "
                                + ("(equal)" if nv["M_eval"] == nv["M_log"] else "(differ: the costs use the eval "
                                   "read's per-example file)") + f" {lab}"})
        for key in latent_keys:
            r = R[key]
            st = r.get("eval_stats") or {}
            if not r.get("feedback"):
                continue
            sw = st.get("sweep") or {}
            chk = st.get("n0_zero")
            if not sw:
                out.append({"id": f"sweep:{key}", "kind": "sweep", "run": r["run"], "outcome": "UNTESTED",
                            "text": f"Step sweep (7.3; section 8), {r['run']}: UNTESTED: no sweep read {lab}"})
                continue
            missing = [nn for nn in SWEEP if nn not in sw]
            ind = [sw[nn]["k"] for nn in sorted(sw) if nn <= r["L"]]
            nondec = all(x <= y for x, y in zip(ind, ind[1:]))
            pts_txt = ", ".join(
                f"n = {nn} {sw[nn]['points']:.1f}" + (" (the plain read: the trained count)" if sw[nn]["plain_read"]
                                                       else "")
                + (f" (out-of-distribution: {nn} > {r['L']})" if sw[nn]["ood"] else "") for nn in sorted(sw))
            if chk is None:
                ck, co = "the n = 0 against zero-latent check: UNTESTED (a read is missing)", "check UNTESTED"
            elif chk["agree"]:
                ck, co = (f"the n = 0 read and the zero-latent read agree on every example (0 of {n} differ in any "
                          f"column of examples.csv; a check of the eval path, not a read; A10.4)"), "n = 0 = zero-latent"
            else:
                ck, co = (f"the n = 0 read and the zero-latent read DIFFER on {chk['examples_differing']} of {n} "
                          f"examples ({', '.join(chk['fields'])}): BUG trigger (A10.4)"), "n = 0 != zero-latent: BUG"
            dep = ("; for this arm (dep_loss) the zero-latent and n = 0 reads are diagnostics that decide nothing "
                   "(A6.1 rule 3; a collapse is reported as such, not killed: A3.4's degenerate-solution note)"
                   if r["config"].get("dep_loss") else "")
            out.append({"id": f"sweep:{key}", "kind": "sweep", "run": r["run"], "outcome": f"reported; {co}",
                        "text": f"Step sweep (7.3, section 8: reported, decides nothing), {r['run']}: {pts_txt}"
                                + (f"; missing: n = {', '.join(map(str, missing))}" if missing else "")
                                + f"; non-decreasing from n = 0 to the trained n = {r['L']}: "
                                + ("yes" if nondec else "no") + f" (descriptive); {ck}{dep} {lab}"})
        co = R.get("coconut")
        r3 = ((co or {}).get("eval_stats") or {}).get("rule3")
        if co is not None and r3:
            n_L = self.dims("openai-community/gpt2", V_LATENT).n_L
            parts, over = [], []
            for kind in sorted(r3, key=lambda k_: int(k_.split("bs")[1])):
                x = r3[kind]
                parts.append(f"band [0, {n_L}] at batch {x['batch']} {x['k']} of {n} ({f1(pts(x['k'], n))}; "
                             f"{x['entry']}): {len(x['correct_differ'])} examples differ in correctness from the "
                             f"HF-path plain read {x['k_hf']} of {n} ({x['hf_entry']}, batch {x['hf_batch']})"
                             + (f" (idx {', '.join(map(str, x['correct_differ']))})" if x["correct_differ"] else "")
                             + f", net {x['net']:+d}, {x['emitted_differ']} differ in E'")
                if x["over_line"]:
                    over.append(f"batch {x['batch']}")
            o = (f"over the line at {', '.join(over)}: deviations.md" if over else "within the line") \
                + "; logits check pending"
            if self.labels3 is not None:                   # the third table (THIRD_LABELS: the task's label rows)
                L3, T3 = self.labels3, THIRD_LABELS[self.task]
                dv = ("explained" if T3["form"] == "15:26Z" else
                      "shape, not path; logits-level " + ("confirmed" if self.diag64_meets
                                                           else "confirmation pending"))
                o = ((f"over the line at {', '.join(over)}: deviations.md {T3['deviation']}, {dv} "
                      f"({T3['batch1']})" if over else "within the line")
                     + f"; logits check {T3['logits_status'].replace('{dev64}', str(self.diag_where))} "
                       f"({T3['logits']})")
                out.append({"id": "rule3:coconut", "kind": "rule3", "run": co["run"], "outcome": o,
                            "text": f"A6.2 rule 3 like-for-like check, {co['run']}: " + "; ".join(parts) + ": "
                                    + (f"more than {RULE3_MAX_DIFF} in {n} at {', '.join(over)}: deviations.md "
                                       f"{T3['deviation']}; its label of record ({T3['batch1']}): \"{L3['deviation']}\"; "
                                       f"the timing is still reported" if over else
                                       f"at most {RULE3_MAX_DIFF} in {n}: within the line")
                                    + f"; {T3['logits_check'].replace('{dev64}', str(self.diag_where))}, label of "
                                      f"record ({T3['logits']}): \"{L3['logits']}\" "
                                    + lab})
            else:
                out.append({"id": "rule3:coconut", "kind": "rule3", "run": co["run"], "outcome": o,
                            "text": f"A6.2 rule 3 like-for-like check, {co['run']}: " + "; ".join(parts) + ": "
                                    + (f"more than {RULE3_MAX_DIFF} in {n} at {', '.join(over)}: goes to deviations.md "
                                       f"with the operator's explanation (this tool writes no deviation); the timing is "
                                       f"still reported" if over else f"at most {RULE3_MAX_DIFF} in {n}: within the line")
                                    + "; rule 3's strict-fp32 first-token logits check: pending (not in this job) "
                                    + lab})
        for key in latent_keys:
            r = R[key]
            tm = ((r.get("eval_stats") or {}).get("timing")) or {}
            own = tm.get("timed_bs1")
            if own is not None:
                per = (own["prefill_ms"] * own["batches"] + own["latent_ms"] * own["latent_passes"]
                       + own["token_ms"] * own["token_passes"]) / n
                o = "measured; comparison with the bar UNTESTED"
                txt = (f"5.5 (b) wall-clock at batch 1 (2.2), {r['run']} on {own['device']} (torch {own['torch']}), "
                       f"its own path ({own['entry']}): {per:.2f} ms per example in timed passes (prefill "
                       f"{own['prefill_ms']:.3f} + {own['latent_passes'] / n:g} latent x {own['latent_ms']:.3f} + "
                       f"{own['token_passes'] / n:g} token x {own['token_ms']:.3f}; the latent_last pass is not "
                       f"recorded), {1000 * own['eval_seconds'] / n:.2f} ms per example for the whole read")
            elif key in ("coconut", "coconut_no_curriculum") and tm.get("band_bs1") is not None:
                bd = tm["band_bs1"]
                hf = tm.get("none")
                per = (bd["prefill_ms"] * bd["batches"] + bd["latent_ms"] * bd["latent_passes"]
                       + bd["token_ms"] * bd["token_passes"]) / n
                o = "UNTESTED (HF path timed at batch 32 only)"
                what = ("A6.2's denominator, never this arm's 5.5 (b), rule 1" if key == "coconut" else
                        "the full-depth band path, as the coconut control's; never this arm's 5.5 (b), A6.2 rule 1")
                txt = (f"5.5 (b) wall-clock at batch 1 (2.2), {r['run']}: its own HF path was timed at batch "
                       f"{hf['batch'] if hf else 'none'} only"
                       + (f" ({hf['latent_ms']:.3f} ms per latent pass, {hf['token_ms']:.3f} per token pass)" if hf
                          else "") + f": UNTESTED at batch 1; the band [0, {r.get('n_L', 12)}] batch-1 read "
                       f"({bd['entry']}; {what}) gives {per:.2f} ms per example in timed passes")
            else:
                o = "UNTESTED"
                txt = f"5.5 (b) wall-clock at batch 1 (2.2), {r['run']}: UNTESTED: no batch-1 timed read"
            if self.labels3 is not None:                   # the third table: the batch-1 reads' label (14:54Z row 2)
                b1 = (tm.get("timed_bs1") or tm.get("band_bs1") or {}).get("entry") if (
                    own is not None or key == "coconut") else None
                if b1 in self.labels3["timed"]:
                    txt += (f"; batch-1 read label ({THIRD_LABELS[self.task]['batch1']}): "
                            f"\"{self.labels3['timed'][b1]}\"")
            out.append({"id": f"wallclock:{key}", "kind": "wallclock", "run": r["run"], "outcome": o,
                        "text": txt + f"; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison {lab}"})
        return out

    def _card_lines(self, R: dict, sigma: float, lab: str) -> list[dict]:
        n, S, out = self.n, Fraction(sigma), []

        def P(r):
            return pts(r["k"], n)

        def d(a, b):
            x = P(a) - P(b)
            return x, f"{x.numerator / x.denominator:+.1f}"

        other = "the other screen task is not in this table"
        coco = R.get("coconut")
        for key in ("c1b_k8", "c1b_k4"):
            r = R.get(key)
            if r is None or coco is None:
                out.append({"id": f"card:{key}", "kind": "card", "run": None, "outcome": "UNTESTED",
                            "text": f"C1b {key[-2:]} (A3.4): UNTESTED on {self.spec['title']} (no record of "
                                    f"{'the arm' if r is None else 'full-depth coconut'})"})
                continue
            x, xs = d(r, coco)
            pass_acc = x >= -S
            kill_acc = x < -2 * S
            if self.ev is not None:
                out.append(self._c1b_card_eval(key, r, coco, R.get("pause"), x, xs, pass_acc, kill_acc, sigma, lab,
                                               other))
                continue
            o = "pending K2" if pass_acc else "not met on this task"
            out.append({"id": f"card:{key}", "kind": "card", "run": r["run"],
                        "outcome": f"pass: {o}; kill part: {'met' if kill_acc else 'not met'}",
                        "text": f"C1b k = {key[-1]} (card line, A3.4), {r['run']} - {coco['run']} = {xs}: pass "
                                f"needs within sigma_screen {f2(sigma)} of full-depth coconut (or better) and K2 not "
                                f"fired: accuracy part {'met' if pass_acc else 'not met'}, K2 pending, pass {o}; kill "
                                f"needs both tasks: this task's accuracy part (below by more than 2 sigma_screen = "
                                f"{f2(2 * sigma)}) {'met' if kill_acc else 'not met'}, K2 part pending; {other}; "
                                f"per-step cost nominal {lab}"})
        c2, sh = R.get("c2"), R.get("c2_shuffled")
        if c2 is None or sh is None or coco is None:
            out.append({"id": "card:c2", "kind": "card", "run": None, "outcome": "UNTESTED",
                        "text": f"C2 (A3.4): UNTESTED on {self.spec['title']} (a record of C2, C2-shuffled or "
                                f"full-depth coconut is missing)"})
        else:
            x, xs = d(c2, sh)
            y, ys = d(c2, coco)
            pass_acc = x > S
            within = abs(x) <= S
            below = y < -2 * S
            if self.ev is not None:
                out.append(self._c2_card_eval(c2, sh, coco, R.get("pause"), xs, ys, pass_acc, within, below, sigma,
                                              lab, other))
            else:
                out.append({"id": "card:c2", "kind": "card", "run": c2["run"],
                            "outcome": f"pass: {'pending K2' if pass_acc else 'not met on this task'}; kill parts: "
                                       f"within-null {'met' if within else 'not met'}, below-coconut "
                                       f"{'met' if below else 'not met'}",
                            "text": f"C2 (card line, A3.4), {c2['run']} - {sh['run']} = {xs} (the gap includes "
                                    f"learnable vs unlearnable auxiliary target), {c2['run']} - {coco['run']} = {ys}: "
                                    f"pass needs above C2-shuffled by more than sigma_screen {f2(sigma)} with K2 not "
                                    f"fired: {'accuracy part met, K2 pending' if pass_acc else 'not met on this task'}; "
                                    f"kill needs one clause on both tasks: this task's parts: within sigma_screen of "
                                    f"C2-shuffled {'met' if within else 'not met'}, below full-depth coconut by more "
                                    f"than 2 sigma_screen {'met' if below else 'not met'}, K2 pending; {other} {lab}"})
        for key, null in (("c3", "c3_foreign"), ("c2c3", "c3_foreign")):
            r = R.get(key)
            if r is None:
                out.append({"id": f"card:{key}", "kind": "card", "run": None, "outcome": "UNTESTED",
                            "text": f"{'C3' if key == 'c3' else 'C2+C3'} (A3.4): UNTESTED on {self.spec['title']} "
                                    f"(no run record)"})
                continue
            if self.ev is not None:
                out.append(self._c3_card_eval(key, r, R, sigma, lab, other))
                continue
            y, ys = d(r, coco) if coco else (None, "n/a")
            out.append({"id": f"card:{key}", "kind": "card", "run": r["run"], "outcome": "pending",
                        "text": f"{'C3' if key == 'c3' else 'C2+C3'} (card line, A3.4), {r['run']}: the scrambled-drop "
                                f"parts of its pass and kill lines are pending a scrambled-at-inference valid read "
                                f"(K2 not independent, A6.1); accuracy vs full-depth coconut {ys}; {other} {lab}"})
        return out

    def _k2_brief(self, r: dict) -> tuple[str, str]:
        """(status, text) of an arm's K2 at the screen as this lines() call decided it."""
        o = r.get("_k2", "pending")
        k2 = (r.get("eval_stats") or {}).get("k2")
        if k2 is None or o not in ("fired", "not fired"):
            return o, f"K2 {o}"
        return o, (f"K2 {o} (Delta_scr {float(k2['delta']):+.1f}, interval [{k2['lo']:+.1f}, {k2['hi']:+.1f}])")

    def _c1b_card_eval(self, key, r, coco, pz, x, xs, pass_acc, kill_acc, sigma, lab, other) -> dict:
        """A3.4's C1b lines with K2 decided at the screen (k = 8 and k = 4 judged independently by the same lines)."""
        S = Fraction(sigma)
        k2o, k2t = self._k2_brief(r)
        if not pass_acc:
            po = "not met on this task"
        elif k2o == "not fired":
            po = "met on this task"
        elif k2o == "fired":
            po = "not met on this task: K2 fired"
        else:
            po = f"pending K2 ({k2o})"
        kill_k2 = {"fired": "met", "not fired": "not met"}.get(k2o, "pending")
        cur = ""
        if k2o == "fired":
            cur = ("; K2 fired at the screen: the arm's latent claim is killed at the screen on this task (A3.4)"
                   + (f"; CURRICULUM-ONLY (pause comparison {float(pts(r['k'], self.n) - pts(pz['k'], self.n)):+.1f} "
                      f"within sigma_screen {f2(sigma)})" if pz is not None and
                      abs(pts(r["k"], self.n) - pts(pz["k"], self.n)) <= S else ""))
        fm = ((r.get("eval_stats") or {}).get("f_meas") or {}).get("bs1") or {}
        if self.labels3 is not None and fm.get("label"):        # critic entry 2026-10-20T15:26Z row 3, verbatim
            o_cost = fm["label"]
        elif "f_meas" in fm:
            o_cost = (f"f_meas {fm['f_meas']:.4f} at batch 1 against the line {fm['line']:.2f}: "
                      + ("threshold met" if fm["threshold_met"] else "nominal")
                      + "; no saving label (rule 3 logits check pending)")
        else:
            o_cost = "nominal (f_meas UNTESTED); rule 3 logits check pending"
        return {"id": f"card:{key}", "kind": "card", "run": r["run"],
                "outcome": f"pass: {po}; kill parts: accuracy {'met' if kill_acc else 'not met'}, K2 {kill_k2}",
                "text": f"C1b k = {key[-1]} (card line, A3.4), {r['run']} - {coco['run']} = {xs}: pass needs within "
                        f"sigma_screen {f2(sigma)} of full-depth coconut (or better) and K2 not fired: accuracy part "
                        f"{'met' if pass_acc else 'not met'}, {k2t}: card pass line {po} (6.3's screen rule is its "
                        f"own line){cur}; kill needs both tasks (below full-depth coconut by more than 2 "
                        f"sigma_screen = {f2(2 * sigma)}, or K2 fired): this task's accuracy part "
                        f"{'met' if kill_acc else 'not met'}, K2 part {kill_k2}; {other}; per-step cost: {o_cost} {lab}"}

    def _c2_card_eval(self, c2, sh, coco, pz, xs, ys, pass_acc, within, below, sigma, lab, other) -> dict:
        S = Fraction(sigma)
        k2o, k2t = self._k2_brief(c2)
        if not pass_acc:
            po = "not met on this task"
        elif k2o == "not fired":
            po = "met on this task"
        elif k2o == "fired":
            po = "not met on this task: K2 fired"
        else:
            po = f"pending K2 ({k2o})"
        kill_k2 = {"fired": "met", "not fired": "not met"}.get(k2o, "pending")
        cur = ""
        if k2o == "fired":
            cur = "; K2 fired at the screen: the arm's latent claim is killed at the screen on this task (A3.4)"
            if pz is not None and abs(pts(c2["k"], self.n) - pts(pz["k"], self.n)) <= S:
                cur += "; CURRICULUM-ONLY"
        return {"id": "card:c2", "kind": "card", "run": c2["run"],
                "outcome": f"pass: {po}; kill parts: within-null {'met' if within else 'not met'}, below-coconut "
                           f"{'met' if below else 'not met'}, K2 {kill_k2}",
                "text": f"C2 (card line, A3.4), {c2['run']} - {sh['run']} = {xs} (the gap includes learnable vs "
                        f"unlearnable auxiliary target), {c2['run']} - {coco['run']} = {ys} "
                        f"({self.spec['c2_sentences']['c2'].split(';')[0]}): pass needs above C2-shuffled by more than "
                        f"sigma_screen {f2(sigma)} with K2 not fired: accuracy part {'met' if pass_acc else 'not met'}, "
                        f"{k2t}: pass {po}{cur}; kill needs one clause on both tasks: this task's parts: within "
                        f"sigma_screen of C2-shuffled {'met' if within else 'not met'}, below full-depth coconut by more "
                        f"than 2 sigma_screen {'met' if below else 'not met'}, K2 fired {kill_k2}; {other} {lab}"}

    def _sensitivity(self, lines: dict) -> list[dict]:
        reg = {ln["id"]: ln["outcome"] for ln in lines["registered"]}
        out = []
        for name, ls in lines.items():
            if name == "registered":
                continue
            for ln in ls:
                if ln["id"] in reg and ln["kind"] != "sigma" and ln["outcome"] != reg[ln["id"]]:
                    out.append({"reading": name, "id": ln["id"], "registered": reg[ln["id"]], "alternative":
                                ln["outcome"]})
        return out

    def _receipt_checks(self, receipts: list[dict], runs: dict) -> dict:
        env = [{"attempt": r.get("attempt"), "job": r.get("job"), "status": r.get("status"),
                "env_reference": next((s.get("ok") and not s.get("mismatches") for s in r.get("steps", [])
                                       if s.get("step") == "env_reference"), None)} for r in receipts]
        named = sorted({a for r in runs.values() for a in r.get("attempts", [])})
        have = {r.get("attempt") for r in receipts}
        admitted = {}
        for r in receipts:
            t = ((r.get("spec") or {}).get("_launch") or {}).get("admitted")
            for x in (r.get("spec") or {}).get("runs", []):
                if t and (x["run"] not in admitted or t < admitted[x["run"]]):
                    admitted[x["run"]] = t
        bar_runs = [s["run"] for s in self.spec["runs"] if s["key"].startswith("cot_lr")]
        lat_runs = [r["run"] for r in runs.values() if r["status"] == "read" and r.get("latent")]
        a5 = None
        if all(b in admitted for b in bar_runs) and all(x in admitted for x in lat_runs) and lat_runs:
            a5 = max(admitted[b] for b in bar_runs) <= min(admitted[x] for x in lat_runs)
        return {"env_reference": env, "attempts_named_by_runs": named,
                "attempts_without_receipt": sorted(set(named) - have),
                "first_admission": admitted, "a5_item8_bar_before_latent": a5}

    def _notes(self) -> list[str]:
        return list(self.spec["notes"]) + [
            "Readings chosen (the alternatives are recomputed in section 4): sigma_screen as A3.4 is written (the "
            "ProsQA cot 3-seed SD term included); 'ahead by 1 sigma' (6.3) as a difference >= sigma; 'within sigma' as "
            "an absolute difference <= sigma, except C1b's pass ('within sigma of full depth') read as not below by more "
            "than sigma; CURRICULUM-ONLY (7.5) applied whenever the pause difference is within sigma; the fade's 0.5 B "
            "read at both neighbouring epochs when B is odd; the 7.2 legs on the selected epoch's logged means.",
            "Costs: |q| per example is exact (the hash-checked valid file through the pinned tokenizer); per-example "
            "emitted tokens were not recorded by training, so mean FLOPs and reference-cost FLOPs are printed as the "
            "exact range over every assignment of per-example E' in [0, max_new_tokens] with the logged total; a cost "
            "comparison is decided only when the ranges decide it. V as the arm's head: 50,257 (cot, no_cot), 50,260 "
            "(latent arms) (A11.2).",
            "Training FLOPs (5.5 (d)): the rows of record's `flops`; GSM-50k latent arms are also charged the lr 1e-4 "
            "cot run they start from (5.2). 'Same epochs' (training-heavy at > 1.25x) is read per epoch: the arm's "
            "charged FLOPs over the bar's per-epoch FLOPs times the arm's charged epoch count.",
            "Pending until the screen eval job runs on valid: the scrambled-at-inference, zero-latent and step-sweep "
            "reads (K2, the null triggers of 7.10), A6.2's timed read (f_meas for C1b), the batch-1 wall-clock (5.5 "
            "(b)) and the per-example cost files that make the FLOP means exact.",
        ] if self.ev is None else list(self.spec["notes"]) + [
            "Readings chosen (the sigma alternatives are recomputed in section 5): sigma_screen as A3.4 is written; "
            "'ahead by 1 sigma' (6.3) as a difference >= sigma; 'within sigma' as an absolute difference <= sigma, "
            "except C1b's pass read as not below by more than sigma; CURRICULUM-ONLY (7.5) whenever the pause "
            "difference is within sigma; the fade's 0.5 B read at both neighbouring epochs.",
            "K2 (7.5, A3.4) and the zero-latent drop: Delta in points from the two reads' per-example files, paired by "
            "example; the interval is 6.2's Delta +- 1.96 sqrt(Var_ex), Var_ex the variance (ddof 1) of Delta over "
            "10,000 paired example draws (numpy default_rng(0), drawn as eval.summarize draws), without 6.2's seed "
            "term (both reads score one checkpoint); the percentile interval of the same draws is printed beside it; "
            "'includes 0' is inclusive. 6.3's scrambled leg is Delta_scr >= sigma_screen.",
            "Null triggers (7.10): each scrambled and zero-latent read against the bar + 2 sigma_screen and against "
            "the arm + sigma_screen, the arm read both as its selection and as its plain eval read (either exceeded "
            "triggers); 'above' is strict.",
            "f_meas (A6.2 rules 1, 2, 4): the C1b read's latent ms per pass over the full-depth coconut checkpoint's "
            "band [0, 12] read's, at batch 1 (of record) and at the eval batch 32 (beside), same device and torch; "
            "run_eval.py's summary keeps the pooled ms per pass, which equals the mean of summarize_timing's per-batch "
            "latent_ms_mean because every batch has the same number of latent passes (checked); it keeps no "
            "latent_last mean, so that column is 'not recorded'. " + (
                "Rule 4's two readings, the printed 0.77 (k = 8) and 0.43 (k = 4) and the formula (b - a) / L + 0.10 "
                "(0.7667, 0.4333), are both printed; the line of record is the stricter of the two (critic entry "
                "2026-10-20T12:40Z (a)): 0.7667 for k = 8, 0.43 for k = 4. " + THIRD_LABELS[self.task]["fmeas_note"]
                if self.third else
                "Rule 4's line of record is the printed 0.77 (k = 8) and 0.43 (k = 4); the formula (b - a) / L + 0.10 "
                "(0.7667, 0.4333) is evaluated beside it. No saving label is written while rule 3's strict-fp32 logits "
                "check is pending."),
            "Costs: the latent arms' means are exact from the plain eval read's per-example E' (every row's FLOPs, "
            "reference FLOPs, prefill tokens and passes recomputed by accounting.inference_record from the hashed "
            "valid |q| equal the file's); the bar and no_cot keep the first table's exact ranges (no per-example read "
            "of either exists), so a ratio to the bar is an interval and a leg is decided only when it decides. C1b "
            "FLOPs at the nominal (b - a) / L, at the measured min(1, f_meas) and at 1; inference_record charges a "
            "fraction as the nearest fraction with denominator <= 1000. V as the arm's head (A11.2).",
            "The step sweep (7.3, section 8) decides nothing; its n = 6 point is the plain read (the trained count), "
            "not a sweep read; n = 8 is out-of-distribution; the n = 0 against zero-latent check compares every column "
            "of the two per-example files (A10.4: a difference is a BUG trigger).",
            ("Still pending or UNTESTED here: the latent_last means of A6.2 rule 2 (not in the summaries); 5.5 (b) "
             "for every comparison with the bar (no timed bar read) and coconut's own HF path at batch 1; "
             + THIRD_LABELS[self.task]["pending_both_task"] + "the screen round's single test read (A3.4)." if self.third
             else
             "Still pending or UNTESTED here: rule 3's strict-fp32 first-token logits check (A6.2; no saving label until "
             "it is logged); the latent_last means of A6.2 rule 2 (not in the summaries); 5.5 (b) for every comparison "
             "with the bar (no timed bar read) and coconut's own HF path at batch 1; plain depth 355M (A13.2: the K3 "
             "lines stay 'nearest size'); the ProsQA-hard side of every both-task clause; the screen round's single "
             "test read (A3.4)."),
        ]


# -------------------------------------------------------------------------------------------------- rendering
def _gf(x: float) -> str:
    return f"{x / 1e9:.3f}"


def _rng(t) -> str:
    return f"[{_gf(t[0])}, {_gf(t[1])}]"


def _gfr(t) -> str:
    """GFLOPs of a (lo, hi) mean: one number when exact, else the range."""
    return _gf(t[0]) if abs(t[1] - t[0]) < 1e-6 * max(1.0, abs(t[0])) else _rng(t)


def _sci(x: float, up: bool = False) -> str:
    """x with two significant digits as the critic entries print it ('6.4e-4', '1e-4'); up=True rounds the mantissa up,
    so '<= ' + _sci(x, up=True) is a true bound ('2.3e-13' for 2.2737e-13)."""
    if x == 0:
        return "0"
    e = math.floor(math.log10(abs(x)))
    m = x / 10 ** e
    m = math.ceil(m * 10 - 1e-9) / 10 if up else round(m, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"{m:g}e{e}"


def _short_cite(cite: str) -> str:
    """'critic entry 2026-10-20T15:26Z row 3' -> '15:26Z row 3' (the md header's short form)."""
    return re.sub(r"^critic entries? \d{4}-\d\d-\d\dT", "", cite)


DIAG_NOT_RUN = "not run (--checks)"     # run_diag.py --checks: the verdict of a check the run did not request


def _diag_complete(s: dict) -> bool:
    """A band_logits record is read only when complete: `complete` true, or every check in `checks_requested` present
    (in `verdicts` and in results.checks, with a verdict other than DIAG_NOT_RUN). run_diag.py writes summary.json
    once at the end with `complete` true and its per-check partial form (`complete` false) to partial.json, which is
    never a record (critic entry 2026-10-20T22:36Z (1)); this refuses a partial form wherever it is found. A record with
    neither field was written before --checks (the GSM-50k records at rev15 and rev16, ProsQA-hard's strict-fp32
    record at rev23) and is complete."""
    if s.get("complete") is True:
        return True
    req = s.get("checks_requested")
    if req is None:
        return "complete" not in s
    v = s.get("verdicts") or {}
    ch = (s.get("results") or {}).get("checks") or {}
    return bool(req) and all(k in v and k in ch and v[k] != DIAG_NOT_RUN for k in req)


def _diag64_where(s: dict) -> str | None:
    """The float64 record's device clause for its labels (critic entry 2026-10-20T21:33Z (2)): "CPU" for device cpu,
    "<GPU short name> (device cuda)" for cuda with a gpu_type; None (refused by _diag) when the device is neither, the
    dtypes are not exactly DIAG64_DTYPES, or the strict set's read-back at the start or the end is not TF32 off,
    "highest", no autocast."""
    dev = str(s.get("device") or "")
    if s.get("dtypes") != DIAG64_DTYPES:
        return None
    st = (s.get("numerics") or {}).get("strict") or {}
    for k in ("torch_at_start", "torch_at_end"):
        f = st.get(k) or {}
        if (f.get("cuda.matmul.allow_tf32") is not False or f.get("cudnn.allow_tf32") is not False
                or f.get("float32_matmul_precision") != "highest" or f.get("autocast_enabled") is not False):
            return None
    if dev == "cpu":
        return "CPU"
    if dev.split(":")[0] == "cuda" and s.get("gpu_type") and s.get("gpu_type") != dev:
        return f"{_gpu_short(s['gpu_type'])} (device cuda)"
    return None


def _gpu_short(name: str | None) -> str:
    """'NVIDIA A100-SXM4-40GB' -> 'A100-40GB' (the critic entries' device label)."""
    m = re.search(r"(A100|H100|L4|T4|V100)[A-Za-z0-9-]*?-(\d+GB)", str(name))
    return f"{m.group(1)}-{m.group(2)}" if m else str(name)


def to_markdown(t: dict) -> str:
    if "eval" in t:
        return _md_with_eval(t)
    n = t["n_valid"]
    L = [f"# {t['title']} screen table: {SCREEN_LABEL}, valid split", "",
         f"{t['data_sentence']} ({n} items). Seed 0, GPT-2 124M, " + "; ".join(t["labels"][3:]) + ". Labels on every "
         f"number and line: {SCREEN_LABEL}; {BAR_LABEL}; {BAR_VALUE_LABEL}. Screen decisions use valid only (A3.4); "
         "screen cells are screen numbers, never verdicts.", "",
         "## 1. Selection and valid accuracy", "",
         "| run | role | pool (epochs) | selected epoch (stage) | valid | vs bar | all-epoch best (3.3 item 3) | "
         "resumes | GPU | start pin (A6.3 rule 4) | record checks |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    bar = next(r for r in t["runs"] if r["key"] == t["bar"]["key"])
    for r in t["runs"]:
        if "selected" not in r or r["status"] != "read":
            st = "UNTESTED" if r["status"] == "UNTESTED" else r.get("exclusion", r["status"])
            L.append(f"| {r['run']} | {r['role']} | | | {st} | | | | | | {r.get('why') or '; '.join(r['problems'])} |")
            continue
        s = r["selected"]
        pool = (f"{r['pool'][0] + 1}-{r['pool'][1] + 1}" if r.get("pool") else "all")
        vs = r["points"] - bar["points"]
        best = (f"{100 * r['best_all']['val_acc']:.1f} at epoch {r['best_all']['epoch']} (stage "
                f"{r['best_all']['stage']}, n latents = {r['best_all']['n_latent']}): earlier stage better on valid"
                if r.get("earlier_stage_better") else "")
        pin = "" if "start_pin" not in r else ("ok" if r["start_pin"]["ok"] else "MISMATCH: speculative, unused")
        L.append(f"| {r['run']} | {r['role']} | {pool} | {s['epoch']} ({s['stage']}) | {r['points']:.1f} ({r['k']} of "
                 f"{n}) | {vs:+.1f} | {best} | {r.get('resumes', 0)} | {r.get('gpu_type', '')} | {pin} | "
                 f"{'ok (' + str(len(r['checks'])) + ')' if not r['problems'] else '; '.join(r['problems'])} |")
    L += ["", "## 2. Cost per valid example (5.1, 5.3, 5.5; A6.4)", "",
          "E' and sequential passes are exact means from the selected epoch's log row; FLOPs (GFLOPs) are the exact "
          "range over the unrecorded per-example E' (notes). Training FLOPs in PFLOPs.", "",
          "| run | L | E' mean | E' / bar | passes mean | passes / bar | FLOPs mean | FLOPs / bar | reference-cost "
          "FLOPs mean | reference / bar | training FLOPs own / charged | charged / bar (same epochs) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bc = bar.get("cost")
    for r in t["runs"]:
        if "cost" not in r:
            continue
        c = r["cost"]

        def rat(a, b):
            return f"[{a[0] / b[1]:.3f}, {a[1] / b[0]:.3f}]"
        fl = _rng(c["flops_mean"])
        flr = rat(c["flops_mean"], bc["flops_mean"])
        if "cost_nominal" in r:
            cn = r["cost_nominal"]
            fl = f"{_rng(cn['flops_mean'])} nominal; {fl} at 1"
            flr = f"{rat(cn['flops_mean'], bc['flops_mean'])} nominal; {flr} at 1"
        tr = r.get("train", {})
        L.append(f"| {r['run']} | {c['L']} | {c['emitted_mean']:.2f} | {c['emitted_mean'] / bc['emitted_mean']:.3f} | "
                 f"{c['passes_mean']:.2f} | {c['passes_mean'] / bc['passes_mean']:.3f} | {fl} | {flr} | "
                 f"{_rng(c['reference_flops_mean'])} | {rat(c['reference_flops_mean'], bc['reference_flops_mean'])} | "
                 f"{tr.get('own', 0) / 1e15:.1f} / {tr.get('charged', 0) / 1e15:.1f} | "
                 f"{tr.get('ratio_same_epochs', 0):.2f}{' training-heavy' if tr.get('heavy') else ''} |")
    L += ["", "## 3. Registered lines", ""]
    order = ["bar", "sigma", "screen", "card", "k2", "k3", "pause", "cost", "c1b_cost", "c2_label", "too_good", "fade",
             "untested"]
    titles = {"bar": "Bar", "sigma": "sigma_screen", "screen": "Screen rule 6.3 (card arms)",
              "card": "Card lines (A3.4)", "k2": "K2 (7.5; A3.4)", "k3": "K3 (7.5; 7.7; A13.2)",
              "pause": "Pause comparison (7.5)", "cost": "7.2 cost legs and 5.5", "c1b_cost": "C1b per-step cost "
              "(A3.4; A6.2)", "c2_label": "C2 labels (A5 item 5; A6.7 items 4, 5)", "too_good": "Too-good triggers "
              "(7.10; A3.4)", "fade": "Fade instrument (7.6)", "untested": "UNTESTED and excluded"}
    for kind in order:
        ls = [ln for ln in t["lines"] if ln["kind"] == kind]
        if ls:
            L.append(f"### {titles[kind]}")
            L += [f"- {ln['text']}" for ln in ls] + [""]
    rc = t["receipts"]
    L += ["### Receipts and launch condition (A3.2 (i); A5 item 8)",
          "- env_reference per receipt: " + ", ".join(
              f"{e['attempt']} " + ("ok" if e["env_reference"] else "no env step (status " + str(e["status"]) + ")"
                                    if e["env_reference"] is None else "NOT ok") for e in rc["env_reference"]),
          f"- attempts named by the runs without a receipt in this read: {', '.join(rc['attempts_without_receipt']) or 'none'}",
          "- A5 item 8 (the bar's three cot trials admitted no later than the first latent arm, from the receipts' "
          f"admissions): {'met' if rc['a5_item8_bar_before_latent'] else 'not shown by the receipts read' if rc['a5_item8_bar_before_latent'] is None else 'NOT met'}",
          "", "## 4. Readings and sensitivity", "",
          f"- sigma_screen of record {t['sigma_screen']['registered']:.4f} (terms: floor 1.0, binomial "
          f"{t['sigma_screen']['terms']['binomial']:.4f}, ProsQA cot 3-seed SD "
          f"{t['sigma_screen']['terms']['prosqa_cot_3seed_sd']}); alternatives recomputed: "
          + ", ".join(f"{k} {v:.4f}" for k, v in t["sigma_screen"]["alternatives"].items()) + "."]
    if t["sensitivity"]:
        L.append("- Lines whose outcome changes under an alternative reading:")
        L += [f"  - {s['id']} under '{s['reading']}': {s['registered']} -> {s['alternative']}" for s in t["sensitivity"]]
    else:
        L.append("- No line changes outcome under the alternative readings.")
    L += ["", "## 5. Notes", ""] + [f"- {x}" for x in t["notes"]]
    return "\n".join(L) + "\n"


LINE_ORDER_EVAL = ["bar", "sigma", "eval_check", "screen", "card", "k2", "too_good", "null_trigger", "k3", "pause",
                   "curriculum", "cost", "c1b_cost", "wallclock", "rule3", "sweep", "c2_label", "fade", "untested"]
LINE_TITLES_EVAL = {
    "bar": "Bar", "sigma": "sigma_screen", "eval_check": "The eval reads: job, entry checks, plain read against the "
    "selection", "screen": "Screen rule 6.3 (card arms)", "card": "Card lines (A3.4)", "k2": "K2 (7.5; A3.4)",
    "too_good": "Too-good triggers (7.10; A3.4)", "null_trigger": "Null triggers (7.10; A3.4)",
    "k3": "K3 (7.5; 7.7; A13.2)", "pause": "Pause comparison (7.5)",
    "curriculum": "Curriculum (3.1's control; A2.2 (b); critic entry 2026-10-21T20:21Z (b) (5), (6))",
    "cost": "7.2 cost legs and 5.5",
    "c1b_cost": "C1b per-step cost: f_meas (A3.4; A6.2)", "wallclock": "5.5 (b) wall-clock at batch 1",
    "rule3": "A6.2 rule 3 like-for-like check", "sweep": "Step sweep (7.3; section 8; decides nothing)",
    "c2_label": "C2 labels (A5 item 5; A6.7 items 4, 5)", "fade": "Fade instrument (7.6)",
    "untested": "UNTESTED and excluded"}


def _ms(x) -> str:
    return "none" if x is None else f"{x:.3f}"


def _md_with_eval(t: dict) -> str:
    """The second table ("with the eval reads"): the first table's sections with the eval reads filled in."""
    n = t["n_valid"]
    ev, cit = t["eval"], t.get("citations") or {}
    job = ev["job"]
    ft = cit.get("first_table") or {}
    ec = cit.get("eval") or {}
    st2 = cit.get("second_table")
    st3 = cit.get("third_table")
    eh = t.get("eval_hinge") or {}
    hkeys = HINGE.get(t["task"], {}).get("keys", ()) if st3 else ()
    pdb = t.get("plain_depth") or {}
    stp = cit.get("prior_table")
    xkeys = [k for k in CLOSING if any(r["key"] == k for r in t["runs"])] if stp else []
    if stp:                                        # the no-curriculum reissue (critic entry 2026-10-21T20:21Z (b) (5))
        ex, xc, xj = t.get("eval_extra") or {}, cit.get("extra_eval") or {}, (t.get("eval_extra") or {}).get("job") or {}
        sp = CLOSING[xkeys[0]] if xkeys else {}
        ordn, (_, pord) = sp.get("ordinal", {}).get(t["task"], "next"), sp.get("prior", {}).get(t["task"], ("", "prior"))
        xr = [r for r in t["runs"] if r["key"] in xkeys]
        L = [f"# {t['title']} screen table with the eval reads, the plain-depth 355M control"
             + (", the hinge arms" if st3 else "") + f" and the no-curriculum control: {SCREEN_LABEL}, valid split", "",
             f"{t['data_sentence']} ({n} items). Seed 0, GPT-2 124M; the control GPT-2 355M; bf16 autocast over the TF32 "
             f"flags (A10.5). Every line carries: {EVAL_LABEL}. Screen decisions use valid only (A3.4); screen cells "
             "are screen numbers, never verdicts.", "",
             f"The {ordn} write-once table: the {pord} table's reissue over the no-curriculum control "
             f"{', '.join(r['run'] for r in xr)} of {sp.get('jobs', {}).get(t['task'])} (A6.4's reissue pattern; PLAN.md "
             f"A2.2 item 1 (b); critic entry {CLOSING_ROW} (b) parts (1) to (6)). It cites the {pord} table "
             f"`{', '.join(stp.get('files', []))}` written once at `{str(stp.get('commit'))[:7]}` ({stp.get('commit')}; "
             f"files unchanged since: {'yes' if stp.get('unchanged') else 'NO'}) and every table that table cites "
             f"(`{', '.join(str(cit[k].get('commit'))[:7] for k in ('first_table', 'second_table', 'third_table') if k in cit)}`"
             f"; their files unchanged since: "
             f"{'yes' if all(cit[k].get('unchanged') for k in ('first_table', 'second_table', 'third_table') if k in cit) else 'NO'}), "
             f"the eval reads `{ec.get('dir')}` of job `{job['job']}` (results commit `{str(ec.get('commit'))[:7]}`)"
             + (f", the hinge eval reads `{(cit.get('hinge_eval') or {}).get('dir')}` (results commit "
                f"`{str((cit.get('hinge_eval') or {}).get('commit'))[:7]}`)" if st3 else "")
             + f", the no-curriculum eval reads `{xc.get('dir')}` of job `{xj.get('job')}`, attempt `{xj.get('attempt')}`, "
             f"registered read row {xj.get('registered_read')}, results commit `{str(xc.get('commit'))[:7]}` "
             f"({len(ex.get('entries') or [])} reads; receipt sha256 {str(xj.get('receipt_sha256'))[:12]}...) and the "
             f"arm's records (its training project's bucket, the completing attempt's receipt). Against the {pord} table "
             f"only these change: the arm's rows in sections 1 and 2 and its eval reads in section 3, its plain-read, K2, "
             f"K3, pause, 7.2, 5.5 (b), fade and sweep lines, the Curriculum line (row (b) (5)), the 7.10 lines, its row of "
             f"the A_pd block, its eval job's line, the receipts line, the K2 seed-term note and the note appended; every "
             f"other cell is the {pord} table's (checked by the tool, listed in section 5). The earlier tables' files are "
             f"not modified.", ""]
    elif st3:                                      # the fourth table (critic entries 2026-10-20T17:12Z (5), 17:36Z (5))
        jh, hc = eh.get("job") or {}, cit.get("hinge_eval") or {}
        L = [f"# {t['title']} screen table with the eval reads, the plain-depth 355M control and the hinge arms: "
             f"{SCREEN_LABEL}, valid split", "",
             f"{t['data_sentence']} ({n} items). Seed 0, GPT-2 124M; the control GPT-2 355M; bf16 autocast over the TF32 "
             f"flags (A10.5). Every line carries: {EVAL_LABEL}. Screen decisions use valid only (A3.4); screen cells "
             "are screen numbers, never verdicts.", "",
             f"The fourth write-once table: the third table's reissue over the three hinge arms of "
             f"{HINGE[t['task']]['job']} (A6.4's reissue pattern; critic entries {ROW_HINGE} (5), (6) and (9), {DONOR_ROW} "
             f"(5)). It cites the first table `{', '.join(ft.get('files', []))}` written once at `{str(ft.get('commit'))[:7]}` "
             f"({ft.get('commit')}; files unchanged since: {'yes' if ft.get('unchanged') else 'NO'}), the second table "
             f"`{', '.join(st2.get('files', []))}` written once at `{str(st2.get('commit'))[:7]}` ({st2.get('commit')}; "
             f"files unchanged since: {'yes' if st2.get('unchanged') else 'NO'}), the third table "
             f"`{', '.join(st3.get('files', []))}` written once at `{str(st3.get('commit'))[:7]}` ({st3.get('commit')}; "
             f"files unchanged since: {'yes' if st3.get('unchanged') else 'NO'}), the eval reads `{ec.get('dir')}` of job "
             f"`{job['job']}`, attempt `{job['attempt']}`, registered read row {job['registered_read']}, results commit "
             f"`{str(ec.get('commit'))[:7]}` ({len(ev['entries'])} entries; receipt sha256 {job['receipt_sha256'][:12]}...), "
             f"the hinge eval reads `{hc.get('dir')}` of job `{jh.get('job')}`, attempt `{jh.get('attempt')}`, registered "
             f"read row {jh.get('registered_read')}, results commit `{str(hc.get('commit'))[:7]}` "
             f"({len(eh.get('entries') or [])} reads and {len(eh.get('directional') or [])} donor-answer records; receipt "
             f"sha256 {str(jh.get('receipt_sha256'))[:12]}...), the control's records (attempt "
             f"`{(pdb.get('probe') or {}).get('run_attempt')}`) and the hinge runs' records (their own bucket, the "
             f"completing attempt's receipt per run). Against the third table only these change: the three hinge arms' "
             f"rows in sections 1 and 2 (UNTESTED there), their eval reads in section 3 and their directional reads "
             f"(3.6), their K2, 7.10, 7.2, 5.5 (b), K3 and card lines (and the 6.3 screen, pause, fade, plain-read and "
             f"sweep lines the six arms carry), their rows of the A_pd block, the hinge eval job's line, the receipts "
             f"line, the K2 seed-term note, the notes appended and the three UNTESTED lines (removed); every other cell "
             f"is the third table's (checked by the tool, listed in section 5). The first three tables' files are not "
             f"modified.", ""]
    elif st2:                                      # the third table (critic entries 2026-10-20T15:53Z)
        L = [f"# {t['title']} screen table with the eval reads and the plain-depth 355M control: {SCREEN_LABEL}, valid "
             f"split", "",
             f"{t['data_sentence']} ({n} items). Seed 0, GPT-2 124M; the control GPT-2 355M; bf16 autocast over the TF32 "
             f"flags (A10.5). Every line carries: {EVAL_LABEL}. Screen decisions use valid only (A3.4); screen cells "
             "are screen numbers, never verdicts.", "",
             f"The third write-once table (A6.4's reissue pattern; critic entries {ROW_PD} (two), "
             f"{THIRD_LABELS[t['task']]['md_rows']}). It cites the first table `{', '.join(ft.get('files', []))}` written once at "
             f"`{str(ft.get('commit'))[:7]}` ({ft.get('commit')}; files unchanged since: "
             f"{'yes' if ft.get('unchanged') else 'NO'}), the second table `{', '.join(st2.get('files', []))}` written "
             f"once at `{str(st2.get('commit'))[:7]}` ({st2.get('commit')}; files unchanged since: "
             f"{'yes' if st2.get('unchanged') else 'NO'}), the eval reads `{ec.get('dir')}` of job `{job['job']}`, "
             f"attempt `{job['attempt']}`, registered read row {job['registered_read']}, results commit "
             f"`{str(ec.get('commit'))[:7]}` ({len(ev['entries'])} entries; receipt sha256 "
             f"{job['receipt_sha256'][:12]}...), and the control's records (attempt "
             f"`{(pdb.get('probe') or {}).get('run_attempt')}`). Against the second table only these change: the "
             f"control's rows in sections 1 and 2, the A_pd block and the six K3 lines, the receipts line, the f_meas "
             f"labels and the C1b cards' per-step cost ({_short_cite(THIRD_LABELS[t['task']]['fmeas'])}), the batch-1 "
             f"labels ({_short_cite(THIRD_LABELS[t['task']]['batch1'])}) on rule 3's "
             f"line, the wall-clock lines and section 3.3, the two notes they touch and the UNTESTED line of the control "
             f"(removed); every other cell is the second table's (checked by the tool, listed in section 5). The first "
             f"two tables' files are not modified.", ""]
    else:
        L = [f"# {t['title']} screen table with the eval reads: {SCREEN_LABEL}, valid split", "",
             f"{t['data_sentence']} ({n} items). Seed 0, GPT-2 124M; bf16 autocast over the TF32 flags (A10.5). Every "
             f"line carries: {EVAL_LABEL}. Screen decisions use valid only (A3.4); screen cells are screen numbers, never "
             "verdicts.", "",
             f"The second write-once table (A6.4's reissue pattern; critic entry {job['registered_read']} part (6)). It "
             f"cites the first table `{', '.join(ft.get('files', []))}` written once at `{str(ft.get('commit'))[:7]}` "
             f"({ft.get('commit')}; files unchanged since: {'yes' if ft.get('unchanged') else 'NO'}), and the eval reads "
             f"`{ec.get('dir')}` of job `{job['job']}`, attempt `{job['attempt']}`, registered read row "
             f"{job['registered_read']}, results commit `{str(ec.get('commit'))[:7]}` ({len(ev['entries'])} entries; "
             f"receipt sha256 {job['receipt_sha256'][:12]}...). The first table's files are not modified.", ""]
    L += [
         "## 1. Selection, valid accuracy and the plain eval read", "",
         "| run | role | pool: epochs (epoch_index) | selected epoch (stage) | valid, training log (selection) | plain "
         "read, eval | vs bar | all-epoch best (3.3 item 3) | resumes | GPU (training) | start pin (A6.3 rule 4) | "
         "record checks |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bar = next(r for r in t["runs"] if r["key"] == t["bar"]["key"])
    per = ev.get("per_run") or {}
    for r in t["runs"]:
        if "selected" not in r or r["status"] != "read":
            st = "UNTESTED" if r["status"] == "UNTESTED" else r.get("exclusion", r["status"])
            L.append(f"| {r['run']} | {r['role']} | | | {st} | | | | | | | {r.get('why') or '; '.join(r['problems'])} |")
            continue
        s_ = r["selected"]
        pool = (f"{r['pool'][0] + 1}-{r['pool'][1] + 1} ({r['pool'][0]}-{r['pool'][1]})" if r.get("pool")
                else "all")
        nv = (per.get(r["key"]) or {}).get("none_vs_log")
        pr = (f"{100 * nv['k_eval'] / n:.1f} ({nv['k_eval']} of {n}): {'agrees' if nv['agree'] else 'DIFFERS'}"
              if nv else "no eval read")
        best = (f"{100 * r['best_all']['val_acc']:.1f} at epoch {r['best_all']['epoch']} (stage "
                f"{r['best_all']['stage']}, n latents = {r['best_all']['n_latent']}): earlier stage better on valid"
                if r.get("earlier_stage_better") else "")
        pin = "" if "start_pin" not in r else ("ok" if r["start_pin"]["ok"] else "MISMATCH: speculative, unused")
        role = r["role"]
        if r.get("hinge"):                         # the hinge arms (critic entry 2026-10-21T11:07Z (1)(f) and (5))
            h = r["hinge"]
            gpu = (f"trained and read on {h['trained_on']}" if h.get("read_on") == [h["trained_on"]] else
                   f"trained on {h['trained_on']}, read on {', '.join(h.get('read_on') or []) or 'no read'}")
            role += f" [{h['label']}; start pin {h['start_pin']}; {gpu}]"
        if r.get("label"):                         # the control: its label, model pin and file of record
            role += (f" [{r['label']}; {r.get('model_id')} {str(r.get('model_revision'))[:8]}; ckpt_best_final.pt "
                     f"{str(r.get('checkpoint'))[:8]}...; selection over every epoch (3.3 item 1, the no_cot rule); valid "
                     f"only; no test read]")
            pool = "all (every epoch)"
        stg = str(s_["stage"])
        if r.get("closing"):                       # the no-curriculum control (critic entry 2026-10-21T20:21Z (b))
            c_ = r["closing"]
            role += (f" [selection over every trained epoch (3.3 item 1; row (b) (2)); pinned stage {c_['pinned_stage']}, "
                     f"n latents {c_['n_latent']} at every epoch (the log's stage column is the schedule's arithmetic); "
                     f"resumed {r.get('resumes', 0)} times; valid only; no test read]")
            pool = "all (every epoch)"
            stg = f"pinned {c_['pinned_stage']}; logged {s_['stage']}"
        L.append(f"| {r['run']} | {role} | {pool} | {s_['epoch']} ({stg}) | {r['points']:.1f} ({r['k']} of "
                 f"{n}) | {pr} | {r['points'] - bar['points']:+.1f} | {best} | {r.get('resumes', 0)} | "
                 f"{r.get('gpu_type', '')} | {pin} | "
                 f"{'ok (' + str(len(r['checks'])) + ')' if not r['problems'] else '; '.join(r['problems'])} |")
    L += ["", "## 2. Cost per valid example (5.1, 5.3, 5.5; A6.4)", "",
          "Latent arms: exact means from the plain eval read's per-example file (E', passes, FLOPs and reference-cost "
          "FLOPs; every row recomputed by `accounting.inference_record` from the hashed valid |q| and the row's own "
          "L and E' equals the file). `cot` and `no_cot` have no per-example read: their FLOPs stay the first table's "
          "exact range. L is latent steps; pause's 6 are pause positions (5.3), not latent steps. GFLOPs; training "
          "FLOPs in PFLOPs.", "",
          "| run | L | E' mean | E' / bar | passes mean | passes / bar | FLOPs mean | FLOPs / bar | first table's FLOPs "
          "range | reference-cost FLOPs mean | reference / bar | first table's reference range | training FLOPs own / "
          "charged | charged / bar (same epochs) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    bc = bar.get("cost")

    def rat(a, b):
        return f"[{a[0] / b[1]:.3f}, {a[1] / b[0]:.3f}]"

    def val(x):
        return f"{x[0] / 1e9:.3f}" if abs(x[1] - x[0]) < 1e-6 * max(1.0, abs(x[0])) else _rng(x)
    for r in t["runs"]:
        if "cost" not in r:
            continue
        c = r["cost"]
        fl, flr = val(c["flops_mean"]), rat(c["flops_mean"], bc["flops_mean"])
        if "cost_nominal" in r:
            parts = [(r["cost_nominal"], f"nominal {r['cost_nominal']['layer_fraction']:.4f}")]
            if "cost_measured" in r:
                parts.append((r["cost_measured"], f"measured {r['cost_measured']['layer_fraction']:.4f}"))
            parts.append((c, "at 1"))
            fl = "; ".join(f"{val(x['flops_mean'])} {lbl}" for x, lbl in parts)
            flr = "; ".join(f"{rat(x['flops_mean'], bc['flops_mean'])} {lbl}" for x, lbl in parts)
        old = r.get("cost_range") or {}
        old_fl = _rng(old["flops_mean"]) if old else ("(UNTESTED in the first two tables)" if r.get("label") else
                                                      "(same)")
        if r["key"] in hkeys and old:              # the fourth table: the hinge arms were UNTESTED in tables 1 to 3
            old_fl = f"{old_fl} (the training log's range; UNTESTED in the first three tables)"
        if r["key"] in xkeys and old:              # the no-curriculum reissue: the arm is in no earlier table
            old_fl = f"{old_fl} (the training log's range; in no earlier table)"
        if "cost_nominal_range" in r:
            old_fl = f"{_rng(r['cost_nominal_range']['flops_mean'])} nominal; {old_fl} at 1"
        old_rf = _rng(old["reference_flops_mean"]) if old else ("(UNTESTED in the first two tables)" if r.get("label")
                                                                else "(same)")
        tr = r.get("train", {})
        Lc = f"{c['L']} (pause positions)" if r["key"] == "pause" else str(c["L"])
        rn = r["run"] + (f" [{r['label']}; {r.get('model_id')}: N_ne {c['N_ne']:,}, V {c['V']:,}]" if r.get("label")
                         else "")
        L.append(f"| {rn} | {Lc} | {c['emitted_mean']:.3f} | {c['emitted_mean'] / bc['emitted_mean']:.3f} | "
                 f"{c['passes_mean']:.3f} | {c['passes_mean'] / bc['passes_mean']:.3f} | {fl} | {flr} | {old_fl} | "
                 f"{val(c['reference_flops_mean'])} | {rat(c['reference_flops_mean'], bc['reference_flops_mean'])} | "
                 f"{old_rf} | {tr.get('own', 0) / 1e15:.1f} / {tr.get('charged', 0) / 1e15:.1f} | "
                 f"{tr.get('ratio_same_epochs', 0):.2f}{' training-heavy' if tr.get('heavy') else ''} |")
    L += ["", "## 3. The eval reads", "", "### 3.1 Accuracy on valid per read (points; k of 500 in the json)", "",
          "| run | plain read (n = trained) | scrambled | zero-latent | n = 0 | n = 1 | n = 2 | n = 4 | n = 8 "
          "(out-of-distribution) | n = 0 against zero-latent |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["runs"]:
        st = per.get(r["key"])
        if not st or not st.get("reads"):
            continue
        rd = st.get("reads") or {}

        def cell(k):
            return f"{rd[k]['points']:.1f}" if k in rd else ""
        chk = st.get("n0_zero")
        ck = "" if chk is None else ("agree on every example" if chk["agree"] else
                                     f"DIFFER on {chk['examples_differing']}: BUG trigger")
        L.append(f"| {r['run']} | {cell('none')} | {cell('scrambled')} | {cell('zero_latent')} | {cell('n0')} | "
                 f"{cell('n1')} | {cell('n2')} | {cell('n4')} | {cell('n8')} | {ck} |")
    L += ["", f"### 3.2 K2 statistics (6.2 paired bootstrap: 10,000 draws, default_rng(0)); sigma_screen "
              f"{t['sigma_screen']['registered']:.2f}", "",
          "| run | Delta_scr | 95% interval (Delta +- 1.96 SE) | percentile interval | correct only own / only "
          "scrambled | K2 at the screen | zero-latent drop [interval] |", "|---|---|---|---|---|---|---|"]
    k2o = {ln["run"]: ln["outcome"] for ln in t["lines"] if ln["kind"] == "k2"}
    for r in t["runs"]:
        st = per.get(r["key"]) or {}
        k2 = st.get("k2")
        if not k2:
            continue
        zd = st.get("zero_drop")
        L.append(f"| {r['run']} | {k2['delta']:+.1f} | [{k2['lo']:+.2f}, {k2['hi']:+.2f}] | [{k2['pct_lo']:+.1f}, "
                 f"{k2['pct_hi']:+.1f}] | {k2['only_a']} / {k2['only_b']} | {k2o.get(r['run'], '')} | "
                 + (f"{zd['delta']:+.1f} [{zd['lo']:+.2f}, {zd['hi']:+.2f}]" if zd else "") + " |")
    L += ["", "### 3.3 Timed reads (A6.2; 5.5 (b)): ms per pass from each summary's timing block (the plain reads, "
              "the batch-1 reads and the band [0, 12] reads; the sweep and intervention reads' timings are in the json)",
          "",
          "| entry | path | batch | timed batches | prefill ms | latent ms (passes) | token ms (passes) | latent_last | "
          "device | torch | read seconds |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in t["runs"]:
        tm = (per.get(r["key"]) or {}).get("timing") or {}
        for kind in sorted(k_ for k_ in tm if k_ == "none" or k_.startswith(("timed_bs", "band_bs"))):
            x = tm[kind]
            path = (f"band {x['latent_band']}" if x["latent_band"] else "HF path")
            L.append(f"| {x['entry']} | {path} | {x['batch']} | {x['batches']} | {_ms(x['prefill_ms'])} | "
                     f"{_ms(x['latent_ms'])} ({x['latent_passes']}) | {_ms(x['token_ms'])} ({x['token_passes']}) | "
                     f"{'not recorded' if x['latent_last_ms'] is None else format(x['latent_last_ms'], '.3f')} | "
                     f"{x['device']} | {x['torch']} | {x['eval_seconds']:.2f} |")
    if t.get("timed_read_labels"):                 # the third table: 14:54Z row 2's label on the batch-1 timed reads
        L += ["", f"Labels on the batch-1 timed reads ({THIRD_LABELS[t['task']]['batch1']}; batch 1 is the timing of "
                  f"record, the batch-32 reads the accuracy of record):"]
        L += [f"- {k}: {v}" for k, v in t["timed_read_labels"].items()]
    L += ["", "### 3.4 Step sweep costs (7.3: accuracy against mean FLOPs, passes and emitted tokens; decides nothing)", "",
          "| run | n | accuracy | E' mean | passes mean | FLOPs mean (GFLOPs) | entry |", "|---|---|---|---|---|---|---|"]
    for r in t["runs"]:
        sw = (per.get(r["key"]) or {}).get("sweep") or {}
        for nn in sorted(sw, key=int):
            x = sw[nn]
            lab_ = " (plain read)" if x["plain_read"] else (" (out-of-distribution)" if x["ood"] else "")
            L.append(f"| {r['run']} | {nn}{lab_} | {x['points']:.1f} | {x['emitted_mean']:.3f} | {x['passes_mean']:.3f} | "
                     f"{x['flops_mean'] / 1e9:.3f} | {x['entry']} |")
    bad = [e for e in ev["entries"] if not e["ok"]]
    if stp:                                        # the arm's own eval job (row (b) (4))
        xb = [e for e in (t.get("eval_extra") or {}).get("entries") or [] if not e["ok"]]
        nx = len((t.get("eval_extra") or {}).get("entries") or [])
    L += ["", f"### 3.5 Entry checks: {len(ev['entries']) - len(bad)} of {len(ev['entries'])} entries pass every "
              "check" + (" (excluded: " + "; ".join(f"{e['name']}: {'; '.join(e['problems'])}" for e in bad) + ")"
                         if bad else ""), "",
          "Per entry: the receipt row (done, rc 0); summary.json, examples.csv and eval.log hashed (provenance); "
          "examples.csv equal to the summary's hash; the valid split's hash equal to the manifest's; one row per "
          "example in file order; the summary's accuracy and means equal to the file's; precision bf16 with A10.5's "
          "flags (A10.1); the registered row; the checkpoint sha256 equal to the run's RUN_DONE selected file and "
          "named by the row; the selected epoch named by the entry; run, seed, config, intervention and n_latent; the "
          "band; every row's 5.1 / 5.3 cost.", ""]
    if stp:
        L += [f"The no-curriculum eval job ({((t.get('eval_extra') or {}).get('job') or {}).get('job')}): {nx - len(xb)} "
              f"of {nx} reads pass every check (the same checks, against its registered row "
              f"{((t.get('eval_extra') or {}).get('job') or {}).get('registered_read')})" + (" (excluded: " + "; ".join(
                  f"{e['name']}: {'; '.join(e['problems'])}" for e in xb) + ")" if xb else "") + ".", ""]
    if st3:                                        # the fourth table: the hinge eval job's entries and records
        hb = [e for e in eh.get("entries") or [] if not e["ok"]]
        hd = eh.get("directional") or []
        db = [e for e in hd if not e.get("ok")]
        L += [f"The hinge eval job ({(eh.get('job') or {}).get('job')}): {len(eh.get('entries') or []) - len(hb)} of "
              f"{len(eh.get('entries') or [])} reads pass every check (the same checks, against its registered row "
              f"{(eh.get('job') or {}).get('registered_read')}); {len(hd) - len(db)} of {len(hd)} donor-answer records "
              f"pass every check (the registering row {DONOR_ROW}; the run's selected checkpoint named by a row; the "
              f"valid split's hash; the registered set's numerics; the class pinned by ruling 1; the registered set's "
              f"statistic recomputed from examples.csv)" + (" (excluded: " + "; ".join(
                  f"{e['name']}: {'; '.join(e['problems'])}" for e in hb + db) + ")" if hb or db else "") + ".", ""]
        L += ["### 3.6 The directional read (A6.1 rule 4 (B), the donor-answer read; critic entry "
              f"{DONOR_ROW}): the registered set (bf16 autocast over TF32 on), valid, the scored set (the class minus "
              "the ambiguous pairs), 10,000 paired draws from default_rng(0)", "",
              "| run | record | status | scored | R_dir (k_dir) | R_own (k_own) | Delta_dir [6.2 form] | percentile | "
              "G_chg (k) against R_dir: rule 5's reading | identification (none / scrambled of 500) | K2 cell |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        k2c = {ln["run"]: ln["outcome"] for ln in t["lines"] if ln["kind"] == "k2"}

        def drow(run, d, cell):
            if d.get("status") not in ("present", "absent"):
                i = d.get("identification")
                i = i if isinstance(i, str) else (i or {}).get("status", "")    # a label (11:07Z (3)) or its status
                return f"| {run} | {d.get('entry')} | {d.get('status')} | | | | | | | {i} | {cell} |"
            i = d.get("identification") or {}
            thin = " (thin)" if d.get("thin") and d["status"] == "present" else ""
            return (f"| {run} | {d['entry']} | {d['status']}{thin} | {d['n_scored']} of "
                    f"{d['n_class']} | {d['r_dir']:.3f} ({d['k_dir']}) | {d['r_own']:.3f} ({d['k_own']}) | "
                    f"{d['delta_dir']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}] | [{d['pct_lo']:+.3f}, {d['pct_hi']:+.3f}] | "
                    f"{d['g_chg']:.3f} ({d['k_g_chg']}): {'exceeded' if d['garbage_exceeds'] else 'not exceeded'} | "
                    f"{i.get('status')} ({i.get('none_diff')} / {i.get('scrambled_diff')}) | {cell} |")
        for r in t["runs"]:
            d = (per.get(r["key"]) or {}).get("directional")
            if d:
                L.append(drow(r["run"], d, k2c.get(r["run"], "")))
        if eh.get("reference"):
            L.append(drow(eh["reference"].get("run"), eh["reference"], A61["reference"]))
        blocks = sorted({d.get("label_block") for d in [(per.get(r["key"]) or {}).get("directional") or {}
                                                         for r in t["runs"]] + [eh.get("reference") or {}]
                         if d.get("label_block")})
        if blocks:
            L += ["", f"Labels on every number of these records (critic entry {DONOR_ROW} (2), its <gpu_type> slot as the "
                      f"read row {(eh.get('job') or {}).get('registered_read')} fills it): " + "; ".join(
                          f'"{b}"' for b in blocks)]
        L += ["", "The gold-literal and excess readings of the garbage clause are in the json and decide nothing "
                  f"(critic entry {DONOR_ROW} ruling 3); the complement class and the secondary rates are in the records "
                  "and decide nothing (section 10).", ""]
    L += ["## 4. Registered lines", ""]
    for kind in LINE_ORDER_EVAL:
        ls = [ln for ln in t["lines"] if ln["kind"] == kind]
        if kind == "k3" and pdb.get("points"):
            L += _md_apd(pdb, t["sigma_screen"]["registered"])
        if ls:
            L.append(f"### {LINE_TITLES_EVAL[kind]}")
            L += [f"- {ln['text']}" for ln in ls] + [""]
    rc = t["receipts"]
    L += ["### Receipts and launch condition (A3.2 (i); A5 item 8)",
          "- env_reference per training receipt: " + ", ".join(
              f"{e['attempt']} " + ("ok" if e["env_reference"] else "no env step (status " + str(e["status"]) + ")"
                                    if e["env_reference"] is None else "NOT ok") for e in rc["env_reference"])
          + f"; the eval job's receipt ({job['attempt']}): {'ok' if job['env_reference'] else 'NOT ok'}"
          + (f"; the hinge eval job's receipt ({eh['job']['attempt']}): {'ok' if eh['job']['env_reference'] else 'NOT ok'}"
             if st3 else "")
          + (f"; the no-curriculum eval job's receipt ({t['eval_extra']['job']['attempt']}): "
             f"{'ok' if t['eval_extra']['job']['env_reference'] else 'NOT ok'}" if stp else "") + f" [{EVAL_LABEL}]",
          f"- attempts named by the runs without a receipt in this read: "
          f"{', '.join(rc['attempts_without_receipt']) or 'none'} [{EVAL_LABEL}]",
          "- A5 item 8 (the bar's three cot trials admitted no later than the first latent arm, from the receipts' "
          f"admissions): {'met' if rc['a5_item8_bar_before_latent'] else 'not shown by the receipts read' if rc['a5_item8_bar_before_latent'] is None else 'NOT met'} [{EVAL_LABEL}]",
          "", "## 5. Readings and sensitivity", "",
          f"- sigma_screen of record {t['sigma_screen']['registered']:.4f} (terms: floor 1.0, binomial "
          f"{t['sigma_screen']['terms']['binomial']:.4f}, ProsQA cot 3-seed SD "
          f"{t['sigma_screen']['terms']['prosqa_cot_3seed_sd']}); alternatives recomputed: "
          + ", ".join(f"{k} {v:.4f}" for k, v in t["sigma_screen"]["alternatives"].items()) + "."]
    if t["sensitivity"]:
        L.append("- Lines whose outcome changes under an alternative reading:")
        L += [f"  - {s_['id']} under '{s_['reading']}': {s_['registered']} -> {s_['alternative']}"
              for s_ in t["sensitivity"]]
    else:
        L.append("- No line changes outcome under the alternative readings.")
    vf = t.get("vs_first_table") or []
    L.append("- Against the first table: " + ("every line it decided keeps its outcome; the lines it left pending "
                                               "are decided here." if not vf else "outcomes that changed: " +
                                               "; ".join(f"{x['id']}: {x['first']} -> {x['second']}" for x in vf)))
    if "vs_second_table" in t:
        vs = t["vs_second_table"]
        L.append("- Against the second table: " + ("every line keeps its outcome" if not vs else "outcomes that "
                 "changed: " + "; ".join(f"{x['id']}: {x['first']} -> {x['second']}" for x in vs))
                 + "; the json cells that differ (compare_second_table, each inside the reissue's allowed set): "
                 + ", ".join(t.get("diff_vs_second_table") or []) + "."
                 + (" (The third table's comparison, carried: every cell it describes is unchanged.)" if st3 else ""))
    if "vs_third_table" in t:
        vs = t["vs_third_table"]
        L.append("- Against the third table: " + ("every line keeps its outcome" if not vs else "outcomes that "
                 "changed: " + "; ".join(f"{x['id']}: {x['first']} -> {x['second']}" for x in vs))
                 + "; the json cells that differ (compare_third_table, each inside the reissue's allowed set): "
                 + ", ".join(t.get("diff_vs_third_table") or []) + "."
                 + (" (The fourth table's comparison, carried: every cell it describes is unchanged.)" if stp else ""))
    if "vs_prior_table" in t:
        vs = t["vs_prior_table"]
        L.append(f"- Against the {pord} table (the prior): " + ("every line keeps its outcome" if not vs else
                 "outcomes that changed: " + "; ".join(f"{x['id']}: {x['first']} -> {x['second']}" for x in vs))
                 + "; the json cells that differ (compare_prior_table, each inside the reissue's allowed set): "
                 + ", ".join(t.get("diff_vs_prior_table") or []) + ".")
    L += ["", "## 6. Notes", ""] + [f"- {x}" for x in t["notes"]]
    return "\n".join(L) + "\n"


def _md_apd(pdb: dict, sigma: float) -> list[str]:
    """The A_pd block of the third table (7.7; A13.2; critic entries 2026-10-20T15:53Z): the two points, the construction
    and the value at every latent arm's FLOPs."""
    L = [f"### A_pd(F): the plain-depth curve from the two sizes (7.7; A13.2; critic entries {ROW_PD})", "",
         f"Control label (every row and K3 line that uses it): {pdb['label']}.", "",
         "| size | run | model (revision) | N_ne | V | d | n_L | d_a | selection | valid | E' mean (total) | |q| mean | "
         "FLOPs per example, GFLOPs (exact range over the unrecorded per-example E') | ckpt_best_final.pt |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in pdb["points"]:
        d = p["dims"]
        L.append(f"| {p['size']} | {p['run']} | {p['model_id']} ({str(p['model_revision'])[:8]}) | {d['N_ne']:,} | "
                 f"{d['V']:,} | {d['d']} | {d['n_L']} | {d['d_a']} | epoch {p['epoch']} of {p['epochs']} (every epoch, "
                 f"3.3 item 1) | {p['points']:.1f} ({p['k']} of {p['n']}) | {p['emitted_mean']:.3f} "
                 f"({p['emitted_total']}) | {p['q_mean']:.2f} | [{p['flops_mean'][0] / 1e9:.4f}, "
                 f"{p['flops_mean'][1] / 1e9:.4f}] | {str(p['checkpoint'])[:12]}... |")
    p0, p1 = pdb["points"]
    L += ["", f"- {pdb['construction']}. Here A_pd(F) = {p0['points']:.1f} + {p1['points'] - p0['points']:.1f} x "
              f"ln(F / F_124M) / ln(F_355M / F_124M), with ln(F_355M / F_124M) in [{pdb['ln_ratio'][0]:.4f}, "
              f"{pdb['ln_ratio'][1]:.4f}] (the 355M point costs {pdb['ratio'][0]:.3f} to {pdb['ratio'][1]:.3f} times "
              f"the 124M point per example).", "",
          f"| arm | valid | F, GFLOPs (fraction) | w = ln(F / F_124M) / ln(F_355M / F_124M) | A_pd(F) | A_pd - "
          f"sigma_screen {sigma:.2f} | acc - A_pd | K3 (registered reading) | nearest size |",
          "|---|---|---|---|---|---|---|---|---|"]
    for k, a in pdb.get("arms", {}).items():
        F = " / ".join(f"{_gfr(x['flops_mean'])}" + ("" if x["fraction"] == "the arm's" else f" ({x['fraction']})")
                       for x in a["parts"])
        lo, hi = a["a_pd"]
        L.append(f"| {a['run']} | {a['points']:.1f} | {F} | [{a['w'][0]:.4f}, {a['w'][1]:.4f}] | [{lo:.4f}, {hi:.4f}] | "
                 f"[{lo - sigma:.4f}, {hi - sigma:.4f}] | [{a['points'] - hi:+.4f}, {a['points'] - lo:+.4f}] | {a['k3']} | "
                 f"{'yes' if a['nearest_size'] else 'no'} |")
    return L + [""]


def committed_sha256(commit: str, path: str) -> str | None:
    """sha256 of `path` as committed at `commit` (git show), or None when git cannot show it."""
    rel = os.path.relpath(os.path.abspath(path), REPO)
    try:
        data = subprocess.run(["git", "-C", REPO, "show", f"{commit}:{rel}"], capture_output=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(data).hexdigest()


def git_path_state(path: str) -> dict:
    """The last commit touching `path` and whether it has uncommitted changes (None outside the repository)."""
    ap = os.path.abspath(path)
    if not ap.startswith(REPO + os.sep):
        return {"in_repo": False, "commit": None, "dirty": None}
    rel = os.path.relpath(ap, REPO)
    try:
        commit = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%H", "--", rel], capture_output=True,
                                text=True, check=True).stdout.strip() or None
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=all", "--", rel],
                               capture_output=True, text=True, check=True).stdout.strip()
        full = subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "--quiet", "HEAD"], capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return {"in_repo": True, "commit": None, "dirty": None}
    return {"in_repo": True, "commit": commit, "dirty": bool(dirty), "head": full or None}


def resolve_commit(commit: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", REPO, "rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}"],
                              capture_output=True, text=True, check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def check_words(*texts: str) -> None:
    for tx in texts:
        low = tx.lower()
        bad = [w for w in FORBIDDEN if w in low]
        if bad:
            raise SystemExit(f"REFUSED: output contains A11.3's forbidden wording {bad}")


def _git_head() -> dict:
    try:
        head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain", "--", os.path.relpath(__file__, REPO)],
                               capture_output=True, text=True, check=True).stdout.strip()
        return {"head": head, "tool_uncommitted": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "tool_uncommitted": None}


def cite_first_table(sc: "Screen", first_json: str, commit: str) -> dict:
    """The first table the second cites (A6.4's reissue pattern): its three files must hash as committed at `commit`
    (refused otherwise: the first table is write-once and never modified); its json is read for the comparison."""
    return {"first_table": _cite_table(sc, first_json, commit, "the first table")}


def cite_second_table(sc: "Screen", second_json: str, commit: str) -> dict:
    """The second table the third cites, as cite_first_table: its three files must hash as committed at `commit`."""
    return {"second_table": _cite_table(sc, second_json, commit, "the second table")}


def _cite_table(sc: "Screen", first_json: str, commit: str, what: str) -> dict:
    base = first_json[:-len(".json")] if first_json.endswith(".json") else first_json
    files = [base + ".json", base + ".md", base + "_provenance.json"]
    full = resolve_commit(commit)
    if full is None:
        raise SystemExit(f"REFUSED: --first-commit {commit} is not a commit of this repository")
    got = {}
    for f in files:
        if not os.path.isfile(f):
            raise SystemExit(f"REFUSED: {what}'s file {f} is absent")
        want = committed_sha256(full, f)
        have = sc.inp.sha256(f, f"{what} (cited)")
        if want != have:
            raise SystemExit(f"REFUSED: {f} differs from its content at {commit} ({what} is write-once)")
        got[os.path.basename(f)] = have
    t1 = sc.inp.read_json(files[0], f"{what} (cited)")
    return {"files": [os.path.relpath(f, REPO) if os.path.abspath(f).startswith(REPO + os.sep) else f for f in files],
            "commit": full, "sha256": got, "unchanged": True, "table": t1}


def compare_first_table(t1: dict, t2: dict, new_runs=()) -> list[dict]:
    """The second table rebuilds the first from the same records: the bar, sigma_screen and every run's status and
    selection must be equal (refused otherwise); a line the first table decided whose outcome differs is listed.
    new_runs: runs UNTESTED in the first table that the later table reads (the third table's control, the fourth
    table's hinge arms); their lines were left UNTESTED by the first table and are decided by the later one."""
    for k in ("task", "n_valid", "bar"):
        if t1.get(k) != t2.get(k):
            raise SystemExit(f"REFUSED: the first table's {k} differs from the records read now")
    if abs(t1["sigma_screen"]["registered"] - t2["sigma_screen"]["registered"]) > 1e-12:
        raise SystemExit("REFUSED: sigma_screen differs from the first table's")
    r1 = {r["key"]: r for r in t1["runs"]}
    for r in t2["runs"]:
        a = r1.get(r["key"]) or {}
        if r["key"] in new_runs and a.get("status") in ("UNTESTED", None):    # None: not in the first table at all
            continue
        if (a.get("status"), a.get("k"), (a.get("selected") or {}).get("epoch")) != \
                (r.get("status"), r.get("k"), (r.get("selected") or {}).get("epoch")):
            raise SystemExit(f"REFUSED: run {r['run']}: status or selection differs from the first table's")
    first = {ln["id"]: ln["outcome"] for ln in t1["lines"]}
    new_names = {r["run"] for r in t2["runs"] if r["key"] in new_runs and (r1.get(r["key"]) or {}).get("status") in
                 ("UNTESTED", None)}
    out = []
    for ln in t2["lines"]:
        if ln.get("run") in new_names:
            continue
        o1 = first.get(ln["id"])
        o2 = ln["outcome"]
        if o1 is None or o2.startswith(o1) or o1 in ("pending", "nominal") or o1.startswith("pass: pending K2"):
            continue
        out.append({"id": ln["id"], "first": o1, "second": ln["outcome"]})
    return out


def compare_outcomes(t_old: dict, t_new: dict) -> list[dict]:
    """Every line of the older table whose outcome differs in the newer one (missing lines listed as 'removed')."""
    new = {ln["id"]: ln["outcome"] for ln in t_new["lines"]}
    return [{"id": ln["id"], "first": ln["outcome"], "second": new.get(ln["id"], "removed")} for ln in t_old["lines"]
            if new.get(ln["id"], "removed") != ln["outcome"]]


def _list_key(xs: list):
    """The key of a list of dicts that json_diff matches by key (lines by id, runs by key, entries by name, receipts by
    attempt, sensitivity by (reading, id)); None for a plain list."""
    if not xs or not all(isinstance(x, dict) for x in xs):
        return None
    for ks in (("reading", "id"), ("id",), ("key",), ("name",), ("attempt",)):
        if all(all(k in x for k in ks) for x in xs):
            vals = ["|".join(str(x[k]) for k in ks) for x in xs]
            if len(set(vals)) == len(vals):
                return ks
    return None


def json_diff(a, b, path: tuple = ()) -> list[tuple[tuple, str, object, object]]:
    """Every leaf where two json values differ: (path, 'changed' | 'added' | 'removed' | 'order', old, new). Lists of
    keyed dicts are matched by key (and their common keys' order compared); other lists by index."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(a) + [k for k in b if k not in a]:
            if k not in b:
                out.append((path + (k,), "removed", a[k], None))
            elif k not in a:
                out.append((path + (k,), "added", None, b[k]))
            else:
                out += json_diff(a[k], b[k], path + (k,))
        return out
    if isinstance(a, list) and isinstance(b, list):
        ka, kb = _list_key(a), _list_key(b)
        if not a or not b:                         # one side empty: match by the other side's key
            ka = kb = ka or kb
        if ka is not None and ka == kb:
            def key(x):
                return "|".join(str(x[k]) for k in ka)
            da, db = {key(x): x for x in a}, {key(x): x for x in b}
            out = [(path + (k,), "removed", da[k], None) for k in da if k not in db]
            out += [(path + (k,), "added", None, db[k]) for k in db if k not in da]
            for k in da:
                if k in db:
                    out += json_diff(da[k], db[k], path + (k,))
            common = [k for k in da if k in db]
            if common != [k for k in db if k in da]:
                out.append((path, "order", None, None))
            return out
        if len(a) != len(b):
            return [(path, "changed", a, b)]
        return [x for i, (u, v) in enumerate(zip(a, b)) for x in json_diff(u, v, path + (i,))]
    return [] if a == b else [(path, "changed", a, b)]


def _is_insertion(old: str, new: str) -> bool:
    """new = old with one segment inserted (common prefix plus common suffix cover old)."""
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    s = 0
    while s < min(len(old), len(new)) - p and old[-1 - s] == new[-1 - s]:
        s += 1
    return p + s >= len(old) and len(new) >= len(old)


def compare_second_table(t2: dict, t3: dict, pd_runs=(), pd_attempts=()) -> list[str]:
    """The third table against the second (critic entry 2026-10-20T15:53Z (4)): every json cell must be equal except
    the reissue's allowed set: the control's run entry and the plain_depth block; the six K3 lines and their sensitivity
    entries; the control's UNTESTED line (removed); the receipts (the control's attempt and runs added, nothing else);
    the f_meas lines and fields and the C1b cards' per-step cost text (15:26Z row 3); rule 3's line and the batch-1 reads'
    wall-clock lines (14:54Z row 2 labels, inserted); the f_meas note and the pending note; the citation of the second
    table, the batch-1 labels, generated_utc and the comparison keys. Returns the differing paths; refuses otherwise."""
    bad, seen = [], []
    for path, kind, old, new in json_diff(t2, t3):
        p = ".".join(str(x) for x in path)
        seen.append(p)
        top = path[0] if path else ""
        sub = path[1] if len(path) > 1 else None
        ok = False
        if top in ("generated_utc", "timed_read_labels", "vs_second_table", "diff_vs_second_table", "plain_depth"):
            ok = True
        elif top == "citations":
            ok = sub == "second_table" and kind == "added"
        elif top == "runs":
            ok = sub == "plain_depth_355m"
        elif top == "lines":
            lid, fld = str(sub), (path[2] if len(path) > 2 else None)
            if lid.startswith("k3:"):
                ok = kind == "changed" and fld in ("text", "outcome")
            elif lid == "untested:plain_depth_355m":
                ok = kind == "removed"
            elif lid in ("c1b_cost:c1b_k8", "c1b_cost:c1b_k4"):
                ok = kind == "changed" and (fld == "outcome" or (fld == "text" and old.split("; saving line f_meas")[0]
                                                                 == new.split("; rule 4's line under both")[0]))
            elif lid in ("card:c1b_k8", "card:c1b_k4"):
                ok = kind == "changed" and fld == "text" and old.split("; per-step cost: ")[0] == \
                    new.split("; per-step cost: ")[0]
            elif lid == "rule3:coconut":
                ok = kind == "changed" and (fld == "outcome" or (fld == "text" and old.split(": more than")[0] ==
                                                                 new.split(": more than")[0]))
            elif lid in ("wallclock:coconut", "wallclock:c1b_k8", "wallclock:c1b_k4"):
                ok = kind == "changed" and fld == "text" and _is_insertion(old, new)
        elif top == "sensitivity":
            ok = "|k3:" in str(sub)
        elif top == "receipts":
            if sub == "env_reference":
                ok = kind == "added" and len(path) == 3 and str(path[2]) in pd_attempts
            elif sub == "first_admission":
                ok = kind == "added" and len(path) == 3 and path[2] in pd_runs
            elif sub == "attempts_named_by_runs" and kind == "changed" and len(path) == 2:
                ok = set(old) <= set(new) and set(new) - set(old) <= set(pd_attempts)
        elif top == "notes":
            o2 = (t2.get("notes") or [None] * (int(sub) + 1))[int(sub)] if isinstance(sub, int) else None
            ok = kind == "changed" and isinstance(o2, str) and o2.startswith(("f_meas (A6.2", "Still pending"))
        elif top == "eval" and len(path) >= 6 and path[1:2] == ("per_run",) and path[3] == "f_meas":
            ok = path[2] in ("c1b_k8", "c1b_k4") and path[-1] in ("saving_label", "rule3", "line_of_record", "label")
        elif top == "eval" and len(path) == 3 and path[1] == "per_run" and path[2] == "plain_depth_355m":
            ok = kind == "added" and new == {"reads": {}}           # the control has no eval read (as no_cot)
        if not ok:
            bad.append(f"{p} ({kind})")
    if bad:
        raise SystemExit("REFUSED: the third table differs from the second outside the reissue's allowed cells "
                         "(critic entry 2026-10-20T15:53Z (4)): " + "; ".join(bad))
    return sorted(set(seen))


FOURTH_LINE_KINDS = ("cost", "k2", "k3", "pause", "screen", "fade", "eval_none", "sweep", "wallclock")
FOURTH_NOTES = ("The hinge arms (critic entry", "A6.1 rules 2 and 3 on the hinge arms", A61["reference"])
ALT_NOTE = "Not a registered reading, printed for the critic"


def compare_third_table(t3: dict, t4: dict, hinge_keys=(), hinge_runs=(), hinge_attempts=()) -> list[str]:
    """The fourth table against the third (critic entry 2026-10-20T17:12Z (5): "only the three arms' run rows, their cost
    rows, the UNTESTED lines they replace (sections 1 and the "UNTESTED and excluded" list), their K2, 7.10, 7.2, 5.5
    (b), K3 and card lines, the receipts line and the notes those lines touch may change; every other cell equals the
    third table's"). Allowed, and nothing else: the three runs' entries (UNTESTED to read) and their eval.per_run and
    plain_depth.arms entries; their lines (cost, K2, K3, pause, the 6.3 screen line, fade, the plain read, the sweep,
    5.5 (b): added as the six arms carry them), the C3 and C2+C3 card lines (changed), their three UNTESTED lines
    (removed when the run is read; its exclusion when a record check fails), the 7.10 lines (null triggers,
    too-good), the hinge eval job's line and the lines of its excluded
    entries (added), the sensitivity entries of those lines; the receipts (the hinge job's attempts, admissions and
    the attempts its runs name: added); the K2 seed-term note (the hinge arms' intervals inserted) and the fourth
    table's own notes (appended); the eval_hinge block, the third table's and the hinge eval job's citations, the
    comparison with the third table and generated_utc. Returns the differing paths; refuses otherwise."""
    hk, hr, ha = set(hinge_keys), set(hinge_runs), set(hinge_attempts)
    added = {f"{k}:{key}" for k in FOURTH_LINE_KINDS for key in hk}
    cards = {f"card:{key}" for key in hk if f"card:{key}" in {ln["id"] for ln in t3.get("lines") or []}}
    untested = {f"untested:{key}" for key in hk}
    eh = t4.get("eval_hinge") or {}
    h_names = {e["name"] for e in eh.get("entries") or []} | {e["name"] for e in eh.get("directional") or []}
    bad, seen = [], []
    a = {k: v for k, v in t3.items() if k != "notes"}
    b = {k: v for k, v in t4.items() if k != "notes"}
    for path, kind, old, new in json_diff(a, b):
        p = ".".join(str(x) for x in path)
        seen.append(p)
        top = path[0] if path else ""
        sub = path[1] if len(path) > 1 else None
        ok = False
        if top == "generated_utc":
            ok = kind == "changed"
        elif top in ("eval_hinge", "vs_third_table", "diff_vs_third_table"):
            ok = kind == "added" and len(path) == 1
        elif top == "citations":
            ok = sub in ("third_table", "hinge_eval") and kind == "added" and len(path) == 2
        elif top == "runs":
            ok = sub in hk and len(path) >= 3
        elif top == "plain_depth":
            ok = sub == "arms" and len(path) == 3 and path[2] in hk and kind == "added"
        elif top == "eval":
            ok = len(path) == 3 and sub == "per_run" and path[2] in hk and kind == "added"
        elif top == "lines":
            lid, fld = str(sub), (path[2] if len(path) > 2 else None)
            if lid in added or lid == "eval_job:hinge" or (lid.startswith("eval_excluded:") and
                                                           lid.split(":", 1)[1] in h_names):
                ok = kind == "added" and fld is None
            elif lid in cards:
                ok = kind == "changed" and fld in ("text", "outcome", "run")
            elif lid in untested:                  # read: removed; a run that fails a check: its exclusion line
                ok = (kind == "removed" and fld is None) or (kind == "changed" and fld in ("text", "outcome"))
            elif lid in ("null_trigger", "too_good"):
                ok = kind == "changed" and fld in ("text", "outcome")
        elif top == "sensitivity":
            sid = str(sub).split("|", 1)[-1]
            ok = sid in added or sid in cards or sid in ("null_trigger", "too_good")
        elif top == "receipts":
            if sub == "env_reference":
                ok = kind == "added" and len(path) == 3 and str(path[2]) in ha
            elif sub == "first_admission":
                ok = kind == "added" and len(path) == 3 and path[2] in hr
            elif sub in ("attempts_named_by_runs", "attempts_without_receipt") and kind == "changed" and len(path) == 2:
                ok = set(old) <= set(new) and set(new) - set(old) <= ha
        if not ok:
            bad.append(f"{p} ({kind})")
    n3, n4 = list(t3.get("notes") or []), list(t4.get("notes") or [])
    if len(n4) < len(n3):
        seen.append("notes")
        bad.append("notes (removed)")
    for i, (x, y) in enumerate(zip(n3, n4)):
        if x != y:
            seen.append(f"notes.{i}")
            if not (x.startswith(ALT_NOTE) and y.startswith(ALT_NOTE) and _is_insertion(x, y)):
                bad.append(f"notes.{i} (changed)")
    for i in range(len(n3), len(n4)):
        seen.append(f"notes.{i}")
        if not n4[i].startswith(FOURTH_NOTES):
            bad.append(f"notes.{i} (added)")
    if bad:
        raise SystemExit("REFUSED: the fourth table differs from the third outside the reissue's allowed cells "
                         "(critic entry 2026-10-20T17:12Z (5)): " + "; ".join(bad))
    return sorted(set(seen))


def compare_prior_table(tp: dict, tn: dict, x_keys=(), x_runs=(), x_attempts=(), x_entries=()) -> list[str]:
    """The no-curriculum reissue against its prior table (the GSM-50k fourth, the ProsQA-hard third; critic entry
    2026-10-21T20:21Z (b) (5): "only the new arm's rows and lines are added; every existing row, line, label and outcome
    byte-equal"). Allowed, and nothing else: the arm's run entry and its eval.per_run and plain_depth.arms entries
    (added); its lines (cost, K2, K3, pause, fade, the plain read, the sweep, 5.5 (b), the Curriculum line; its UNTESTED
    line when it enters no cell: added), its eval job's line and the lines of that job's excluded entries (added), the
    7.10 lines (null triggers, too-good: changed), the sensitivity entries of those lines; the receipts line (the arm's
    train-job attempts, its admission, the attempts it names: added; A5 item 8's flag); the K2 seed-term note (the
    arm's interval inserted) and the reissue's own note (appended); the eval_extra block, the prior table's and the
    arm's eval job's citations, the comparison fields (vs_first_table, vs_prior_table, diff_vs_prior_table) and
    generated_utc. Every other json cell must equal the prior table's. Returns the differing paths; refuses otherwise."""
    xk, xr, xa, xe = set(x_keys), set(x_runs), set(x_attempts), set(x_entries)
    added = {f"{k}:{key}" for k in CLOSING_LINE_KINDS for key in xk} | {"eval_job:extra"} | {
        f"eval_excluded:{nm}" for nm in xe}
    bad, seen = [], []
    a = {k: v for k, v in tp.items() if k != "notes"}
    b = {k: v for k, v in tn.items() if k != "notes"}
    for path, kind, old, new in json_diff(a, b):
        p = ".".join(str(x) for x in path)
        seen.append(p)
        top = path[0] if path else ""
        sub = path[1] if len(path) > 1 else None
        ok = False
        if top == "generated_utc":
            ok = kind == "changed"
        elif top in ("vs_first_table", "vs_prior_table", "diff_vs_prior_table"):
            ok = True
        elif top == "eval_extra":
            ok = kind == "added" and len(path) == 1
        elif top == "citations":
            ok = sub in ("prior_table", "extra_eval") and kind == "added" and len(path) == 2
        elif top == "runs":
            ok = sub in xk and kind == "added" and len(path) == 2
        elif top == "plain_depth":
            ok = sub == "arms" and len(path) == 3 and path[2] in xk and kind == "added"
        elif top == "eval":
            ok = len(path) == 3 and sub == "per_run" and path[2] in xk and kind == "added"
        elif top == "lines":
            lid, fld = str(sub), (path[2] if len(path) > 2 else None)
            if lid in added:
                ok = kind == "added" and fld is None
            elif lid in ("null_trigger", "too_good"):
                ok = kind == "changed" and fld in ("text", "outcome")
        elif top == "sensitivity":
            sid = str(sub).split("|", 1)[-1]
            ok = sid in added or sid in ("null_trigger", "too_good")
        elif top == "receipts":
            if sub == "env_reference":
                ok = kind == "added" and len(path) == 3 and str(path[2]) in xa
            elif sub == "first_admission":
                ok = kind == "added" and len(path) == 3 and path[2] in xr
            elif sub in ("attempts_named_by_runs", "attempts_without_receipt") and kind == "changed" and len(path) == 2:
                ok = set(old) <= set(new) and set(new) - set(old) <= xa
            elif sub == "a5_item8_bar_before_latent":
                ok = kind == "changed" and len(path) == 2
        if not ok:
            bad.append(f"{p} ({kind})")
    n3, n4 = list(tp.get("notes") or []), list(tn.get("notes") or [])
    if len(n4) < len(n3):
        seen.append("notes")
        bad.append("notes (removed)")
    for i, (x, y) in enumerate(zip(n3, n4)):
        if x != y:
            seen.append(f"notes.{i}")
            if not (x.startswith(ALT_NOTE) and y.startswith(ALT_NOTE) and _is_insertion(x, y)):
                bad.append(f"notes.{i} (changed)")
    for i in range(len(n3), len(n4)):
        seen.append(f"notes.{i}")
        if not n4[i].startswith(CLOSING_NOTES):
            bad.append(f"notes.{i} (added)")
    if bad:
        raise SystemExit("REFUSED: the no-curriculum reissue differs from its prior table outside the allowed cells "
                         f"(critic entry {CLOSING_ROW} (b) (5)): " + "; ".join(bad))
    return sorted(set(seen))


def add_training(t: dict, runs_raw: dict, spec: dict) -> None:
    """5.5 (d) training FLOPs: own rows; GSM-50k latent arms also charged the start run (5.2); ratio to the bar per epoch."""
    by = {r["key"]: r for r in t["runs"]}
    bar = by[t["bar"]["key"]]
    bar_epochs = len(bar["curve"])
    per_epoch = bar["train_flops_own"] / bar_epochs
    src = None
    if spec["start_pin"]:
        src = next((r for r in t["runs"] if r["run"] == spec["start_pin"]["source_run"]), None)
    for r in t["runs"]:
        if "train_flops_own" not in r or r["status"] != "read":
            continue
        own, ep = r["train_flops_own"], len(r["curve"])
        charged, ep_c = own, ep
        if src is not None and r.get("start_pin"):
            charged += src["train_flops_own"]
            ep_c += len(src["curve"])
        ratio = charged / (per_epoch * ep_c)
        r["train"] = {"own": own, "charged": charged, "charged_epochs": ep_c, "ratio_same_epochs": ratio,
                      "ratio_to_bar_total": charged / bar["train_flops_own"], "heavy": ratio > 1.25}


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--records", required=True, help="local mirror of the bucket's silent/ prefix (small files only)")
    ap.add_argument("--receipts", nargs="*", default=[], help="JOB_RECEIPT.json files or globs")
    ap.add_argument("--source-uri", default=None, help="the bucket prefix the records mirror (recorded per file)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--table-name", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--critic-log", default=None)
    ap.add_argument("--k1-table", default=None)
    ap.add_argument("--valid-file", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--eval-dir", default=None,
                    help="the screen eval job's results (results/evals/<job>/): writes the second table, 'with the eval "
                         "reads' (critic entry 2026-10-20T10:51Z part (6)); needs --first-commit")
    ap.add_argument("--first-table", default=None,
                    help="the first table's json (default results/screens/<task>/<task>_screen.json)")
    ap.add_argument("--first-commit", default=None, help="the commit that wrote the first table (e.g. rev10)")
    ap.add_argument("--plain-depth-dir", default=None,
                    help="the plain-depth control's records (a mirror of the bucket's silent/ prefix: runs/<run>/ of both "
                         "plain-depth runs and receipts/<job>/<attempt>/ of their job): writes the third table, 'with the "
                         "plain-depth 355M control' (critic entries 2026-10-20T15:53Z); needs --eval-dir, --first-commit "
                         "and --second-commit")
    ap.add_argument("--second-table", default=None,
                    help="the second table's json (default results/screens/<task>/<task>_screen_with_eval.json)")
    ap.add_argument("--second-commit", default=None, help="the commit that wrote the second table (e.g. rev11)")
    ap.add_argument("--diag-gpu", default=None,
                    help="A6.2 rule 3's strict-fp32 GPU logits record (summary.json; optional, printed beside; default "
                         "per task, DIAG_RECORDS)")
    ap.add_argument("--diag-float64", default=None,
                    help="A6.2 rule 3's float64 CPU logits record (summary.json; required by the third table, the line; "
                         "default per task, DIAG_RECORDS)")
    ap.add_argument("--hinge-eval-dir", default=None,
                    help="the hinge eval job's results (results/evals/screen-eval-gsm50k-hinge-main/): writes the fourth "
                         "table, 'with the hinge arms' (critic entries 2026-10-20T17:12Z (5), (6), (9) and 17:36Z (5)); needs "
                         "every option of the third table plus --third-commit, --hinge-records and --registered-read")
    ap.add_argument("--hinge-records", default=None,
                    help="the hinge runs' records (a mirror of AcctA's bucket's silent/ prefix: runs/<run>/ of the three "
                         "hinge runs and receipts/screen-gsm50k-accta/<attempt>/JOB_RECEIPT.json)")
    ap.add_argument("--hinge-source-uri", default=None, help="the bucket prefix --hinge-records mirrors (recorded per file)")
    ap.add_argument("--registered-read", default=None,
                    help="the critic entry (YYYY-MM-DDTHH:MMZ) registering the hinge eval job's reads (the 10:51Z pattern)")
    ap.add_argument("--third-table", default=None,
                    help="the third table's json (default results/screens/<task>/<task>_screen_with_plain_depth.json)")
    ap.add_argument("--third-commit", default=None, help="the commit that wrote the third table (e.g. rev12)")
    ap.add_argument("--extra-arm", default=None, choices=sorted(CLOSING),
                    help="the closing plan's no-curriculum control (critic entry 2026-10-21T20:21Z (b)): writes the "
                         "reissue 'with the no-curriculum control' (GSM-50k: the fourth table's, ProsQA-hard: the "
                         "third's); needs every option of the prior table plus --extra-records, --extra-eval-dir, "
                         "--extra-registered-read and --prior-commit")
    ap.add_argument("--extra-records", default=None,
                    help="the arm's records (a mirror of its training project's bucket's silent/ prefix: runs/<run>/ "
                         "and receipts/<its train job>/<attempt>/JOB_RECEIPT.json)")
    ap.add_argument("--extra-source-uri", default=None, help="the bucket prefix --extra-records mirrors")
    ap.add_argument("--extra-eval-dir", default=None,
                    help="the arm's eval job's results (results/evals/screen-eval-<task>-nocur-<project>/)")
    ap.add_argument("--extra-registered-read", default=None,
                    help="the critic entry (YYYY-MM-DDTHH:MMZ) registering the arm's eval reads and naming its checkpoint")
    ap.add_argument("--prior-table", default=None,
                    help="the prior table's json (default results/screens/<task>/<the fourth GSM-50k or the third "
                         "ProsQA-hard table>.json)")
    ap.add_argument("--prior-commit", default=None,
                    help="the commit that wrote the prior table (GSM-50k rev9, ProsQA-hard rev18)")
    a = ap.parse_args(argv)
    closing_only = [o for o, v in (("--extra-records", a.extra_records), ("--extra-source-uri", a.extra_source_uri),
                                   ("--extra-eval-dir", a.extra_eval_dir),
                                   ("--extra-registered-read", a.extra_registered_read),
                                   ("--prior-table", a.prior_table), ("--prior-commit", a.prior_commit)) if v]
    if closing_only and not a.extra_arm:
        ap.error(f"{', '.join(closing_only)} belong to the no-curriculum reissue: they need --extra-arm")
    if a.extra_arm:
        miss = [o for o, v in (("--extra-records", a.extra_records), ("--extra-eval-dir", a.extra_eval_dir),
                               ("--extra-registered-read", a.extra_registered_read),
                               ("--prior-commit", a.prior_commit), ("--eval-dir", a.eval_dir),
                               ("--first-commit", a.first_commit), ("--plain-depth-dir", a.plain_depth_dir),
                               ("--second-commit", a.second_commit))
                + ((("--hinge-eval-dir", a.hinge_eval_dir),) if a.task in HINGE else ()) if not v]
        if miss:
            ap.error(f"--extra-arm writes the reissue of the {CLOSING[a.extra_arm]['prior'][a.task][1]} table: it needs "
                     f"{', '.join(miss)}")
        if not CRITIC_TS_RE.fullmatch(a.extra_registered_read):
            ap.error(f"--extra-registered-read {a.extra_registered_read!r} is not a critic entry stamp (YYYY-MM-DDTHH:MMZ)")
    name = a.table_name or (CLOSING_NAME.format(task=a.task) if a.extra_arm else
                            f"{a.task}_screen_with_hinge" if a.hinge_eval_dir else
                            f"{a.task}_screen_with_plain_depth" if a.plain_depth_dir else
                            f"{a.task}_screen_with_eval" if a.eval_dir else f"{a.task}_screen")
    if not TABLE_NAME_RE.fullmatch(name):
        ap.error(f"--table-name {name!r} must match {TABLE_NAME_RE.pattern}")
    if a.eval_dir and not a.first_commit:
        ap.error("--eval-dir needs --first-commit: the second table cites the commit that wrote the first")
    if a.eval_dir and name == f"{a.task}_screen":
        ap.error("the second table needs a name of its own; the first table's files are never rewritten")
    if a.plain_depth_dir and not (a.eval_dir and a.second_commit):
        ap.error("--plain-depth-dir writes the third table: it needs --eval-dir, --first-commit and --second-commit "
                 "(it cites both earlier tables)")
    if a.plain_depth_dir and name in (f"{a.task}_screen", f"{a.task}_screen_with_eval"):
        ap.error("the third table needs a name of its own; the first two tables' files are never rewritten")
    fourth_only = [o for o, v in (("--hinge-records", a.hinge_records), ("--hinge-source-uri", a.hinge_source_uri),
                                  ("--registered-read", a.registered_read), ("--third-commit", a.third_commit),
                                  ("--third-table", a.third_table)) if v]
    if fourth_only and not a.hinge_eval_dir:
        ap.error(f"{', '.join(fourth_only)} belong to the fourth table: they need --hinge-eval-dir")
    if a.hinge_eval_dir:
        if a.task not in HINGE:
            ap.error(f"no hinge job on {a.task} (A2.1; critic entry {ROW_HINGE} (8): C3 and C2+C3 on ProsQA-hard close "
                     f"UNTESTED): no fourth table")
        miss = [o for o, v in (("--eval-dir", a.eval_dir), ("--first-commit", a.first_commit),
                               ("--plain-depth-dir", a.plain_depth_dir), ("--second-commit", a.second_commit),
                               ("--third-commit", a.third_commit), ("--hinge-records", a.hinge_records),
                               ("--registered-read", a.registered_read)) if not v]
        if miss:
            ap.error(f"--hinge-eval-dir writes the fourth table (the third table's reissue over the hinge arms): it "
                     f"needs {', '.join(miss)}")
        if not CRITIC_TS_RE.fullmatch(a.registered_read):
            ap.error(f"--registered-read {a.registered_read!r} is not a critic entry stamp (YYYY-MM-DDTHH:MMZ)")
        if name in (f"{a.task}_screen", f"{a.task}_screen_with_eval", f"{a.task}_screen_with_plain_depth"):
            ap.error("the fourth table needs a name of its own; the first three tables' files are never rewritten")
    if a.extra_arm and name in (f"{a.task}_screen", f"{a.task}_screen_with_eval", f"{a.task}_screen_with_plain_depth",
                                f"{a.task}_screen_with_hinge"):
        ap.error("the no-curriculum reissue needs a name of its own; the earlier tables' files are never rewritten")
    out = a.out or os.path.join(EXP, "results", "screens", a.task)
    paths = {ext: os.path.join(out, f"{name}{suf}") for ext, suf in
             (("json", ".json"), ("md", ".md"), ("prov", "_provenance.json"))}
    if any(os.path.exists(p) for p in paths.values()):
        raise SystemExit(f"REFUSED: {' / '.join(p for p in paths.values() if os.path.exists(p))} exists; the screen "
                         f"table is write-once (A3.4; the registration's write-once rule for verdict files)")
    if a.hinge_eval_dir and not os.path.isdir(a.hinge_eval_dir):
        raise SystemExit(f"REFUSED: the hinge eval dir {a.hinge_eval_dir} is absent: the fourth table needs the hinge eval "
                         f"job's reads (critic entry {ROW_HINGE} (5) and (6); its registered read row "
                         f"{a.registered_read}); nothing written")
    if a.hinge_records and not os.path.isdir(os.path.join(a.hinge_records, "runs")):
        raise SystemExit(f"REFUSED: --hinge-records {a.hinge_records} has no runs/ directory")
    if a.extra_arm and not os.path.isdir(a.extra_eval_dir):
        raise SystemExit(f"REFUSED: the no-curriculum eval dir {a.extra_eval_dir} is absent: the reissue needs the arm's "
                         f"eval reads (critic entry {CLOSING_ROW} (b) (4); its registered read row "
                         f"{a.extra_registered_read}); nothing written")
    if a.extra_arm and not os.path.isdir(os.path.join(a.extra_records, "runs")):
        raise SystemExit(f"REFUSED: --extra-records {a.extra_records} has no runs/ directory")
    diag = {k: os.path.abspath(v) for k, v in (("gpu", a.diag_gpu), ("float64", a.diag_float64)) if v} or None
    sc = Screen(a.task, a.records, a.receipts, source_uri=a.source_uri, manifest=a.manifest, critic_log=a.critic_log,
                k1_table=a.k1_table, valid_file=a.valid_file, tokenizer=a.tokenizer, eval_dir=a.eval_dir,
                plain_depth_dir=a.plain_depth_dir, diag=diag, hinge_records=a.hinge_records,
                hinge_source_uri=a.hinge_source_uri, hinge_eval_dir=a.hinge_eval_dir, hinge_read=a.registered_read,
                extra_arm=a.extra_arm, extra_records=a.extra_records, extra_source_uri=a.extra_source_uri,
                extra_eval_dir=a.extra_eval_dir, extra_read=a.extra_registered_read)
    citations = None
    if a.eval_dir:
        citations = cite_first_table(sc, a.first_table or os.path.join(EXP, "results", "screens", a.task,
                                                                        f"{a.task}_screen.json"), a.first_commit)
        if a.plain_depth_dir:
            citations.update(cite_second_table(sc, a.second_table or os.path.join(
                EXP, "results", "screens", a.task, f"{a.task}_screen_with_eval.json"), a.second_commit))
        if a.hinge_eval_dir:
            citations["third_table"] = _cite_table(sc, a.third_table or os.path.join(
                EXP, "results", "screens", a.task, f"{a.task}_screen_with_plain_depth.json"), a.third_commit,
                "the third table")
        est = git_path_state(a.eval_dir)
        if est["in_repo"] and (est["dirty"] or not est["commit"]):
            raise SystemExit(f"REFUSED: {a.eval_dir} is not committed clean (write-once eval results, A10.2)")
        citations["eval"] = {"dir": os.path.relpath(os.path.abspath(a.eval_dir), REPO) if est["in_repo"]
                             else os.path.abspath(a.eval_dir), "commit": est["commit"]}
        if a.hinge_eval_dir:
            hst = git_path_state(a.hinge_eval_dir)
            if hst["in_repo"] and (hst["dirty"] or not hst["commit"]):
                raise SystemExit(f"REFUSED: {a.hinge_eval_dir} is not committed clean (write-once eval results, A10.2)")
            citations["hinge_eval"] = {"dir": os.path.relpath(os.path.abspath(a.hinge_eval_dir), REPO)
                                       if hst["in_repo"] else os.path.abspath(a.hinge_eval_dir), "commit": hst["commit"]}
        if a.extra_arm:                            # the prior table (write-once at --prior-commit) and the arm's reads
            pname, pord = CLOSING[a.extra_arm]["prior"][a.task]
            citations["prior_table"] = _cite_table(sc, a.prior_table or os.path.join(
                EXP, "results", "screens", a.task, f"{pname}.json"), a.prior_commit, f"the {pord} table")
            xst = git_path_state(a.extra_eval_dir)
            if xst["in_repo"] and (xst["dirty"] or not xst["commit"]):
                raise SystemExit(f"REFUSED: {a.extra_eval_dir} is not committed clean (write-once eval results, A10.2)")
            citations["extra_eval"] = {"dir": os.path.relpath(os.path.abspath(a.extra_eval_dir), REPO)
                                       if xst["in_repo"] else os.path.abspath(a.extra_eval_dir), "commit": xst["commit"]}
    table = sc.build()
    add_training(table, {}, sc.spec)
    if citations is not None:
        citations["eval"].update(job=table["eval"]["job"]["job"], attempt=table["eval"]["job"]["attempt"],
                                 registered_read=table["eval"]["job"]["registered_read"])
        if sc.fourth:
            jh = table["eval_hinge"]["job"]
            citations["hinge_eval"].update(job=jh["job"], attempt=jh["attempt"], registered_read=jh["registered_read"])
        if sc.extra:
            jx = table["eval_extra"]["job"]
            citations["extra_eval"].update(job=jx["job"], attempt=jx["attempt"], registered_read=jx["registered_read"])
        table["citations"] = citations
        table["vs_first_table"] = compare_first_table(
            citations["first_table"]["table"], table, new_runs=(("plain_depth_355m",) if sc.third else ())
            + (tuple(sc.hinge["keys"]) if sc.fourth else ()) + tuple(sc.x_keys))
        citations["first_table"].pop("table")
    table["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if sc.extra:                                   # the reissue: every other cell is the prior table's
        tp = citations["prior_table"].pop("table")
        for c in ("second_table", "third_table"):
            if c in citations:
                citations[c].pop("table", None)
        for k in ("vs_second_table", "diff_vs_second_table", "vs_third_table", "diff_vs_third_table"):
            if k in tp:                            # the prior table's comparisons, carried: every cell they describe
                table[k] = tp[k]                   # is unchanged (below)
        table["vs_prior_table"] = compare_outcomes(tp, table)
        x_att = sorted({e["attempt"] for e in table["receipts"]["env_reference"] if e.get("job") == sc.x_job}
                       | {x for r in table["runs"] if r["key"] in sc.x_keys for x in r.get("attempts") or []})
        table["diff_vs_prior_table"] = compare_prior_table(
            tp, json.loads(json.dumps(table, default=float)), x_keys=sc.x_keys, x_runs=tuple(sc.x_runs),
            x_attempts=tuple(x_att), x_entries=tuple(e["name"] for e in table["eval_extra"]["entries"]))
    elif sc.fourth:                                # the fourth table: every other cell is the third table's
        t2 = citations["second_table"].pop("table")
        t3 = citations["third_table"].pop("table")
        table["vs_second_table"] = t3.get("vs_second_table")       # the third table's comparison, carried: every cell
        table["diff_vs_second_table"] = t3.get("diff_vs_second_table")   # it describes is unchanged (below)
        table["vs_third_table"] = compare_outcomes(t3, table)
        keys = tuple(sc.hinge["keys"])
        h_att = sorted({e["attempt"] for e in table["receipts"]["env_reference"] if e.get("job") == sc.hinge["job"]}
                       | {x for r in table["runs"] if r["key"] in keys for x in r.get("attempts") or []})
        table["diff_vs_third_table"] = compare_third_table(
            t3, json.loads(json.dumps(table, default=float)), hinge_keys=keys,
            hinge_runs=tuple(r for r, k in sc.h_runs.items()), hinge_attempts=tuple(h_att))
    elif sc.third:                                 # the third table: every other cell is the second table's
        t2 = citations["second_table"].pop("table")
        table["vs_second_table"] = compare_outcomes(t2, table)
        pd_att = [x for x in [(sc.pd_probe or {}).get("run_attempt")] if x]      # the control's receipt joins
        table["diff_vs_second_table"] = compare_second_table(
            t2, json.loads(json.dumps(table, default=float)), pd_runs=tuple(sc.spec["plain_depth_all"]),
            pd_attempts=tuple(pd_att))
    md = to_markdown(table)
    import platform
    versions = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    prov = {"task": a.task, "table": name, "generated_utc": table["generated_utc"], "argv": list(argv or sys.argv[1:]),
            "records_root": os.path.abspath(a.records), "source_uri": a.source_uri, "receipt_globs": a.receipts,
            "git": _git_head(), "versions": versions, "inputs": sc.inp.listing(),
            "outputs": {"json": os.path.basename(paths["json"]), "md": os.path.basename(paths["md"])}}
    if citations is not None:
        prov["eval_dir"] = os.path.abspath(a.eval_dir)
        prov["citations"] = citations
    if a.plain_depth_dir:
        prov["plain_depth_dir"] = os.path.abspath(a.plain_depth_dir)
    if a.hinge_eval_dir:
        prov.update(hinge_eval_dir=os.path.abspath(a.hinge_eval_dir), hinge_records_root=os.path.abspath(a.hinge_records),
                    hinge_source_uri=a.hinge_source_uri, registered_read=a.registered_read)
    if a.extra_arm:
        prov.update(extra_arm=a.extra_arm, extra_eval_dir=os.path.abspath(a.extra_eval_dir),
                    extra_records_root=os.path.abspath(a.extra_records), extra_source_uri=a.extra_source_uri,
                    extra_registered_read=a.extra_registered_read)
    js = json.dumps(table, indent=1, default=float)
    pj = json.dumps(prov, indent=1)
    check_words(md, js, pj)
    os.makedirs(out, exist_ok=True)
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
