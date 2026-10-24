# Silent reasoning: deviations, incidents and operational decisions

Relative to `governance/registration.md` (sections 0-14 and any amendments below its line). Times UTC. Every entry
names the registration section it touches, what had been seen when it was made, and whether any verdict rule, scoring
file or registered comparison changed (if yes, it is also an amendment below the registration's line, dated, and the
original verdict is still printed). An entry is written on the day it happens, by the operator, and the critic reads it at
the next log row.

**Standing rule: speculative runs never enter a verdict.** A run that was not called for by the registered run order
(section 9), the registered seeds (6.3) or a registered adaptive rule (4.4, 7.4, 7.6) is printed as "speculative,
unused" and enters no pick, verdict, sigma or oracle cell, even if it scores better, unless the registered rule would
have run it anyway.

## Protocol deviations

- **A3.6 resume point for `cot` / `no_cot` (2026-10-19 02:40 UTC, lead; seen: the three resumed runs below, no
  accuracy read of any run).** A3.6 names `ckpt_last.pt` as the resume point. `cot` and `no_cot` run with the
  reference's `save_only_improve` (registration 3.2), under which `train.py` writes no `ckpt_last.pt`; their resume
  point is the last improving epoch's `ckpt_best.pt`, and `optimizer_last.pth` plus `ckpt_state.json` are written at
  that same epoch (D17: weights and AdamW state from one epoch). Epochs after the best are re-run from that boundary
  with their (seed, epoch) data order; the trackers continue from `ckpt_state.json`; both rows of a re-run epoch stay
  in `log.jsonl` (`resumes`, `resumed_from`). The critic's smoke row (2026-10-19T01:09Z) already recorded a `cot`
  resume "from ckpt_best.pt". Latent arms (per-epoch optimizer reset, `ckpt_last.pt` every epoch) resume exactly.
  No verdict rule, scoring file or registered comparison changed; a critic entry confirming the reading is requested.

(Format: **<section> (<date time>, <who directed it>, <what had been seen>).** What changed, why, and the
statement "no verdict rule, scoring file or registered comparison changed" or the pointer to the amendment.)

## Operational incidents

- **2026-10-19 01:58:27 UTC: spot preemption, AcctB `silent-train-20261019-012616-b84e` (us-central1-a).** Run
  prosqa_cot_s0, epoch 5 synced (best epoch 4). Relaunch `020057-c989` preempted at 02:04:19 UTC after three minutes;
  `020701-70b1` never existed (us-central1-b has no a2-ultragpu-1g); `020831-f917` running in us-central1-c since
  02:08:50 UTC. Resumed after epoch 4 from `ckpt_best.pt`; epoch 5 re-run (val_acc 0.6933 vs 0.6967, train_loss
  relative difference 7e-6: GPU nondeterminism). The partial epoch is redone from the boundary; the first epoch-5 row
  is kept in `log.jsonl` and enters no number.
- **2026-10-19 about 02:10 UTC: spot preemptions, AcctC `silent-train-20261019-013859-4882` and AcctA `silent-
  train-20261019-013940-64e2` (both us-central1-a).** AcctC: prosqa_coconut_s1 after epoch 6, exact resume from
  `ckpt_last.pt` on `021142-08be` (us-central1-c, 02:12:09 UTC); AcctA: prosqa_cot_lr3e-5_s0 after epoch 6 (best epoch 3),
  resumed after epoch 3 from `ckpt_best.pt` on `021320-dca5` (us-central1-c, 02:13:48 UTC); epochs 4 to 6 re-run (about 10
  min); the first rows 4 to 6 are kept and enter no number; Both resumed runs are the same runs under A3.6 (one resume each,
  not fragmented).
- **2026-10-19 about 03:49 UTC: spot preemption, AcctA `silent-train-20261019-021320-dca5` (us-central1-c).** Run
  prosqa_cot_lr3e-5_s0 at epoch 26 of 50 (best 0.79). The operator's container had been reclaimed between wakes, so the
  supervisor was dead; Receipt status at the cut: running. The resume point is the last improving epoch's `ckpt_best.pt`
  (A8.1); the superseded rows stay in `log.jsonl` and enter no number (A8.2). Second resume of this run (not fragmented).
- **2026-10-19 about 03:50 UTC: spot preemption, AcctC `silent-train-20261019-021142-08be` (us-central1-c).** Run
  prosqa_coconut_s1 at epoch 26 of 50 (stage 5). Latent arm: exact resume after epoch 26 from `ckpt_last.pt`
  (`resumed_from` row on 93eb); second resume of this run, not fragmented. Receipt status at the cut: running. AcctA's run
  of the previous line resumed after epoch 16 (its last improving epoch, best 0.79 on valid); the original rows 17 to 26
  are superseded by resume (A8.2) and enter no number; about 35 min re-run.
- **2026-10-19 05:00 to 05:55 UTC: spot-capacity storm in us-central1 (seven preemptions, two stockouts), all handled by the supervisor; The run has
  5 resumes (02:20Z, 04:01Z, 05:10Z, 05:21Z, 05:40Z), so it is **fragmented** under A3.6 (reported, not void); its superseded rows (17 to 26 of the
  0749 attempt, 17 to 18 of later ones) enter no number (A8.2). coconut s0 resumed exactly after epoch 5 from `ckpt_last.pt` at 05:37Z, two resumes,
  not fragmented). [Corrected 2026-10-19 12:15 UTC from the run's records at the critic's 12:07Z catch: the first text said "epoch 9", the epoch
  reached on 2e2c at 05:53Z, not the epoch at the loss.] AcctC `035551-93eb` (-a) and main `014019-f8e7` (-a, 40 GB) untouched. Action: the
  supervisor takes a per-job zone order from the environment (`SILENT_SUPERVISE_ZONES`); AcctA's and AcctB's jobs go east-first until the storm
  passes.
- Relaunched by the supervisor as `091728-b58c` at 09:19:37 UTC in **us-east4-c** (east-first zone order, factor 1.2, code
  rev60). pause_s0 resumes from its last improving epoch's `ckpt_best.pt` (A8.1). Third preemption of this job today.
- the ProsQA-hard screen's relaunch counter is spent (11 real attempts, MAX_RELAUNCH 10, A8.4 "a human looks").** The
  supervisor holds the job; the operator relaunches it by hand under the same 8-hour short admission when the window admits
  it (after a close at actual), central zones unless the author directs east; recorded as a deliberate human relaunch past
  the counter, cause spot churn, no job defect.
- **2026-10-20 06:50 UTC: spot preemption, AcctB `silent-train-20261020-041910-11f9` (us-central1-a; the job retries in the
  central zones after the supervisor's backoff.
- **2026-10-20 about 03:17 UTC: spot preemption, AcctC `silent-train-20261020-025152-49e0` (us-central1-c; the
  GSM-50k latent screen's relaunch, 25 min after creation; `c1b_k4` resumed twice on it).** Relaunched by the supervisor
  as `031905-622a` at 03:19:40Z (us-central1-c). Twelve spot preemptions across AcctB and AcctC since 2026-10-19 15:46Z.
- **2026-10-20 02:55 UTC: the operator's container was restarted** (the harness's doing); the supervisor process died and was
  restarted at 02:56Z; the working tree, the scratchpad records and the credentials survived; no VM, run, read or
  ledger record affected; cost none.
- **2026-10-20 02:50 UTC: spot preemption, AcctC `silent-train-20261019-204820-d318` (us-central1-a; the GSM-50k latent screen
  after 22,032 s: `pause`, `coconut` and `c1b_k8` done, `c1b_k4` at epoch 13; the remaining runs resume from their bucket
  checkpoints (latent arms exactly from `ckpt_last.pt`, A3.6).
- **2026-10-19 about 21:32 UTC: spot preemption, AcctB `silent-train-20261019-195159-1790` (us-central1-c; the
  ProsQA-hard `cot` lr1e-4 trial at epoch 13; the lr3e-5 trial had finished on it).** Relaunched by the supervisor as
  `213321-0ade` (21:34:09Z, us-central1-a), the seventh attempt of the job (counter 7 of 10).
- **2026-10-19 20:37 UTC: spot preemption, AcctC `silent-train-20261019-203304-7fd9` (us-central1-a; the GSM-50k latent
  screen's first VM, 546 s after creation, no run started;
- **2026-10-19 19:58 UTC: spot preemption, main `silent-eval-20261019-195307-7f7d` (us-central1-a; det1's registered test
  read, A13.3 (f)) during its boot; no entry ran, no summary pushed (write-once untouched); relaunched by hand as
  `202638-6c28` (us-central1-c, 20:26:38Z) under the same registered read; the read is spent once, at the attempt that
  writes the summary.
- **2026-10-19 19:51 UTC: spot preemption, AcctB `silent-train-20261019-191352-4285` (us-central1-a; the ProsQA-hard `cot`
  lr3e-5 trial at epoch 22 after two resumes; The east zones are requested from the author for this job.
- **2026-10-19 18:53 to 19:14 UTC: two more spot preemptions, AcctB `silent-train-20261019-185217-196b` and
  `185619-a829` (us-central1-a), each within minutes of creation; relaunched by the supervisor as `191352-4285`
  (19:14:47Z). Five attempts of the ProsQA-hard screen since 17:29Z;
- the `cot` trial resumes from its last improving epoch's `ckpt_best.pt`, A8.1). Central zones only for this job (the critic's
  15:35Z ruling on the east figure). Seven spot preemptions in us-central1 today across AcctB and AcctC;
- **2026-10-19 17:05:37 and 17:09:18 UTC: spot preemptions, AcctC `silent-train-20261019-160323-4299` (us-central1-c,
  GSM-50k bar, first `cot` trial about epoch 20 of 25) and its relaunch `170634-13b2` (us-central1-a, two minutes after
  creation).** The supervisor's creation grace period held the gone 13b2 as "creating" for 20 minutes (fixed in
  `rev370`: the grace is creation-only); relaunched as `172741-7fff` at 17:28:22Z (us-central1-a). The bar runs resume
  from their bucket checkpoints (A3.6; `cot` from the last improving epoch's `ckpt_best.pt`, A8.1).
- **2026-10-19 17:27 UTC: a dirty-tarball hand launch, AcctB `silent-train-20261019-172701-3acf`** (`screen-prosqa-hard-
  acctb`), launched from the working tree with `--allow-dirty` while an agent's uncommitted launcher and
  `common/latent/` edits were in it (tarball `rev371-dirty`). Deleted at 17:28Z during its startup install; no run
  started; Relaunched from a clean detached worktree at `rev237` as `172918-8046`. Rule: hand launches run from a
  detached worktree at the cited commit.
- **2026-10-19 16:47 UTC: `prep-pin-medium-main` (`silent-prep-20261019-164733-2c65`, main) reported "failed"** because
  the operator's spec omitted `smoke_config` and the launcher's default 124M smoke ran; it died on a missing
  `data/prosqa_train.json` (a pin-only prep fetches no data) and `calibrate_real` on the unreachable hub, as the critic
  read from the logs (17:40Z row; the first text here said "at the epoch-1 boundary", the launcher's wrapper verdict). The
  pin step itself succeeded (CHANGELOG 17:35Z, 17:55Z) and is the job's purpose. No run or number affected.
- **2026-10-19 15:57 UTC: launcher defect, AcctC `silent-train-20261019-154904-c083` (`screen-gsm50k-bar-acctc`) aborted before
  any run** ("cannot tell the task of train_path data/gsm50k_train.json": the launcher's data step did not know the screen
  datasets of A5 item 1, and the files existed only in the operator's reclaimed container). Fixed in commit `rev155` (screen
  tasks from the manifest's `screen` block; the bucket copy verified, never generated on a VM); the data regenerated with hashes
  equal to the manifest and uploaded to the three budget buckets; relaunched by hand as `silent-train-20261019-160323-4299` (us-
  central1-c, 16:03:23Z), whose data step verified the hashes. No run, read or number affected; Found in the same code path and
  fixed in the same commit: the launcher overrode every run's `model_id` with the 124M pin, so the 355M plain-depth control
  would have trained the 124M model; the launcher now pins each run's own model and refuses an unpinned one; the plain-depth
  runs wait for a gpt2-medium pin row (CHANGELOG 16:15Z). No plain-depth run had started.
- the earlier eval jobs ran on main, whose sprint service account reads every bucket (A5 plumbing; lead's oversight when
  naming the fallback project).** No read was spent (no entry ran; A10.1 counts a read at its summary). Resolution without
  any permission or billing change: the four checkpoints were downloaded by the operator's session (sha256 equal to the
  pins) and uploaded byte for byte into AcctB's bucket under `silent/mirror/<source bucket>/runs/<run>/ckpt_best_final.pt`;
  `make_jobs --checkpoint-mirror` rewrites each entry's `checkpoint` URI to the mirror, keeps the source URI as
  `checkpoint_source` and leaves the sha256 pin (verified by the VM on every read) and every other field unchanged, so the
  record is the same read of the same bytes; Registration sections touched: none (A10.2's pin is the sha256, not the URI);
  the critic reads this entry at the K2 row.
- **2026-10-20 21:10 UTC: A6.2 rule 3's like-for-like check at batch 1 on ProsQA-hard differs on 2 of 300 (idx 21, 268).**
  The ProsQA-hard coconut checkpoint `rev241` read through the band path with `latent_band=[0,12]` against its HF-path
  `none` read (220 of 300 at batch 32): at batch 32 the band read differs on **0 of 300**; at batch 1 it differs on **2 of
  300** (idx 21 and 268), and the critic's 21:03Z row 3 recomputed that the same two examples differ inside the band path
  across the two shapes, so this is the A14.1 padded-shape effect under bf16 (as on GSM-50k, deviations 12:05Z, closed
  by the 14:54Z row), not a path difference. The batch-1 timing stays of record (rule 2); the batch-32 read is the
  accuracy of record. Rule 3's function-equality check on this checkpoint (the float64 CPU record) follows; until it is
  logged the ProsQA-hard f_meas lines carry "rule 3 logits check pending on this task's denominator".
- **2026-10-20 19:25 UTC: a resume refusal on AcctB `silent-train-20261020-144340-865b` (record item from the critic's
  19:14Z row).** `prosqa_hard_c2_s0` had started on `a26f` (preempted 13:54Z) without finishing an epoch; on `865b` the
  launcher refused that partial snapshot ("config.json does not match the snapshot sha256", receipt `resume.ok false`,
  `exact true`), archived it under `runs_partial/`, and started the run from scratch (zero resumes on `865b`, 30 epochs in
  6,408 s; `prosqa_hard_c2.json` unchanged between the two attempts' commits `rev192` and `rev372`). No finished epoch
  was lost and no number is affected; the field that differed in the snapshot's effective config was not identified (the
  archived snapshot was not opened). The refusal is the launcher doing what A8.2 asks: a snapshot whose config does not
  hash-match is never resumed. The entry stands as the record of the restart.
- **2026-10-20 14:45 UTC: spot preemption, AcctB `silent-train-20261020-130426-a26f` (us-central1-a) at 13:54:37Z**, the
  ProsQA-hard screen's fourteenth real attempt (hand-launched 13:04:26Z; `prosqa_hard_c1b_k4_s0` finished its 30 epochs
  on it in 2,302 s; `prosqa_hard_c2_s0` had started without a logged epoch). The supervisor does not relaunch this job
  (counter spent); the operator's loop relaunches by hand under the same 8-hour admission terms and the fourteen rows, each
  attempt logged in the ledger; this entry covers the loop's relaunches of 14:45Z onwards as deliberate human relaunches
  (A8.4 "a human looks": cause spot churn, nothing in the job). Remaining: `c2` (30 epochs), `c2_shuffled` (30 epochs),
  about 4.5 h.
- **2026-10-20 12:45 UTC: deliberate human relaunch of `screen-prosqa-hard-acctb` past the supervisor's counter (A8.4 "a human looks").** The
  job's eleventh real attempt (`1e3c`, spot-preempted 07:02Z) spent MAX_RELAUNCH (CHANGELOG 07:55Z); the cause across the eleven is spot
  churn in us-central1 for A100-80GB (nine preemptions, two capacity refusals), not a job defect: every attempt that ran resumed exactly and
  seven of the ten arms are done on valid. Remaining arms: `c1b_k4` from its epoch-17 checkpoint, `c2`, `c2_shuffled` (about 5 h). The
  supervisor will not relaunch this job again (counter spent): the operator's loop relaunches a boot preemption, and a later preemption needs
  another hand relaunch, each logged here.
- **2026-10-20 12:05 UTC: A6.2 rule 3's like-for-like check at batch 1 differs on 2 of 500 (lead's explanation; the critic
  rules).** The coconut checkpoint `rev180` read through the band path with `latent_band=[0,12]` (the f_meas denominator)
  against its HF-path `none` read (55 of 500 correct, batch 32): at batch 32 the band read is 55 of 500 with **0** examples
  differing; at batch 1 it is 53 of 500 with **2** differing (idx 273 and 497), over rule 3's "more than 1 in 500" line. The
  explanation: the two reads at the same batch size agree on every example, so the band path is like-for-like with the HF
  path; the two flips appear only when the batch size changes, which is the padded-shape kernel effect A14.1 describes
  (B = 1 against B = batch changes the padded shape; under bf16 autocast over TF32 near-tie generations can flip; the
  companion-independence test is the same-shape swap, 0 of 300). The batch-1 timing is still of record (rule 2) and reported;
  f_meas is unaffected (a timing, not an accuracy). Rule 3's second check (one batch at strict fp32 with TF32 off, HF path vs
  band [0, 12] first-token logits at 1e-4) has not run and would settle it; until it is logged every f_meas line carries
  "rule 3 logits check pending" and no saving label is written. Table: `results/screens/gsm50k/gsm50k_screen_with_eval.md`
  (`rev11`).
- launcher now refuses).** `silent-eval-20261020-105717-03e7` (10:57:17Z) and `silent-eval-20261020-111541-bf75` (11:15:41Z), the
  screen eval job, were launched by hand with an environment that lacked `SILENT_SERVICE_ACCOUNT`, so each VM booted as main's
  default compute account, could not reach the bucket (no startup.log, no job download), and its startup script's exit trap shut it
  down about 70 s after creation (GCE `guestTerminate`; not preemptions). Every earlier main launch and the supervisor set the
  account (`MAIN_SERVICE_ACCOUNT`); the omission is the operator's. its startup log uploaded at 11:23:33Z. Fix: `gcp.py` refuses a
  launch into main without the account before any ledger row (test added; launcher suite 123).
- **2026-10-20 11:05 UTC: the plain-depth attempt `silent-train-20261020-104337-3293` ran without the batch-16 probe (a
  launcher defect, fixed).** The launcher treats an existing `prep/<job>/memory_probe.json` as "already measured by an earlier
  attempt" and skipped the probe; the file there was attempt `e910`'s failed 32 x 1024 probe (uploaded regardless of its
  result). So the A13.2 (b) fallback's peak at micro-batch 16 x 1024 is not in `3293`'s receipt; the OOM at 32 x 1024 is (e910's
  receipt; deviations 09:58Z). The run trains at 16 x 8 as allowed (epochs 1 and 2 done, bf16 autocast, TF32 on). Fix: a failed
  probe is no longer uploaded under prep/, and a prep file with an error row never counts as measured (`jobs.py`
  `probe_measured`; test extended; launcher suite 123). The 16 x 1024 peak will be in the receipt of any later attempt of this
  job (the fix is on HEAD; a relaunch tars HEAD); if none is needed, the control's record carries "probe at 16: not measured;
  OOM at 32 measured (e910)" on every row that uses it.
- **2026-10-20 09:58 UTC: plain-depth 355M control at micro-batch 16 x grad_accum 8 (A13.2 (b); a registered deviation of 3.2's micro-batch
  clause, logged before the run).** The launcher's memory probe for `screen-plain-depth-355m-acctc` (attempt `silent-
  train-20261020-094517-e910`, A100-SXM4-80GB, gpt2-medium at the pinned revision `rev125`, weights `rev121`; the probe shape `--shape
  gpt2-medium`, batch 32, seq 1024, 0 latents) ran out of memory in all three precisions (fp32: 79.16 GiB in use when a 512 MiB allocation
  failed; fp16 and bf16: 76.7 GiB in use when a 6.13 GiB allocation failed, of 79.25 GiB); the parity line passed (cuda fp32 vs cpu float64 max
  abs logit diff 1.4e-7). Receipt status `memory_probe_failed`; Applied as A13.2 allows: both configs
  (`configs/screen/gsm50k_plain_depth_355m.json`, `configs/screen/prosqa_hard_plain_depth_355m.json`) go from `batch_size` 32 / `grad_accum` 4
  to **16 / 8** (effective 128 unchanged; nothing else of the recipe changes), the job's probe runs at batch 16 so the peak lands in the
  receipt, and every row and K3 line that uses the control prints "micro-batch 16 x 8". Relaunched by hand from a clean worktree at the commit
  below (the supervisor does not relaunch a job after a refused precondition).
- **2026-10-19 15:46:02 UTC: spot preemption, AcctC `silent-train-20261019-153923-d3bb` (us-central1-a) during boot**
  (`screen-gsm50k-bar-acctc`; the startup script was installing; no run started, nothing to resume). Relaunched by the
  supervisor at 15:49:44Z as `silent-train-20261019-154904-c083` (us-central1-c; the first try found no capacity in -c
  and -a).
- the supervisor's pass fell between the row and the creation, found no VM, closed the admission at the full reserved worst case
  (seq 60, "no termination evidence") and attempted a relaunch, which the launcher refused at the quota check (no second VM, no
  second ledger row). The VM was created at 15:39:40Z and runs; its runs are untouched. Fix: a 20-minute creation grace period in
  the supervisor (`GRACE_S`, commit `rev101`, test), supervisor restarted 15:44Z. Record: seq 60 stands (rows are never edited)
  and is reconciled to the VM's actual runtime cost at its end; until then the committed figure equals an open admission at its
  worst case. No run, read or number affected;
- **2026-10-19 between 02:14 and 02:25 UTC: the operator's session was deleted by accident**, taking the scratchpad
  supervisor process (preemption relaunches) with it. No VM, run, checkpoint or ledger record was affected; the four
  VMs ran on and synced every epoch. Recovered at 02:27 UTC (CHANGELOG 02:40 UTC); the supervisor is rebuilt as repo
  code. Cost: none beyond the VMs' normal running.

(Format: **<date time>:** what happened (preemption, crash, Mac slot contention, restart from scratch,
misqueued stage), the run affected, its partial output's fate ("kept, enters no number"), and the cost.)

## Speculative runs

None yet. (Format: the run, why it started early, its status under the standing rule: "speculative, unused" or "became
the registered run because the rule called for it".)

## Analysis additions

None yet. (Reporting only; nothing here changes a verdict. Format: what is computed, by which code at which commit,
and the statement that no registered number moved.)

## Known biases

None yet. (From audits and from the registration's own arithmetic: direction, size and which comparisons it touches.
Already known before any run, from the registration: selection of the tuned bar on seed 0 inflates the bar slightly
(against the latent arms); a 2 sigma K1 tolerance has a false-miss rate of roughly 8% per cell under a perfect
reproduction; ProntoQA's regenerated data is not byte-identical to the paper's; the head-cost confound of section 5.4
favours latent steps at small sizes.)

- **Selection-split overlap, GSM8k-Aug (critic, 2026-10-17 21:29 UTC; registration A2 item 4).** 7 of the 500 valid items share a
  word 13-gram with a train record's question + steps (1.4%; 2.2% with steps and answer on the valid side), against
  0.53% on test. Valid is the selection split, so checkpoint and LR choices see a slightly optimistic valid accuracy;
  the bias is the same for every arm and for every trial of the tuned bar, so no comparison tilts. No exact duplicate
  questions. The 7 items are listed by idx in the report. No action beyond reporting.
