# Silent reasoning: standardized report

**Status: FINAL** (2026-10-22). Closed on the 72-hour scope (PLAN.md A1): clean negative; the single end-of-round test read (A3.4) is done
(`results/screens/test_read.md`, `rev1`; reporting only, no decision attached) · **Dates:** 2026-10-17 (plan approved) to
2026-10-21; the closing plan (A2.2, A2.2.1) ran 2026-10-21 20:21Z to 2026-10-22 09:57Z, its results in the dated section below · **Compute:** window total within a fixed compute cap (`governance/run_log.jsonl` at the close; nothing open; the cap of A2 raised by A2.1 for one hinge job).

## Question
Does a GPT-2 124M model that reasons in continuous latent steps (Coconut, and three registered cards built on it) match or beat a
tuned written-CoT bar on verifiable tasks with at least 5x fewer emitted tokens at equal total FLOPs, with latents that carry state
under intervention (K2) and an advantage that survives a bigger plain model at equal FLOPs (K3)? Prior before any run (PLAN.md
section 0): P(full goal) about 0.10, P(tie with tuned CoT at equal FLOPs with 3-5x fewer emitted tokens) about 0.30, P(clean
negative) about 0.60.

## Method
Binding text: `governance/registration.md` (sections 0-14; amendments A1-A15) under the plan `PLAN.md` (A1-A3, A2.1). Scope of
the window (PLAN.md A1; registration A3.3): Phase 1 oracle cut to ProsQA (`cot`, `coconut` at seeds 0-2; `no_cot`, `pause` at seed 0)
and GSM8k-Aug (`cot`, then `coconut` from it, seed 0), kill rule K1 against the published cells (4.1, A7.2); Phase 2 screens at 124M,
seed 0, bf16 autocast over TF32 (A10.5), on ProsQA-hard (depth 5-7, 1.7x edges; 10,000 / 300 / 500) and GSM-50k (the first 50,000 of
a seed-0 permutation of GSM8k-Aug train; valid 500, test 1,319) against a 3-LR tuned `cot` bar, `no_cot`, full-depth `coconut`,
pause-same-curriculum, scrambled-at-inference, zero-latent, the step sweep {0, 1, 2, 4, 8} and plain depth = GPT-2 355M `no_cot`
(A3.4, A13.2). Cards (A3.4): C1b (latent steps through the middle k of 12 layers, k in {8, 4}; null = full-depth `coconut`; the
wall-clock fraction f_meas of A6.2), C2 (verifiable-state supervision by an auxiliary decoder; null = C2-shuffled), C3 (a
latent-dependence hinge; null = C3-foreign; K2 "not independent" under A6.1, with the donor-answer directional read of rule 4 (B)),
and C2+C3; the C3 arms ran on GSM-50k only (A2.1). Screen decisions use valid only; sigma_screen = 3.60 on both tasks (A3.4 as
written; critic entry 2026-10-20T10:49Z); a screen filters and is never a verdict (6.3); K2 at the screen fires iff Delta_scr < 3.60 or
its 10,000-draw paired example-bootstrap interval includes 0 (7.5; 12:40Z (b)); K3 fires iff acc < A_pd(F) - 3.60 with A_pd linear in
ln F between the 124M and 355M `no_cot` points (7.7). Costs charge every forward pass (5.1, 5.3); passes and emitted tokens are
reported beside FLOPs (7.2). Every table is written once by `code/read_oracle.py` or `code/read_screen.py` from hashed per-example
files and reissued, never edited (A6.4); the critic recomputed every verdict from the files. UNTESTED, not estimated (A1 item 4):
ProntoQA, `coconut_no_curriculum`, C4-C6, Phase 3, Phase 4, the 2 B fade run, the readout positive control, every confirmation seed,
the C3 arms on ProsQA-hard (A2.1).

## Results
Every number is cited from its table of record with its commit and carries the labels the critic's rows attach; the forms are the
critic's (rows 2026-10-19T16:59Z, 2026-10-20T14:08Z, 12:40Z, 16:31Z, 21:03Z, 23:37Z, 2026-10-21T14:57Z). The Phase 2 sentences are
the tables' own lines and entered this file after (iv), the single test read (critic entries 2026-10-20T12:40Z row 3, 23:37Z,
2026-10-21T14:57Z (5), 16:33Z; lead row 16:40Z). Outcome: a clean negative on the 72-hour scope. No card arm advances past either screen; K2 fired or counts as fired on every card
arm but C1b k = 8; K3 fires on nothing; no WIN, tie or pass sentence was reached.

### Phase 1: K1 (the oracle)
Tables of record: `results/k1/k1_table_final.md` (numbers write-once at `rev2`; labels `rev3`, `rev4`, `rev5`) and
`results/k1/k1_table_window.md` (`rev6`; label-only edit `rev7`). Verdict sentences, verbatim:
- "K1 ProsQA: pass (cot, no_cot, coconut, pause) [cot and coconut three seeds; no_cot and pause single seed; audit: cot s1 3 of 500
  differ at batch 1, explained: TF32 batch-shape rounding flips near-ties (top-2 margins 1e-3 to 1e-2) along the greedy chain; no
  cross-row dependence (same-shape swap 0 of 300 at both numerics); strict fp32 0 of 300 divergences (diag `rev8`; A14.2)]".
- "K1 on GSM8k-Aug, 1 seed: pass (cot 40.94, coconut 34.65) against 42.9 / 34.1, tolerances 2.0 / 3.0 [single seed; no_cot UNTESTED;
  KNOWN (reproduction); fp32-tf32 (footnote: the reference's probable strict fp32, A4; d_prec 0.0 on ProsQA does not transfer to GSM
  and is not claimed for it); cot trained and read on A100-40GB; coconut: cot stage on A100-40GB, coconut stage on A100-80GB, read on
  A100-40GB (A3.2 (i) to (iii) on the eval VM); images v20261016 / v20261019; zero resumes; audit: 0 of 1,319 differ at batch 1 on
  both checkpoints (7.10 item 2, A10.3; cot: one example's generation length differs, answer unchanged); cot on the edge: 540 of
  1,319, abs(diff) 1.96 of 2.0, 539 would have missed; secondary oracle flagged on cot: paper-style count 30.6 vs 25.0, checked: the
  reference test chains count 30.1 under 2.2's definition, the paper's convention unstated (critic entry 2026-10-20T14:08Z)]".
- "K1 for the window (A3.3) = K1 on ProsQA: pass; GSM8k-Aug: pass [1 seed]"; "K1 ProntoQA: UNTESTED (A3.3)"; "K1 (registered,
  ProsQA and ProntoQA): UNTESTED; ProntoQA UNTESTED (A3.3)"; "K1 GSM8k-Aug: UNTESTED (no_cot not read) [single seed]".
- Phase 1 matched the published `cot` and `coconut` cells on ProsQA (three seeds) and on GSM8k-Aug (one seed) and the published
  `no_cot` and `pause` cells on ProsQA (one seed) within the registered tolerances (4.1, A7.2); GSM8k-Aug `no_cot`, ProntoQA and
  `coconut_no_curriculum` are UNTESTED (A3.3; A1 item 1). KNOWN (reproduction).

| task | arm | seeds (test) | mean | SD | sigma_cell | tolerance | published | verdict | labels |
|---|---|---|---|---|---|---|---|---|---|
| ProsQA | cot | 75.0 / 71.2 / 78.4 | 74.87 | 3.60 | 2.88 | 5.76 | 77.50 | matched | three seeds; audit label above |
| ProsQA | no_cot | 78.0 | 78.00 | | 1.00 | 2.00 | 76.70 | matched | single seed |
| ProsQA | coconut | 96.8 / 96.2 / 95.8 | 96.27 | 0.50 | 1.00 | 2.00 | 97.00 | matched | three seeds |
| ProsQA | pause | 97.0 | 97.00 | | 1.00 | 2.00 | 96.60 | matched | single seed |
| GSM8k-Aug | cot | 40.9 (540 of 1,319) | 40.94 | | 1.00 | 2.00 | 42.90 | matched | single seed; on the edge; secondary oracle flagged |
| GSM8k-Aug | coconut | 34.6 (457 of 1,319) | 34.65 | | 1.50 | 3.00 | 34.10 | matched | single seed |

Every cell: KNOWN (reproduction); fp32-tf32; ProsQA read on A100-40GB of A100-80GB checkpoints; resumes per seed cot 1 / 0 / 0,
coconut 2 / 2 / 0, pause 1, no_cot 0, none fragmented. Precision pair (A4 item 2; critic entry 2026-10-19T19:55Z): d_prec = 0.0 (ProsQA
coconut seed 0, strict 96.8 against fp32-tf32 96.8; per-example agreement 476 of 500), fp32-equivalent. Determinism pair: det1 read once
(CHANGELOG 2026-10-19 20:55Z: 75.8 against the oracle seed-0 run's 75.0; A13.3); det2 waited for the author and did not run, so
sigma_task does not exist on either task (critic entry 2026-10-20T14:08Z) and no 6.2 verdict, 13.10 scrambled-drop sentence or WIN /
tie sentence exists anywhere. The released-ProsQA tuned-CoT bar read (3.4 as A3.3 narrows it) was not registered or read in the window
(critic entries 2026-10-19T16:59Z (6), 2026-10-20T14:08Z row 2 (vi)); its LR trials trained and carry no registered test number. Diagnostic on the three ProsQA Coconut checkpoints
(section 8; A12.4, A14.3; the table's sentence): at this read the three selected Coconut checkpoints' answers show no measurable
dependence on their latents: scrambled drop 0.0 points on every seed (0 of 500 test examples change on each; paired 95% CI [0.0,
0.0]), zero-latent drop -0.4 / -0.2 / 0.0, step sweeps flat within 0.4 points from n = 0 to n = 8; the scrambled pass fed every example
another example's latents (verified bit for bit on the valid split); the latents are read by the model (the next-token logits move on
300 of 300 valid examples) and decide no emitted token on any of them; KNOWN (reproduction), fp32-tf32, three seeds, diagnostic,
ProsQA only (GSM8k-Aug untested). On GSM8k-Aug (one seed, test, diagnostic, no verdict; critic entry 2026-10-20T14:08Z (7)): Coconut
own 34.65; scrambled 3.94 (paired drop 30.71 [28.13, 33.36]); zero-latent 6.52 (28.13 [25.47, 30.71]); the GSM control numbers are
printed, not interpreted (14:08Z row 2 (iii)).

### Phase 2, GSM-50k screen (124M, seed 0, valid 500; bar LR-tuned 3 trials; arm 1 trial; sigma_screen 3.60; no test read)
Table of record: `results/screens/gsm50k/gsm50k_screen_with_hinge.md` (the fourth table, `rev9`, citing `rev10`, `rev11`,
`rev12`; eval reads `rev13`, `rev14`). Every line: screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials;
arm 1 trial. Bar: `cot` lr 3e-4 25.6 (128 of 500) at epoch 17 (trials 21.0 / 23.8 / 25.6) [bar: single seed, LR-tuned (3 trials)];
`no_cot` 6.2 (31 of 500); plain depth 355M `no_cot` 8.0 (40 of 500), epoch 24 [plain depth 355M, 1 seed; micro-batch 16 x 8; probe at
16 not measured, OOM at 32 measured (e910)].

| arm | valid | vs bar | Delta_scr [interval] | K2 at the screen | zero-latent drop | pause comparison | passes; FLOP ratio to bar |
|---|---|---|---|---|---|---|---|
| pause | 9.6 | -16.0 | (no fed-back vector; no K2 read) | | | | 3.210; [0.671, 0.673] |
| coconut | 11.0 | -14.6 | +7.0 [+4.11, +9.89] | not fired | +10.0 [+7.24, +12.76] | +1.4: CURRICULUM-ONLY | 9.396; [0.673, 0.675]; billing claim only |
| C1b k = 8 | 9.4 | -16.2 | +4.0 [+1.70, +6.30] | not fired | +1.4 [-0.90, +3.70] | -0.2: CURRICULUM-ONLY | 9.280; [0.652, 0.654] nominal; billing claim only |
| C1b k = 4 | 9.4 | -16.2 | +1.4 [-0.12, +2.92] | fired | +1.2 [-0.03, +2.43] | -0.2: CURRICULUM-ONLY | 9.284; [0.632, 0.634] nominal; billing claim only |
| C2 | 12.0 | -13.6 | +8.6 [+5.70, +11.50] | not fired | +11.0 [+8.19, +13.81] | +2.4: CURRICULUM-ONLY | 9.310; [0.672, 0.674]; billing claim only |
| C2-shuffled | 11.6 | -14.0 | +8.6 [+5.74, +11.46] | not fired | +10.0 [+7.00, +13.00] | +2.0: CURRICULUM-ONLY | 9.230; [0.671, 0.673]; billing claim only |
| C3 | 10.4 | -15.2 | +10.4 [+7.71, +13.09] | not independent; counts as fired (rule 5) | +10.4 [+7.71, +13.09] (diagnostic) | +0.8: CURRICULUM-ONLY | 9.226; [0.671, 0.673]; billing claim only |
| C3-foreign | 10.0 | -15.6 | +2.0 [+0.00, +4.00] | not independent; fired on the drop leg | +2.6 [+0.49, +4.71] (diagnostic) | +0.4: CURRICULUM-ONLY | 9.216; [0.671, 0.673]; billing claim only |
| C2+C3 | 12.0 | -13.6 | +11.8 [+8.99, +14.61] | not independent; counts as fired (rule 5) | +12.0 [+9.16, +14.84] (diagnostic) | +2.4: CURRICULUM-ONLY | 9.268; [0.672, 0.673]; billing claim only |

- 6.3: no card arm advances, on the bar leg alone (9.4 / 9.4 / 12.0 / 10.4 / 12.0 against 25.6 - 3.60 = 22.00); the pause leg is not met
  either; FLOP and token legs met by every arm (E' 2.21 to 2.40 against 0.2 x 30.44 = 6.09); the passes criterion (<= 6.29) met by pause
  only, so every feedback arm carries "billing claim only" (7.2). 7.10: no too-good trigger, no null trigger. 7.6: no arm fading.
- K2 (critic entry 2026-10-20T12:40Z (f) form): "K2 fired at the screen (7.5; A3.4): the arm's latent claim is killed at the screen on
  GSM-50k; it continues only as a labelled curriculum or token-budget effect; CURRICULUM-ONLY (pause comparison -0.2)" on C1b k = 4;
  "not fired" on coconut, C1b k = 8, C2 and C2-shuffled. C2's scrambled drop (+8.6) does not separate it from its null (+8.6) or from
  coconut (+7.0) at this screen.
- K3 (critic entries 2026-10-20T15:53Z, 16:31Z): A_pd 6.29 to 6.39 from the 124M `no_cot` point (6.2 at [10.261, 10.269] GFLOPs) and the
  355M point (8.0 at [35.948, 35.972] GFLOPs); fires on no arm; pause, C1b k = 8 and k = 4 "no better than plain depth" (within 3.60 of
  A_pd); coconut, C2 and C2-shuffled 4.6 to 5.6 above A_pd, a screen number against a 1-seed control and no K3 verdict either way.
- Card lines (A3.4): C1b k = 8: -1.6 from full-depth coconut, within 3.60, K2 not fired: card pass line met on this task as a screen
  number (6.3 governs; does not advance). C1b k = 4: -1.6, K2 fired: pass not met. C2: +0.4 over C2-shuffled (the gap includes
  learnable vs unlearnable auxiliary target; A5 item 5: the last supervised target contains the answer in 0.770 of the GSM-50k records
  at 3 latent steps), +1.0 over coconut: pass not met; the kill's within-null clause met on this task.
- C1b per-step cost (A6.2): k = 8 "saving shows: f_meas 0.7200 <= 0.7667 at batch 1 (A6.2; A100-40GB, 1 seed, valid; rule 3: 0 of 500
  differ at batch 32; logits 6.4e-4 at strict fp32 on A100, <= 2.3e-13 at float64, explained; diag `rev15`, float64 record
  `rev16`)" (0.7396 at batch 32 beside); k = 4 "nominal: f_meas 0.4326 against the line 0.43" (0.4785 at batch 32). "Saving shows" is
  (critic entry 2026-10-20T15:26Z row 3) A6.2 rule 4's line met on the card's wall-clock number, one seed, valid, on an arm that does not
  advance; it is not a K3 outcome, not a 7.2 claim, not a statement about what any latent carries, not a test number.
- Every latent arm's all-epoch best on valid sits at stage 1 (epochs 5 or 6, n latents = 2: 11.0 to 14.6) above its final-regime
  selection: "earlier stage better on valid" (3.3 item 3), a flag, not a cell.

#### C3, C3-foreign, C2+C3 on GSM-50k (critic entries 2026-10-21T14:57Z, three; the fourth table stands as the table of record)
The critic's reading in one line (14:57Z row 1 (5)): "on GSM-50k at 124M, 1 seed, valid, C3 and C2+C3 sit within sigma_screen of
full-depth coconut (-0.6, +1.0) and 15.2 and 13.6 points under the bar; their scrambled drops (+10.4, +11.8) have no direction on the
donor-answer read (0 of 289 donor answers; 288 and 287 of 289 outputs changed), so K2 counts as fired and both are killed as latent
claims at the screen on this task; C3-foreign's drop (+2.0) fires K2 on the drop leg; the C3 minus C3-foreign drop difference (+8.4) is
"dependence on own latents (content or fingerprint)"; K3 fires on none; ProsQA-hard UNTESTED for both cards (A2.1)". Runs
`screen-gsm50k-accta`: "resumed 3 / 0 / 2 times, exact boundary resumes, same GPU type; start pin rev17... epoch 25;
trained and read on A100-80GB"; eval registered read 2026-10-21T11:07Z (after one failed attempt, `db49`). Every directional
number: "screen number: 1 seed, valid 500 (scored 289); bar LR-tuned 3 trials; arm 1 trial;
diagnostic entry, no 4.5 read spent; bf16 autocast over TF32 on (strict fp32 beside, diagnostic); read on A100-80GB, trained on
A100-80GB; no test read; ProsQA-hard UNTESTED (A2.1)".
- Selection: C3 10.4 (52 of 500, epoch 25), C3-foreign 10.0 (50 of 500, epoch 16), C2+C3 12.0 (60 of 500, epoch 21); vs full-depth
  coconut 11.0: -0.6 / -1.0 / +1.0; vs the bar: -15.2 / -15.6 / -13.6 (6.3's bar leg fails on all three; no advancement).
- K2 (A6.1 rules 1, 5 and 6), as the table prints it: C3 "K2 not independent (A6.1); counts as fired (rule 5): drop without direction
  (fingerprint-consistent)"; the drop +10.4 [+7.71, +13.09] does not fire 7.5's K2 and its directional read is absent: donor-answer
  rate 0 of 289 = 0.000 vs 4 of 289 = 0.014 under own latents, Delta_dir -0.014 [-0.027, -0.000]; garbage G_chg 288 of 289 = 0.997 vs
  R_dir 0.000: exceeded. C2+C3: the same words; drop +11.8 [+8.99, +14.61]; 0 of 289 vs 2 of 289 = 0.007, Delta_dir -0.007 [-0.016,
  +0.003]; G_chg 287 of 289 = 0.993: exceeded. C3-foreign: "K2 not independent; fired on the drop leg (7.5; A3.4)", Delta_scr +2.0
  [+0.00, +4.00] (the exact lower bound +0.0008; the drop leg fires on Delta_scr < sigma_screen alone); directional read absent (4 of
  289 vs 3 of 289, Delta_dir +0.003 [-0.003, +0.010]); G_chg 0.481: exceeded. Each arm's latent claim is killed at the screen on
  GSM-50k and cannot be A1's positive read on this task. What the read shows (14:57Z row 1 (1)): a checkpoint that detects foreign
  latents and does not read their content; "fingerprint-consistent" is a label, not a finding that the latents are a fingerprint: the
  read cannot tell a fingerprint from reasoning state the head declines to express under a mismatch, and no sentence beyond the label
  is licensed.
- Dependence on own latents (content or fingerprint) (A6.1 rule 2): C3 minus C3-foreign scrambled-drop difference +8.4 [+5.11, +11.69];
  C2+C3 minus C3-foreign +9.8 [+6.46, +13.14]. Zero-latent reads (diagnostics that decide nothing for these arms, rule 3): C3 0.0,
  C2+C3 0.0, C3-foreign 7.4. The coconut reference entry (no hinge; diagnostic; decides nothing): donor-answer rate 7 of 289 vs 2 of
  289, Delta_dir +0.017 [-0.003, +0.038], absent; G_chg 0.830.
- Card lines (A3.4): on C3 and C2+C3 every pass part is met on this task (drop above 10; accuracy within 3.60 of coconut; the drop
  exceeding C3-foreign's by more than 3.60) and the pass is not met because K2 counts as fired (rule 5): a pass line written on the
  scrambled drop cannot be met by a drop that has no direction, since that drop is the quantity the hinge trained against (14:57Z row 1
  (3)); it does not mean the hinge failed at its one job in rule 1's sense (the drop is large), it means the drop carries no readable
  content; no kill part is met;
  "ProsQA-hard UNTESTED (A2.1): no both-tasks clause can be met as written (critic entry 2026-10-20T17:12Z (8))". C2+C3's C2-side line:
  +0.4 over C2-shuffled, not above it by 3.60; attribution set: +0.0 vs C2, +1.6 vs C3, +1.0 vs coconut.
- K3: not fired (A_pd 6.38; margins +4.02 / +3.62 / +5.62). Pause comparison +0.8 / +0.4 / +2.4: CURRICULUM-ONLY. 7.2: FLOP and token
  legs met, passes criterion not met (9.226 / 9.216 / 9.268 against 6.29): billing claim only. Training FLOPs charged / bar 1.30 / 1.68 /
  1.32, "training-heavy" (5.5 (d)). Not fading; no too-good or null trigger.

### Phase 2, ProsQA-hard screen (124M, seed 0, valid 300; bar LR-tuned 3 trials, at the LR-grid edge; arm 1 trial; sigma_screen 3.60; no test read)
Table of record: `results/screens/prosqa_hard/prosqa_hard_screen_with_plain_depth.md` (the third table, `rev18`, citing `rev19`,
`rev20`; eval reads `rev21`; diag records `rev22`, `rev23`). Bar: `cot` lr 3e-5 80.3 (241 of 300) at epoch 14 (trials 80.3 /
70.7 / 73.0) [bar: single seed, LR-tuned (3 trials), at the LR-grid edge (3e-5)]; `no_cot` 57.7 (173 of 300); plain depth 355M `no_cot`
61.7 (185 of 300), epoch 28 [the same control label]. C3, C3-foreign, C2+C3: UNTESTED on ProsQA-hard (no run; A2.1).

| arm | valid | vs bar | Delta_scr [interval] | K2 at the screen | zero-latent drop | pause comparison | passes; FLOP ratio to bar |
|---|---|---|---|---|---|---|---|
| pause | 74.7 | -5.7 | (no fed-back vector; no K2 read) | | | | 9.153; [0.869, 0.872] |
| coconut | 73.3 | -7.0 | +0.0 [+0.00, +0.00] | fired | +0.7 [-0.26, +1.59] | -1.3: CURRICULUM-ONLY | 16.147; [0.869, 0.872]; billing claim only |
| C1b k = 8 | 71.7 | -8.7 | +8.0 [+3.55, +12.45] | not fired | +7.7 [+3.60, +11.73] | -3.0: CURRICULUM-ONLY | 16.163; [0.865, 0.868] nominal; billing claim only |
| C1b k = 4 | 70.7 | -9.7 | -0.3 [-0.99, +0.32] | fired | +0.3 [-0.32, +0.99] | -4.0: not within sigma | 16.147; [0.862, 0.865] nominal; billing claim only |
| C2 | 72.3 | -8.0 | +0.3 [-0.32, +0.99] | fired | +0.7 [-0.26, +1.59] | -2.3: CURRICULUM-ONLY | 16.180; [0.869, 0.872]; billing claim only |
| C2-shuffled | 71.3 | -9.0 | +0.3 [-0.33, +1.00] | fired | +1.3 [+0.03, +2.64] | -3.3: CURRICULUM-ONLY | 16.147; [0.869, 0.872]; billing claim only |

- 6.3: no card arm advances (bar legs 71.7 / 70.7 / 72.3 against 80.3 - 3.60 = 76.73; pause legs -3.0 / -4.0 / -2.3); FLOP and token
  legs met by every arm (E' 8.15 to 8.18 against 0.2 x 76.07 = 15.21); the passes criterion (<= 15.41) met by pause only: "billing
  claim only" on the five feedback arms. 7.10: no arm above 99.0; no null trigger. 7.6: no arm fading.
- K2: fired at the screen on coconut, C1b k = 4, C2 and C2-shuffled (every interval including 0) and not on C1b k = 8; the controls'
  sentence (critic entry 2026-10-20T21:03Z row 2 (4)): "latents not used at inference on ProsQA-hard at the screen (K2 fired at the
  screen, 7.5 / A3.4: Delta_scr +0.0 (coconut) / +0.3 (C2-shuffled), intervals including 0; zero-latent drop +0.7 / +1.3;
  CURRICULUM-ONLY, pause comparison -1.3 / -3.3; 1 seed, valid 300; this checkpoint at this read)", where "not used at inference"
  means the registered scrambled and zero-latent reads not moving the output at this read; the coconut line adds that the scrambled
  pass fed other examples' vectors by construction and by the A14.3 record on three other coconut checkpoints (diag `rev8`), not
  measured on this checkpoint. Sweeps flat within 1.0 point on four arms; C1b k = 8 rises from 64.0 at n = 0 to 72.3 at n = 4 (7.3).
- The observation form (critic entry 2026-10-20T21:03Z row 2 (5), the only one that travels): "On ProsQA-hard at the screen (124M, 1
  seed, valid 300, no test read) every latent arm's selected checkpoint scores 13.0 to 17.0 points above no_cot (57.7): pause 74.7,
  coconut 73.3, c2 72.3, c1b_k8 71.7, c2_shuffled 71.3, c1b_k4 70.7. On four of the five feedback arms the registered scrambled read
  and the zero-latent read sit within 1.3 points of the own read (K2 fired at the screen on coconut, c1b_k4, c2 and c2_shuffled), and
  pause, which feeds nothing back, is the highest arm; c1b_k8 is the exception (scrambled -8.0, zero-latent -7.7, sweep rising to
  n = 4). At this read the gain over no_cot does not depend measurably on what the fed-back vectors carry on those four arms; the
  pause comparison (CURRICULUM-ONLY on coconut, c1b_k8, c2, c2_shuffled) is the registered reading of it, and the staged chain
  supervision that the pause arm shares, the curriculum's training data, is the candidate the registered controls point to, untested
  as a cause (no no-curriculum arm ran on this task; 1 seed)."
- K3 (critic entries 2026-10-20T21:03Z row 4, 23:37Z): A_pd 57.69 to 57.72 (124M `no_cot` 57.7 at [110.239, 110.376] GFLOPs; 355M 61.7
  at [384.111, 384.477] GFLOPs, 3.48x); fires on no arm; "no better than plain depth" on no arm (margins +12.95 to +16.95); a screen
  number against a 1-seed control and no K3 verdict either way.
- C1b per-step cost: k = 8 "saving shows: f_meas 0.7280 <= 0.7667 at batch 1 (A6.2; A100-40GB, 1 seed, valid; rule 3: 0 of 300 differ
  at batch 32; logits <= 1.7e-12 at float64 on A100-40GB (device cuda) on this checkpoint, 0 argmax differences; the fp32 kernel-order
  explanation as the GSM-50k records rev15 / rev16; float64 record rev22; strict-fp32 GPU figure: 7.6e-4 on A100-40GB (32 of 32
  over 1e-4; diag rev23))" (0.7499 at batch 32 beside); k = 4 "nominal: f_meas 0.4391 against the line 0.43" (0.5291 at batch 32).
  The GSM-50k 0.7200 is printed beside and never pooled.
- Earlier-stage flags (3.3 item 3; critic entry 2026-10-20T19:14Z (4)): "on ProsQA-hard every latent arm's best valid accuracy over all
  30 epochs sits at an earlier stage (2 to 5 latents, 23 to 57 written tokens) above its final-regime selection by 3.3 to 9.3 points
  (1 seed, valid 300, no test read)" (coconut 76.7 at epoch 8, c1b_k8 75.0 at 14, c1b_k4 74.3 at 18, c2 81.7 at 9, c2_shuffled 76.7
  at 9); none meets 7.2's token leg; a flag, not a selection. 13-gram overlap vs its own train 37.0% on valid (A6.7 item 2).

### Cross-table card verdicts (A3.4; critic entry 2026-10-20T21:03Z row 2; both screens, 1 seed, valid)
- C1b k = 4: "killed at the screen (A3.4 kill clause: K2 fired on both tasks; 1 seed, valid, both tasks); CURRICULUM-ONLY on GSM-50k
  (pause -0.2), below pause by 4.0 on ProsQA-hard (no label); f_meas nominal on both tasks (0.4326 / 0.4391 against 0.43)".
- C2: killed at the screen under A3.4 kill clause 1 ("within sigma_screen of C2-shuffled on both tasks": +0.4 on GSM-50k, +1.0 on
  ProsQA-hard, against 3.60; each gap includes learnable vs unlearnable auxiliary target); K2 fired on it on ProsQA-hard and not on
  GSM-50k; on neither task does the scrambled drop separate C2 from its null (+8.6 against +8.6; +0.3 against +0.3). The 7.9 R3
  readout sentence for its decoder stays unwritten (its positive control never ran).
- C1b k = 8: "card pass line met on both tasks as a screen number (A3.4; 1 seed, valid): the band at 8 of 12 layers keeps full-depth
  coconut's accuracy within sigma_screen (-1.6 on GSM-50k, -1.7 on ProsQA-hard) with K2 not fired on either (+4.0 / +8.0); f_meas
  0.7200 on GSM-50k (saving shows) and 0.7280 on ProsQA-hard (saving shows, the rule 3 record met at float64); 6.3 failed on both tasks
  (bar legs -16.2 / -8.6; pause legs -0.2 / -3.0); does not advance; CURRICULUM-ONLY on both; billing claim only on both (7.2 passes
  criterion)". It owes no confirmation seed and carries no sentence about what its latents carry.
- C3 and C2+C3: killed as latent claims at the screen on GSM-50k by K2 (counts as fired, rule 5; critic entry 2026-10-21T14:57Z);
  UNTESTED on ProsQA-hard (A2.1); no both-tasks clause of A3.4 can be met as written; the F1 patch directional read is UNTESTED for both.
- Readout (7.9): coverage and load-bearing fraction were measured on no arm (no probe, F1 patch or bit-width ladder ran on any screen
  checkpoint; the positive control never ran), so no "explained" or "partly explained" label exists; every arm's readout is UNTESTED.

### Predictions against outcomes (section 11, A3.4, 17:36Z (3); the critic's, printed unedited)
ProsQA coconut 95.5 [92, 98]: 96.27; cot 78 (SD 2.0): 74.87; P(K1 ProsQA) 0.70: pass. GSM8k-Aug cot 42: 40.94; coconut 32 [27, 36]:
34.65; P(K1 GSM) 0.50: pass at the window's scope. ProsQA-hard bar 60 [40, 80]: 80.3, above the interval; no_cot 55: 57.7; coconut 70
[50, 90]: 73.3; pause within 3 of coconut: +1.3, yes. GSM-50k bar 30 [24, 36]: 25.6; no_cot 12: 6.2; coconut from cot 22 [15, 30]:
11.0, below; P(K2 fires on full-depth coconut on GSM-50k) 0.4: did not (it fired on ProsQA-hard). C1b k = 8 minus coconut -1 [-4, +2]
/ -2 [-5, +1]: -1.7 / -1.6; k = 4 -5 [-10, 0] / -6 [-12, -1]: -2.7 / -1.6; P(K2 fires on k = 8) 0.4: did not on either task;
wall-clock within 10% of 8/12: 0.7200 / 0.7280, yes. C2 minus coconut 0 [-3, +2] / +2 [-2, +6]: -1.0 / +1.0; C2 minus C2-shuffled
+1 [-2, +4] / +2 [-1, +5]: +1.0 / +0.4; C2's scrambled drop on GSM-50k 8 [2, 20]: 8.6; P(C2 passes) 0.45: killed. C3 on GSM-50k
(critic entry 2026-10-21T14:57Z (2)): drop above 15 (P 0.45): 10.4, not met; accuracy +1 [-4, +4]: -0.6, inside; P(C3-foreign's drop
exceeds 10) 0.25: 2.0, not met; C2+C3 minus coconut +3 [-2, +9]: +1.0, inside; P(dead by K2) 0.2: dead, by rule 5; R_own 0.010
[0.003, 0.025] on every arm: met; Delta_dir C3 +0.03 [-0.01, +0.12] (P(present) 0.35): -0.014, outside the interval on the low side,
absent; C2+C3 +0.05 [-0.01, +0.20]: -0.007, inside, absent; C3-foreign +0.01 [-0.01, +0.05]: +0.003, absent; coconut +0.02 [-0.01,
+0.08]: +0.017, absent; G_chg C3 0.30 [0.10, 0.70], C2+C3 0.25 [0.08, 0.60], C3-foreign 0.12 [0.03, 0.40], coconut 0.10 [0.03, 0.30]:
0.997, 0.993, 0.481, 0.830, outside every interval on the high side (the critic's miss, recorded: on this task at 1 seed the scrambled
pass changes almost every output, so the garbage clause's second leg fires on any arm; it changes no reading); P(at least one hinge
arm "drop without direction") 0.65: met (two). P(a card passes its screen on at least one task) 0.6: none advanced under 6.3; P(the 124M read
is positive under A1's definition) 0.15: it is not.

### Test read
Done once, for reporting, with no decision attached (A3.4; 4.5 one registered read per checkpoint; critic entries 2026-10-21T15:25Z /
15:27Z, the table `results/screens/test_read.md` / `.json` + provenance, `rev1`, written by `code/read_test.py` over the two eval
jobs `screen-test-gsm50k-acctc` (AcctC, `e5fa`, 21 entries) and `screen-test-prosqa-hard-acctb` (AcctB, `0999`, 14 entries); the
hinge files and the ProsQA-hard 355M file read from byte-identical mirrors in the job's own bucket, pinned by sha256). Labels on every
number: end-of-round test read (A3.4): reporting only, no decision attached; 1 seed; one registered read per checkpoint (4.5); the
interval is 6.2's form for one accuracy (binomial SE); test minus valid is descriptive (valid is the selection split, test the scoring
split, 1.3). Primary reads (the selected `ckpt_best_final.pt`, the final stage's latent count), test accuracy [95% interval], with the
valid of record beside:

| run | GSM-50k test (1,319) | valid | ProsQA-hard test (500) | valid |
|---|---|---|---|---|
| tuned bar (cot) | 21.0 [18.8, 23.2] | 25.6 | 83.4 [80.1, 86.7] | 80.3 |
| no_cot | 6.4 [5.1, 7.8] | 6.2 | 53.2 [48.8, 57.6] | 57.7 |
| plain depth 355M | 7.4 [6.0, 8.8] | 8.0 | 60.4 [56.1, 64.7] | 61.7 |
| pause | 6.9 [5.5, 8.3] | 9.6 | 74.8 [71.0, 78.6] | 74.7 |
| coconut | 9.2 [7.6, 10.7] | 11.0 | 72.8 [68.9, 76.7] | 73.3 |
| C1b k = 8 | 8.3 [6.8, 9.7] | 9.4 | 73.6 [69.7, 77.5] | 71.7 |
| C1b k = 4 | 7.5 [6.1, 8.9] | 9.4 | 71.2 [67.2, 75.2] | 70.7 |
| C2 | 10.2 [8.5, 11.8] | 12.0 | 75.8 [72.0, 79.6] | 72.3 |
| C2-shuffled | 9.9 [8.3, 11.5] | 11.6 | 72.6 [68.7, 76.5] | 71.3 |
| C3 | 8.9 [7.4, 10.5] | 10.4 | UNTESTED (A2.1) | |
| C3-foreign | 9.0 [7.5, 10.6] | 10.0 | UNTESTED (A2.1) | |
| C2+C3 | 9.3 [7.8, 10.9] | 12.0 | UNTESTED (A2.1) | |

Secondary reads (3.3 item 3: the all-epoch best `ckpt_best.pt`, an earlier stage, at that stage's latent count; decides nothing): on
GSM-50k the nine stage-1 checkpoints (2 latents) read 11.8 to 13.9 on test (valid 11.0 to 14.6) with 21 to 23 emitted tokens against
2.2 to 2.4 for the final-stage reads; on ProsQA-hard the five earlier-stage checkpoints read 71.2 to 78.8 on test (valid 74.3 to 81.7)
with 25 to 57 emitted tokens against 8.2. Every test minus valid gap is inside its interval's width (the largest: the GSM-50k bar -4.6,
the ProsQA-hard no_cot -4.5, C2 +3.5, the bar +3.1); none is a result, and no line of this report changes with it. Test emitted tokens:
the GSM-50k bar 30.7 per answer against 2.2 to 2.4 for every latent arm (a 13x token saving on an arm 11 to 14 points under the bar);
the ProsQA-hard bar 77.8 against 8.2.

### Costs and incidents
Compute: within a fixed compute cap at the close (`governance/run_log.jsonl`), nothing open; the two test-read jobs e5fa and 0999
had two boot preemptions; every admission and close is in `governance/run_log.jsonl`. Incidents (`governance/deviations.md`):
spot preemptions on every project (two storms on 2026-10-19; `prosqa_cot_lr3e-5_s0` fragmented at 5 resumes; the ProsQA-hard screen's
relaunch counter spent, then hand relaunches); the plain-depth 355M micro-batch 16 x 8 fallback (A13.2 (b)); A6.2 rule 3 like-for-like
2 of 500 (GSM-50k) and 2 of 300 (ProsQA-hard) at batch 1, explained as bf16 near-tie flips across the padded shape and confirmed at
float64; the strict-fp32 band-logits "BUG at 1e-4" verdicts (6.4e-4, 7.6e-4) kept printed beside their float64 explanations (2.3e-13,
1.7e-12; A15); five operator errors (service account, launcher data step, prep smoke, dirty
tarball, hinge eval bucket permission); no checkpoint, read or number was lost.

## Closing facts (dated section, 2026-10-22; PLAN.md A2.2 and A2.2.1; no earlier table is edited)

The author's closing plan (A2.2, 2026-10-21; cap raised by A2.2.1) asked for three cheap facts, each registered by the
critic with a numeric prediction before it ran (critic entries 2026-10-21T20:21Z (a), (b), (c)). All three ran and are on the record; the
sentences below are the critic's verification rows' sentences, verbatim (rows 2026-10-22T07:28Z, 07:10Z, 09:56Z, 07:38Z), and nothing
here changes a table or verdict above. The earlier tables stand as written; the fifth GSM-50k table and the fourth ProsQA-hard table are
reissues that add one arm each and are the screen tables of record from their verification rows.

### (a) det2 and sigma_task(ProsQA) (`results/k1/prosqa_cot_same_seed_pair.md`, `rev24`; E1 `results/evals/det2-eval-acctc/`, `rev25`; critic entries 06:52Z, 06:54Z, 07:28Z)
The same-seed pair on ProsQA (13.6; critic entry 2026-10-21T20:21Z (a)) is complete: prosqa_cot_s0 / det1 / det2, each read once on test, score 75.0 / 75.8 / 77.8 (375 / 379 / 389 of 500; batch-1 audit flips 0 / 1 / 0 of 500); the per-example churn is 106 (s0 / det1), 86 (det1 / det2) and 98 (s0 / det2) of 500; under the strict reading of record d_nd = |77.8 - 75.8| = 2.0 > 0.72, so sigma_task(ProsQA) = sqrt(3.60^2 + 2.0^2) = 4.12, with the author's 'one run' reading (d_nd 0.8, 3.69) printed beside and not amended [pair read on two GPU types: det1 on A100-40GB, det2 on A100-80GB, both trained on A100-80GB; A3.2 (ii) parity passed on both by the project-level references (det1 AcctB's, det2 AcctC's 4882, the same reference checkpoint), no per-job step on either; env_reference ok with no mismatches and image v20261019 on both; det2's accuracy enters no cell, no seed pool and no sigma_cell; rises, never falls].

sigma_task 4.12 is the figure of record for every ProsQA verdict not yet written (6.2 beats / loses / tie, the 13.10 sentence, any WIN, LOSS or confirmation seed: none exists and none is written here) and changes no cell or verdict of record: the K1 table is unchanged (sigma_cell is 4.1's quantity), the GSM-50k (`rev26`) and ProsQA-hard (`rev18`) screen tables were read at sigma_screen 3.60, which carries no determinism term (A3.4), are write-once and are not reissued, and no sigma_task exists on any other task.

For the record, three registered screen margins sit between 3.60 and 4.12 and would read differently had those tables been read at 4.12: on GSM-50k C1b k = 8's Delta_scr +4.0 [+1.70, +6.30] (K2 'not fired' at 3.60; at 4.12 the first clause of 7.5 holds, and with it the card pass line 'met on this task' and the 6.3 scrambled leg 'met' would not), on ProsQA-hard C1b k = 4's pause comparison -4.0 ('not within sigma_screen' at 3.60; within at 4.12, CURRICULUM-ONLY), and on GSM-50k the descriptive K3 label 'no better than plain depth' would attach to C3 (+4.02 over A_pd) and C3-foreign (+3.62); every other registered outcome (K2, K3, pause, card, 6.3, 7.10, fade, the Curriculum line's BOTH) reads the same at either value.

### (b) Coconut without the curriculum, seed 0, the same judge (GSM-50k: `results/screens/gsm50k/gsm50k_screen_with_no_curriculum.md`, `rev26`, E2 `results/evals/screen-eval-gsm50k-nocur-acctc/`, `rev27`, critic entries 03:18Z, 07:10Z; ProsQA-hard: `results/screens/prosqa_hard/prosqa_hard_screen_with_no_curriculum.md`, `rev28`, E3 `results/evals/screen-eval-prosqa-hard-nocur-acctb/`, `rev29`, critic entries 09:30Z, 09:56Z)
On GSM-50k the no-curriculum control gsm50k_coconut_no_curriculum_s0 (6 latents at every trained epoch; selection over every epoch, epoch 20) scores 9.8 (49 of 500) vs no_cot 6.2 (+3.6) and vs full-depth coconut 11.0 (-1.2), sigma_screen 3.60: BOTH; within sigma_screen of both no_cot and coconut; the coconut-vs-no_cot gap on this task (4.8) is itself 1.3 sigma_screen: unresolved at 1 seed. Its own K2 fired at the screen (Delta_scr +1.6 [-0.06, +3.26]; zero-latent drop +1.0 [-0.70, +2.70]), its pause comparison is +0.2 (CURRICULUM-ONLY), K3 is not fired and 7.10 is not triggered; no other arm's K2, K3, card, pass or screen outcome changes and nothing here transfers to ProsQA-hard [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial].

On ProsQA-hard the no-curriculum control prosqa_hard_coconut_no_curriculum_s0 (7 latents at every trained epoch; selection over every epoch, epoch 2 of 30; one exact boundary resume) scores 58.0 (174 of 300) vs no_cot 57.7 (+0.3) and vs full-depth coconut 73.3 (-15.3), sigma_screen 3.60: NEAR NO_COT; at this read the same latent layout trained without the staged curriculum scores with no_cot: the gain of the latent arms over no_cot does not appear without the curriculum; the curriculum is necessary for it at this read (1 seed). Its own K2 fired at the screen (Delta_scr -1.0 [-2.74, +0.74]; zero-latent drop +1.0 [-0.47, +2.47]), its pause comparison is -16.7 (not within sigma_screen; no CURRICULUM-ONLY label), K3 is not fired (58.0 vs A_pd 57.71: no better than plain depth) and 7.10 is not triggered; no other arm's K2, K3, card, pass or screen outcome changes [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial].

The two Curriculum zones, each on its own task and neither read into the other: GSM-50k BOTH (9.8 within sigma_screen of both no_cot 6.2 and coconut 11.0; the gap 4.8 is 1.3 sigma_screen: unresolved at 1 seed) and ProsQA-hard NEAR NO_COT (58.0 with no_cot 57.7 and 15.3 below coconut 73.3; the gap 15.7 is 4.4 sigma_screen: necessary at this read, 1 seed).

### (c) The Pareto read (`results/screens/pareto_read.md`, `rev30`; critic entry 07:38Z)
The Pareto read (`results/screens/pareto_read.{md,json}`, `rev30`; 7.3: secondary, reported, decides nothing; a compilation from the committed records: no new training, no new read of any checkpoint) places every point of both tasks on accuracy against mean emitted tokens E', mean FLOPs (5.1; the 5.5 (c) reference cost beside) and mean sequential passes (5.4 (b)), valid and test apart: on GSM-50k the nine earlier-stage checkpoints (stage 1, 2 latents, epochs 5 to 6; secondary read, 3.3 item 3) score 11.0 to 14.6 on valid at 21.3 to 22.9 tokens and 15.7 to 16.2 GFLOPs (from the training log's means (range); no per-example record on valid) against their selections' 9.4 to 12.0 at 2.2 to 2.4 tokens and 11.6 to 11.7 GFLOPs, and 11.8 to 13.9 on test at 21.0 to 23.2 tokens and 15.9 to 16.5 GFLOPs against 6.9 to 10.2 at 2.2 to 2.4 and 11.3 to 11.9; the tuned bar sits at 25.6 (valid; 30.4 tokens, 17.3 GFLOPs) and 21.0 (test; 30.7, 17.7).

On ProsQA-hard the five earlier-stage checkpoints (2 to 5 latents, epochs 8 to 18) score 74.3 to 81.7 on valid at 22.8 to 57.1 tokens and 115.5 to 124.6 GFLOPs against their selections' 70.7 to 73.3 at 8.1 to 8.2 tokens and 112.0 GFLOPs, and 71.2 to 78.8 on test at 25.2 to 56.8 tokens and 116.8 to 125.5 GFLOPs against 71.2 to 75.8 at 8.2 and 112.3 to 113.2 (C1b k = 4's two test points both 71.2); the bar 80.3 (valid; 76.1 tokens, 128.5 to 129.0 GFLOPs) and 83.4 (test; 77.8, 130.4).

Every gap is descriptive [1 seed; test points: end-of-round test read (A3.4): reporting only, no decision attached; one registered read per checkpoint (4.5)]; valid and test never share an axis; the 7.3 FLOP multiplier and the budgeted-CoT curve are UNTESTED (no budgeted-CoT read exists); no line of any table changes.

### The plan's predictions against outcomes (the critic's, scored in rows 06:52Z, 07:10Z, 07:28Z, 07:38Z, 09:30Z, 09:56Z)
Row (a): five of five (det2 test 77.8 inside [72.5, 78.5]; d_nd > 0.72 realized; sigma_task 4.12 inside [3.60, 4.3]; selected epoch
and the 4.0 h cap). Row (b): fully scored on both tasks, one miss each (GSM-50k: the selected epoch in the first half; both tasks: the
joint "both runs finish inside 3.5 h without a resume"); the zones BOTH (P 0.45) and NEAR NO_COT (P 0.7) were the modal predictions.
Row (c): accuracy above selection 14 of 14 on valid and 13 of 14 on test; tokens 3 to 10x: direction 28 of 28, the stated range 20 of 28;
FLOPs above selection 14 of 14 on test; the bar highest on GSM-50k (both splits) and on ProsQA-hard test, as foreseen.

### Costs and incidents of the closing plan
Within the plan's cap (`governance/run_log.jsonl`, 295 records, nothing open; zero VMs in every project at the close). Completing runs: det2
4dc1 (AcctC, 2.9 h, 0 resumes), J2 cad5 (AcctC, 2.7 h, 0 resumes), J3 f421 + aa8f (AcctB; one exact boundary
resume at epoch 27), E1, E2, E3. Waste: a spot storm on AcctA preempted det2 eight times before an epoch
and the supervisor relaunched it against the cap row, shrank its cap and admitted it twice more after the author moved it to
AcctC (the supervisor was then stopped and every closing admission was by hand); J3 took ten admissions (three launch
deferrals, 9df1 / 7d20 / 0550, because the runner compares the whole-run estimate with the time left under the cap, also on a resume;
four preemptions before an epoch; d4fb's fifteen epochs lost when its last checkpoint and state file were uploaded in separate syncs
and the exact-resume rule refused the pair, so f421 ran from epoch 1 (critic entry 09:30Z, verified from the archived files); the
record is f421 + aa8f). Two bookkeeping deviations (critic entry 09:30Z): 589d and aa8f were admitted while a dead VM's worst case was
still unreleased, so the plan's committed figure stood over the cap on paper at those two admissions; the realized figure never did.
One fact of record without a label (rows 07:10Z, 09:56Z): the no-curriculum arms' band [0, 12] reads differ from their HF-path plain
reads on 1 of 500 (GSM-50k, both batch sizes) and on 4 and 8 of 300 (ProsQA-hard, batch 1 and 32); the arms' band reads were
registered for timing only and their accuracy enters no cell.

## What it rules out
- Within the registered 72-hour scope at 124M, seed 0, on valid: no card arm advances past either screen (6.3), so the conditional
  continuation of PLAN.md A1 (scale at 355M / 774M, then code transfer) is not triggered and the project closes with this report.
- A latent claim at the screen for C1b k = 4 (K2 fired on both tasks), for C2 (within its null on both tasks; K2 fired on ProsQA-hard)
  and, on GSM-50k, for C3 and C2+C3 (K2 counts as fired, rule 5: the donor's answer appears 0 of 289 times under donor latents while
  288 and 287 of 289 scored outputs change to something that is neither the own-pass prediction nor the donor's answer; what the read
  shows is a checkpoint that detects foreign latents and does not read their content).
- On GSM-50k every latent arm sits 13.6 to 16.2 under the 3-LR bar and within 3.60 of the pause arm (CURRICULUM-ONLY on all eight);
  the critic's "coconut from cot 22 [15, 30]" came in at 11.0, below the interval. K3 fires on nothing on either task: no arm loses to
  plain depth at equal FLOPs (pause, C1b k = 8 and k = 4 "no better than plain depth" on GSM-50k).
- On released ProsQA the three Coconut checkpoints' answers show no measurable dependence on their latents at the registered reads
  (scrambled drop 0.0 on every seed), as the A12.4 sentence above states and no further.

## What it does not rule out / caveats
- Every Phase 2 number is a screen number: 1 seed, valid only, bar LR-tuned 3 trials, arm 1 trial, no test read. A screen filters and
  never verdicts; no 6.2 verdict, tie, WIN, LOSS or 13.10 scrambled-drop sentence exists on any task (sigma_task's determinism term was
  never fixed; no confirmation seed ran). The 7.2 claim test was never reached: no arm met the accuracy leg; the token and FLOP legs
  held for every latent arm (E' 0.07 to 0.11 of the bar's), the sequential-passes criterion failed on every feedback arm.
- The bars are single-seed and LR-tuned only (the schedule axis UNTESTED); the ProsQA-hard bar sits at the LR-grid edge (3e-5); the
  released-ProsQA tuned bar was not read; the K1 GSM cot cell is a single-seed match at the tolerance floor (540 of 1,319; 539 would
  have missed).
- UNTESTED, not estimated: ProntoQA; `coconut_no_curriculum`; C4-C6; Phase 3 (no slope, no FLOP multiplier); Phase 4; C3, C3-foreign
  and C2+C3 on ProsQA-hard (and the F1 patch directional read for both); the 2 B fade run; readout coverage and load-bearing fraction
  on every arm (the positive control never ran); the bit-width ladder; every confirmation seed; GSM8k-Aug `no_cot`; the 355M control's
  three seeds; 5.5 (b) wall-clock against the bar (no timed bar read).
- The pause-same-curriculum control sits within sigma_screen of every latent arm but C1b k = 4 on ProsQA-hard; the curriculum's
  training data is the candidate the registered controls point to, untested as a cause (no no-curriculum arm ran on either screen task).
- K2 on C3 and C2+C3 is "not independent" by construction (A6.1): "drop without direction (fingerprint-consistent)" labels a large drop
  whose directional read has an interval including 0 and whose changed-output garbage fraction exceeds the donor-answer rate; it does
  not say what the latents hold, only that the donor's answer is not read out of them at this screen.
- The strict-fp32 band-logits records missed the registered 1e-4 line on both tasks; the verdicts stay printed; the float64 records show
  the two forward paths are one function (A15). Known biases (deviations.md): the GSM valid set shares 13-grams with train on 7 of 500
  items (equal for every arm); the head-cost confound of 5.4 favours latent steps at small sizes; bar selection on seed 0 inflates it.
- The test read is reporting only (A3.4); its gaps to valid are descriptive and change no number above.

## References
- [PLAN.md](PLAN.md) (the plan; A1-A3, A2.1), [CHANGELOG.md](CHANGELOG.md) (lab notes, newest first), [README.md](README.md),
  [ARTIFACTS.md](ARTIFACTS.md), [docs/prior_art.md](docs/prior_art.md), [docs/theory_memo.md](docs/theory_memo.md).
- Governance: [registration.md](governance/registration.md) (sections 0-14, A1-A15), [critic_log.md](governance/critic_log.md)
  (every verdict row), [deviations.md](governance/deviations.md), [run_log.jsonl](governance/run_log.jsonl),
  [designs/a6_1_directional_read.md](governance/designs/a6_1_directional_read.md).
- Tables of record: `results/k1/k1_table_final.md` (`rev2`), `results/k1/k1_table_window.md` (`rev6`),
  `results/screens/gsm50k/gsm50k_screen_with_hinge.md` (`rev9`), `results/screens/prosqa_hard/prosqa_hard_screen_with_plain_depth.md`
  (`rev18`); diag records under `results/diag/` (`rev8`, `rev15`, `rev16`, `rev23`, `rev22`); eval reads under `results/evals/`.
- Reference work: Hao et al. 2024 (Coconut; facebookresearch/coconut at rev31); the critiques and families of PLAN.md 1.2-1.4.
