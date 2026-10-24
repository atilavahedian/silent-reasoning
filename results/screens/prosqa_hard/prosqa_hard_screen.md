# ProsQA-hard screen table: screen numbers (1 seed), valid split

ProsQA-hard: depth 5-7, 1.7x edges, 300-word questions (released: depth 3-6, 36 edges, 180 words) (300 items). Seed 0, GPT-2 124M, valid split only (A3.4); no test read; bf16 autocast over the TF32 flags (A10.5). Labels on every number and line: screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial; bar: single seed, LR-tuned (3 trials). Screen decisions use valid only (A3.4); screen cells are screen numbers, never verdicts.

## 1. Selection and valid accuracy

| run | role | pool (epochs) | selected epoch (stage) | valid | vs bar | all-epoch best (3.3 item 3) | resumes | GPU | start pin (A6.3 rule 4) | record checks |
|---|---|---|---|---|---|---|---|---|---|---|
| prosqa_hard_cot_lr3e-5_s0 | bar trial: cot, lr 3e-5 | all | 14 (0) | 80.3 (241 of 300) | +0.0 |  | 3 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| prosqa_hard_cot_lr1e-4_s0 | bar trial: cot, lr 1e-4 | all | 10 (0) | 70.7 (212 of 300) | -9.7 |  | 1 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| prosqa_hard_cot_lr3e-4_s0 | bar trial: cot, lr 3e-4 | all | 17 (0) | 73.0 (219 of 300) | -7.3 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| prosqa_hard_no_cot_s0 | floor; the 124M plain-depth point (7.7) | all | 8 (0) | 57.7 (173 of 300) | -22.7 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| prosqa_hard_plain_depth_355m_s0 | plain depth: 355M no_cot (7.7, A13.2) | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |
| prosqa_hard_pause_s0 | control: pause-same-curriculum | 22-30 | 30 (9) | 74.7 (224 of 300) | -5.7 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (14) |
| prosqa_hard_coconut_s0 | control: full-depth coconut (C1b null; C2 / C3 base) | 22-30 | 30 (9) | 73.3 (220 of 300) | -7.0 | 76.7 at epoch 8 (stage 2, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB |  | ok (15) |
| prosqa_hard_c1b_k8_s0 | card C1b, k = 8 (band [2, 10)) | 22-30 | 27 (8) | 71.7 (215 of 300) | -8.7 | 75.0 at epoch 14 (stage 4, n latents = 4): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB |  | ok (15) |
| prosqa_hard_c1b_k4_s0 | card C1b, k = 4 (band [4, 8)) | 22-30 | 30 (9) | 70.7 (212 of 300) | -9.7 | 74.3 at epoch 18 (stage 5, n latents = 5): earlier stage better on valid | 2 | NVIDIA A100-SXM4-80GB |  | ok (15) |
| prosqa_hard_c2_s0 | card C2 | 22-30 | 27 (8) | 72.3 (217 of 300) | -8.0 | 81.7 at epoch 9 (stage 2, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB |  | ok (15) |
| prosqa_hard_c2_shuffled_s0 | null: C2-shuffled | 22-30 | 30 (9) | 71.3 (214 of 300) | -9.0 | 76.7 at epoch 9 (stage 2, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB |  | ok (15) |
| prosqa_hard_c3_s0 | card C3 | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |
| prosqa_hard_c3_foreign_s0 | null: C3-foreign | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |
| prosqa_hard_c2c3_s0 | card C2+C3 | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |

## 2. Cost per valid example (5.1, 5.3, 5.5; A6.4)

E' and sequential passes are exact means from the selected epoch's log row; FLOPs (GFLOPs) are the exact range over the unrecorded per-example E' (notes). Training FLOPs in PFLOPs.

| run | L | E' mean | E' / bar | passes mean | passes / bar | FLOPs mean | FLOPs / bar | reference-cost FLOPs mean | reference / bar | training FLOPs own / charged | charged / bar (same epochs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| prosqa_hard_cot_lr3e-5_s0 | 0 | 76.07 | 1.000 | 77.07 | 1.000 | [128.523, 128.960] | [0.997, 1.003] | [11546.838, 14766.932] | [0.782, 1.279] | 157.8 / 157.8 | 1.00 |
| prosqa_hard_cot_lr1e-4_s0 | 0 | 73.06 | 0.960 | 74.06 | 0.961 | [127.702, 128.145] | [0.990, 0.997] | [11008.807, 14271.679] | [0.746, 1.236] | 157.8 / 157.8 | 1.00 |
| prosqa_hard_cot_lr3e-4_s0 | 0 | 78.00 | 1.025 | 79.00 | 1.025 | [129.049, 129.482] | [1.001, 1.007] | [11894.983, 15082.136] | [0.806, 1.306] | 157.8 / 157.8 | 1.00 |
| prosqa_hard_no_cot_s0 | 0 | 8.19 | 0.108 | 9.19 | 0.119 | [110.239, 110.376] | [0.855, 0.859] | [1047.183, 2057.251] | [0.071, 0.178] | 142.1 / 142.1 | 0.90 |
| prosqa_hard_pause_s0 | 7 | 8.15 | 0.107 | 9.15 | 0.119 | [111.962, 112.099] | [0.868, 0.872] | [1064.521, 2072.631] | [0.072, 0.179] | 149.3 / 149.3 | 0.95 |
| prosqa_hard_coconut_s0 | 7 | 8.15 | 0.107 | 16.15 | 0.210 | [111.961, 112.097] | [0.868, 0.872] | [2146.479, 3153.881] | [0.145, 0.273] | 149.3 / 149.3 | 0.95 |
| prosqa_hard_c1b_k8_s0 | 7 | 8.16 | 0.107 | 16.16 | 0.210 | [111.516, 111.653] nominal; [111.965, 112.102] at 1 | [0.865, 0.869] nominal; [0.868, 0.872] at 1 | [2148.524, 3157.361] | [0.145, 0.273] | 149.1 / 149.1 | 0.94 |
| prosqa_hard_c1b_k4_s0 | 7 | 8.15 | 0.107 | 16.15 | 0.210 | [111.063, 111.200] nominal; [111.961, 112.097] at 1 | [0.861, 0.865] nominal; [0.868, 0.872] at 1 | [2146.479, 3153.881] | [0.145, 0.273] | 148.9 / 148.9 | 0.94 |
| prosqa_hard_c2_s0 | 7 | 8.18 | 0.108 | 16.18 | 0.210 | [111.969, 112.106] | [0.868, 0.872] | [2150.569, 3160.864] | [0.146, 0.274] | 149.4 / 149.4 | 0.95 |
| prosqa_hard_c2_shuffled_s0 | 7 | 8.15 | 0.107 | 16.15 | 0.210 | [111.961, 112.097] | [0.868, 0.872] | [2146.479, 3153.881] | [0.145, 0.273] | 149.4 / 149.4 | 0.95 |

## 3. Registered lines

### Bar
- Bar (ProsQA-hard, valid): prosqa_hard_cot_lr3e-5_s0 80.3 (241 of 300) at epoch 14; trials prosqa_hard_cot_lr3e-5_s0 80.3, prosqa_hard_cot_lr1e-4_s0 70.7, prosqa_hard_cot_lr3e-4_s0 73.0 [bar: single seed, LR-tuned (3 trials); screen numbers (1 seed)]

### sigma_screen
- sigma_screen = 3.60 points (A3.4: max(1.0, 100 sqrt(p (1 - p) / n_valid) at the bar's p = 241/300, the ProsQA cot 3-seed SD once it exists))

### Screen rule 6.3 (card arms)
- Screen rule (6.3), prosqa_hard_c1b_k8_s0 (C1b): does not advance. Legs: bar: not met (71.7 >= 80.3 - 3.60 = 76.73); scrambled null + sigma: pending (no scrambled-at-inference valid read); pause + sigma: not met (-3.0 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Screen rule (6.3), prosqa_hard_c1b_k4_s0 (C1b): does not advance. Legs: bar: not met (70.7 >= 80.3 - 3.60 = 76.73); scrambled null + sigma: pending (no scrambled-at-inference valid read); pause + sigma: not met (-4.0 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Screen rule (6.3), prosqa_hard_c2_s0 (C2): does not advance. Legs: bar: not met (72.3 >= 80.3 - 3.60 = 76.73); scrambled null + sigma: pending (no scrambled-at-inference valid read); pause + sigma: not met (-2.3 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Card lines (A3.4)
- C1b k = 8 (card line, A3.4), prosqa_hard_c1b_k8_s0 - prosqa_hard_coconut_s0 = -1.7: pass needs within sigma_screen 3.60 of full-depth coconut (or better) and K2 not fired: accuracy part met, K2 pending, pass pending K2; kill needs both tasks: this task's accuracy part (below by more than 2 sigma_screen = 7.20) not met, K2 part pending; the other screen task is not in this table; per-step cost nominal [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C1b k = 4 (card line, A3.4), prosqa_hard_c1b_k4_s0 - prosqa_hard_coconut_s0 = -2.7: pass needs within sigma_screen 3.60 of full-depth coconut (or better) and K2 not fired: accuracy part met, K2 pending, pass pending K2; kill needs both tasks: this task's accuracy part (below by more than 2 sigma_screen = 7.20) not met, K2 part pending; the other screen task is not in this table; per-step cost nominal [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C2 (card line, A3.4), prosqa_hard_c2_s0 - prosqa_hard_c2_shuffled_s0 = +1.0 (the gap includes learnable vs unlearnable auxiliary target), prosqa_hard_c2_s0 - prosqa_hard_coconut_s0 = -1.0: pass needs above C2-shuffled by more than sigma_screen 3.60 with K2 not fired: not met on this task; kill needs one clause on both tasks: this task's parts: within sigma_screen of C2-shuffled met, below full-depth coconut by more than 2 sigma_screen not met, K2 pending; the other screen task is not in this table [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C3 (A3.4): UNTESTED on ProsQA-hard (no run record)
- C2+C3 (A3.4): UNTESTED on ProsQA-hard (no run record)

### K2 (7.5; A3.4)
- K2 (7.5; A3.4: every latent arm at the screen), prosqa_hard_coconut_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), prosqa_hard_c1b_k8_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), prosqa_hard_c1b_k4_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), prosqa_hard_c2_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), prosqa_hard_c2_shuffled_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### K3 (7.5; 7.7; A13.2)
- K3 (7.5 at the screen), prosqa_hard_pause_s0: 74.7 vs A_pd 57.67 (the 124M no_cot point alone, prosqa_hard_no_cot_s0 epoch 8) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), prosqa_hard_coconut_s0: 73.3 vs A_pd 57.67 (the 124M no_cot point alone, prosqa_hard_no_cot_s0 epoch 8) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), prosqa_hard_c1b_k8_s0: 71.7 vs A_pd 57.67 (the 124M no_cot point alone, prosqa_hard_no_cot_s0 epoch 8) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), prosqa_hard_c1b_k4_s0: 70.7 vs A_pd 57.67 (the 124M no_cot point alone, prosqa_hard_no_cot_s0 epoch 8) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), prosqa_hard_c2_s0: 72.3 vs A_pd 57.67 (the 124M no_cot point alone, prosqa_hard_no_cot_s0 epoch 8) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), prosqa_hard_c2_shuffled_s0: 71.3 vs A_pd 57.67 (the 124M no_cot point alone, prosqa_hard_no_cot_s0 epoch 8) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Pause comparison (7.5)
- Pause comparison (7.5), prosqa_hard_coconut_s0 - prosqa_hard_pause_s0 = -1.3: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), prosqa_hard_c1b_k8_s0 - prosqa_hard_pause_s0 = -3.0: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), prosqa_hard_c1b_k4_s0 - prosqa_hard_pause_s0 = -4.0: not within sigma_screen 3.60 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), prosqa_hard_c2_s0 - prosqa_hard_pause_s0 = -2.3: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), prosqa_hard_c2_shuffled_s0 - prosqa_hard_pause_s0 = -3.3: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### 7.2 cost legs and 5.5
- 7.2 cost legs, prosqa_hard_pause_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.868, 0.872]); token leg (E' <= 0.2 x bar's) met (8.15 vs 0.2 x 76.07 = 15.21); passes criterion (<= 0.2 x bar's) met (9.15 vs 15.41); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, prosqa_hard_coconut_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.868, 0.872]); token leg (E' <= 0.2 x bar's) met (8.15 vs 0.2 x 76.07 = 15.21); passes criterion (<= 0.2 x bar's) not met (16.15 vs 15.41); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, prosqa_hard_c1b_k8_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.868, 0.872]; nominal layer fraction: [0.865, 0.869]); token leg (E' <= 0.2 x bar's) met (8.16 vs 0.2 x 76.07 = 15.21); passes criterion (<= 0.2 x bar's) not met (16.16 vs 15.41); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, prosqa_hard_c1b_k4_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.868, 0.872]; nominal layer fraction: [0.861, 0.865]); token leg (E' <= 0.2 x bar's) met (8.15 vs 0.2 x 76.07 = 15.21); passes criterion (<= 0.2 x bar's) not met (16.15 vs 15.41); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, prosqa_hard_c2_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.868, 0.872]); token leg (E' <= 0.2 x bar's) met (8.18 vs 0.2 x 76.07 = 15.21); passes criterion (<= 0.2 x bar's) not met (16.18 vs 15.41); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, prosqa_hard_c2_shuffled_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.868, 0.872]); token leg (E' <= 0.2 x bar's) met (8.15 vs 0.2 x 76.07 = 15.21); passes criterion (<= 0.2 x bar's) not met (16.15 vs 15.41); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### C1b per-step cost (A3.4; A6.2)
- C1b per-step cost, prosqa_hard_c1b_k8_s0: nominal: f_meas UNTESTED (A6.2's timed read, band [0, 12] on the full-depth coconut checkpoint, has not run; the training-time val_latent_step_ms rows are not A6.2's fraction, rule 1); saving line f_meas <= 0.77; FLOPs printed at the nominal 8/12 and at 1 (the largest min(1, f_meas)); the FLOP leg is read at 1 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C1b per-step cost, prosqa_hard_c1b_k4_s0: nominal: f_meas UNTESTED (A6.2's timed read, band [0, 12] on the full-depth coconut checkpoint, has not run; the training-time val_latent_step_ms rows are not A6.2's fraction, rule 1); saving line f_meas <= 0.43; FLOPs printed at the nominal 4/12 and at 1 (the largest min(1, f_meas)); the FLOP leg is read at 1 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### C2 labels (A5 item 5; A6.7 items 4, 5)
- prosqa_hard_c2_s0: A5 item 5: the last supervised target contains the answer in 1.00 of the ProsQA-hard records; printed beside every C2 number; config.json _state_targets answer_in_last_target_frac 1.0 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- prosqa_hard_c2_shuffled_s0: A6.7 item 4: C2-shuffled's donor frontier contains the record's target name 34.8% of the time, symmetric between target and negative target, so the null has no answer signal; config.json _state_targets answer_in_last_target_frac 1.0, answer_in_last_used_target_frac 0.3483 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- train_state_loss by epoch 1-30 (A6.7 item 5): C2 0.584 0.449 0.429 0.503 0.453 0.431 0.476 0.466 0.450 0.504 0.475 0.437 0.467 0.425 0.402 0.391 0.395 0.364 0.379 0.357 0.345 0.336 0.324 0.324 0.315 0.319 0.304; C2-shuffled 0.587 0.462 0.454 0.536 0.536 0.534 0.574 0.572 0.571 0.589 0.588 0.589 0.597 0.597 0.598 0.599 0.602 0.600 0.601 0.600 0.601 0.600 0.600 0.599 0.600 0.600 0.599; every C2 minus C2-shuffled sentence carries: the gap includes learnable vs unlearnable auxiliary target [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Too-good triggers (7.10; A3.4)
- Too-good triggers (7.10; A3.4), any arm above 99%: none; the null triggers (scrambled-at-inference, zero-latent) are pending their valid reads [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Fade instrument (7.6)
- Fade instrument (7.6), prosqa_hard_pause_s0: lead over the bar's curve at epoch 15 -8.0, epoch 30 +2.3 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), prosqa_hard_coconut_s0: lead over the bar's curve at epoch 15 -3.3, epoch 30 +1.0 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), prosqa_hard_c1b_k8_s0: lead over the bar's curve at epoch 15 -8.3, epoch 30 -1.0 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), prosqa_hard_c1b_k4_s0: lead over the bar's curve at epoch 15 -9.3, epoch 30 -1.7 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), prosqa_hard_c2_s0: lead over the bar's curve at epoch 15 -8.0, epoch 30 -0.7 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), prosqa_hard_c2_shuffled_s0: lead over the bar's curve at epoch 15 -6.7, epoch 30 -1.0 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), prosqa_hard_no_cot_s0: lead over the bar's curve at epoch 15 -26.3, epoch 30 -22.7 (0.5 B = 15): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### UNTESTED and excluded
- prosqa_hard_plain_depth_355m_s0 (plain depth: 355M no_cot (7.7, A13.2)): UNTESTED: no run record (RUN_DONE.json absent); enters no cell
- prosqa_hard_c3_s0 (card C3): UNTESTED: no run record (RUN_DONE.json absent); enters no cell
- prosqa_hard_c3_foreign_s0 (null: C3-foreign): UNTESTED: no run record (RUN_DONE.json absent); enters no cell
- prosqa_hard_c2c3_s0 (card C2+C3): UNTESTED: no run record (RUN_DONE.json absent); enters no cell

### Receipts and launch condition (A3.2 (i); A5 item 8)
- env_reference per receipt: silent-train-20261019-172918-8046 ok, silent-train-20261019-185619-a829 ok, silent-train-20261019-191352-4285 ok, silent-train-20261019-195159-1790 ok, silent-train-20261019-213321-0ade ok, silent-train-20261020-041910-11f9 ok, silent-train-20261020-065429-1e3c ok, silent-train-20261020-130426-a26f ok, silent-train-20261020-144340-865b ok
- attempts named by the runs without a receipt in this read: none
- A5 item 8 (the bar's three cot trials admitted no later than the first latent arm, from the receipts' admissions): met

## 4. Readings and sensitivity

- sigma_screen of record 3.6019 (terms: floor 1.0, binomial 2.2948, ProsQA cot 3-seed SD 3.601851375797361); alternatives recomputed: without the ProsQA cot SD term 2.2948, floor only 1.0000.
- Lines whose outcome changes under an alternative reading:
  - pause:c1b_k8 under 'without the ProsQA cot SD term': CURRICULUM-ONLY -> not within sigma
  - pause:c2 under 'without the ProsQA cot SD term': CURRICULUM-ONLY -> not within sigma
  - pause:c2_shuffled under 'without the ProsQA cot SD term': CURRICULUM-ONLY -> not within sigma
  - card:c1b_k4 under 'without the ProsQA cot SD term': pass: pending K2; kill part: not met -> pass: not met on this task; kill part: not met
  - pause:coconut under 'floor only': CURRICULUM-ONLY -> not within sigma
  - pause:c1b_k8 under 'floor only': CURRICULUM-ONLY -> not within sigma
  - pause:c2 under 'floor only': CURRICULUM-ONLY -> not within sigma
  - pause:c2_shuffled under 'floor only': CURRICULUM-ONLY -> not within sigma
  - card:c1b_k8 under 'floor only': pass: pending K2; kill part: not met -> pass: not met on this task; kill part: not met
  - card:c1b_k4 under 'floor only': pass: pending K2; kill part: not met -> pass: not met on this task; kill part: met

## 5. Notes

- A6.7 item 2: ProsQA-hard 13-gram overlap vs its own train 37.0% valid, 36.0% test (template sentences over a tiny vocabulary); duplicate questions 0 and duplicate graphs 0 across splits; never read as leakage.
- Readings chosen (the alternatives are recomputed in section 4): sigma_screen as A3.4 is written (the ProsQA cot 3-seed SD term included); 'ahead by 1 sigma' (6.3) as a difference >= sigma; 'within sigma' as an absolute difference <= sigma, except C1b's pass ('within sigma of full depth') read as not below by more than sigma; CURRICULUM-ONLY (7.5) applied whenever the pause difference is within sigma; the fade's 0.5 B read at both neighbouring epochs when B is odd; the 7.2 legs on the selected epoch's logged means.
- Costs: |q| per example is exact (the hash-checked valid file through the pinned tokenizer); per-example emitted tokens were not recorded by training, so mean FLOPs and reference-cost FLOPs are printed as the exact range over every assignment of per-example E' in [0, max_new_tokens] with the logged total; a cost comparison is decided only when the ranges decide it. V as the arm's head: 50,257 (cot, no_cot), 50,260 (latent arms) (A11.2).
- Training FLOPs (5.5 (d)): the rows of record's `flops`; GSM-50k latent arms are also charged the lr 1e-4 cot run they start from (5.2). 'Same epochs' (training-heavy at > 1.25x) is read per epoch: the arm's charged FLOPs over the bar's per-epoch FLOPs times the arm's charged epoch count.
- Pending until the screen eval job runs on valid: the scrambled-at-inference, zero-latent and step-sweep reads (K2, the null triggers of 7.10), A6.2's timed read (f_meas for C1b), the batch-1 wall-clock (5.5 (b)) and the per-example cost files that make the FLOP means exact.
