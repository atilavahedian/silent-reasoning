# GSM-50k screen table: screen numbers (1 seed), valid split

GSM-50k: training on the first 50,000 records of a seed-0 permutation of the GSM8k-Aug train set (A3.4.5); valid = the full GSM8k-Aug valid set (500 items). Seed 0, GPT-2 124M, valid split only (A3.4); no test read; bf16 autocast over the TF32 flags (A10.5). Labels on every number and line: screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial; bar: single seed, LR-tuned (3 trials). Screen decisions use valid only (A3.4); screen cells are screen numbers, never verdicts.

## 1. Selection and valid accuracy

| run | role | pool (epochs) | selected epoch (stage) | valid | vs bar | all-epoch best (3.3 item 3) | resumes | GPU | start pin (A6.3 rule 4) | record checks |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm50k_cot_lr3e-5_s0 | bar trial: cot, lr 3e-5 | all | 23 (0) | 21.0 (105 of 500) | -4.6 |  | 1 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_cot_lr1e-4_s0 | bar trial: cot, lr 1e-4 | all | 25 (0) | 23.8 (119 of 500) | -1.8 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_cot_lr3e-4_s0 | bar trial: cot, lr 3e-4 | all | 17 (0) | 25.6 (128 of 500) | +0.0 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_no_cot_s0 | floor; the 124M plain-depth point (7.7) | all | 22 (0) | 6.2 (31 of 500) | -19.4 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_plain_depth_355m_s0 | plain depth: 355M no_cot (7.7, A13.2) | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |
| gsm50k_pause_s0 | control: pause-same-curriculum | 13-25 | 22 (7) | 9.6 (48 of 500) | -16.0 | 13.6 at epoch 5 (stage 1, n latents = 2): earlier stage better on valid | 1 | NVIDIA A100-SXM4-80GB | ok | ok (14) |
| gsm50k_coconut_s0 | control: full-depth coconut (C1b null; C2 / C3 base) | 13-25 | 17 (5) | 11.0 (55 of 500) | -14.6 | 12.6 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c1b_k8_s0 | card C1b, k = 8 (band [2, 10)) | 13-25 | 19 (6) | 9.4 (47 of 500) | -16.2 | 14.0 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c1b_k4_s0 | card C1b, k = 4 (band [4, 8)) | 13-25 | 15 (4) | 9.4 (47 of 500) | -16.2 | 14.6 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 3 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c2_s0 | card C2 | 13-25 | 20 (6) | 12.0 (60 of 500) | -13.6 | 12.6 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c2_shuffled_s0 | null: C2-shuffled | 13-25 | 17 (5) | 11.6 (58 of 500) | -14.0 | 13.0 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c3_s0 | card C3 | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |
| gsm50k_c3_foreign_s0 | null: C3-foreign | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |
| gsm50k_c2c3_s0 | card C2+C3 | | | UNTESTED | | | | | | no run record (RUN_DONE.json absent) |

## 2. Cost per valid example (5.1, 5.3, 5.5; A6.4)

E' and sequential passes are exact means from the selected epoch's log row; FLOPs (GFLOPs) are the exact range over the unrecorded per-example E' (notes). Training FLOPs in PFLOPs.

| run | L | E' mean | E' / bar | passes mean | passes / bar | FLOPs mean | FLOPs / bar | reference-cost FLOPs mean | reference / bar | training FLOPs own / charged | charged / bar (same epochs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm50k_cot_lr3e-5_s0 | 0 | 29.83 | 0.980 | 30.83 | 0.981 | [17.162, 17.206] | [0.989, 0.994] | [499.120, 800.036] | [0.614, 1.563] | 64.8 / 64.8 | 1.00 |
| gsm50k_cot_lr1e-4_s0 | 0 | 28.92 | 0.950 | 29.92 | 0.952 | [16.932, 16.976] | [0.976, 0.981] | [479.798, 780.255] | [0.590, 1.524] | 64.8 / 64.8 | 1.00 |
| gsm50k_cot_lr3e-4_s0 | 0 | 30.44 | 1.000 | 31.44 | 1.000 | [17.312, 17.357] | [0.997, 1.003] | [511.913, 812.749] | [0.630, 1.588] | 64.8 / 64.8 | 1.00 |
| gsm50k_no_cot_s0 | 0 | 2.21 | 0.072 | 3.21 | 0.102 | [10.261, 10.269] | [0.591, 0.593] | [35.270, 94.780] | [0.043, 0.185] | 43.3 / 43.3 | 0.67 |
| gsm50k_pause_s0 | 6 | 2.21 | 0.073 | 3.21 | 0.102 | [11.641, 11.650] | [0.671, 0.673] | [41.711, 101.367] | [0.051, 0.198] | 46.7 / 111.5 | 0.92 |
| gsm50k_coconut_s0 | 6 | 2.40 | 0.079 | 9.40 | 0.299 | [11.687, 11.697] | [0.673, 0.676] | [132.388, 196.232] | [0.163, 0.383] | 46.7 / 111.5 | 0.92 |
| gsm50k_c1b_k8_s0 | 6 | 2.28 | 0.075 | 9.28 | 0.295 | [11.314, 11.323] nominal; [11.658, 11.667] at 1 | [0.652, 0.654] nominal; [0.672, 0.674] at 1 | [130.899, 192.180] | [0.161, 0.375] | 45.8 / 110.6 | 0.91 |
| gsm50k_c1b_k4_s0 | 6 | 2.28 | 0.075 | 9.28 | 0.295 | [10.970, 10.979] nominal; [11.659, 11.668] at 1 | [0.632, 0.634] nominal; [0.672, 0.674] at 1 | [130.950, 192.343] | [0.161, 0.376] | 44.8 / 109.6 | 0.90 |
| gsm50k_c2_s0 | 6 | 2.31 | 0.076 | 9.31 | 0.296 | [11.666, 11.675] | [0.672, 0.674] | [131.279, 193.353] | [0.162, 0.378] | 48.9 / 113.7 | 0.93 |
| gsm50k_c2_shuffled_s0 | 6 | 2.23 | 0.073 | 9.23 | 0.294 | [11.646, 11.655] | [0.671, 0.673] | [130.266, 190.317] | [0.160, 0.372] | 48.9 / 113.7 | 0.93 |

## 3. Registered lines

### Bar
- Bar (GSM-50k, valid): gsm50k_cot_lr3e-4_s0 25.6 (128 of 500) at epoch 17; trials gsm50k_cot_lr3e-5_s0 21.0, gsm50k_cot_lr1e-4_s0 23.8, gsm50k_cot_lr3e-4_s0 25.6 [bar: single seed, LR-tuned (3 trials); screen numbers (1 seed)]

### sigma_screen
- sigma_screen = 3.60 points (A3.4: max(1.0, 100 sqrt(p (1 - p) / n_valid) at the bar's p = 128/500, the ProsQA cot 3-seed SD once it exists))

### Screen rule 6.3 (card arms)
- Screen rule (6.3), gsm50k_c1b_k8_s0 (C1b): does not advance. Legs: bar: not met (9.4 >= 25.6 - 3.60 = 22.00); scrambled null + sigma: pending (no scrambled-at-inference valid read); pause + sigma: not met (-0.2 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Screen rule (6.3), gsm50k_c1b_k4_s0 (C1b): does not advance. Legs: bar: not met (9.4 >= 25.6 - 3.60 = 22.00); scrambled null + sigma: pending (no scrambled-at-inference valid read); pause + sigma: not met (-0.2 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Screen rule (6.3), gsm50k_c2_s0 (C2): does not advance. Legs: bar: not met (12.0 >= 25.6 - 3.60 = 22.00); scrambled null + sigma: pending (no scrambled-at-inference valid read); pause + sigma: not met (+2.4 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Card lines (A3.4)
- C1b k = 8 (card line, A3.4), gsm50k_c1b_k8_s0 - gsm50k_coconut_s0 = -1.6: pass needs within sigma_screen 3.60 of full-depth coconut (or better) and K2 not fired: accuracy part met, K2 pending, pass pending K2; kill needs both tasks: this task's accuracy part (below by more than 2 sigma_screen = 7.20) not met, K2 part pending; the other screen task is not in this table; per-step cost nominal [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C1b k = 4 (card line, A3.4), gsm50k_c1b_k4_s0 - gsm50k_coconut_s0 = -1.6: pass needs within sigma_screen 3.60 of full-depth coconut (or better) and K2 not fired: accuracy part met, K2 pending, pass pending K2; kill needs both tasks: this task's accuracy part (below by more than 2 sigma_screen = 7.20) not met, K2 part pending; the other screen task is not in this table; per-step cost nominal [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C2 (card line, A3.4), gsm50k_c2_s0 - gsm50k_c2_shuffled_s0 = +0.4 (the gap includes learnable vs unlearnable auxiliary target), gsm50k_c2_s0 - gsm50k_coconut_s0 = +1.0: pass needs above C2-shuffled by more than sigma_screen 3.60 with K2 not fired: not met on this task; kill needs one clause on both tasks: this task's parts: within sigma_screen of C2-shuffled met, below full-depth coconut by more than 2 sigma_screen not met, K2 pending; the other screen task is not in this table [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C3 (A3.4): UNTESTED on GSM-50k (no run record)
- C2+C3 (A3.4): UNTESTED on GSM-50k (no run record)

### K2 (7.5; A3.4)
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_coconut_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c1b_k8_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c1b_k4_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c2_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c2_shuffled_s0: pending: no scrambled-at-inference read on valid exists for its selected checkpoint [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### K3 (7.5; 7.7; A13.2)
- K3 (7.5 at the screen), gsm50k_pause_s0: 9.6 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired; no better than plain depth (7.5) [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_coconut_s0: 11.0 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c1b_k8_s0: 9.4 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired; no better than plain depth (7.5) [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c1b_k4_s0: 9.4 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired; no better than plain depth (7.5) [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c2_s0: 12.0 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c2_shuffled_s0: 11.6 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Pause comparison (7.5)
- Pause comparison (7.5), gsm50k_coconut_s0 - gsm50k_pause_s0 = +1.4: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c1b_k8_s0 - gsm50k_pause_s0 = -0.2: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c1b_k4_s0 - gsm50k_pause_s0 = -0.2: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c2_s0 - gsm50k_pause_s0 = +2.4: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c2_shuffled_s0 - gsm50k_pause_s0 = +2.0: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### 7.2 cost legs and 5.5
- 7.2 cost legs, gsm50k_pause_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.671, 0.673]); token leg (E' <= 0.2 x bar's) met (2.21 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) met (3.21 vs 6.29); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_coconut_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.673, 0.676]); token leg (E' <= 0.2 x bar's) met (2.40 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.40 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c1b_k8_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.672, 0.674]; nominal layer fraction: [0.652, 0.654]); token leg (E' <= 0.2 x bar's) met (2.28 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.28 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c1b_k4_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.672, 0.674]; nominal layer fraction: [0.632, 0.634]); token leg (E' <= 0.2 x bar's) met (2.28 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.28 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c2_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.672, 0.674]); token leg (E' <= 0.2 x bar's) met (2.31 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.31 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c2_shuffled_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (ratio in [0.671, 0.673]); token leg (E' <= 0.2 x bar's) met (2.23 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.23 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: UNTESTED [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### C1b per-step cost (A3.4; A6.2)
- C1b per-step cost, gsm50k_c1b_k8_s0: nominal: f_meas UNTESTED (A6.2's timed read, band [0, 12] on the full-depth coconut checkpoint, has not run; the training-time val_latent_step_ms rows are not A6.2's fraction, rule 1); saving line f_meas <= 0.77; FLOPs printed at the nominal 8/12 and at 1 (the largest min(1, f_meas)); the FLOP leg is read at 1 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- C1b per-step cost, gsm50k_c1b_k4_s0: nominal: f_meas UNTESTED (A6.2's timed read, band [0, 12] on the full-depth coconut checkpoint, has not run; the training-time val_latent_step_ms rows are not A6.2's fraction, rule 1); saving line f_meas <= 0.43; FLOPs printed at the nominal 4/12 and at 1 (the largest min(1, f_meas)); the FLOP leg is read at 1 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### C2 labels (A5 item 5; A6.7 items 4, 5)
- gsm50k_c2_s0: A5 item 5: the last supervised target contains the answer in 0.770 of the GSM-50k records at 3 latent steps; printed beside every C2 number; config.json _state_targets answer_in_last_target_frac 0.76956 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- gsm50k_c2_shuffled_s0: A6.7 item 4: on GSM-50k the C2-shuffled donor shares the answer 430 of 50,000 = 0.86% (chance); config.json _state_targets answer_in_last_target_frac 0.76956, answer_in_last_used_target_frac 0.00878 [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- train_state_loss by epoch 4-25 (A6.7 item 5): C2 3.616 2.491 1.991 2.461 2.164 1.980 2.219 2.081 1.975 1.936 1.866 1.803 1.741 1.685 1.632 1.583 1.536 1.490 1.446 1.408 1.360 1.331; C2-shuffled 4.372 4.172 4.145 4.229 4.184 4.170 4.219 4.198 4.184 4.174 4.163 4.145 4.127 4.108 4.087 4.065 4.042 4.017 3.991 3.965 3.937 3.905; every C2 minus C2-shuffled sentence carries: the gap includes learnable vs unlearnable auxiliary target [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Too-good triggers (7.10; A3.4)
- Too-good triggers (7.10; A3.4), any arm more than 10 points over the bar: none; the null triggers (scrambled-at-inference, zero-latent) are pending their valid reads [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### Fade instrument (7.6)
- Fade instrument (7.6), gsm50k_pause_s0: lead over the bar's curve at epoch 12 -16.4, epoch 13 -16.6, epoch 25 -14.2 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_coconut_s0: lead over the bar's curve at epoch 12 -13.8, epoch 13 -14.8, epoch 25 -14.4 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c1b_k8_s0: lead over the bar's curve at epoch 12 -14.0, epoch 13 -14.8, epoch 25 -14.8 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c1b_k4_s0: lead over the bar's curve at epoch 12 -15.8, epoch 13 -16.4, epoch 25 -14.4 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c2_s0: lead over the bar's curve at epoch 12 -13.4, epoch 13 -14.4, epoch 25 -11.8 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c2_shuffled_s0: lead over the bar's curve at epoch 12 -13.2, epoch 13 -12.6, epoch 25 -14.2 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_no_cot_s0: lead over the bar's curve at epoch 12 -17.8, epoch 13 -18.8, epoch 25 -17.6 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); bar LR-tuned 3 trials; arm 1 trial]

### UNTESTED and excluded
- gsm50k_plain_depth_355m_s0 (plain depth: 355M no_cot (7.7, A13.2)): UNTESTED: no run record (RUN_DONE.json absent); enters no cell
- gsm50k_c3_s0 (card C3): UNTESTED: no run record (RUN_DONE.json absent); enters no cell
- gsm50k_c3_foreign_s0 (null: C3-foreign): UNTESTED: no run record (RUN_DONE.json absent); enters no cell
- gsm50k_c2c3_s0 (card C2+C3): UNTESTED: no run record (RUN_DONE.json absent); enters no cell

### Receipts and launch condition (A3.2 (i); A5 item 8)
- env_reference per receipt: silent-train-20261019-153923-d3bb ok, silent-train-20261019-154904-c083 ok, silent-train-20261019-160323-4299 ok, silent-train-20261019-172741-7fff ok, silent-train-20261019-203810-7ebd ok, silent-train-20261019-204820-d318 ok, silent-train-20261020-025152-49e0 ok, silent-train-20261020-030355-f2b5 ok, silent-train-20261020-031905-622a ok
- attempts named by the runs without a receipt in this read: none
- A5 item 8 (the bar's three cot trials admitted no later than the first latent arm, from the receipts' admissions): met

## 4. Readings and sensitivity

- sigma_screen of record 3.6019 (terms: floor 1.0, binomial 1.9517, ProsQA cot 3-seed SD 3.601851375797361); alternatives recomputed: without the ProsQA cot SD term 1.9517, floor only 1.0000.
- Lines whose outcome changes under an alternative reading:
  - k3:pause under 'without the ProsQA cot SD term': not fired; no better than plain depth -> not fired
  - k3:c1b_k8 under 'without the ProsQA cot SD term': not fired; no better than plain depth -> not fired
  - k3:c1b_k4 under 'without the ProsQA cot SD term': not fired; no better than plain depth -> not fired
  - pause:c2 under 'without the ProsQA cot SD term': CURRICULUM-ONLY -> not within sigma
  - pause:c2_shuffled under 'without the ProsQA cot SD term': CURRICULUM-ONLY -> not within sigma
  - k3:pause under 'floor only': not fired; no better than plain depth -> not fired
  - k3:c1b_k8 under 'floor only': not fired; no better than plain depth -> not fired
  - k3:c1b_k4 under 'floor only': not fired; no better than plain depth -> not fired
  - pause:coconut under 'floor only': CURRICULUM-ONLY -> not within sigma
  - pause:c2 under 'floor only': CURRICULUM-ONLY -> not within sigma
  - pause:c2_shuffled under 'floor only': CURRICULUM-ONLY -> not within sigma
  - card:c1b_k8 under 'floor only': pass: pending K2; kill part: not met -> pass: not met on this task; kill part: not met
  - card:c1b_k4 under 'floor only': pass: pending K2; kill part: not met -> pass: not met on this task; kill part: not met
  - fade:c2_shuffled under 'floor only': not fading -> fading at one 0.5 B reading

## 5. Notes

- A2 item 4 (c): the GSM valid set shares a 13-gram with the full GSM8k-Aug train on 7 of 500 items (the 50k subset at most that); a known bias, valid slightly optimistic, equally for every arm.
- Readings chosen (the alternatives are recomputed in section 4): sigma_screen as A3.4 is written (the ProsQA cot 3-seed SD term included); 'ahead by 1 sigma' (6.3) as a difference >= sigma; 'within sigma' as an absolute difference <= sigma, except C1b's pass ('within sigma of full depth') read as not below by more than sigma; CURRICULUM-ONLY (7.5) applied whenever the pause difference is within sigma; the fade's 0.5 B read at both neighbouring epochs when B is odd; the 7.2 legs on the selected epoch's logged means.
- Costs: |q| per example is exact (the hash-checked valid file through the pinned tokenizer); per-example emitted tokens were not recorded by training, so mean FLOPs and reference-cost FLOPs are printed as the exact range over every assignment of per-example E' in [0, max_new_tokens] with the logged total; a cost comparison is decided only when the ranges decide it. V as the arm's head: 50,257 (cot, no_cot), 50,260 (latent arms) (A11.2).
- Training FLOPs (5.5 (d)): the rows of record's `flops`; GSM-50k latent arms are also charged the lr 1e-4 cot run they start from (5.2). 'Same epochs' (training-heavy at > 1.25x) is read per epoch: the arm's charged FLOPs over the bar's per-epoch FLOPs times the arm's charged epoch count.
- Pending until the screen eval job runs on valid: the scrambled-at-inference, zero-latent and step-sweep reads (K2, the null triggers of 7.10), A6.2's timed read (f_meas for C1b), the batch-1 wall-clock (5.5 (b)) and the per-example cost files that make the FLOP means exact.
