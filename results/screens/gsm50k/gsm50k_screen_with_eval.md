# GSM-50k screen table with the eval reads: screen numbers (1 seed), valid split

GSM-50k: training on the first 50,000 records of a seed-0 permutation of the GSM8k-Aug train set (A3.4.5); valid = the full GSM8k-Aug valid set (500 items). Seed 0, GPT-2 124M; bf16 autocast over the TF32 flags (A10.5). Every line carries: screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial. Screen decisions use valid only (A3.4); screen cells are screen numbers, never verdicts.

The second write-once table (A6.4's reissue pattern; critic entry 2026-10-20T10:51Z part (6)). It cites the first table `results/screens/gsm50k/gsm50k_screen.json, results/screens/gsm50k/gsm50k_screen.md, results/screens/gsm50k/gsm50k_screen_provenance.json` written once at `rev10` (a6e282ceb7bbc7c2e60524f91d88a8ad69c7b847; files unchanged since: yes), and the eval reads `results/evals/screen-eval-gsm50k-main` of job `screen-eval-gsm50k-main`, attempt `silent-eval-20261020-112122-f7da`, registered read row 2026-10-20T10:51Z, results commit `rev13` (45 entries; receipt sha256 5cc22e035ec8...). The first table's files are not modified.

## 1. Selection, valid accuracy and the plain eval read

| run | role | pool: epochs (epoch_index) | selected epoch (stage) | valid, training log (selection) | plain read, eval | vs bar | all-epoch best (3.3 item 3) | resumes | GPU (training) | start pin (A6.3 rule 4) | record checks |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm50k_cot_lr3e-5_s0 | bar trial: cot, lr 3e-5 | all | 23 (0) | 21.0 (105 of 500) | no eval read | -4.6 |  | 1 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_cot_lr1e-4_s0 | bar trial: cot, lr 1e-4 | all | 25 (0) | 23.8 (119 of 500) | no eval read | -1.8 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_cot_lr3e-4_s0 | bar trial: cot, lr 3e-4 | all | 17 (0) | 25.6 (128 of 500) | no eval read | +0.0 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_no_cot_s0 | floor; the 124M plain-depth point (7.7) | all | 22 (0) | 6.2 (31 of 500) | no eval read | -19.4 |  | 0 | NVIDIA A100-SXM4-80GB |  | ok (13) |
| gsm50k_plain_depth_355m_s0 | plain depth: 355M no_cot (7.7, A13.2) | | | UNTESTED | | | | | | | no run record (RUN_DONE.json absent) |
| gsm50k_pause_s0 | control: pause-same-curriculum | 13-25 (12-24) | 22 (7) | 9.6 (48 of 500) | 9.6 (48 of 500): agrees | -16.0 | 13.6 at epoch 5 (stage 1, n latents = 2): earlier stage better on valid | 1 | NVIDIA A100-SXM4-80GB | ok | ok (14) |
| gsm50k_coconut_s0 | control: full-depth coconut (C1b null; C2 / C3 base) | 13-25 (12-24) | 17 (5) | 11.0 (55 of 500) | 11.0 (55 of 500): agrees | -14.6 | 12.6 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c1b_k8_s0 | card C1b, k = 8 (band [2, 10)) | 13-25 (12-24) | 19 (6) | 9.4 (47 of 500) | 9.4 (47 of 500): agrees | -16.2 | 14.0 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c1b_k4_s0 | card C1b, k = 4 (band [4, 8)) | 13-25 (12-24) | 15 (4) | 9.4 (47 of 500) | 9.4 (47 of 500): agrees | -16.2 | 14.6 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 3 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c2_s0 | card C2 | 13-25 (12-24) | 20 (6) | 12.0 (60 of 500) | 12.0 (60 of 500): agrees | -13.6 | 12.6 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c2_shuffled_s0 | null: C2-shuffled | 13-25 (12-24) | 17 (5) | 11.6 (58 of 500) | 11.6 (58 of 500): agrees | -14.0 | 13.0 at epoch 6 (stage 1, n latents = 2): earlier stage better on valid | 0 | NVIDIA A100-SXM4-80GB | ok | ok (15) |
| gsm50k_c3_s0 | card C3 | | | UNTESTED | | | | | | | no run record (RUN_DONE.json absent) |
| gsm50k_c3_foreign_s0 | null: C3-foreign | | | UNTESTED | | | | | | | no run record (RUN_DONE.json absent) |
| gsm50k_c2c3_s0 | card C2+C3 | | | UNTESTED | | | | | | | no run record (RUN_DONE.json absent) |

## 2. Cost per valid example (5.1, 5.3, 5.5; A6.4)

Latent arms: exact means from the plain eval read's per-example file (E', passes, FLOPs and reference-cost FLOPs; every row recomputed by `accounting.inference_record` from the hashed valid |q| and the row's own L and E' equals the file). `cot` and `no_cot` have no per-example read: their FLOPs stay the first table's exact range. L is latent steps; pause's 6 are pause positions (5.3), not latent steps. GFLOPs; training FLOPs in PFLOPs.

| run | L | E' mean | E' / bar | passes mean | passes / bar | FLOPs mean | FLOPs / bar | first table's FLOPs range | reference-cost FLOPs mean | reference / bar | first table's reference range | training FLOPs own / charged | charged / bar (same epochs) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gsm50k_cot_lr3e-5_s0 | 0 | 29.834 | 0.980 | 30.834 | 0.981 | [17.162, 17.206] | [0.989, 0.994] | (same) | [499.120, 800.036] | [0.614, 1.563] | (same) | 64.8 / 64.8 | 1.00 |
| gsm50k_cot_lr1e-4_s0 | 0 | 28.916 | 0.950 | 29.916 | 0.952 | [16.932, 16.976] | [0.976, 0.981] | (same) | [479.798, 780.255] | [0.590, 1.524] | (same) | 64.8 / 64.8 | 1.00 |
| gsm50k_cot_lr3e-4_s0 | 0 | 30.436 | 1.000 | 31.436 | 1.000 | [17.312, 17.357] | [0.997, 1.003] | (same) | [511.913, 812.749] | [0.630, 1.588] | (same) | 64.8 / 64.8 | 1.00 |
| gsm50k_no_cot_s0 | 0 | 2.206 | 0.072 | 3.206 | 0.102 | [10.261, 10.269] | [0.591, 0.593] | (same) | [35.270, 94.780] | [0.043, 0.185] | (same) | 43.3 / 43.3 | 0.67 |
| gsm50k_pause_s0 | 6 (pause positions) | 2.210 | 0.073 | 3.210 | 0.102 | 11.643 | [0.671, 0.673] | [11.641, 11.650] | 52.644 | [0.065, 0.103] | [41.711, 101.367] | 46.7 / 111.5 | 0.92 |
| gsm50k_coconut_s0 | 6 | 2.396 | 0.079 | 9.396 | 0.299 | 11.689 | [0.673, 0.675] | [11.687, 11.697] | 145.338 | [0.179, 0.284] | [132.388, 196.232] | 46.7 / 111.5 | 0.92 |
| gsm50k_c1b_k8_s0 | 6 | 2.280 | 0.075 | 9.280 | 0.295 | 11.315 nominal 0.6667; 11.371 measured 0.7200; 11.660 at 1 | [0.652, 0.654] nominal 0.6667; [0.655, 0.657] measured 0.7200; [0.672, 0.674] at 1 | [11.314, 11.323] nominal; [11.658, 11.667] at 1 | 141.963 | [0.175, 0.277] | [130.899, 192.180] | 45.8 / 110.6 | 0.91 |
| gsm50k_c1b_k4_s0 | 6 | 2.284 | 0.075 | 9.284 | 0.295 | 10.972 nominal 0.3333; 11.074 measured 0.4326; 11.661 at 1 | [0.632, 0.634] nominal 0.3333; [0.638, 0.640] measured 0.4326; [0.672, 0.674] at 1 | [10.970, 10.979] nominal; [11.659, 11.668] at 1 | 141.971 | [0.175, 0.277] | [130.950, 192.343] | 44.8 / 109.6 | 0.90 |
| gsm50k_c2_s0 | 6 | 2.310 | 0.076 | 9.310 | 0.296 | 11.668 | [0.672, 0.674] | [11.666, 11.675] | 143.060 | [0.176, 0.279] | [131.279, 193.353] | 48.9 / 113.7 | 0.93 |
| gsm50k_c2_shuffled_s0 | 6 | 2.230 | 0.073 | 9.230 | 0.294 | 11.648 | [0.671, 0.673] | [11.646, 11.655] | 141.074 | [0.174, 0.276] | [130.266, 190.317] | 48.9 / 113.7 | 0.93 |

## 3. The eval reads

### 3.1 Accuracy on valid per read (points; k of 500 in the json)

| run | plain read (n = trained) | scrambled | zero-latent | n = 0 | n = 1 | n = 2 | n = 4 | n = 8 (out-of-distribution) | n = 0 against zero-latent |
|---|---|---|---|---|---|---|---|---|---|
| gsm50k_pause_s0 | 9.6 |  |  |  |  |  |  |  |  |
| gsm50k_coconut_s0 | 11.0 | 4.0 | 1.0 | 1.0 | 7.6 | 9.0 | 10.6 | 11.6 | agree on every example |
| gsm50k_c1b_k8_s0 | 9.4 | 5.4 | 8.0 | 8.0 | 8.8 | 8.8 | 9.6 | 9.4 | agree on every example |
| gsm50k_c1b_k4_s0 | 9.4 | 8.0 | 8.2 | 8.2 | 8.6 | 8.8 | 9.2 | 9.6 | agree on every example |
| gsm50k_c2_s0 | 12.0 | 3.4 | 1.0 | 1.0 | 5.8 | 9.2 | 11.2 | 12.0 | agree on every example |
| gsm50k_c2_shuffled_s0 | 11.6 | 3.0 | 1.6 | 1.6 | 6.2 | 6.6 | 11.0 | 11.8 | agree on every example |

### 3.2 K2 statistics (6.2 paired bootstrap: 10,000 draws, default_rng(0)); sigma_screen 3.60

| run | Delta_scr | 95% interval (Delta +- 1.96 SE) | percentile interval | correct only own / only scrambled | K2 at the screen | zero-latent drop [interval] |
|---|---|---|---|---|---|---|
| gsm50k_coconut_s0 | +7.0 | [+4.11, +9.89] | [+4.2, +10.0] | 46 / 11 | not fired | +10.0 [+7.24, +12.76] |
| gsm50k_c1b_k8_s0 | +4.0 | [+1.70, +6.30] | [+1.8, +6.4] | 28 / 8 | not fired | +1.4 [-0.90, +3.70] |
| gsm50k_c1b_k4_s0 | +1.4 | [-0.12, +2.92] | [+0.0, +3.0] | 11 / 4 | fired | +1.2 [-0.03, +2.43] |
| gsm50k_c2_s0 | +8.6 | [+5.70, +11.50] | [+5.8, +11.6] | 51 / 8 | not fired | +11.0 [+8.19, +13.81] |
| gsm50k_c2_shuffled_s0 | +8.6 | [+5.74, +11.46] | [+5.8, +11.6] | 50 / 7 | not fired | +10.0 [+7.00, +13.00] |

### 3.3 Timed reads (A6.2; 5.5 (b)): ms per pass from each summary's timing block (the plain reads, the batch-1 reads and the band [0, 12] reads; the sweep and intervention reads' timings are in the json)

| entry | path | batch | timed batches | prefill ms | latent ms (passes) | token ms (passes) | latent_last | device | torch | read seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| gsm50k_coconut_s0_valid_band0-12_timed_bs1 | band [0, 12] | 1 | 500 | 10.994 | 11.120 (3000) | 10.894 (1196) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 60.02 |
| gsm50k_coconut_s0_valid_band0-12_timed_bs32 | band [0, 12] | 32 | 16 | 34.829 | 12.046 (96) | 11.274 (124) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 3.85 |
| gsm50k_coconut_s0_valid_none | HF path | 32 | 16 | 42.340 | 16.543 (80) | 15.912 (124) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 4.73 |
| gsm50k_c1b_k8_s0_valid_none | band [2, 10] | 32 | 16 | 34.669 | 8.909 (96) | 11.341 (66) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 2.92 |
| gsm50k_c1b_k8_s0_valid_timed_bs1 | band [2, 10] | 1 | 500 | 10.983 | 8.006 (3000) | 10.914 (1136) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 50.07 |
| gsm50k_c1b_k4_s0_valid_none | band [4, 8] | 32 | 16 | 35.032 | 5.764 (96) | 11.730 (67) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 2.72 |
| gsm50k_c1b_k4_s0_valid_timed_bs1 | band [4, 8] | 1 | 500 | 11.013 | 4.810 (3000) | 10.949 (1143) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 40.60 |
| gsm50k_c2_s0_valid_none | HF path | 32 | 16 | 41.688 | 16.443 (80) | 15.906 (87) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 4.16 |
| gsm50k_c2_shuffled_s0_valid_none | HF path | 32 | 16 | 42.409 | 16.769 (80) | 16.231 (62) | not recorded | NVIDIA A100-SXM4-40GB | 2.14.1+cu130 | 3.78 |

### 3.4 Step sweep costs (7.3: accuracy against mean FLOPs, passes and emitted tokens; decides nothing)

| run | n | accuracy | E' mean | passes mean | FLOPs mean (GFLOPs) | entry |
|---|---|---|---|---|---|---|
| gsm50k_pause_s0 | 6 (plain read) | 9.6 | 2.210 | 3.210 | 11.643 | gsm50k_pause_s0_valid_none |
| gsm50k_coconut_s0 | 0 | 1.0 | 5.422 | 6.422 | 11.410 | gsm50k_coconut_s0_valid_n0 |
| gsm50k_coconut_s0 | 1 | 7.6 | 2.504 | 4.504 | 10.854 | gsm50k_coconut_s0_valid_n1 |
| gsm50k_coconut_s0 | 2 | 9.0 | 2.274 | 5.274 | 10.969 | gsm50k_coconut_s0_valid_n2 |
| gsm50k_coconut_s0 | 4 | 10.6 | 2.280 | 7.280 | 11.315 | gsm50k_coconut_s0_valid_n4 |
| gsm50k_coconut_s0 | 6 (plain read) | 11.0 | 2.396 | 9.396 | 11.689 | gsm50k_coconut_s0_valid_none |
| gsm50k_coconut_s0 | 8 (out-of-distribution) | 11.6 | 2.262 | 11.262 | 12.001 | gsm50k_coconut_s0_valid_n8 |
| gsm50k_c1b_k8_s0 | 0 | 8.0 | 2.254 | 3.254 | 10.619 | gsm50k_c1b_k8_s0_valid_n0 |
| gsm50k_c1b_k8_s0 | 1 | 8.8 | 2.264 | 4.264 | 10.736 | gsm50k_c1b_k8_s0_valid_n1 |
| gsm50k_c1b_k8_s0 | 2 | 8.8 | 2.274 | 5.274 | 10.854 | gsm50k_c1b_k8_s0_valid_n2 |
| gsm50k_c1b_k8_s0 | 4 | 9.6 | 2.260 | 7.260 | 11.080 | gsm50k_c1b_k8_s0_valid_n4 |
| gsm50k_c1b_k8_s0 | 6 (plain read) | 9.4 | 2.280 | 9.280 | 11.315 | gsm50k_c1b_k8_s0_valid_none |
| gsm50k_c1b_k8_s0 | 8 (out-of-distribution) | 9.4 | 2.282 | 11.282 | 11.546 | gsm50k_c1b_k8_s0_valid_n8 |
| gsm50k_c1b_k4_s0 | 0 | 8.2 | 2.296 | 3.296 | 10.629 | gsm50k_c1b_k4_s0_valid_n0 |
| gsm50k_c1b_k4_s0 | 1 | 8.6 | 2.306 | 4.306 | 10.689 | gsm50k_c1b_k4_s0_valid_n1 |
| gsm50k_c1b_k4_s0 | 2 | 8.8 | 2.292 | 5.292 | 10.743 | gsm50k_c1b_k4_s0_valid_n2 |
| gsm50k_c1b_k4_s0 | 4 | 9.2 | 2.292 | 7.292 | 10.859 | gsm50k_c1b_k4_s0_valid_n4 |
| gsm50k_c1b_k4_s0 | 6 (plain read) | 9.4 | 2.284 | 9.284 | 10.972 | gsm50k_c1b_k4_s0_valid_none |
| gsm50k_c1b_k4_s0 | 8 (out-of-distribution) | 9.6 | 2.276 | 11.276 | 11.085 | gsm50k_c1b_k4_s0_valid_n8 |
| gsm50k_c2_s0 | 0 | 1.0 | 5.884 | 6.884 | 11.525 | gsm50k_c2_s0_valid_n0 |
| gsm50k_c2_s0 | 1 | 5.8 | 2.392 | 4.392 | 10.826 | gsm50k_c2_s0_valid_n1 |
| gsm50k_c2_s0 | 2 | 9.2 | 2.312 | 5.312 | 10.978 | gsm50k_c2_s0_valid_n2 |
| gsm50k_c2_s0 | 4 | 11.2 | 2.386 | 7.386 | 11.342 | gsm50k_c2_s0_valid_n4 |
| gsm50k_c2_s0 | 6 (plain read) | 12.0 | 2.310 | 9.310 | 11.668 | gsm50k_c2_s0_valid_none |
| gsm50k_c2_s0 | 8 (out-of-distribution) | 12.0 | 2.306 | 11.306 | 12.012 | gsm50k_c2_s0_valid_n8 |
| gsm50k_c2_shuffled_s0 | 0 | 1.6 | 5.166 | 6.166 | 11.346 | gsm50k_c2_shuffled_s0_valid_n0 |
| gsm50k_c2_shuffled_s0 | 1 | 6.2 | 2.238 | 4.238 | 10.787 | gsm50k_c2_shuffled_s0_valid_n1 |
| gsm50k_c2_shuffled_s0 | 2 | 6.6 | 2.236 | 5.236 | 10.959 | gsm50k_c2_shuffled_s0_valid_n2 |
| gsm50k_c2_shuffled_s0 | 4 | 11.0 | 2.216 | 7.216 | 11.299 | gsm50k_c2_shuffled_s0_valid_n4 |
| gsm50k_c2_shuffled_s0 | 6 (plain read) | 11.6 | 2.230 | 9.230 | 11.648 | gsm50k_c2_shuffled_s0_valid_none |
| gsm50k_c2_shuffled_s0 | 8 (out-of-distribution) | 11.8 | 2.226 | 11.226 | 11.992 | gsm50k_c2_shuffled_s0_valid_n8 |

### 3.5 Entry checks: 45 of 45 entries pass every check

Per entry: the receipt row (done, rc 0); summary.json, examples.csv and eval.log hashed (provenance); examples.csv equal to the summary's hash; the valid split's hash equal to the manifest's; one row per example in file order; the summary's accuracy and means equal to the file's; precision bf16 with A10.5's flags (A10.1); the registered row; the checkpoint sha256 equal to the run's RUN_DONE selected file and named by the row; the selected epoch named by the entry; run, seed, config, intervention and n_latent; the band; every row's 5.1 / 5.3 cost.

## 4. Registered lines

### Bar
- Bar (GSM-50k, valid): gsm50k_cot_lr3e-4_s0 25.6 (128 of 500) at epoch 17; trials gsm50k_cot_lr3e-5_s0 21.0, gsm50k_cot_lr1e-4_s0 23.8, gsm50k_cot_lr3e-4_s0 25.6 [bar: single seed, LR-tuned (3 trials); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### sigma_screen
- sigma_screen = 3.60 points (A3.4: max(1.0, 100 sqrt(p (1 - p) / n_valid) at the bar's p = 128/500, the ProsQA cot 3-seed SD once it exists)) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### The eval reads: job, entry checks, plain read against the selection
- Eval reads: job screen-eval-gsm50k-main, attempt silent-eval-20261020-112122-f7da, registered row 2026-10-20T10:51Z, receipt complete (2026-10-20T11:27:07Z to 2026-10-20T11:49:43Z, code rev183, env_reference ok): 45 of 45 entries pass every check (receipt row, file hashes, valid split hash, A10.5 numerics, the selected checkpoint's sha256 and epoch, the 5.1 / 5.3 cost of every row) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Plain read against the selection, gsm50k_pause_s0: gsm50k_pause_s0_valid_none 48 of 500 (9.6) vs the training log's selection 48 of 500 (9.6, epoch 22; same checkpoint, same numerics): agree; emitted tokens 1105 vs the log's 1105 (equal) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Plain read against the selection, gsm50k_coconut_s0: gsm50k_coconut_s0_valid_none 55 of 500 (11.0) vs the training log's selection 55 of 500 (11.0, epoch 17; same checkpoint, same numerics): agree; emitted tokens 1198 vs the log's 1198 (equal) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Plain read against the selection, gsm50k_c1b_k8_s0: gsm50k_c1b_k8_s0_valid_none 47 of 500 (9.4) vs the training log's selection 47 of 500 (9.4, epoch 19; same checkpoint, same numerics): agree; emitted tokens 1140 vs the log's 1140 (equal) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Plain read against the selection, gsm50k_c1b_k4_s0: gsm50k_c1b_k4_s0_valid_none 47 of 500 (9.4) vs the training log's selection 47 of 500 (9.4, epoch 15; same checkpoint, same numerics): agree; emitted tokens 1142 vs the log's 1142 (equal) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Plain read against the selection, gsm50k_c2_s0: gsm50k_c2_s0_valid_none 60 of 500 (12.0) vs the training log's selection 60 of 500 (12.0, epoch 20; same checkpoint, same numerics): agree; emitted tokens 1155 vs the log's 1155 (equal) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Plain read against the selection, gsm50k_c2_shuffled_s0: gsm50k_c2_shuffled_s0_valid_none 58 of 500 (11.6) vs the training log's selection 58 of 500 (11.6, epoch 17; same checkpoint, same numerics): agree; emitted tokens 1115 vs the log's 1115 (equal) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Screen rule 6.3 (card arms)
- Screen rule (6.3), gsm50k_c1b_k8_s0 (C1b): does not advance. Legs: bar: not met (9.4 >= 25.6 - 3.60 = 22.00); scrambled null + sigma: met (Delta_scr +4.0 vs 3.60); pause + sigma: not met (-0.2 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Screen rule (6.3), gsm50k_c1b_k4_s0 (C1b): does not advance. Legs: bar: not met (9.4 >= 25.6 - 3.60 = 22.00); scrambled null + sigma: not met (Delta_scr +1.4 vs 3.60); pause + sigma: not met (-0.2 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Screen rule (6.3), gsm50k_c2_s0 (C2): does not advance. Legs: bar: not met (12.0 >= 25.6 - 3.60 = 22.00); scrambled null + sigma: met (Delta_scr +8.6 vs 3.60); pause + sigma: not met (+2.4 vs 3.60); FLOP leg: met (7.2 leg 2); token leg: met (7.2 leg 3) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Card lines (A3.4)
- C1b k = 8 (card line, A3.4), gsm50k_c1b_k8_s0 - gsm50k_coconut_s0 = -1.6: pass needs within sigma_screen 3.60 of full-depth coconut (or better) and K2 not fired: accuracy part met, K2 not fired (Delta_scr +4.0, interval [+1.7, +6.3]): card pass line met on this task (6.3's screen rule is its own line); kill needs both tasks (below full-depth coconut by more than 2 sigma_screen = 7.20, or K2 fired): this task's accuracy part not met, K2 part not met; the other screen task is not in this table; per-step cost: f_meas 0.7200 at batch 1 against the line 0.77: threshold met; no saving label (rule 3 logits check pending) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- C1b k = 4 (card line, A3.4), gsm50k_c1b_k4_s0 - gsm50k_coconut_s0 = -1.6: pass needs within sigma_screen 3.60 of full-depth coconut (or better) and K2 not fired: accuracy part met, K2 fired (Delta_scr +1.4, interval [-0.1, +2.9]): card pass line not met on this task: K2 fired (6.3's screen rule is its own line); K2 fired at the screen: the arm's latent claim is killed at the screen on this task (A3.4); CURRICULUM-ONLY (pause comparison -0.2 within sigma_screen 3.60); kill needs both tasks (below full-depth coconut by more than 2 sigma_screen = 7.20, or K2 fired): this task's accuracy part not met, K2 part met; the other screen task is not in this table; per-step cost: f_meas 0.4326 at batch 1 against the line 0.43: nominal; no saving label (rule 3 logits check pending) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- C2 (card line, A3.4), gsm50k_c2_s0 - gsm50k_c2_shuffled_s0 = +0.4 (the gap includes learnable vs unlearnable auxiliary target), gsm50k_c2_s0 - gsm50k_coconut_s0 = +1.0 (A5 item 5: the last supervised target contains the answer in 0.770 of the GSM-50k records at 3 latent steps): pass needs above C2-shuffled by more than sigma_screen 3.60 with K2 not fired: accuracy part not met, K2 not fired (Delta_scr +8.6, interval [+5.7, +11.5]): pass not met on this task; kill needs one clause on both tasks: this task's parts: within sigma_screen of C2-shuffled met, below full-depth coconut by more than 2 sigma_screen not met, K2 fired not met; the other screen task is not in this table [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- C3 (A3.4): UNTESTED on GSM-50k (no run record) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- C2+C3 (A3.4): UNTESTED on GSM-50k (no run record) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### K2 (7.5; A3.4)
- K2 (7.5; A3.4), gsm50k_pause_s0: no K2 read is defined for pause: section 8's scrambled construction replaces fed-back vectors and pause has none (5.3: latent steps 0, its 6 positions are pause tokens); its content test is the 7.5 pause comparison itself (critic entry 2026-10-20T10:49Z (4)(a)) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_coconut_s0: Delta_scr = 11.0 - 4.0 = +7.0 (55 - 20 of 500, paired per example: 46 correct only with own latents, 11 only with scrambled; gsm50k_coconut_s0_valid_none vs gsm50k_coconut_s0_valid_scrambled), 95% interval [+4.11, +9.89] (6.2: Delta +- 1.96 sqrt(Var_ex), Var_ex over 10,000 paired example draws, default_rng(0); percentile [+4.2, +10.0]) against sigma_screen 3.60: not fired (Delta_scr >= sigma_screen and the interval excludes 0); beside it: zero-latent drop 11.0 - 1.0 = +10.0 [+7.24, +12.76] (gsm50k_coconut_s0_valid_zero_latent); pause comparison gsm50k_coconut_s0 - gsm50k_pause_s0 = +1.4, within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c1b_k8_s0: Delta_scr = 9.4 - 5.4 = +4.0 (47 - 27 of 500, paired per example: 28 correct only with own latents, 8 only with scrambled; gsm50k_c1b_k8_s0_valid_none vs gsm50k_c1b_k8_s0_valid_scrambled), 95% interval [+1.70, +6.30] (6.2: Delta +- 1.96 sqrt(Var_ex), Var_ex over 10,000 paired example draws, default_rng(0); percentile [+1.8, +6.4]) against sigma_screen 3.60: not fired (Delta_scr >= sigma_screen and the interval excludes 0); beside it: zero-latent drop 9.4 - 8.0 = +1.4 [-0.90, +3.70] (gsm50k_c1b_k8_s0_valid_zero_latent); pause comparison gsm50k_c1b_k8_s0 - gsm50k_pause_s0 = -0.2, within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c1b_k4_s0: Delta_scr = 9.4 - 8.0 = +1.4 (47 - 40 of 500, paired per example: 11 correct only with own latents, 4 only with scrambled; gsm50k_c1b_k4_s0_valid_none vs gsm50k_c1b_k4_s0_valid_scrambled), 95% interval [-0.12, +2.92] (6.2: Delta +- 1.96 sqrt(Var_ex), Var_ex over 10,000 paired example draws, default_rng(0); percentile [+0.0, +3.0]) against sigma_screen 3.60: fired (Delta_scr < sigma_screen; the interval includes 0): the arm's latent claim is killed at the screen on this task (A3.4); it continues only as a labelled curriculum or token-budget effect (7.5; its parenthetical label is withheld under A11.3); beside it: zero-latent drop 9.4 - 8.2 = +1.2 [-0.03, +2.43] (gsm50k_c1b_k4_s0_valid_zero_latent); pause comparison gsm50k_c1b_k4_s0 - gsm50k_pause_s0 = -0.2, within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c2_s0 (A5 item 5: the last supervised target contains the answer in 0.770 of the GSM-50k records at 3 latent steps): Delta_scr = 12.0 - 3.4 = +8.6 (60 - 17 of 500, paired per example: 51 correct only with own latents, 8 only with scrambled; gsm50k_c2_s0_valid_none vs gsm50k_c2_s0_valid_scrambled), 95% interval [+5.70, +11.50] (6.2: Delta +- 1.96 sqrt(Var_ex), Var_ex over 10,000 paired example draws, default_rng(0); percentile [+5.8, +11.6]) against sigma_screen 3.60: not fired (Delta_scr >= sigma_screen and the interval excludes 0); beside it: zero-latent drop 12.0 - 1.0 = +11.0 [+8.19, +13.81] (gsm50k_c2_s0_valid_zero_latent); pause comparison gsm50k_c2_s0 - gsm50k_pause_s0 = +2.4, within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K2 (7.5; A3.4: every latent arm at the screen), gsm50k_c2_shuffled_s0: Delta_scr = 11.6 - 3.0 = +8.6 (58 - 15 of 500, paired per example: 50 correct only with own latents, 7 only with scrambled; gsm50k_c2_shuffled_s0_valid_none vs gsm50k_c2_shuffled_s0_valid_scrambled), 95% interval [+5.74, +11.46] (6.2: Delta +- 1.96 sqrt(Var_ex), Var_ex over 10,000 paired example draws, default_rng(0); percentile [+5.8, +11.6]) against sigma_screen 3.60: not fired (Delta_scr >= sigma_screen and the interval excludes 0); beside it: zero-latent drop 11.6 - 1.6 = +10.0 [+7.00, +13.00] (gsm50k_c2_shuffled_s0_valid_zero_latent); pause comparison gsm50k_c2_shuffled_s0 - gsm50k_pause_s0 = +2.0, within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Too-good triggers (7.10; A3.4)
- Too-good triggers (7.10; A3.4), any arm more than 10 points over the bar: none; the null triggers (scrambled-at-inference, zero-latent) are the next line (eval reads) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Null triggers (7.10; A3.4)
- Null triggers (7.10; A3.4), the scrambled-at-inference and zero-latent reads on valid against the bar + 2 sigma_screen = 25.6 + 7.20 = 32.80 and against the arm + sigma_screen: gsm50k_coconut_s0 scrambled 4.0, zero-latent 1.0 (arm 11.0 + 3.60 = 14.60); gsm50k_c1b_k8_s0 scrambled 5.4, zero-latent 8.0 (arm 9.4 + 3.60 = 13.00); gsm50k_c1b_k4_s0 scrambled 8.0, zero-latent 8.2 (arm 9.4 + 3.60 = 13.00); gsm50k_c2_s0 scrambled 3.4, zero-latent 1.0 (arm 12.0 + 3.60 = 15.60); gsm50k_c2_shuffled_s0 scrambled 3.0, zero-latent 1.6 (arm 11.6 + 3.60 = 15.20): none above either line [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### K3 (7.5; 7.7; A13.2)
- K3 (7.5 at the screen), gsm50k_pause_s0: 9.6 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired; no better than plain depth (7.5) [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_coconut_s0: 11.0 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c1b_k8_s0: 9.4 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired; no better than plain depth (7.5) [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c1b_k4_s0: 9.4 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired; no better than plain depth (7.5) [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c2_s0: 12.0 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- K3 (7.5 at the screen), gsm50k_c2_shuffled_s0: 11.6 vs A_pd 6.20 (the 124M no_cot point alone, gsm50k_no_cot_s0 epoch 22) - sigma_screen 3.60: not fired [nearest size; plain depth 355M UNTESTED (A13.2); screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Pause comparison (7.5)
- Pause comparison (7.5), gsm50k_coconut_s0 - gsm50k_pause_s0 = +1.4: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c1b_k8_s0 - gsm50k_pause_s0 = -0.2: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c1b_k4_s0 - gsm50k_pause_s0 = -0.2: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c2_s0 - gsm50k_pause_s0 = +2.4: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Pause comparison (7.5), gsm50k_c2_shuffled_s0 - gsm50k_pause_s0 = +2.0: within sigma_screen 3.60: CURRICULUM-ONLY [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### 7.2 cost legs and 5.5
- 7.2 cost legs, gsm50k_pause_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (exact mean from gsm50k_pause_s0_valid_none's per-example E': 11.643 GFLOPs, ratio [0.671, 0.673]; against the bar's range [17.312, 17.357] GFLOPs, whose per-example E' was not recorded); token leg (E' <= 0.2 x bar's) met (2.210 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) met (3.210 vs 6.29); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED (the arm's own batch-1 timing, where it exists, is its 5.5 (b) line) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_coconut_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (exact mean from gsm50k_coconut_s0_valid_none's per-example E': 11.689 GFLOPs, ratio [0.673, 0.675]; against the bar's range [17.312, 17.357] GFLOPs, whose per-example E' was not recorded); token leg (E' <= 0.2 x bar's) met (2.396 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.396 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED (the arm's own batch-1 timing, where it exists, is its 5.5 (b) line) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c1b_k8_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (exact mean from gsm50k_c1b_k8_s0_valid_none's per-example E': nominal 8/12: 11.315 GFLOPs, ratio [0.652, 0.654]; measured min(1, f_meas) = 0.7200: 11.371 GFLOPs, ratio [0.655, 0.657]; at 1: 11.660 GFLOPs, ratio [0.672, 0.674]; against the bar's range [17.312, 17.357] GFLOPs, whose per-example E' was not recorded); token leg (E' <= 0.2 x bar's) met (2.280 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.280 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED (the arm's own batch-1 timing, where it exists, is its 5.5 (b) line) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c1b_k4_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (exact mean from gsm50k_c1b_k4_s0_valid_none's per-example E': nominal 4/12: 10.972 GFLOPs, ratio [0.632, 0.634]; measured min(1, f_meas) = 0.4326: 11.074 GFLOPs, ratio [0.638, 0.640]; at 1: 11.661 GFLOPs, ratio [0.672, 0.674]; against the bar's range [17.312, 17.357] GFLOPs, whose per-example E' was not recorded); token leg (E' <= 0.2 x bar's) met (2.284 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.284 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED (the arm's own batch-1 timing, where it exists, is its 5.5 (b) line) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c2_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (exact mean from gsm50k_c2_s0_valid_none's per-example E': 11.668 GFLOPs, ratio [0.672, 0.674]; against the bar's range [17.312, 17.357] GFLOPs, whose per-example E' was not recorded); token leg (E' <= 0.2 x bar's) met (2.310 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.310 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED (the arm's own batch-1 timing, where it exists, is its 5.5 (b) line) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 7.2 cost legs, gsm50k_c2_shuffled_s0 vs the bar: FLOP leg (mean FLOPs <= bar's) met (exact mean from gsm50k_c2_shuffled_s0_valid_none's per-example E': 11.648 GFLOPs, ratio [0.671, 0.673]; against the bar's range [17.312, 17.357] GFLOPs, whose per-example E' was not recorded); token leg (E' <= 0.2 x bar's) met (2.230 vs 0.2 x 30.44 = 6.09); passes criterion (<= 0.2 x bar's) not met (9.230 vs 6.29); label: billing claim only (7.2); 5.5 (a), (c): no flip; (b) wall-clock at batch 1: the bar has no timed read, so (b) decides no comparison: UNTESTED (the arm's own batch-1 timing, where it exists, is its 5.5 (b) line) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### C1b per-step cost: f_meas (A3.4; A6.2)
- C1b per-step cost (A3.4; A6.2), gsm50k_c1b_k8_s0: f_meas = latent_ms_mean 8.006 ms (band [2, 10), gsm50k_c1b_k8_s0_valid_timed_bs1) / 11.120 ms (full-depth coconut through band [0, 12], gsm50k_coconut_s0_valid_band0-12_timed_bs1) = 0.7200 at batch 1, the fraction of record (latent passes 3000 / 3000 over 500 / 500 timed batches; prefill 10.983 / 10.994 ms, token 10.914 / 10.894 ms; latent_last: not recorded (run_eval.py's summary keeps prefill, latent and token only); NVIDIA A100-SXM4-40GB, torch 2.14.1+cu130, both reads); at the eval batch 32 beside it: 8.909 / 12.046 = 0.7396 (gsm50k_c1b_k8_s0_valid_none / gsm50k_coconut_s0_valid_band0-12_timed_bs32, 16 / 16 timed batches); saving line f_meas <= 0.77 (rule 4) met at batch 1; no saving label is written: rule 3 logits check pending (critic entry 2026-10-20T10:51Z part (3)); FLOPs printed at the nominal 8/12, at the measured 0.7200 and at 1, and the FLOP leg holds only if it holds at all three [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- C1b per-step cost (A3.4; A6.2), gsm50k_c1b_k4_s0: f_meas = latent_ms_mean 4.810 ms (band [4, 8), gsm50k_c1b_k4_s0_valid_timed_bs1) / 11.120 ms (full-depth coconut through band [0, 12], gsm50k_coconut_s0_valid_band0-12_timed_bs1) = 0.4326 at batch 1, the fraction of record (latent passes 3000 / 3000 over 500 / 500 timed batches; prefill 11.013 / 10.994 ms, token 10.949 / 10.894 ms; latent_last: not recorded (run_eval.py's summary keeps prefill, latent and token only); NVIDIA A100-SXM4-40GB, torch 2.14.1+cu130, both reads); at the eval batch 32 beside it: 5.764 / 12.046 = 0.4785 (gsm50k_c1b_k4_s0_valid_none / gsm50k_coconut_s0_valid_band0-12_timed_bs32, 16 / 16 timed batches); saving line f_meas <= 0.43 (rule 4) not met at batch 1: nominal, layer_fraction = min(1, f_meas) = 0.4326 in K3 and 7.2; the formula (b - a) / L + 0.10 = 0.4333 would be met: the two readings of rule 4 differ at this value (for the critic); no saving label is written: rule 3 logits check pending (critic entry 2026-10-20T10:51Z part (3)); FLOPs printed at the nominal 4/12, at the measured 0.4326 and at 1, and the FLOP leg holds only if it holds at all three [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### 5.5 (b) wall-clock at batch 1
- 5.5 (b) wall-clock at batch 1 (2.2), gsm50k_pause_s0: UNTESTED: no batch-1 timed read; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 5.5 (b) wall-clock at batch 1 (2.2), gsm50k_coconut_s0: its own HF path was timed at batch 32 only (16.543 ms per latent pass, 15.912 per token pass): UNTESTED at batch 1; the band [0, 12] batch-1 read (gsm50k_coconut_s0_valid_band0-12_timed_bs1; A6.2's denominator, never this arm's 5.5 (b), rule 1) gives 103.77 ms per example in timed passes; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 5.5 (b) wall-clock at batch 1 (2.2), gsm50k_c1b_k8_s0 on NVIDIA A100-SXM4-40GB (torch 2.14.1+cu130), its own path (gsm50k_c1b_k8_s0_valid_timed_bs1): 83.82 ms per example in timed passes (prefill 10.983 + 6 latent x 8.006 + 2.272 token x 10.914; the latent_last pass is not recorded), 100.14 ms per example for the whole read; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 5.5 (b) wall-clock at batch 1 (2.2), gsm50k_c1b_k4_s0 on NVIDIA A100-SXM4-40GB (torch 2.14.1+cu130), its own path (gsm50k_c1b_k4_s0_valid_timed_bs1): 64.90 ms per example in timed passes (prefill 11.013 + 6 latent x 4.810 + 2.286 token x 10.949; the latent_last pass is not recorded), 81.20 ms per example for the whole read; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 5.5 (b) wall-clock at batch 1 (2.2), gsm50k_c2_s0: UNTESTED: no batch-1 timed read; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- 5.5 (b) wall-clock at batch 1 (2.2), gsm50k_c2_shuffled_s0: UNTESTED: no batch-1 timed read; the bar has no batch-1 timed read, so 5.5 (b) decides no comparison [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### A6.2 rule 3 like-for-like check
- A6.2 rule 3 like-for-like check, gsm50k_coconut_s0: band [0, 12] at batch 1 53 of 500 (10.6; gsm50k_coconut_s0_valid_band0-12_timed_bs1): 2 examples differ in correctness from the HF-path plain read 55 of 500 (gsm50k_coconut_s0_valid_none, batch 32) (idx 273, 497), net -2, 1 differ in E'; band [0, 12] at batch 32 55 of 500 (11.0; gsm50k_coconut_s0_valid_band0-12_timed_bs32): 0 examples differ in correctness from the HF-path plain read 55 of 500 (gsm50k_coconut_s0_valid_none, batch 32), net +0, 3 differ in E': more than 1 in 500 at batch 1: goes to deviations.md with the operator's explanation (this tool writes no deviation); the timing is still reported; rule 3's strict-fp32 first-token logits check: pending (not in this job) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Step sweep (7.3; section 8; decides nothing)
- Step sweep (7.3, section 8: reported, decides nothing), gsm50k_coconut_s0: n = 0 1.0, n = 1 7.6, n = 2 9.0, n = 4 10.6, n = 6 11.0 (the plain read: the trained count), n = 8 11.6 (out-of-distribution: 8 > 6); non-decreasing from n = 0 to the trained n = 6: yes (descriptive); the n = 0 read and the zero-latent read agree on every example (0 of 500 differ in any column of examples.csv; a check of the eval path, not a read; A10.4) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Step sweep (7.3, section 8: reported, decides nothing), gsm50k_c1b_k8_s0: n = 0 8.0, n = 1 8.8, n = 2 8.8, n = 4 9.6, n = 6 9.4 (the plain read: the trained count), n = 8 9.4 (out-of-distribution: 8 > 6); non-decreasing from n = 0 to the trained n = 6: no (descriptive); the n = 0 read and the zero-latent read agree on every example (0 of 500 differ in any column of examples.csv; a check of the eval path, not a read; A10.4) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Step sweep (7.3, section 8: reported, decides nothing), gsm50k_c1b_k4_s0: n = 0 8.2, n = 1 8.6, n = 2 8.8, n = 4 9.2, n = 6 9.4 (the plain read: the trained count), n = 8 9.6 (out-of-distribution: 8 > 6); non-decreasing from n = 0 to the trained n = 6: yes (descriptive); the n = 0 read and the zero-latent read agree on every example (0 of 500 differ in any column of examples.csv; a check of the eval path, not a read; A10.4) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Step sweep (7.3, section 8: reported, decides nothing), gsm50k_c2_s0: n = 0 1.0, n = 1 5.8, n = 2 9.2, n = 4 11.2, n = 6 12.0 (the plain read: the trained count), n = 8 12.0 (out-of-distribution: 8 > 6); non-decreasing from n = 0 to the trained n = 6: yes (descriptive); the n = 0 read and the zero-latent read agree on every example (0 of 500 differ in any column of examples.csv; a check of the eval path, not a read; A10.4) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Step sweep (7.3, section 8: reported, decides nothing), gsm50k_c2_shuffled_s0: n = 0 1.6, n = 1 6.2, n = 2 6.6, n = 4 11.0, n = 6 11.6 (the plain read: the trained count), n = 8 11.8 (out-of-distribution: 8 > 6); non-decreasing from n = 0 to the trained n = 6: yes (descriptive); the n = 0 read and the zero-latent read agree on every example (0 of 500 differ in any column of examples.csv; a check of the eval path, not a read; A10.4) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### C2 labels (A5 item 5; A6.7 items 4, 5)
- gsm50k_c2_s0: A5 item 5: the last supervised target contains the answer in 0.770 of the GSM-50k records at 3 latent steps; printed beside every C2 number; config.json _state_targets answer_in_last_target_frac 0.76956 [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- gsm50k_c2_shuffled_s0: A6.7 item 4 with the run's own value (critic entry 2026-10-20T10:49Z): on GSM-50k the C2-shuffled donor shares the answer 439 of 50,000 = 0.88% (chance; the review's replica at another derangement seed gave 430); config.json _state_targets answer_in_last_target_frac 0.76956, answer_in_last_used_target_frac 0.00878 [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- train_state_loss by epoch 4-25 (A6.7 item 5): C2 3.616 2.491 1.991 2.461 2.164 1.980 2.219 2.081 1.975 1.936 1.866 1.803 1.741 1.685 1.632 1.583 1.536 1.490 1.446 1.408 1.360 1.331; C2-shuffled 4.372 4.172 4.145 4.229 4.184 4.170 4.219 4.198 4.184 4.174 4.163 4.145 4.127 4.108 4.087 4.065 4.042 4.017 3.991 3.965 3.937 3.905; every C2 minus C2-shuffled sentence carries: the gap includes learnable vs unlearnable auxiliary target [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Fade instrument (7.6)
- Fade instrument (7.6), gsm50k_pause_s0: lead over the bar's curve at epoch 12 -16.4, epoch 13 -16.6, epoch 25 -14.2 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_coconut_s0: lead over the bar's curve at epoch 12 -13.8, epoch 13 -14.8, epoch 25 -14.4 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c1b_k8_s0: lead over the bar's curve at epoch 12 -14.0, epoch 13 -14.8, epoch 25 -14.8 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c1b_k4_s0: lead over the bar's curve at epoch 12 -15.8, epoch 13 -16.4, epoch 25 -14.4 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c2_s0: lead over the bar's curve at epoch 12 -13.4, epoch 13 -14.4, epoch 25 -11.8 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_c2_shuffled_s0: lead over the bar's curve at epoch 12 -13.2, epoch 13 -12.6, epoch 25 -14.2 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- Fade instrument (7.6), gsm50k_no_cot_s0: lead over the bar's curve at epoch 12 -17.8, epoch 13 -18.8, epoch 25 -17.6 (0.5 B = 12.5 read at epochs 12 and 13): not fading (sigma_screen 3.60) [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### UNTESTED and excluded
- gsm50k_plain_depth_355m_s0 (plain depth: 355M no_cot (7.7, A13.2)): UNTESTED: no run record (RUN_DONE.json absent); enters no cell [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- gsm50k_c3_s0 (card C3): UNTESTED: no run record (RUN_DONE.json absent); enters no cell [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- gsm50k_c3_foreign_s0 (null: C3-foreign): UNTESTED: no run record (RUN_DONE.json absent); enters no cell [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- gsm50k_c2c3_s0 (card C2+C3): UNTESTED: no run record (RUN_DONE.json absent); enters no cell [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

### Receipts and launch condition (A3.2 (i); A5 item 8)
- env_reference per training receipt: silent-train-20261019-153923-d3bb ok, silent-train-20261019-154904-c083 ok, silent-train-20261019-160323-4299 ok, silent-train-20261019-172741-7fff ok, silent-train-20261019-203810-7ebd ok, silent-train-20261019-204820-d318 ok, silent-train-20261020-025152-49e0 ok, silent-train-20261020-030355-f2b5 ok, silent-train-20261020-031905-622a ok; the eval job's receipt (silent-eval-20261020-112122-f7da): ok [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- attempts named by the runs without a receipt in this read: none [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]
- A5 item 8 (the bar's three cot trials admitted no later than the first latent arm, from the receipts' admissions): met [screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial]

## 5. Readings and sensitivity

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
  - card:c1b_k8 under 'floor only': pass: met on this task; kill parts: accuracy not met, K2 not met -> pass: not met on this task; kill parts: accuracy not met, K2 not met
  - card:c1b_k4 under 'floor only': pass: not met on this task: K2 fired; kill parts: accuracy not met, K2 met -> pass: not met on this task; kill parts: accuracy not met, K2 met
  - fade:c2_shuffled under 'floor only': not fading -> fading at one 0.5 B reading
- Against the first table: every line it decided keeps its outcome; the lines it left pending are decided here.

## 6. Notes

- A2 item 4 (c): the GSM valid set shares a 13-gram with the full GSM8k-Aug train on 7 of 500 items (the 50k subset at most that); a known bias, valid slightly optimistic, equally for every arm.
- Readings chosen (the sigma alternatives are recomputed in section 5): sigma_screen as A3.4 is written; 'ahead by 1 sigma' (6.3) as a difference >= sigma; 'within sigma' as an absolute difference <= sigma, except C1b's pass read as not below by more than sigma; CURRICULUM-ONLY (7.5) whenever the pause difference is within sigma; the fade's 0.5 B read at both neighbouring epochs.
- K2 (7.5, A3.4) and the zero-latent drop: Delta in points from the two reads' per-example files, paired by example; the interval is 6.2's Delta +- 1.96 sqrt(Var_ex), Var_ex the variance (ddof 1) of Delta over 10,000 paired example draws (numpy default_rng(0), drawn as eval.summarize draws), without 6.2's seed term (both reads score one checkpoint); the percentile interval of the same draws is printed beside it; 'includes 0' is inclusive. 6.3's scrambled leg is Delta_scr >= sigma_screen.
- Null triggers (7.10): each scrambled and zero-latent read against the bar + 2 sigma_screen and against the arm + sigma_screen, the arm read both as its selection and as its plain eval read (either exceeded triggers); 'above' is strict.
- f_meas (A6.2 rules 1, 2, 4): the C1b read's latent ms per pass over the full-depth coconut checkpoint's band [0, 12] read's, at batch 1 (of record) and at the eval batch 32 (beside), same device and torch; run_eval.py's summary keeps the pooled ms per pass, which equals the mean of summarize_timing's per-batch latent_ms_mean because every batch has the same number of latent passes (checked); it keeps no latent_last mean, so that column is 'not recorded'. Rule 4's line of record is the printed 0.77 (k = 8) and 0.43 (k = 4); the formula (b - a) / L + 0.10 (0.7667, 0.4333) is evaluated beside it. No saving label is written while rule 3's strict-fp32 logits check is pending.
- Costs: the latent arms' means are exact from the plain eval read's per-example E' (every row's FLOPs, reference FLOPs, prefill tokens and passes recomputed by accounting.inference_record from the hashed valid |q| equal the file's); the bar and no_cot keep the first table's exact ranges (no per-example read of either exists), so a ratio to the bar is an interval and a leg is decided only when it decides. C1b FLOPs at the nominal (b - a) / L, at the measured min(1, f_meas) and at 1; inference_record charges a fraction as the nearest fraction with denominator <= 1000. V as the arm's head (A11.2).
- The step sweep (7.3, section 8) decides nothing; its n = 6 point is the plain read (the trained count), not a sweep read; n = 8 is out-of-distribution; the n = 0 against zero-latent check compares every column of the two per-example files (A10.4: a difference is a BUG trigger).
- Still pending or UNTESTED here: rule 3's strict-fp32 first-token logits check (A6.2; no saving label until it is logged); the latent_last means of A6.2 rule 2 (not in the summaries); 5.5 (b) for every comparison with the bar (no timed bar read) and coconut's own HF path at batch 1; plain depth 355M (A13.2: the K3 lines stay 'nearest size'); the ProsQA-hard side of every both-task clause; the screen round's single test read (A3.4).
- Not a registered reading, printed for the critic: 7.5 names K2's interval an example-bootstrap interval and the 10:51Z row the 10,000-draw example bootstrap, so 6.2's seed term sigma^2 (1/n_A + 1/n_B) is not in it (both reads score one checkpoint); with that term at one seed (+ 2 sigma_screen^2) the intervals would be: gsm50k_coconut_s0 +7.0 +- 10.39 (includes 0); gsm50k_c1b_k8_s0 +4.0 +- 10.25 (includes 0); gsm50k_c1b_k4_s0 +1.4 +- 10.10 (includes 0); gsm50k_c2_s0 +8.6 +- 10.40 (includes 0); gsm50k_c2_shuffled_s0 +8.6 +- 10.39 (includes 0).
