# The Pareto read (7.3: secondary, reported, decides nothing): accuracy against cost per task, valid and test apart

Critic entry 2026-10-21T20:21Z (c). A compilation from the committed records (row (c) (2)): no VM, no new read of any checkpoint; it decides nothing and changes no line of any table. Valid and test are separate panels and never share an axis; every test point carries the test read's label: end-of-round test read (A3.4): reporting only, no decision attached; 1 seed; one registered read per checkpoint (4.5).

Labels of the points (row (c) (3), verbatim): "final selection (3.3 item 1): n latents = the trained count"; "earlier-stage checkpoint, n latents = its stage; writes the remaining steps (E' as recorded: 21 to 23 tokens on GSM-50k test, 25 to 57 on ProsQA-hard test); secondary read (3.3 item 3); decides nothing"; "step sweep n = k of the final checkpoint (7.3; n = 8 out-of-distribution)"; "tuned bar: the pick (3 LRs)"; "no_cot (floor; the 124M plain-depth point)"; "pause (control)"; "plain depth 355M (7.7)"; "from the training log's means (range); no per-example record on valid"; "no band".

## GSM-50k

Valid of record: results/screens/gsm50k/gsm50k_screen_with_hinge.json at `rev9` (valid 500); test: results/screens/test_read.json at `rev1` (test 1319; end-of-round test read (A3.4): reporting only, no decision attached; 1 seed; one registered read per checkpoint (4.5)).

### GSM-50k, valid: accuracy against mean emitted tokens E'

| point | run | epoch | n latents | accuracy | band (accuracy) | E' mean | band (E') |
|---|---|---|---|---|---|---|---|
| step sweep n = 8 (OOD) | gsm50k_c3_s0 | 25 | 8 | 10.4 | [7.8, 13.2] | 2.204 | [2.156, 2.254] |
| step sweep n = 4 | gsm50k_c3_s0 | 25 | 4 | 10.6 | [8.0, 13.4] | 2.206 | [2.160, 2.256] |
| no_cot (from means) | gsm50k_no_cot_s0 | 22 | 0 | 6.2 | no band | 2.206 | no band |
| pause (control) | gsm50k_pause_s0 | 22 | 6 | 9.6 | [7.0, 12.4] | 2.210 | [2.162, 2.262] |
| step sweep n = 4 | gsm50k_c2_shuffled_s0 | 17 | 4 | 11.0 | [8.2, 13.8] | 2.216 | [2.170, 2.264] |
| final selection | gsm50k_c3_foreign_s0 | 16 | 6 | 10.0 | [7.4, 12.8] | 2.216 | [2.168, 2.268] |
| step sweep n = 1 | gsm50k_c3_foreign_s0 | 16 | 1 | 9.0 | [6.6, 11.6] | 2.220 | [2.174, 2.270] |
| step sweep n = 2 | gsm50k_c3_foreign_s0 | 16 | 2 | 9.8 | [7.4, 12.4] | 2.222 | [2.176, 2.272] |
| step sweep n = 4 | gsm50k_c3_foreign_s0 | 16 | 4 | 10.4 | [7.8, 13.2] | 2.222 | [2.176, 2.274] |
| step sweep n = 8 (OOD) | gsm50k_c3_foreign_s0 | 16 | 8 | 9.8 | [7.4, 12.4] | 2.222 | [2.174, 2.274] |
| step sweep n = 8 (OOD) | gsm50k_c2_shuffled_s0 | 17 | 8 | 11.8 | [9.0, 14.8] | 2.226 | [2.178, 2.276] |
| final selection | gsm50k_c3_s0 | 25 | 6 | 10.4 | [7.8, 13.2] | 2.226 | [2.176, 2.278] |
| final selection | gsm50k_c2_shuffled_s0 | 17 | 6 | 11.6 | [8.8, 14.4] | 2.230 | [2.182, 2.282] |
| step sweep n = 2 | gsm50k_c2_shuffled_s0 | 17 | 2 | 6.6 | [4.4, 8.8] | 2.236 | [2.190, 2.286] |
| step sweep n = 1 | gsm50k_c2_shuffled_s0 | 17 | 1 | 6.2 | [4.2, 8.4] | 2.238 | [2.192, 2.288] |
| step sweep n = 8 (OOD) | gsm50k_c2c3_s0 | 21 | 8 | 12.2 | [9.4, 15.2] | 2.244 | [2.194, 2.298] |
| step sweep n = 0 | gsm50k_c1b_k8_s0 | 19 | 0 | 8.0 | [5.8, 10.4] | 2.254 | [2.202, 2.310] |
| step sweep n = 4 | gsm50k_c1b_k8_s0 | 19 | 4 | 9.6 | [7.0, 12.4] | 2.260 | [2.208, 2.316] |
| step sweep n = 8 (OOD) | gsm50k_coconut_s0 | 17 | 8 | 11.6 | [8.8, 14.4] | 2.262 | [2.210, 2.318] |
| step sweep n = 1 | gsm50k_c1b_k8_s0 | 19 | 1 | 8.8 | [6.4, 11.4] | 2.264 | [2.208, 2.322] |
| final selection | gsm50k_c2c3_s0 | 21 | 6 | 12.0 | [9.2, 15.0] | 2.268 | [2.216, 2.324] |
| step sweep n = 2 | gsm50k_c1b_k8_s0 | 19 | 2 | 8.8 | [6.4, 11.4] | 2.274 | [2.218, 2.334] |
| step sweep n = 2 | gsm50k_coconut_s0 | 17 | 2 | 9.0 | [6.6, 11.6] | 2.274 | [2.220, 2.336] |
| step sweep n = 8 (OOD) | gsm50k_c1b_k4_s0 | 15 | 8 | 9.6 | [7.2, 12.2] | 2.276 | [2.222, 2.332] |
| final selection | gsm50k_c1b_k8_s0 | 19 | 6 | 9.4 | [7.0, 12.0] | 2.280 | [2.226, 2.338] |
| step sweep n = 4 | gsm50k_coconut_s0 | 17 | 4 | 10.6 | [8.0, 13.4] | 2.280 | [2.224, 2.342] |
| step sweep n = 8 (OOD) | gsm50k_c1b_k8_s0 | 19 | 8 | 9.4 | [7.0, 12.0] | 2.282 | [2.224, 2.346] |
| step sweep n = 4 | gsm50k_c2c3_s0 | 21 | 4 | 11.6 | [8.8, 14.4] | 2.282 | [2.228, 2.340] |
| step sweep n = 2 | gsm50k_c3_s0 | 25 | 2 | 8.4 | [6.0, 11.0] | 2.282 | [2.222, 2.348] |
| final selection | gsm50k_c1b_k4_s0 | 15 | 6 | 9.4 | [7.0, 12.0] | 2.284 | [2.230, 2.342] |
| step sweep n = 2 | gsm50k_c1b_k4_s0 | 15 | 2 | 8.8 | [6.4, 11.4] | 2.292 | [2.238, 2.350] |
| step sweep n = 4 | gsm50k_c1b_k4_s0 | 15 | 4 | 9.2 | [6.8, 11.8] | 2.292 | [2.236, 2.352] |
| step sweep n = 0 | gsm50k_c1b_k4_s0 | 15 | 0 | 8.2 | [5.8, 10.6] | 2.296 | [2.240, 2.358] |
| step sweep n = 1 | gsm50k_c1b_k4_s0 | 15 | 1 | 8.6 | [6.2, 11.0] | 2.306 | [2.250, 2.368] |
| step sweep n = 8 (OOD) | gsm50k_c2_s0 | 20 | 8 | 12.0 | [9.2, 14.8] | 2.306 | [2.218, 2.430] |
| final selection | gsm50k_c2_s0 | 20 | 6 | 12.0 | [9.2, 15.0] | 2.310 | [2.224, 2.430] |
| step sweep n = 2 | gsm50k_c2_s0 | 20 | 2 | 9.2 | [6.8, 11.8] | 2.312 | [2.238, 2.398] |
| plain depth 355M (from means) | gsm50k_plain_depth_355m_s0 | 24 | 0 | 8.0 | no band | 2.332 | no band |
| step sweep n = 2 | gsm50k_c2c3_s0 | 21 | 2 | 8.2 | [5.8, 10.8] | 2.348 | [2.276, 2.422] |
| step sweep n = 4 | gsm50k_c2_s0 | 20 | 4 | 11.2 | [8.4, 14.0] | 2.386 | [2.222, 2.674] |
| step sweep n = 1 | gsm50k_c2_s0 | 20 | 1 | 5.8 | [3.8, 7.8] | 2.392 | [2.302, 2.496] |
| final selection | gsm50k_coconut_s0 | 17 | 6 | 11.0 | [8.4, 13.8] | 2.396 | [2.232, 2.676] |
| step sweep n = 1 | gsm50k_coconut_s0 | 17 | 1 | 7.6 | [5.4, 10.0] | 2.504 | [2.314, 2.804] |
| step sweep n = 0 | gsm50k_c3_foreign_s0 | 16 | 0 | 7.4 | [5.2, 9.8] | 2.598 | [2.378, 2.920] |
| step sweep n = 1 | gsm50k_c2c3_s0 | 21 | 1 | 6.8 | [4.6, 9.0] | 2.694 | [2.472, 3.006] |
| step sweep n = 1 | gsm50k_c3_s0 | 25 | 1 | 6.2 | [4.2, 8.4] | 2.706 | [2.574, 2.846] |
| step sweep n = 0 | gsm50k_c2_shuffled_s0 | 17 | 0 | 1.6 | [0.6, 2.8] | 5.166 | [4.942, 5.396] |
| step sweep n = 0 | gsm50k_coconut_s0 | 17 | 0 | 1.0 | [0.2, 2.0] | 5.422 | [5.106, 5.790] |
| step sweep n = 0 | gsm50k_c2_s0 | 20 | 0 | 1.0 | [0.2, 2.0] | 5.884 | [5.660, 6.118] |
| step sweep n = 0 | gsm50k_c3_s0 | 25 | 0 | 0.0 | [0.0, 0.0] | 5.960 | [5.466, 6.538] |
| step sweep n = 0 | gsm50k_c2c3_s0 | 21 | 0 | 0.0 | [0.0, 0.0] | 6.424 | [6.018, 6.910] |
| earlier-stage checkpoint (from means) | gsm50k_c2_shuffled_s0 | 6 | 2 | 13.0 | no band | 21.302 | no band |
| earlier-stage checkpoint (from means) | gsm50k_coconut_s0 | 6 | 2 | 12.6 | no band | 21.384 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c2_s0 | 6 | 2 | 12.6 | no band | 21.470 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c3_s0 | 6 | 2 | 13.4 | no band | 21.536 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c1b_k8_s0 | 6 | 2 | 14.0 | no band | 21.724 | no band |
| earlier-stage checkpoint (from means) | gsm50k_pause_s0 | 5 | 2 | 13.6 | no band | 22.194 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c2c3_s0 | 5 | 2 | 12.8 | no band | 22.756 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c3_foreign_s0 | 6 | 2 | 11.0 | no band | 22.856 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c1b_k4_s0 | 6 | 2 | 14.6 | no band | 22.858 | no band |
| tuned bar (from means) | gsm50k_cot_lr3e-4_s0 | 17 | 0 | 25.6 | no band | 30.436 | no band |

### GSM-50k, valid: accuracy against mean FLOPs (5.1; GFLOPs)

| point | run | epoch | n latents | accuracy | band (accuracy) | FLOPs mean | band (FLOPs) | reference-cost FLOPs mean |
|---|---|---|---|---|---|---|---|---|
| no_cot (from means) | gsm50k_no_cot_s0 | 22 | 0 | 6.2 | no band | [10.261, 10.269] | no band | [35.270, 94.780] |
| step sweep n = 0 | gsm50k_c1b_k8_s0 | 19 | 0 | 8.0 | [5.8, 10.4] | 10.619 (the read's layer fraction 0.6667, as the file records) | [10.314, 10.931] | 47.534 |
| step sweep n = 0 | gsm50k_c1b_k4_s0 | 15 | 0 | 8.2 | [5.8, 10.6] | 10.629 (the read's layer fraction 0.3333, as the file records) | [10.322, 10.941] | 48.140 |
| step sweep n = 1 | gsm50k_c1b_k4_s0 | 15 | 1 | 8.6 | [6.2, 11.0] | 10.689 (the read's layer fraction 0.3333, as the file records) | [10.382, 11.001] | 63.325 |
| step sweep n = 0 | gsm50k_c3_foreign_s0 | 16 | 0 | 7.4 | [5.2, 9.8] | 10.705 | [10.390, 11.031] | 55.566 |
| step sweep n = 1 | gsm50k_c1b_k8_s0 | 19 | 1 | 8.8 | [6.4, 11.4] | 10.736 (the read's layer fraction 0.6667, as the file records) | [10.430, 11.048] | 62.816 |
| step sweep n = 2 | gsm50k_c1b_k4_s0 | 15 | 2 | 8.8 | [6.4, 11.4] | 10.743 (the read's layer fraction 0.3333, as the file records) | [10.436, 11.056] | 78.415 |
| step sweep n = 1 | gsm50k_c3_foreign_s0 | 16 | 1 | 9.0 | [6.6, 11.6] | 10.783 | [10.476, 11.093] | 62.033 |
| step sweep n = 1 | gsm50k_c2_shuffled_s0 | 17 | 1 | 6.2 | [4.2, 8.4] | 10.787 | [10.482, 11.098] | 62.249 |
| step sweep n = 1 | gsm50k_c2_s0 | 20 | 1 | 5.8 | [3.8, 7.8] | 10.826 | [10.517, 11.142] | 65.311 |
| step sweep n = 2 | gsm50k_c1b_k8_s0 | 19 | 2 | 8.8 | [6.4, 11.4] | 10.854 (the read's layer fraction 0.6667, as the file records) | [10.545, 11.166] | 78.272 |
| step sweep n = 1 | gsm50k_coconut_s0 | 17 | 1 | 7.6 | [5.4, 10.0] | 10.854 | [10.544, 11.176] | 67.539 |
| step sweep n = 4 | gsm50k_c1b_k4_s0 | 15 | 4 | 9.2 | [6.8, 11.8] | 10.859 (the read's layer fraction 0.3333, as the file records) | [10.550, 11.171] | 109.817 |
| step sweep n = 1 | gsm50k_c2c3_s0 | 21 | 1 | 6.8 | [4.6, 9.0] | 10.901 | [10.588, 11.217] | 70.261 |
| step sweep n = 1 | gsm50k_c3_s0 | 25 | 1 | 6.2 | [4.2, 8.4] | 10.904 | [10.596, 11.218] | 69.938 |
| step sweep n = 2 | gsm50k_c3_foreign_s0 | 16 | 2 | 9.8 | [7.4, 12.4] | 10.956 | [10.649, 11.266] | 77.297 |
| step sweep n = 2 | gsm50k_c2_shuffled_s0 | 17 | 2 | 6.6 | [4.4, 8.8] | 10.959 | [10.653, 11.272] | 77.487 |
| step sweep n = 2 | gsm50k_coconut_s0 | 17 | 2 | 9.0 | [6.6, 11.6] | 10.969 | [10.662, 11.282] | 78.084 |
| step sweep n = 2 | gsm50k_c3_s0 | 25 | 2 | 8.4 | [6.0, 11.0] | 10.971 | [10.665, 11.284] | 78.224 |
| step sweep n = 2 | gsm50k_c2_s0 | 20 | 2 | 9.2 | [6.8, 11.8] | 10.978 | [10.672, 11.292] | 79.022 |
| step sweep n = 2 | gsm50k_c2c3_s0 | 21 | 2 | 8.2 | [5.8, 10.8] | 10.987 | [10.682, 11.297] | 79.150 |
| step sweep n = 4 | gsm50k_c1b_k8_s0 | 19 | 4 | 9.6 | [7.0, 12.4] | 11.080 (the read's layer fraction 0.6667, as the file records) | [10.773, 11.393] | 109.363 |
| step sweep n = 8 (OOD) | gsm50k_c1b_k4_s0 | 15 | 8 | 9.6 | [7.2, 12.2] | 11.085 (the read's layer fraction 0.3333, as the file records) | [10.777, 11.398] | 175.251 |
| step sweep n = 4 | gsm50k_c3_s0 | 25 | 4 | 10.6 | [8.0, 13.4] | 11.296 | [10.991, 11.608] | 108.238 |
| step sweep n = 4 | gsm50k_c2_shuffled_s0 | 17 | 4 | 11.0 | [8.2, 13.8] | 11.299 | [10.993, 11.611] | 108.547 |
| step sweep n = 4 | gsm50k_c3_foreign_s0 | 16 | 4 | 10.4 | [7.8, 13.2] | 11.301 | [10.995, 11.611] | 108.646 |
| step sweep n = 4 | gsm50k_coconut_s0 | 17 | 4 | 10.6 | [8.0, 13.4] | 11.315 | [11.008, 11.627] | 109.678 |
| step sweep n = 4 | gsm50k_c2c3_s0 | 21 | 4 | 11.6 | [8.8, 14.4] | 11.315 | [11.009, 11.628] | 109.652 |
| step sweep n = 4 | gsm50k_c2_s0 | 20 | 4 | 11.2 | [8.4, 14.0] | 11.342 | [11.025, 11.665] | 113.759 |
| step sweep n = 0 | gsm50k_c2_shuffled_s0 | 17 | 0 | 1.6 | [0.6, 2.8] | 11.346 | [11.036, 11.667] | 92.597 |
| step sweep n = 0 | gsm50k_coconut_s0 | 17 | 0 | 1.0 | [0.2, 2.0] | 11.410 | [11.087, 11.742] | 98.785 |
| step sweep n = 0 | gsm50k_c2_s0 | 20 | 0 | 1.0 | [0.2, 2.0] | 11.525 | [11.209, 11.846] | 104.824 |
| step sweep n = 0 | gsm50k_c3_s0 | 25 | 0 | 0.0 | [0.0, 0.0] | 11.545 | [11.209, 11.890] | 109.723 |
| step sweep n = 8 (OOD) | gsm50k_c1b_k8_s0 | 19 | 8 | 9.4 | [7.0, 12.0] | 11.546 (the read's layer fraction 0.6667, as the file records) | [11.239, 11.859] | 175.397 |
| pause (control) | gsm50k_pause_s0 | 22 | 6 | 9.6 | [7.0, 12.4] | 11.643 | [11.335, 11.956] | 52.644 |
| final selection | gsm50k_c3_foreign_s0 | 16 | 6 | 10.0 | [7.4, 12.8] | 11.644 | [11.339, 11.956] | 140.835 |
| final selection | gsm50k_c3_s0 | 25 | 6 | 10.4 | [7.8, 13.2] | 11.647 | [11.341, 11.958] | 140.958 |
| final selection | gsm50k_c2_shuffled_s0 | 17 | 6 | 11.6 | [8.8, 14.4] | 11.648 | [11.341, 11.960] | 141.074 |
| final selection | gsm50k_c2c3_s0 | 21 | 6 | 12.0 | [9.2, 15.0] | 11.657 | [11.351, 11.968] | 141.764 |
| final selection | gsm50k_c1b_k8_s0 | 19 | 6 | 9.4 | [7.0, 12.0] | 11.660 at 1; 11.315 nominal 0.6667; 11.371 measured 0.7200 | [11.008, 11.628] (the plain read's layer fraction 0.6667) | 141.963 |
| step sweep n = 0 | gsm50k_c2c3_s0 | 21 | 0 | 0.0 | [0.0, 0.0] | 11.660 | [11.334, 11.986] | 114.985 |
| final selection | gsm50k_c1b_k4_s0 | 15 | 6 | 9.4 | [7.0, 12.0] | 11.661 at 1; 10.972 nominal 0.3333; 11.074 measured 0.4326 | [10.664, 11.285] (the plain read's layer fraction 0.3333) | 141.971 |
| final selection | gsm50k_c2_s0 | 20 | 6 | 12.0 | [9.2, 15.0] | 11.668 | [11.359, 11.985] | 143.060 |
| final selection | gsm50k_coconut_s0 | 17 | 6 | 11.0 | [8.4, 13.8] | 11.689 | [11.376, 12.013] | 145.338 |
| step sweep n = 8 (OOD) | gsm50k_c3_s0 | 25 | 8 | 10.4 | [7.8, 13.2] | 11.986 | [11.680, 12.297] | 173.919 |
| step sweep n = 8 (OOD) | gsm50k_c3_foreign_s0 | 16 | 8 | 9.8 | [7.4, 12.4] | 11.991 | [11.685, 12.302] | 174.251 |
| step sweep n = 8 (OOD) | gsm50k_c2_shuffled_s0 | 17 | 8 | 11.8 | [9.0, 14.8] | 11.992 | [11.686, 12.305] | 174.367 |
| step sweep n = 8 (OOD) | gsm50k_c2c3_s0 | 21 | 8 | 12.2 | [9.4, 15.2] | 11.996 | [11.691, 12.308] | 174.602 |
| step sweep n = 8 (OOD) | gsm50k_coconut_s0 | 17 | 8 | 11.6 | [8.8, 14.4] | 12.001 | [11.694, 12.313] | 175.073 |
| step sweep n = 8 (OOD) | gsm50k_c2_s0 | 20 | 8 | 12.0 | [9.2, 14.8] | 12.012 | [11.704, 12.329] | 176.395 |
| earlier-stage checkpoint (from means) | gsm50k_c2_shuffled_s0 | 6 | 2 | 13.0 | no band | [15.719, 15.760] | no band | [374.216, 655.105] |
| earlier-stage checkpoint (from means) | gsm50k_coconut_s0 | 6 | 2 | 12.6 | no band | [15.739, 15.781] | no band | [375.807, 657.265] |
| earlier-stage checkpoint (from means) | gsm50k_c2_s0 | 6 | 2 | 12.6 | no band | [15.761, 15.802] | no band | [377.475, 659.171] |
| earlier-stage checkpoint (from means) | gsm50k_c3_s0 | 6 | 2 | 13.4 | no band | [15.777, 15.819] | no band | [378.763, 660.749] |
| earlier-stage checkpoint (from means) | gsm50k_c1b_k8_s0 | 6 | 2 | 14.0 | no band | [15.824, 15.866] at 1; [15.709, 15.751] nominal 0.6667; [15.728, 15.769] measured 0.7200 | no band | [382.458, 665.207] |
| earlier-stage checkpoint (from means) | gsm50k_pause_s0 | 5 | 2 | 13.6 | no band | [15.942, 15.984] | no band | [368.783, 653.482] |
| earlier-stage checkpoint (from means) | gsm50k_c2c3_s0 | 5 | 2 | 12.8 | no band | [16.083, 16.125] | no band | [402.843, 689.704] |
| earlier-stage checkpoint (from means) | gsm50k_c3_foreign_s0 | 6 | 2 | 11.0 | no band | [16.108, 16.150] | no band | [404.833, 691.981] |
| earlier-stage checkpoint (from means) | gsm50k_c1b_k4_s0 | 6 | 2 | 14.6 | no band | [16.108, 16.150] at 1; [15.878, 15.921] nominal 0.3333; [15.913, 15.955] measured 0.4326 | no band | [404.873, 692.030] |
| tuned bar (from means) | gsm50k_cot_lr3e-4_s0 | 17 | 0 | 25.6 | no band | [17.312, 17.357] | no band | [511.913, 812.749] |
| plain depth 355M (from means) | gsm50k_plain_depth_355m_s0 | 24 | 0 | 8.0 | no band | [35.948, 35.972] | no band | [104.829, 283.112] |

### GSM-50k, valid: accuracy against mean sequential passes (5.4 (b): the matched-passes side)

| point | run | epoch | n latents | accuracy | band (accuracy) | passes mean | band (passes) |
|---|---|---|---|---|---|---|---|
| no_cot (from means) | gsm50k_no_cot_s0 | 22 | 0 | 6.2 | no band | 3.206 | no band |
| pause (control) | gsm50k_pause_s0 | 22 | 6 | 9.6 | [7.0, 12.4] | 3.210 | [3.162, 3.262] |
| step sweep n = 0 | gsm50k_c1b_k8_s0 | 19 | 0 | 8.0 | [5.8, 10.4] | 3.254 | [3.202, 3.310] |
| step sweep n = 0 | gsm50k_c1b_k4_s0 | 15 | 0 | 8.2 | [5.8, 10.6] | 3.296 | [3.240, 3.358] |
| plain depth 355M (from means) | gsm50k_plain_depth_355m_s0 | 24 | 0 | 8.0 | no band | 3.332 | no band |
| step sweep n = 0 | gsm50k_c3_foreign_s0 | 16 | 0 | 7.4 | [5.2, 9.8] | 3.598 | [3.378, 3.920] |
| step sweep n = 1 | gsm50k_c3_foreign_s0 | 16 | 1 | 9.0 | [6.6, 11.6] | 4.220 | [4.174, 4.270] |
| step sweep n = 1 | gsm50k_c2_shuffled_s0 | 17 | 1 | 6.2 | [4.2, 8.4] | 4.238 | [4.192, 4.288] |
| step sweep n = 1 | gsm50k_c1b_k8_s0 | 19 | 1 | 8.8 | [6.4, 11.4] | 4.264 | [4.208, 4.322] |
| step sweep n = 1 | gsm50k_c1b_k4_s0 | 15 | 1 | 8.6 | [6.2, 11.0] | 4.306 | [4.250, 4.368] |
| step sweep n = 1 | gsm50k_c2_s0 | 20 | 1 | 5.8 | [3.8, 7.8] | 4.392 | [4.302, 4.496] |
| step sweep n = 1 | gsm50k_coconut_s0 | 17 | 1 | 7.6 | [5.4, 10.0] | 4.504 | [4.314, 4.804] |
| step sweep n = 1 | gsm50k_c2c3_s0 | 21 | 1 | 6.8 | [4.6, 9.0] | 4.694 | [4.472, 5.006] |
| step sweep n = 1 | gsm50k_c3_s0 | 25 | 1 | 6.2 | [4.2, 8.4] | 4.706 | [4.574, 4.846] |
| step sweep n = 2 | gsm50k_c3_foreign_s0 | 16 | 2 | 9.8 | [7.4, 12.4] | 5.222 | [5.176, 5.272] |
| step sweep n = 2 | gsm50k_c2_shuffled_s0 | 17 | 2 | 6.6 | [4.4, 8.8] | 5.236 | [5.190, 5.286] |
| step sweep n = 2 | gsm50k_c1b_k8_s0 | 19 | 2 | 8.8 | [6.4, 11.4] | 5.274 | [5.218, 5.334] |
| step sweep n = 2 | gsm50k_coconut_s0 | 17 | 2 | 9.0 | [6.6, 11.6] | 5.274 | [5.220, 5.336] |
| step sweep n = 2 | gsm50k_c3_s0 | 25 | 2 | 8.4 | [6.0, 11.0] | 5.282 | [5.222, 5.348] |
| step sweep n = 2 | gsm50k_c1b_k4_s0 | 15 | 2 | 8.8 | [6.4, 11.4] | 5.292 | [5.238, 5.350] |
| step sweep n = 2 | gsm50k_c2_s0 | 20 | 2 | 9.2 | [6.8, 11.8] | 5.312 | [5.238, 5.398] |
| step sweep n = 2 | gsm50k_c2c3_s0 | 21 | 2 | 8.2 | [5.8, 10.8] | 5.348 | [5.276, 5.422] |
| step sweep n = 0 | gsm50k_c2_shuffled_s0 | 17 | 0 | 1.6 | [0.6, 2.8] | 6.166 | [5.942, 6.396] |
| step sweep n = 0 | gsm50k_coconut_s0 | 17 | 0 | 1.0 | [0.2, 2.0] | 6.422 | [6.106, 6.790] |
| step sweep n = 0 | gsm50k_c2_s0 | 20 | 0 | 1.0 | [0.2, 2.0] | 6.884 | [6.660, 7.118] |
| step sweep n = 0 | gsm50k_c3_s0 | 25 | 0 | 0.0 | [0.0, 0.0] | 6.960 | [6.466, 7.538] |
| step sweep n = 4 | gsm50k_c3_s0 | 25 | 4 | 10.6 | [8.0, 13.4] | 7.206 | [7.160, 7.256] |
| step sweep n = 4 | gsm50k_c2_shuffled_s0 | 17 | 4 | 11.0 | [8.2, 13.8] | 7.216 | [7.170, 7.264] |
| step sweep n = 4 | gsm50k_c3_foreign_s0 | 16 | 4 | 10.4 | [7.8, 13.2] | 7.222 | [7.176, 7.274] |
| step sweep n = 4 | gsm50k_c1b_k8_s0 | 19 | 4 | 9.6 | [7.0, 12.4] | 7.260 | [7.208, 7.316] |
| step sweep n = 4 | gsm50k_coconut_s0 | 17 | 4 | 10.6 | [8.0, 13.4] | 7.280 | [7.224, 7.342] |
| step sweep n = 4 | gsm50k_c2c3_s0 | 21 | 4 | 11.6 | [8.8, 14.4] | 7.282 | [7.228, 7.340] |
| step sweep n = 4 | gsm50k_c1b_k4_s0 | 15 | 4 | 9.2 | [6.8, 11.8] | 7.292 | [7.236, 7.352] |
| step sweep n = 4 | gsm50k_c2_s0 | 20 | 4 | 11.2 | [8.4, 14.0] | 7.386 | [7.222, 7.674] |
| step sweep n = 0 | gsm50k_c2c3_s0 | 21 | 0 | 0.0 | [0.0, 0.0] | 7.424 | [7.018, 7.910] |
| final selection | gsm50k_c3_foreign_s0 | 16 | 6 | 10.0 | [7.4, 12.8] | 9.216 | [9.168, 9.268] |
| final selection | gsm50k_c3_s0 | 25 | 6 | 10.4 | [7.8, 13.2] | 9.226 | [9.176, 9.278] |
| final selection | gsm50k_c2_shuffled_s0 | 17 | 6 | 11.6 | [8.8, 14.4] | 9.230 | [9.182, 9.282] |
| final selection | gsm50k_c2c3_s0 | 21 | 6 | 12.0 | [9.2, 15.0] | 9.268 | [9.216, 9.324] |
| final selection | gsm50k_c1b_k8_s0 | 19 | 6 | 9.4 | [7.0, 12.0] | 9.280 | [9.226, 9.338] |
| final selection | gsm50k_c1b_k4_s0 | 15 | 6 | 9.4 | [7.0, 12.0] | 9.284 | [9.230, 9.342] |
| final selection | gsm50k_c2_s0 | 20 | 6 | 12.0 | [9.2, 15.0] | 9.310 | [9.224, 9.430] |
| final selection | gsm50k_coconut_s0 | 17 | 6 | 11.0 | [8.4, 13.8] | 9.396 | [9.232, 9.676] |
| step sweep n = 8 (OOD) | gsm50k_c3_s0 | 25 | 8 | 10.4 | [7.8, 13.2] | 11.204 | [11.156, 11.254] |
| step sweep n = 8 (OOD) | gsm50k_c3_foreign_s0 | 16 | 8 | 9.8 | [7.4, 12.4] | 11.222 | [11.174, 11.274] |
| step sweep n = 8 (OOD) | gsm50k_c2_shuffled_s0 | 17 | 8 | 11.8 | [9.0, 14.8] | 11.226 | [11.178, 11.276] |
| step sweep n = 8 (OOD) | gsm50k_c2c3_s0 | 21 | 8 | 12.2 | [9.4, 15.2] | 11.244 | [11.194, 11.298] |
| step sweep n = 8 (OOD) | gsm50k_coconut_s0 | 17 | 8 | 11.6 | [8.8, 14.4] | 11.262 | [11.210, 11.318] |
| step sweep n = 8 (OOD) | gsm50k_c1b_k4_s0 | 15 | 8 | 9.6 | [7.2, 12.2] | 11.276 | [11.222, 11.332] |
| step sweep n = 8 (OOD) | gsm50k_c1b_k8_s0 | 19 | 8 | 9.4 | [7.0, 12.0] | 11.282 | [11.224, 11.346] |
| step sweep n = 8 (OOD) | gsm50k_c2_s0 | 20 | 8 | 12.0 | [9.2, 14.8] | 11.306 | [11.218, 11.430] |
| earlier-stage checkpoint (from means) | gsm50k_pause_s0 | 5 | 2 | 13.6 | no band | 23.194 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c2_shuffled_s0 | 6 | 2 | 13.0 | no band | 24.302 | no band |
| earlier-stage checkpoint (from means) | gsm50k_coconut_s0 | 6 | 2 | 12.6 | no band | 24.384 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c2_s0 | 6 | 2 | 12.6 | no band | 24.470 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c3_s0 | 6 | 2 | 13.4 | no band | 24.536 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c1b_k8_s0 | 6 | 2 | 14.0 | no band | 24.724 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c2c3_s0 | 5 | 2 | 12.8 | no band | 25.756 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c3_foreign_s0 | 6 | 2 | 11.0 | no band | 25.856 | no band |
| earlier-stage checkpoint (from means) | gsm50k_c1b_k4_s0 | 6 | 2 | 14.6 | no band | 25.858 | no band |
| tuned bar (from means) | gsm50k_cot_lr3e-4_s0 | 17 | 0 | 25.6 | no band | 31.436 | no band |

### GSM-50k, test: accuracy against mean emitted tokens E'

| point | run | epoch | n latents | accuracy | band (accuracy) | E' mean | band (E') |
|---|---|---|---|---|---|---|---|
| pause (control) | gsm50k_pause_s0 | 22 | 6 | 6.9 | [5.6, 8.3] | 2.205 | [2.177, 2.235] |
| final selection | gsm50k_c3_s0 | 25 | 6 | 8.9 | [7.4, 10.5] | 2.229 | [2.198, 2.260] |
| final selection | gsm50k_c3_foreign_s0 | 16 | 6 | 9.0 | [7.5, 10.6] | 2.231 | [2.202, 2.262] |
| final selection | gsm50k_c1b_k8_s0 | 19 | 6 | 8.3 | [6.8, 9.8] | 2.256 | [2.223, 2.290] |
| final selection | gsm50k_c2_shuffled_s0 | 17 | 6 | 9.9 | [8.3, 11.6] | 2.259 | [2.228, 2.292] |
| final selection | gsm50k_c2_s0 | 20 | 6 | 10.2 | [8.5, 11.8] | 2.269 | [2.236, 2.303] |
| final selection | gsm50k_c2c3_s0 | 21 | 6 | 9.3 | [7.8, 10.9] | 2.278 | [2.246, 2.312] |
| no_cot | gsm50k_no_cot_s0 | 22 | 0 | 6.4 | [5.2, 7.8] | 2.280 | [2.208, 2.396] |
| final selection | gsm50k_coconut_s0 | 17 | 6 | 9.2 | [7.7, 10.8] | 2.300 | [2.265, 2.336] |
| plain depth 355M | gsm50k_plain_depth_355m_s0 | 24 | 0 | 7.4 | [6.1, 8.9] | 2.351 | [2.312, 2.390] |
| final selection | gsm50k_c1b_k4_s0 | 15 | 6 | 7.5 | [6.1, 8.9] | 2.354 | [2.292, 2.443] |
| earlier-stage checkpoint | gsm50k_c3_s0 | 6 | 2 | 12.5 | [10.8, 14.3] | 20.989 | [20.374, 21.612] |
| earlier-stage checkpoint | gsm50k_c2_s0 | 6 | 2 | 13.5 | [11.7, 15.3] | 21.323 | [20.707, 21.942] |
| earlier-stage checkpoint | gsm50k_c2_shuffled_s0 | 6 | 2 | 13.4 | [11.7, 15.2] | 21.347 | [20.724, 21.988] |
| earlier-stage checkpoint | gsm50k_coconut_s0 | 6 | 2 | 12.7 | [10.8, 14.5] | 21.480 | [20.854, 22.103] |
| earlier-stage checkpoint | gsm50k_c1b_k8_s0 | 6 | 2 | 12.0 | [10.3, 13.7] | 21.904 | [21.305, 22.502] |
| earlier-stage checkpoint | gsm50k_pause_s0 | 5 | 2 | 11.8 | [10.1, 13.6] | 22.455 | [21.826, 23.089] |
| earlier-stage checkpoint | gsm50k_c3_foreign_s0 | 6 | 2 | 11.8 | [10.0, 13.6] | 22.815 | [22.201, 23.445] |
| earlier-stage checkpoint | gsm50k_c1b_k4_s0 | 6 | 2 | 13.9 | [12.1, 15.8] | 23.017 | [22.368, 23.665] |
| earlier-stage checkpoint | gsm50k_c2c3_s0 | 5 | 2 | 12.9 | [11.1, 14.7] | 23.198 | [22.552, 23.859] |
| tuned bar | gsm50k_cot_lr3e-4_s0 | 17 | 0 | 21.0 | [18.9, 23.2] | 30.684 | [30.057, 31.293] |

### GSM-50k, test: accuracy against mean FLOPs (5.1; GFLOPs)

| point | run | epoch | n latents | accuracy | band (accuracy) | FLOPs mean | band (FLOPs) | reference-cost FLOPs mean |
|---|---|---|---|---|---|---|---|---|
| no_cot | gsm50k_no_cot_s0 | 22 | 0 | 6.4 | [5.2, 7.8] | 10.553 | [10.356, 10.756] | 48.702 |
| final selection | gsm50k_c1b_k4_s0 | 15 | 6 | 7.5 | [6.1, 8.9] | 11.262 (the read's layer fraction 0.3333, as the file records) | [11.066, 11.463] | 146.628 |
| final selection | gsm50k_c1b_k8_s0 | 19 | 6 | 8.3 | [6.8, 9.8] | 11.582 (the read's layer fraction 0.6667, as the file records) | [11.387, 11.785] | 144.909 |
| pause (control) | gsm50k_pause_s0 | 22 | 6 | 6.9 | [5.6, 8.3] | 11.914 | [11.719, 12.116] | 53.441 |
| final selection | gsm50k_c3_s0 | 25 | 6 | 8.9 | [7.4, 10.5] | 11.920 | [11.725, 12.122] | 144.468 |
| final selection | gsm50k_c3_foreign_s0 | 16 | 6 | 9.0 | [7.5, 10.6] | 11.921 | [11.725, 12.123] | 144.612 |
| final selection | gsm50k_c2_shuffled_s0 | 17 | 6 | 9.9 | [8.3, 11.6] | 11.928 | [11.732, 12.130] | 145.152 |
| final selection | gsm50k_c2_s0 | 20 | 6 | 10.2 | [8.5, 11.8] | 11.930 | [11.734, 12.132] | 145.157 |
| final selection | gsm50k_c2c3_s0 | 21 | 6 | 9.3 | [7.8, 10.9] | 11.932 | [11.736, 12.135] | 145.384 |
| final selection | gsm50k_coconut_s0 | 17 | 6 | 9.2 | [7.7, 10.8] | 11.938 | [11.741, 12.140] | 145.702 |
| earlier-stage checkpoint | gsm50k_c3_s0 | 6 | 2 | 12.5 | [10.8, 14.3] | 15.927 | [15.633, 16.225] | 462.370 |
| earlier-stage checkpoint | gsm50k_c2_s0 | 6 | 2 | 13.5 | [11.7, 15.3] | 16.010 | [15.718, 16.311] | 470.486 |
| earlier-stage checkpoint | gsm50k_c2_shuffled_s0 | 6 | 2 | 13.4 | [11.7, 15.2] | 16.016 | [15.723, 16.321] | 471.595 |
| earlier-stage checkpoint | gsm50k_c1b_k8_s0 | 6 | 2 | 12.0 | [10.3, 13.7] | 16.041 (the read's layer fraction 0.6667, as the file records) | [15.753, 16.342] | 480.980 |
| earlier-stage checkpoint | gsm50k_coconut_s0 | 6 | 2 | 12.7 | [10.8, 14.5] | 16.049 | [15.757, 16.347] | 472.998 |
| earlier-stage checkpoint | gsm50k_c1b_k4_s0 | 6 | 2 | 13.9 | [12.1, 15.8] | 16.205 (the read's layer fraction 0.3333, as the file records) | [15.905, 16.510] | 507.470 |
| earlier-stage checkpoint | gsm50k_pause_s0 | 5 | 2 | 11.8 | [10.1, 13.6] | 16.294 | [16.000, 16.595] | 469.394 |
| earlier-stage checkpoint | gsm50k_c3_foreign_s0 | 6 | 2 | 11.8 | [10.0, 13.6] | 16.384 | [16.082, 16.691] | 504.636 |
| earlier-stage checkpoint | gsm50k_c2c3_s0 | 5 | 2 | 12.9 | [11.1, 14.7] | 16.480 | [16.180, 16.790] | 513.560 |
| tuned bar | gsm50k_cot_lr3e-4_s0 | 17 | 0 | 21.0 | [18.9, 23.2] | 17.662 | [17.363, 17.970] | 622.787 |
| plain depth 355M | gsm50k_plain_depth_355m_s0 | 24 | 0 | 7.4 | [6.1, 8.9] | 36.930 | [36.234, 37.644] | 141.536 |

### GSM-50k, test: accuracy against mean sequential passes (5.4 (b): the matched-passes side)

| point | run | epoch | n latents | accuracy | band (accuracy) | passes mean | band (passes) |
|---|---|---|---|---|---|---|---|
| pause (control) | gsm50k_pause_s0 | 22 | 6 | 6.9 | [5.6, 8.3] | 3.205 | [3.177, 3.235] |
| no_cot | gsm50k_no_cot_s0 | 22 | 0 | 6.4 | [5.2, 7.8] | 3.280 | [3.208, 3.396] |
| plain depth 355M | gsm50k_plain_depth_355m_s0 | 24 | 0 | 7.4 | [6.1, 8.9] | 3.351 | [3.312, 3.390] |
| final selection | gsm50k_c3_s0 | 25 | 6 | 8.9 | [7.4, 10.5] | 9.229 | [9.198, 9.260] |
| final selection | gsm50k_c3_foreign_s0 | 16 | 6 | 9.0 | [7.5, 10.6] | 9.231 | [9.202, 9.262] |
| final selection | gsm50k_c1b_k8_s0 | 19 | 6 | 8.3 | [6.8, 9.8] | 9.256 | [9.223, 9.290] |
| final selection | gsm50k_c2_shuffled_s0 | 17 | 6 | 9.9 | [8.3, 11.6] | 9.259 | [9.228, 9.292] |
| final selection | gsm50k_c2_s0 | 20 | 6 | 10.2 | [8.5, 11.8] | 9.269 | [9.236, 9.303] |
| final selection | gsm50k_c2c3_s0 | 21 | 6 | 9.3 | [7.8, 10.9] | 9.278 | [9.246, 9.312] |
| final selection | gsm50k_coconut_s0 | 17 | 6 | 9.2 | [7.7, 10.8] | 9.300 | [9.265, 9.336] |
| final selection | gsm50k_c1b_k4_s0 | 15 | 6 | 7.5 | [6.1, 8.9] | 9.354 | [9.292, 9.443] |
| earlier-stage checkpoint | gsm50k_pause_s0 | 5 | 2 | 11.8 | [10.1, 13.6] | 23.455 | [22.826, 24.089] |
| earlier-stage checkpoint | gsm50k_c3_s0 | 6 | 2 | 12.5 | [10.8, 14.3] | 23.989 | [23.374, 24.612] |
| earlier-stage checkpoint | gsm50k_c2_s0 | 6 | 2 | 13.5 | [11.7, 15.3] | 24.323 | [23.707, 24.942] |
| earlier-stage checkpoint | gsm50k_c2_shuffled_s0 | 6 | 2 | 13.4 | [11.7, 15.2] | 24.347 | [23.724, 24.988] |
| earlier-stage checkpoint | gsm50k_coconut_s0 | 6 | 2 | 12.7 | [10.8, 14.5] | 24.480 | [23.854, 25.103] |
| earlier-stage checkpoint | gsm50k_c1b_k8_s0 | 6 | 2 | 12.0 | [10.3, 13.7] | 24.904 | [24.305, 25.502] |
| earlier-stage checkpoint | gsm50k_c3_foreign_s0 | 6 | 2 | 11.8 | [10.0, 13.6] | 25.815 | [25.201, 26.445] |
| earlier-stage checkpoint | gsm50k_c1b_k4_s0 | 6 | 2 | 13.9 | [12.1, 15.8] | 26.017 | [25.368, 26.665] |
| earlier-stage checkpoint | gsm50k_c2c3_s0 | 5 | 2 | 12.9 | [11.1, 14.7] | 26.198 | [25.552, 26.859] |
| tuned bar | gsm50k_cot_lr3e-4_s0 | 17 | 0 | 21.0 | [18.9, 23.2] | 31.684 | [31.057, 32.293] |

### GSM-50k: where the earlier-stage points sit (descriptive; decides nothing)

- valid: the earlier-stage checkpoint of gsm50k_pause_s0 (epoch 5, n latents 2) scores 13.6 at 22.19 tokens and [15.942, 15.984] GFLOPs; its selection (epoch 22, n latents 6) 9.6 at 2.21 and 11.643 (gaps: accuracy +4.0 points, tokens +19.98, FLOPs +4.320 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_coconut_s0 (epoch 6, n latents 2) scores 12.6 at 21.38 tokens and [15.739, 15.781] GFLOPs; its selection (epoch 17, n latents 6) 11.0 at 2.40 and 11.689 (gaps: accuracy +1.6 points, tokens +18.99, FLOPs +4.071 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c1b_k8_s0 (epoch 6, n latents 2) scores 14.0 at 21.72 tokens and [15.824, 15.866] GFLOPs; its selection (epoch 19, n latents 6) 9.4 at 2.28 and 11.660 (gaps: accuracy +4.6 points, tokens +19.44, FLOPs +4.185 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c1b_k4_s0 (epoch 6, n latents 2) scores 14.6 at 22.86 tokens and [16.108, 16.150] GFLOPs; its selection (epoch 15, n latents 6) 9.4 at 2.28 and 11.661 (gaps: accuracy +5.2 points, tokens +20.57, FLOPs +4.468 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c2_s0 (epoch 6, n latents 2) scores 12.6 at 21.47 tokens and [15.761, 15.802] GFLOPs; its selection (epoch 20, n latents 6) 12.0 at 2.31 and 11.668 (gaps: accuracy +0.6 points, tokens +19.16, FLOPs +4.114 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c2_shuffled_s0 (epoch 6, n latents 2) scores 13.0 at 21.30 tokens and [15.719, 15.760] GFLOPs; its selection (epoch 17, n latents 6) 11.6 at 2.23 and 11.648 (gaps: accuracy +1.4 points, tokens +19.07, FLOPs +4.092 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c3_s0 (epoch 6, n latents 2) scores 13.4 at 21.54 tokens and [15.777, 15.819] GFLOPs; its selection (epoch 25, n latents 6) 10.4 at 2.23 and 11.647 (gaps: accuracy +3.0 points, tokens +19.31, FLOPs +4.151 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c3_foreign_s0 (epoch 6, n latents 2) scores 11.0 at 22.86 tokens and [16.108, 16.150] GFLOPs; its selection (epoch 16, n latents 6) 10.0 at 2.22 and 11.644 (gaps: accuracy +1.0 points, tokens +20.64, FLOPs +4.485 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of gsm50k_c2c3_s0 (epoch 5, n latents 2) scores 12.8 at 22.76 tokens and [16.083, 16.125] GFLOPs; its selection (epoch 21, n latents 6) 12.0 at 2.27 and 11.657 (gaps: accuracy +0.8 points, tokens +20.49, FLOPs +4.447 GFLOPs at the range midpoints)
- test: the earlier-stage checkpoint of gsm50k_pause_s0 (epoch 5, n latents 2) scores 11.8 at 22.45 tokens and 16.294 GFLOPs; its selection (epoch 22, n latents 6) 6.9 at 2.21 and 11.914 (gaps: accuracy +4.9 points, tokens +20.25, FLOPs +4.379 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_coconut_s0 (epoch 6, n latents 2) scores 12.7 at 21.48 tokens and 16.049 GFLOPs; its selection (epoch 17, n latents 6) 9.2 at 2.30 and 11.938 (gaps: accuracy +3.5 points, tokens +19.18, FLOPs +4.112 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c1b_k8_s0 (epoch 6, n latents 2) scores 12.0 at 21.90 tokens and 16.041 GFLOPs; its selection (epoch 19, n latents 6) 8.3 at 2.26 and 11.582 (gaps: accuracy +3.7 points, tokens +19.65, FLOPs +4.459 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c1b_k4_s0 (epoch 6, n latents 2) scores 13.9 at 23.02 tokens and 16.205 GFLOPs; its selection (epoch 15, n latents 6) 7.5 at 2.35 and 11.262 (gaps: accuracy +6.4 points, tokens +20.66, FLOPs +4.943 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c2_s0 (epoch 6, n latents 2) scores 13.5 at 21.32 tokens and 16.010 GFLOPs; its selection (epoch 20, n latents 6) 10.2 at 2.27 and 11.930 (gaps: accuracy +3.3 points, tokens +19.05, FLOPs +4.080 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c2_shuffled_s0 (epoch 6, n latents 2) scores 13.4 at 21.35 tokens and 16.016 GFLOPs; its selection (epoch 17, n latents 6) 9.9 at 2.26 and 11.928 (gaps: accuracy +3.5 points, tokens +19.09, FLOPs +4.089 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c3_s0 (epoch 6, n latents 2) scores 12.5 at 20.99 tokens and 15.927 GFLOPs; its selection (epoch 25, n latents 6) 8.9 at 2.23 and 11.920 (gaps: accuracy +3.6 points, tokens +18.76, FLOPs +4.007 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c3_foreign_s0 (epoch 6, n latents 2) scores 11.8 at 22.82 tokens and 16.384 GFLOPs; its selection (epoch 16, n latents 6) 9.0 at 2.23 and 11.921 (gaps: accuracy +2.7 points, tokens +20.58, FLOPs +4.464 GFLOPs)
- test: the earlier-stage checkpoint of gsm50k_c2c3_s0 (epoch 5, n latents 2) scores 12.9 at 23.20 tokens and 16.480 GFLOPs; its selection (epoch 21, n latents 6) 9.3 at 2.28 and 11.932 (gaps: accuracy +3.6 points, tokens +20.92, FLOPs +4.548 GFLOPs)
- The valid earlier-stage FLOPs are ranges over the unrecorded per-example E' (widest 0.263% of the point's FLOPs; under 1%: no valid eval of ckpt_best.pt is needed by row (c) (2))

## ProsQA-hard

Valid of record: results/screens/prosqa_hard/prosqa_hard_screen_with_plain_depth.json at `rev18` (valid 300); test: results/screens/test_read.json at `rev1` (test 500; end-of-round test read (A3.4): reporting only, no decision attached; 1 seed; one registered read per checkpoint (4.5)).

### ProsQA-hard, valid: accuracy against mean emitted tokens E'

| point | run | epoch | n latents | accuracy | band (accuracy) | E' mean | band (E') |
|---|---|---|---|---|---|---|---|
| step sweep n = 0 | prosqa_hard_c2_shuffled_s0 | 30 | 0 | 70.0 | [65.0, 75.0] | 8.137 | [8.080, 8.193] |
| step sweep n = 1 | prosqa_hard_c1b_k8_s0 | 27 | 1 | 70.0 | [64.7, 75.3] | 8.143 | [8.083, 8.203] |
| final selection | prosqa_hard_c1b_k4_s0 | 30 | 7 | 70.7 | [65.3, 75.7] | 8.147 | [8.087, 8.207] |
| step sweep n = 1 | prosqa_hard_c1b_k4_s0 | 30 | 1 | 70.3 | [65.3, 75.3] | 8.147 | [8.087, 8.207] |
| step sweep n = 2 | prosqa_hard_c1b_k4_s0 | 30 | 2 | 70.3 | [65.3, 75.3] | 8.147 | [8.087, 8.207] |
| step sweep n = 4 | prosqa_hard_c1b_k4_s0 | 30 | 4 | 70.7 | [65.3, 75.7] | 8.147 | [8.087, 8.207] |
| step sweep n = 8 (OOD) | prosqa_hard_c1b_k4_s0 | 30 | 8 | 70.7 | [65.3, 75.7] | 8.147 | [8.087, 8.207] |
| step sweep n = 0 | prosqa_hard_c1b_k8_s0 | 27 | 0 | 64.0 | [58.7, 69.3] | 8.147 | [8.087, 8.210] |
| final selection | prosqa_hard_c2_shuffled_s0 | 30 | 7 | 71.3 | [66.3, 76.3] | 8.147 | [8.090, 8.207] |
| step sweep n = 1 | prosqa_hard_c2_shuffled_s0 | 30 | 1 | 71.0 | [66.0, 76.0] | 8.147 | [8.090, 8.207] |
| step sweep n = 2 | prosqa_hard_c2_shuffled_s0 | 30 | 2 | 71.0 | [66.0, 76.0] | 8.147 | [8.090, 8.207] |
| step sweep n = 8 (OOD) | prosqa_hard_c2_shuffled_s0 | 30 | 8 | 70.7 | [65.7, 75.7] | 8.147 | [8.090, 8.207] |
| final selection | prosqa_hard_coconut_s0 | 30 | 7 | 73.3 | [68.3, 78.3] | 8.147 | [8.087, 8.207] |
| step sweep n = 0 | prosqa_hard_coconut_s0 | 30 | 0 | 72.7 | [67.7, 77.7] | 8.147 | [8.087, 8.207] |
| step sweep n = 1 | prosqa_hard_coconut_s0 | 30 | 1 | 73.7 | [69.0, 78.7] | 8.147 | [8.087, 8.207] |
| step sweep n = 2 | prosqa_hard_coconut_s0 | 30 | 2 | 73.3 | [68.3, 78.3] | 8.147 | [8.087, 8.207] |
| step sweep n = 4 | prosqa_hard_coconut_s0 | 30 | 4 | 73.7 | [69.0, 78.7] | 8.147 | [8.087, 8.207] |
| step sweep n = 8 (OOD) | prosqa_hard_coconut_s0 | 30 | 8 | 73.3 | [68.3, 78.3] | 8.147 | [8.087, 8.207] |
| step sweep n = 2 | prosqa_hard_c1b_k8_s0 | 27 | 2 | 71.0 | [66.0, 76.0] | 8.150 | [8.090, 8.213] |
| step sweep n = 4 | prosqa_hard_c2_shuffled_s0 | 30 | 4 | 71.3 | [66.3, 76.3] | 8.150 | [8.093, 8.210] |
| step sweep n = 0 | prosqa_hard_c1b_k4_s0 | 30 | 0 | 70.3 | [65.0, 75.7] | 8.153 | [8.093, 8.213] |
| pause (control) | prosqa_hard_pause_s0 | 30 | 7 | 74.7 | [69.7, 79.7] | 8.153 | [8.093, 8.217] |
| step sweep n = 4 | prosqa_hard_c1b_k8_s0 | 27 | 4 | 72.3 | [67.3, 77.3] | 8.160 | [8.100, 8.223] |
| final selection | prosqa_hard_c1b_k8_s0 | 27 | 7 | 71.7 | [66.7, 76.7] | 8.163 | [8.103, 8.223] |
| step sweep n = 8 (OOD) | prosqa_hard_c1b_k8_s0 | 27 | 8 | 72.0 | [67.0, 77.0] | 8.167 | [8.107, 8.227] |
| final selection | prosqa_hard_c2_s0 | 27 | 7 | 72.3 | [67.3, 77.3] | 8.180 | [8.123, 8.240] |
| step sweep n = 0 | prosqa_hard_c2_s0 | 27 | 0 | 71.7 | [66.7, 76.7] | 8.180 | [8.123, 8.240] |
| step sweep n = 1 | prosqa_hard_c2_s0 | 27 | 1 | 72.0 | [67.0, 77.0] | 8.180 | [8.123, 8.240] |
| step sweep n = 2 | prosqa_hard_c2_s0 | 27 | 2 | 72.0 | [67.0, 77.0] | 8.180 | [8.123, 8.240] |
| step sweep n = 4 | prosqa_hard_c2_s0 | 27 | 4 | 72.3 | [67.3, 77.3] | 8.180 | [8.123, 8.240] |
| step sweep n = 8 (OOD) | prosqa_hard_c2_s0 | 27 | 8 | 72.3 | [67.3, 77.3] | 8.180 | [8.123, 8.240] |
| plain depth 355M (from means) | prosqa_hard_plain_depth_355m_s0 | 28 | 0 | 61.7 | no band | 8.180 | no band |
| no_cot (from means) | prosqa_hard_no_cot_s0 | 8 | 0 | 57.7 | no band | 8.190 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c1b_k4_s0 | 18 | 5 | 74.3 | no band | 22.763 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c1b_k8_s0 | 14 | 4 | 75.0 | no band | 34.307 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_coconut_s0 | 8 | 2 | 76.7 | no band | 54.027 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c2_s0 | 9 | 2 | 81.7 | no band | 55.970 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c2_shuffled_s0 | 9 | 2 | 76.7 | no band | 57.060 | no band |
| tuned bar (from means) | prosqa_hard_cot_lr3e-5_s0 | 14 | 0 | 80.3 | no band | 76.070 | no band |

### ProsQA-hard, valid: accuracy against mean FLOPs (5.1; GFLOPs)

| point | run | epoch | n latents | accuracy | band (accuracy) | FLOPs mean | band (FLOPs) | reference-cost FLOPs mean |
|---|---|---|---|---|---|---|---|---|
| no_cot (from means) | prosqa_hard_no_cot_s0 | 8 | 0 | 57.7 | no band | [110.239, 110.376] | no band | [1047.183, 2057.251] |
| step sweep n = 0 | prosqa_hard_c2_shuffled_s0 | 30 | 0 | 70.0 | [65.0, 75.0] | 110.662 | [108.089, 113.200] | 1418.786 |
| step sweep n = 0 | prosqa_hard_c1b_k8_s0 | 27 | 0 | 64.0 | [58.7, 69.3] | 110.664 (the read's layer fraction 0.6667, as the file records) | [108.091, 113.201] | 1419.839 |
| step sweep n = 0 | prosqa_hard_coconut_s0 | 30 | 0 | 72.7 | [67.7, 77.7] | 110.664 | [108.089, 113.201] | 1420.137 |
| step sweep n = 0 | prosqa_hard_c1b_k4_s0 | 30 | 0 | 70.3 | [65.0, 75.7] | 110.666 (the read's layer fraction 0.3333, as the file records) | [108.093, 113.203] | 1421.842 |
| step sweep n = 0 | prosqa_hard_c2_s0 | 27 | 0 | 71.7 | [66.7, 76.7] | 110.673 | [108.105, 113.209] | 1425.672 |
| step sweep n = 1 | prosqa_hard_c1b_k4_s0 | 30 | 1 | 70.3 | [65.3, 75.3] | 110.729 (the read's layer fraction 0.3333, as the file records) | [108.156, 113.266] | 1577.435 |
| step sweep n = 1 | prosqa_hard_c1b_k8_s0 | 27 | 1 | 70.0 | [64.7, 75.3] | 110.792 (the read's layer fraction 0.6667, as the file records) | [108.219, 113.329] | 1576.291 |
| step sweep n = 2 | prosqa_hard_c1b_k4_s0 | 30 | 2 | 70.3 | [65.3, 75.3] | 110.793 (the read's layer fraction 0.3333, as the file records) | [108.221, 113.331] | 1734.388 |
| step sweep n = 1 | prosqa_hard_c2_shuffled_s0 | 30 | 1 | 71.0 | [66.0, 76.0] | 110.857 | [108.286, 113.395] | 1576.649 |
| step sweep n = 1 | prosqa_hard_coconut_s0 | 30 | 1 | 73.7 | [69.0, 78.7] | 110.857 | [108.281, 113.394] | 1576.821 |
| step sweep n = 1 | prosqa_hard_c2_s0 | 27 | 1 | 72.0 | [67.0, 77.0] | 110.866 | [108.297, 113.402] | 1582.365 |
| step sweep n = 4 | prosqa_hard_c1b_k4_s0 | 30 | 4 | 70.7 | [65.3, 75.7] | 110.922 (the read's layer fraction 0.3333, as the file records) | [108.349, 113.460] | 2049.105 |
| step sweep n = 2 | prosqa_hard_c1b_k8_s0 | 27 | 2 | 71.0 | [66.0, 76.0] | 110.922 (the read's layer fraction 0.6667, as the file records) | [108.349, 113.459] | 1734.629 |
| step sweep n = 2 | prosqa_hard_c2_shuffled_s0 | 30 | 2 | 71.0 | [66.0, 76.0] | 111.049 | [108.478, 113.588] | 1733.602 |
| step sweep n = 2 | prosqa_hard_coconut_s0 | 30 | 2 | 73.3 | [68.3, 78.3] | 111.049 | [108.473, 113.587] | 1733.774 |
| step sweep n = 2 | prosqa_hard_c2_s0 | 27 | 2 | 72.0 | [67.0, 77.0] | 111.058 | [108.489, 113.594] | 1739.328 |
| step sweep n = 8 (OOD) | prosqa_hard_c1b_k4_s0 | 30 | 8 | 70.7 | [65.3, 75.7] | 111.180 (the read's layer fraction 0.3333, as the file records) | [108.606, 113.718] | 2681.775 |
| step sweep n = 4 | prosqa_hard_c1b_k8_s0 | 27 | 4 | 72.3 | [67.3, 77.3] | 111.182 (the read's layer fraction 0.6667, as the file records) | [108.609, 113.720] | 2051.001 |
| step sweep n = 4 | prosqa_hard_coconut_s0 | 30 | 4 | 73.7 | [69.0, 78.7] | 111.434 | [108.857, 113.973] | 2048.490 |
| step sweep n = 4 | prosqa_hard_c2_shuffled_s0 | 30 | 4 | 71.3 | [66.3, 76.3] | 111.435 | [108.863, 113.974] | 2048.817 |
| step sweep n = 4 | prosqa_hard_c2_s0 | 27 | 4 | 72.3 | [67.3, 77.3] | 111.443 | [108.873, 113.981] | 2054.062 |
| step sweep n = 8 (OOD) | prosqa_hard_c1b_k8_s0 | 27 | 8 | 72.0 | [67.0, 77.0] | 111.698 (the read's layer fraction 0.6667, as the file records) | [109.123, 114.238] | 2684.383 |
| final selection | prosqa_hard_c2_shuffled_s0 | 30 | 7 | 71.3 | [66.3, 76.3] | 112.012 | [109.439, 114.554] | 2522.416 |
| final selection | prosqa_hard_coconut_s0 | 30 | 7 | 73.3 | [68.3, 78.3] | 112.012 | [109.434, 114.553] | 2522.588 |
| final selection | prosqa_hard_c1b_k4_s0 | 30 | 7 | 70.7 | [65.3, 75.7] | 112.012 at 1; 111.115 nominal 0.3333; 111.258 measured 0.4391 | [108.542, 113.654] (the plain read's layer fraction 0.3333) | 2523.203 |
| pause (control) | prosqa_hard_pause_s0 | 30 | 7 | 74.7 | [69.7, 79.7] | 112.014 | [109.440, 114.554] | 1441.224 |
| final selection | prosqa_hard_c1b_k8_s0 | 27 | 7 | 71.7 | [66.7, 76.7] | 112.017 at 1; 111.568 nominal 0.6667; 111.651 measured 0.7280 | [108.994, 114.107] (the plain read's layer fraction 0.6667) | 2525.493 |
| final selection | prosqa_hard_c2_s0 | 27 | 7 | 72.3 | [67.3, 77.3] | 112.021 | [109.449, 114.560] | 2528.187 |
| step sweep n = 8 (OOD) | prosqa_hard_c2_shuffled_s0 | 30 | 8 | 70.7 | [65.7, 75.7] | 112.205 | [109.631, 114.747] | 2680.988 |
| step sweep n = 8 (OOD) | prosqa_hard_coconut_s0 | 30 | 8 | 73.3 | [68.3, 78.3] | 112.205 | [109.626, 114.746] | 2681.160 |
| step sweep n = 8 (OOD) | prosqa_hard_c2_s0 | 27 | 8 | 72.3 | [67.3, 77.3] | 112.214 | [109.642, 114.753] | 2686.768 |
| earlier-stage checkpoint (from means) | prosqa_hard_c1b_k4_s0 | 18 | 5 | 74.3 | no band | [115.469, 115.758] at 1; [114.828, 115.118] nominal 0.3333; [114.930, 115.219] measured 0.4391 | no band | [3771.091, 5904.657] |
| earlier-stage checkpoint (from means) | prosqa_hard_c1b_k8_s0 | 14 | 4 | 75.0 | no band | [118.373, 118.741] at 1; [118.117, 118.485] nominal 0.6667; [118.164, 118.532] measured 0.7280 | no band | [5301.922, 8018.030] |
| earlier-stage checkpoint (from means) | prosqa_hard_coconut_s0 | 8 | 2 | 76.7 | no band | [123.309, 123.751] | no band | [8099.721, 11355.349] |
| earlier-stage checkpoint (from means) | prosqa_hard_c2_s0 | 9 | 2 | 81.7 | no band | [123.835, 124.281] | no band | [8422.792, 11703.516] |
| earlier-stage checkpoint (from means) | prosqa_hard_c2_shuffled_s0 | 9 | 2 | 76.7 | no band | [124.131, 124.578] | no band | [8605.173, 11897.129] |
| tuned bar (from means) | prosqa_hard_cot_lr3e-5_s0 | 14 | 0 | 80.3 | no band | [128.523, 128.960] | no band | [11546.838, 14766.932] |
| plain depth 355M (from means) | prosqa_hard_plain_depth_355m_s0 | 28 | 0 | 61.7 | no band | [384.111, 384.477] | no band | [2985.861, 5855.679] |

### ProsQA-hard, valid: accuracy against mean sequential passes (5.4 (b): the matched-passes side)

| point | run | epoch | n latents | accuracy | band (accuracy) | passes mean | band (passes) |
|---|---|---|---|---|---|---|---|
| step sweep n = 0 | prosqa_hard_c2_shuffled_s0 | 30 | 0 | 70.0 | [65.0, 75.0] | 9.137 | [9.080, 9.193] |
| step sweep n = 0 | prosqa_hard_c1b_k8_s0 | 27 | 0 | 64.0 | [58.7, 69.3] | 9.147 | [9.087, 9.210] |
| step sweep n = 0 | prosqa_hard_coconut_s0 | 30 | 0 | 72.7 | [67.7, 77.7] | 9.147 | [9.087, 9.207] |
| step sweep n = 0 | prosqa_hard_c1b_k4_s0 | 30 | 0 | 70.3 | [65.0, 75.7] | 9.153 | [9.093, 9.213] |
| pause (control) | prosqa_hard_pause_s0 | 30 | 7 | 74.7 | [69.7, 79.7] | 9.153 | [9.093, 9.217] |
| step sweep n = 0 | prosqa_hard_c2_s0 | 27 | 0 | 71.7 | [66.7, 76.7] | 9.180 | [9.123, 9.240] |
| plain depth 355M (from means) | prosqa_hard_plain_depth_355m_s0 | 28 | 0 | 61.7 | no band | 9.180 | no band |
| no_cot (from means) | prosqa_hard_no_cot_s0 | 8 | 0 | 57.7 | no band | 9.190 | no band |
| step sweep n = 1 | prosqa_hard_c1b_k8_s0 | 27 | 1 | 70.0 | [64.7, 75.3] | 10.143 | [10.083, 10.203] |
| step sweep n = 1 | prosqa_hard_c1b_k4_s0 | 30 | 1 | 70.3 | [65.3, 75.3] | 10.147 | [10.087, 10.207] |
| step sweep n = 1 | prosqa_hard_c2_shuffled_s0 | 30 | 1 | 71.0 | [66.0, 76.0] | 10.147 | [10.090, 10.207] |
| step sweep n = 1 | prosqa_hard_coconut_s0 | 30 | 1 | 73.7 | [69.0, 78.7] | 10.147 | [10.087, 10.207] |
| step sweep n = 1 | prosqa_hard_c2_s0 | 27 | 1 | 72.0 | [67.0, 77.0] | 10.180 | [10.123, 10.240] |
| step sweep n = 2 | prosqa_hard_c1b_k4_s0 | 30 | 2 | 70.3 | [65.3, 75.3] | 11.147 | [11.087, 11.207] |
| step sweep n = 2 | prosqa_hard_c2_shuffled_s0 | 30 | 2 | 71.0 | [66.0, 76.0] | 11.147 | [11.090, 11.207] |
| step sweep n = 2 | prosqa_hard_coconut_s0 | 30 | 2 | 73.3 | [68.3, 78.3] | 11.147 | [11.087, 11.207] |
| step sweep n = 2 | prosqa_hard_c1b_k8_s0 | 27 | 2 | 71.0 | [66.0, 76.0] | 11.150 | [11.090, 11.213] |
| step sweep n = 2 | prosqa_hard_c2_s0 | 27 | 2 | 72.0 | [67.0, 77.0] | 11.180 | [11.123, 11.240] |
| step sweep n = 4 | prosqa_hard_c1b_k4_s0 | 30 | 4 | 70.7 | [65.3, 75.7] | 13.147 | [13.087, 13.207] |
| step sweep n = 4 | prosqa_hard_coconut_s0 | 30 | 4 | 73.7 | [69.0, 78.7] | 13.147 | [13.087, 13.207] |
| step sweep n = 4 | prosqa_hard_c2_shuffled_s0 | 30 | 4 | 71.3 | [66.3, 76.3] | 13.150 | [13.093, 13.210] |
| step sweep n = 4 | prosqa_hard_c1b_k8_s0 | 27 | 4 | 72.3 | [67.3, 77.3] | 13.160 | [13.100, 13.223] |
| step sweep n = 4 | prosqa_hard_c2_s0 | 27 | 4 | 72.3 | [67.3, 77.3] | 13.180 | [13.123, 13.240] |
| final selection | prosqa_hard_c1b_k4_s0 | 30 | 7 | 70.7 | [65.3, 75.7] | 16.147 | [16.087, 16.207] |
| final selection | prosqa_hard_c2_shuffled_s0 | 30 | 7 | 71.3 | [66.3, 76.3] | 16.147 | [16.090, 16.207] |
| final selection | prosqa_hard_coconut_s0 | 30 | 7 | 73.3 | [68.3, 78.3] | 16.147 | [16.087, 16.207] |
| final selection | prosqa_hard_c1b_k8_s0 | 27 | 7 | 71.7 | [66.7, 76.7] | 16.163 | [16.103, 16.223] |
| final selection | prosqa_hard_c2_s0 | 27 | 7 | 72.3 | [67.3, 77.3] | 16.180 | [16.123, 16.240] |
| step sweep n = 8 (OOD) | prosqa_hard_c1b_k4_s0 | 30 | 8 | 70.7 | [65.3, 75.7] | 17.147 | [17.087, 17.207] |
| step sweep n = 8 (OOD) | prosqa_hard_c2_shuffled_s0 | 30 | 8 | 70.7 | [65.7, 75.7] | 17.147 | [17.090, 17.207] |
| step sweep n = 8 (OOD) | prosqa_hard_coconut_s0 | 30 | 8 | 73.3 | [68.3, 78.3] | 17.147 | [17.087, 17.207] |
| step sweep n = 8 (OOD) | prosqa_hard_c1b_k8_s0 | 27 | 8 | 72.0 | [67.0, 77.0] | 17.167 | [17.107, 17.227] |
| step sweep n = 8 (OOD) | prosqa_hard_c2_s0 | 27 | 8 | 72.3 | [67.3, 77.3] | 17.180 | [17.123, 17.240] |
| earlier-stage checkpoint (from means) | prosqa_hard_c1b_k4_s0 | 18 | 5 | 74.3 | no band | 28.763 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c1b_k8_s0 | 14 | 4 | 75.0 | no band | 39.307 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_coconut_s0 | 8 | 2 | 76.7 | no band | 57.027 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c2_s0 | 9 | 2 | 81.7 | no band | 58.970 | no band |
| earlier-stage checkpoint (from means) | prosqa_hard_c2_shuffled_s0 | 9 | 2 | 76.7 | no band | 60.060 | no band |
| tuned bar (from means) | prosqa_hard_cot_lr3e-5_s0 | 14 | 0 | 80.3 | no band | 77.070 | no band |

### ProsQA-hard, test: accuracy against mean emitted tokens E'

| point | run | epoch | n latents | accuracy | band (accuracy) | E' mean | band (E') |
|---|---|---|---|---|---|---|---|
| plain depth 355M | prosqa_hard_plain_depth_355m_s0 | 28 | 0 | 60.4 | [56.0, 64.6] | 8.172 | [8.124, 8.220] |
| final selection | prosqa_hard_c1b_k4_s0 | 30 | 7 | 71.2 | [67.2, 75.0] | 8.178 | [8.132, 8.226] |
| final selection | prosqa_hard_coconut_s0 | 30 | 7 | 72.8 | [69.0, 76.6] | 8.178 | [8.134, 8.224] |
| final selection | prosqa_hard_c1b_k8_s0 | 27 | 7 | 73.6 | [69.8, 77.4] | 8.180 | [8.134, 8.228] |
| final selection | prosqa_hard_c2_shuffled_s0 | 30 | 7 | 72.6 | [68.6, 76.6] | 8.182 | [8.136, 8.228] |
| pause (control) | prosqa_hard_pause_s0 | 30 | 7 | 74.8 | [71.0, 78.6] | 8.182 | [8.134, 8.230] |
| no_cot | prosqa_hard_no_cot_s0 | 8 | 0 | 53.2 | [48.8, 57.6] | 8.186 | [8.140, 8.234] |
| final selection | prosqa_hard_c2_s0 | 27 | 7 | 75.8 | [72.0, 79.6] | 8.198 | [8.154, 8.244] |
| earlier-stage checkpoint | prosqa_hard_c1b_k4_s0 | 18 | 5 | 71.2 | [67.2, 75.2] | 25.164 | [23.540, 26.872] |
| earlier-stage checkpoint | prosqa_hard_c1b_k8_s0 | 14 | 4 | 76.8 | [73.0, 80.4] | 32.546 | [31.314, 33.828] |
| earlier-stage checkpoint | prosqa_hard_coconut_s0 | 8 | 2 | 73.8 | [70.0, 77.6] | 55.262 | [53.848, 56.770] |
| earlier-stage checkpoint | prosqa_hard_c2_shuffled_s0 | 9 | 2 | 78.8 | [75.2, 82.2] | 55.312 | [53.838, 56.818] |
| earlier-stage checkpoint | prosqa_hard_c2_s0 | 9 | 2 | 78.6 | [74.8, 82.2] | 56.804 | [55.204, 58.476] |
| tuned bar | prosqa_hard_cot_lr3e-5_s0 | 14 | 0 | 83.4 | [80.0, 86.6] | 77.794 | [76.476, 79.140] |

### ProsQA-hard, test: accuracy against mean FLOPs (5.1; GFLOPs)

| point | run | epoch | n latents | accuracy | band (accuracy) | FLOPs mean | band (FLOPs) | reference-cost FLOPs mean |
|---|---|---|---|---|---|---|---|---|
| no_cot | prosqa_hard_no_cot_s0 | 8 | 0 | 53.2 | [48.8, 57.6] | 111.510 | [109.558, 113.431] | 1439.436 |
| final selection | prosqa_hard_c1b_k4_s0 | 30 | 7 | 71.2 | [67.2, 75.0] | 112.345 (the read's layer fraction 0.3333, as the file records) | [110.388, 114.268] | 2555.432 |
| final selection | prosqa_hard_c1b_k8_s0 | 27 | 7 | 73.6 | [69.8, 77.4] | 112.794 (the read's layer fraction 0.6667, as the file records) | [110.839, 114.717] | 2555.520 |
| final selection | prosqa_hard_coconut_s0 | 30 | 7 | 72.8 | [69.0, 76.6] | 113.243 | [111.285, 115.169] | 2556.130 |
| pause (control) | prosqa_hard_pause_s0 | 30 | 7 | 74.8 | [71.0, 78.6] | 113.244 | [111.287, 115.168] | 1460.757 |
| final selection | prosqa_hard_c2_shuffled_s0 | 30 | 7 | 72.6 | [68.6, 76.6] | 113.244 | [111.286, 115.169] | 2555.961 |
| final selection | prosqa_hard_c2_s0 | 27 | 7 | 75.8 | [72.0, 79.6] | 113.248 | [111.291, 115.174] | 2558.519 |
| earlier-stage checkpoint | prosqa_hard_c1b_k4_s0 | 18 | 5 | 71.2 | [67.2, 75.2] | 116.794 (the read's layer fraction 0.3333, as the file records) | [114.922, 118.653] | 4878.190 |
| earlier-stage checkpoint | prosqa_hard_c1b_k8_s0 | 14 | 4 | 76.8 | [73.0, 80.4] | 119.002 (the read's layer fraction 0.6667, as the file records) | [117.001, 120.947] | 6063.189 |
| earlier-stage checkpoint | prosqa_hard_coconut_s0 | 8 | 2 | 73.8 | [70.0, 77.6] | 125.030 | [123.035, 126.975] | 9541.464 |
| earlier-stage checkpoint | prosqa_hard_c2_shuffled_s0 | 9 | 2 | 78.8 | [75.2, 82.2] | 125.045 | [123.044, 127.013] | 9562.289 |
| earlier-stage checkpoint | prosqa_hard_c2_s0 | 9 | 2 | 78.6 | [74.8, 82.2] | 125.452 | [123.429, 127.430] | 9829.984 |
| tuned bar | prosqa_hard_cot_lr3e-5_s0 | 14 | 0 | 83.4 | [80.0, 86.6] | 130.374 | [128.388, 132.291] | 13062.068 |
| plain depth 355M | prosqa_hard_plain_depth_355m_s0 | 28 | 0 | 60.4 | [56.0, 64.6] | 388.459 | [381.696, 395.081] | 4101.042 |

### ProsQA-hard, test: accuracy against mean sequential passes (5.4 (b): the matched-passes side)

| point | run | epoch | n latents | accuracy | band (accuracy) | passes mean | band (passes) |
|---|---|---|---|---|---|---|---|
| plain depth 355M | prosqa_hard_plain_depth_355m_s0 | 28 | 0 | 60.4 | [56.0, 64.6] | 9.172 | [9.124, 9.220] |
| pause (control) | prosqa_hard_pause_s0 | 30 | 7 | 74.8 | [71.0, 78.6] | 9.182 | [9.134, 9.230] |
| no_cot | prosqa_hard_no_cot_s0 | 8 | 0 | 53.2 | [48.8, 57.6] | 9.186 | [9.140, 9.234] |
| final selection | prosqa_hard_c1b_k4_s0 | 30 | 7 | 71.2 | [67.2, 75.0] | 16.178 | [16.132, 16.226] |
| final selection | prosqa_hard_coconut_s0 | 30 | 7 | 72.8 | [69.0, 76.6] | 16.178 | [16.134, 16.224] |
| final selection | prosqa_hard_c1b_k8_s0 | 27 | 7 | 73.6 | [69.8, 77.4] | 16.180 | [16.134, 16.228] |
| final selection | prosqa_hard_c2_shuffled_s0 | 30 | 7 | 72.6 | [68.6, 76.6] | 16.182 | [16.136, 16.228] |
| final selection | prosqa_hard_c2_s0 | 27 | 7 | 75.8 | [72.0, 79.6] | 16.198 | [16.154, 16.244] |
| earlier-stage checkpoint | prosqa_hard_c1b_k4_s0 | 18 | 5 | 71.2 | [67.2, 75.2] | 31.164 | [29.540, 32.872] |
| earlier-stage checkpoint | prosqa_hard_c1b_k8_s0 | 14 | 4 | 76.8 | [73.0, 80.4] | 37.546 | [36.314, 38.828] |
| earlier-stage checkpoint | prosqa_hard_coconut_s0 | 8 | 2 | 73.8 | [70.0, 77.6] | 58.262 | [56.848, 59.770] |
| earlier-stage checkpoint | prosqa_hard_c2_shuffled_s0 | 9 | 2 | 78.8 | [75.2, 82.2] | 58.312 | [56.838, 59.818] |
| earlier-stage checkpoint | prosqa_hard_c2_s0 | 9 | 2 | 78.6 | [74.8, 82.2] | 59.804 | [58.204, 61.476] |
| tuned bar | prosqa_hard_cot_lr3e-5_s0 | 14 | 0 | 83.4 | [80.0, 86.6] | 78.794 | [77.476, 80.140] |

### ProsQA-hard: where the earlier-stage points sit (descriptive; decides nothing)

- valid: the earlier-stage checkpoint of prosqa_hard_coconut_s0 (epoch 8, n latents 2) scores 76.7 at 54.03 tokens and [123.309, 123.751] GFLOPs; its selection (epoch 30, n latents 7) 73.3 at 8.15 and 112.012 (gaps: accuracy +3.3 points, tokens +45.88, FLOPs +11.517 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of prosqa_hard_c1b_k8_s0 (epoch 14, n latents 4) scores 75.0 at 34.31 tokens and [118.373, 118.741] GFLOPs; its selection (epoch 27, n latents 7) 71.7 at 8.16 and 112.017 (gaps: accuracy +3.3 points, tokens +26.14, FLOPs +6.540 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of prosqa_hard_c1b_k4_s0 (epoch 18, n latents 5) scores 74.3 at 22.76 tokens and [115.469, 115.758] GFLOPs; its selection (epoch 30, n latents 7) 70.7 at 8.15 and 112.012 (gaps: accuracy +3.7 points, tokens +14.62, FLOPs +3.601 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of prosqa_hard_c2_s0 (epoch 9, n latents 2) scores 81.7 at 55.97 tokens and [123.835, 124.281] GFLOPs; its selection (epoch 27, n latents 7) 72.3 at 8.18 and 112.021 (gaps: accuracy +9.3 points, tokens +47.79, FLOPs +12.037 GFLOPs at the range midpoints)
- valid: the earlier-stage checkpoint of prosqa_hard_c2_shuffled_s0 (epoch 9, n latents 2) scores 76.7 at 57.06 tokens and [124.131, 124.578] GFLOPs; its selection (epoch 30, n latents 7) 71.3 at 8.15 and 112.012 (gaps: accuracy +5.3 points, tokens +48.91, FLOPs +12.342 GFLOPs at the range midpoints)
- test: the earlier-stage checkpoint of prosqa_hard_coconut_s0 (epoch 8, n latents 2) scores 73.8 at 55.26 tokens and 125.030 GFLOPs; its selection (epoch 30, n latents 7) 72.8 at 8.18 and 113.243 (gaps: accuracy +1.0 points, tokens +47.08, FLOPs +11.787 GFLOPs)
- test: the earlier-stage checkpoint of prosqa_hard_c1b_k8_s0 (epoch 14, n latents 4) scores 76.8 at 32.55 tokens and 119.002 GFLOPs; its selection (epoch 27, n latents 7) 73.6 at 8.18 and 112.794 (gaps: accuracy +3.2 points, tokens +24.37, FLOPs +6.207 GFLOPs)
- test: the earlier-stage checkpoint of prosqa_hard_c1b_k4_s0 (epoch 18, n latents 5) scores 71.2 at 25.16 tokens and 116.794 GFLOPs; its selection (epoch 30, n latents 7) 71.2 at 8.18 and 112.345 (gaps: accuracy +0.0 points, tokens +16.99, FLOPs +4.449 GFLOPs)
- test: the earlier-stage checkpoint of prosqa_hard_c2_s0 (epoch 9, n latents 2) scores 78.6 at 56.80 tokens and 125.452 GFLOPs; its selection (epoch 27, n latents 7) 75.8 at 8.20 and 113.248 (gaps: accuracy +2.8 points, tokens +48.61, FLOPs +12.204 GFLOPs)
- test: the earlier-stage checkpoint of prosqa_hard_c2_shuffled_s0 (epoch 9, n latents 2) scores 78.8 at 55.31 tokens and 125.045 GFLOPs; its selection (epoch 30, n latents 7) 72.6 at 8.18 and 113.244 (gaps: accuracy +6.2 points, tokens +47.13, FLOPs +11.801 GFLOPs)
- The valid earlier-stage FLOPs are ranges over the unrecorded per-example E' (widest 0.359% of the point's FLOPs; under 1%: no valid eval of ckpt_best.pt is needed by row (c) (2))

## Not read here

- The 7.3 FLOP multiplier: UNTESTED (no budgeted-CoT read exists; it would be a new read of the bar checkpoint, outside (c)).
- The 7.3 budgeted-CoT curve: UNTESTED.

## Notes

- A compilation from the committed records (row (c) (2)): no VM, no new read of any checkpoint; it decides nothing and changes no line of any table.
- Valid and test are separate panels and never share an axis; every test point carries the test read's label: end-of-round test read (A3.4): reporting only, no decision attached; 1 seed; one registered read per checkpoint (4.5).
- Valid final selections: accuracy of record (the training log's selection, the screen table's); costs the table's cost row (exact means from the plain read's per-example file); the band from that plain read's per-example file. Valid earlier-stage points: the training log's epoch row (valid accuracy, val_emitted_mean), FLOPs by read_screen.Screen.costs on the logged mean (the exact range over every per-example E' with the logged total), passes exact (1 + latent steps + E'); no band.
- The bar, no_cot and plain depth 355M on valid: the tables' ranges (no per-example read of them on valid); on test every point is per-example (the test read's files). C1b FLOPs: on valid at the fractions the table charges (nominal, measured, at 1); on test at the read's own layer fraction, as the file records.
- Bands: example bootstrap, 10,000 draws, numpy default_rng(0), the 95% percentile interval of each mean on the same draws per point. FLOPs are 5.1's (the head charged as 5.4 says); the reference-cost FLOPs (5.5 (c)) beside them; the passes panel is 5.4 (b)'s matched-passes side.
- Not said here (row (c) (4)): no comparison of a test point with a valid point, no K3 beyond the tables' lines, nothing about what the latents carry (A11.3), no multiplier, no claim about the written steps of an earlier-stage checkpoint.
