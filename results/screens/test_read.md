# Screen round: the end-of-round test read (A3.4)

Labels on every number: end-of-round test read (A3.4): reporting only, no decision attached; 1 seed; one registered read per checkpoint (4.5); valid of record: the screen table's selection (A3.4: screen decisions use valid only); test minus valid: descriptive, decides nothing.

## GSM-50k

Job `screen-test-gsm50k-acctc` (acctc, attempt silent-eval-20261021-153214-e5fa), registered read 2026-10-21T15:27Z; test `data/gsm_test.json` (1319 items, sha256 46ce5b0c44fd...); valid of record: the fourth GSM-50k table `results/screens/gsm50k/gsm50k_screen_with_hinge.json` at 4a0e33d4d9e1 (valid 500 items). Interval: 6.2's form, +-1.96 binomial SE.

| run | read | checkpoint | valid of record | test (95% interval) | test emitted tokens (mean) | test minus valid (descriptive) |
|---|---|---|---|---|---|---|
| gsm50k_cot_lr3e-4_s0 | primary | ckpt_best_final.pt epoch 17, n_latent 0, d549ca7063d5 | 25.6 (128/500) | 21.0 [18.8, 23.2] (277/1319) | 30.68 | -4.6 |
| gsm50k_no_cot_s0 | primary | ckpt_best_final.pt epoch 22, n_latent 0, ad884395e666 | 6.2 (31/500) | 6.4 [5.1, 7.8] (85/1319) | 2.28 | +0.2 |
| gsm50k_plain_depth_355m_s0 | primary | ckpt_best_final.pt epoch 24, n_latent 0, b6ab7ad59a73 | 8.0 (40/500) | 7.4 [6.0, 8.8] (98/1319) | 2.35 | -0.6 |
| gsm50k_pause_s0 | primary | ckpt_best_final.pt epoch 22, n_latent 6, 213e3bb2a0f6 | 9.6 (48/500) | 6.9 [5.5, 8.3] (91/1319) | 2.21 | -2.7 |
| gsm50k_coconut_s0 | primary | ckpt_best_final.pt epoch 17, n_latent 6, ca25839a826f | 11.0 (55/500) | 9.2 [7.6, 10.7] (121/1319) | 2.30 | -1.8 |
| gsm50k_c1b_k8_s0 | primary | ckpt_best_final.pt epoch 19, n_latent 6, 56d06429f8a8 | 9.4 (47/500) | 8.3 [6.8, 9.7] (109/1319) | 2.26 | -1.1 |
| gsm50k_c1b_k4_s0 | primary | ckpt_best_final.pt epoch 15, n_latent 6, 95acb9e0386b | 9.4 (47/500) | 7.5 [6.1, 8.9] (99/1319) | 2.35 | -1.9 |
| gsm50k_c2_s0 | primary | ckpt_best_final.pt epoch 20, n_latent 6, 0a6339b45a2f | 12.0 (60/500) | 10.2 [8.5, 11.8] (134/1319) | 2.27 | -1.8 |
| gsm50k_c2_shuffled_s0 | primary | ckpt_best_final.pt epoch 17, n_latent 6, 93fd0c2e929c | 11.6 (58/500) | 9.9 [8.3, 11.5] (131/1319) | 2.26 | -1.7 |
| gsm50k_c3_s0 | primary | ckpt_best_final.pt epoch 25, n_latent 6, b162dfeeae54 | 10.4 (52/500) | 8.9 [7.4, 10.5] (118/1319) | 2.23 | -1.5 |
| gsm50k_c3_foreign_s0 | primary | ckpt_best_final.pt epoch 16, n_latent 6, 5beb0542b761 | 10.0 (50/500) | 9.0 [7.5, 10.6] (119/1319) | 2.23 | -1.0 |
| gsm50k_c2c3_s0 | primary | ckpt_best_final.pt epoch 21, n_latent 6, 14f7f0cf5485 | 12.0 (60/500) | 9.3 [7.8, 10.9] (123/1319) | 2.28 | -2.7 |
| gsm50k_pause_s0 | secondary | ckpt_best.pt epoch 5, n_latent 2, ea1ddeda26c0 | 13.6 (68/500) | 11.8 [10.1, 13.6] (156/1319) | 22.45 | -1.8 |
| gsm50k_coconut_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, 1834bb109d23 | 12.6 (63/500) | 12.7 [10.9, 14.5] (167/1319) | 21.48 | +0.1 |
| gsm50k_c1b_k8_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, 3e697a6c4e1d | 14.0 (70/500) | 12.0 [10.2, 13.7] (158/1319) | 21.90 | -2.0 |
| gsm50k_c1b_k4_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, 03affb209577 | 14.6 (73/500) | 13.9 [12.1, 15.8] (184/1319) | 23.02 | -0.7 |
| gsm50k_c2_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, 8d4be009a7b0 | 12.6 (63/500) | 13.5 [11.7, 15.3] (178/1319) | 21.32 | +0.9 |
| gsm50k_c2_shuffled_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, d25905753227 | 13.0 (65/500) | 13.4 [11.6, 15.3] (177/1319) | 21.35 | +0.4 |
| gsm50k_c3_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, 9a538227e0f6 | 13.4 (67/500) | 12.5 [10.7, 14.3] (165/1319) | 20.99 | -0.9 |
| gsm50k_c3_foreign_s0 | secondary | ckpt_best.pt epoch 6, n_latent 2, 49bf1b36af14 | 11.0 (55/500) | 11.8 [10.0, 13.5] (155/1319) | 22.82 | +0.8 |
| gsm50k_c2c3_s0 | secondary | ckpt_best.pt epoch 5, n_latent 2, 8ae5028ae755 | 12.8 (64/500) | 12.9 [11.1, 14.7] (170/1319) | 23.20 | +0.1 |

Secondary reads (9): secondary read (3.3 item 3): all-epoch best, an earlier stage; earlier-stage checkpoint, n latents = its stage; decides nothing. Each scores the run's all-epoch best `ckpt_best.pt` at its own stage's latent count once; it never replaces the primary row.

## ProsQA-hard

Job `screen-test-prosqa-hard-acctb` (acctb, attempt silent-eval-20261021-155218-0999), registered read 2026-10-21T15:27Z; test `data/prosqa_hard_test.json` (500 items, sha256 85a26a32ae6f...); valid of record: the third ProsQA-hard table `results/screens/prosqa_hard/prosqa_hard_screen_with_plain_depth.json` at af1f57398a06 (valid 300 items). Interval: 6.2's form, +-1.96 binomial SE.

| run | read | checkpoint | valid of record | test (95% interval) | test emitted tokens (mean) | test minus valid (descriptive) |
|---|---|---|---|---|---|---|
| prosqa_hard_cot_lr3e-5_s0 | primary | ckpt_best_final.pt epoch 14, n_latent 0, aefce29c737f | 80.3 (241/300) | 83.4 [80.1, 86.7] (417/500) | 77.79 | +3.1 |
| prosqa_hard_no_cot_s0 | primary | ckpt_best_final.pt epoch 8, n_latent 0, 0165fbb8e72f | 57.7 (173/300) | 53.2 [48.8, 57.6] (266/500) | 8.19 | -4.5 |
| prosqa_hard_plain_depth_355m_s0 | primary | ckpt_best_final.pt epoch 28, n_latent 0, ae2e48b54ebc | 61.7 (185/300) | 60.4 [56.1, 64.7] (302/500) | 8.17 | -1.3 |
| prosqa_hard_pause_s0 | primary | ckpt_best_final.pt epoch 30, n_latent 7, 0e5f68e85319 | 74.7 (224/300) | 74.8 [71.0, 78.6] (374/500) | 8.18 | +0.1 |
| prosqa_hard_coconut_s0 | primary | ckpt_best_final.pt epoch 30, n_latent 7, 37cd86030812 | 73.3 (220/300) | 72.8 [68.9, 76.7] (364/500) | 8.18 | -0.5 |
| prosqa_hard_c1b_k8_s0 | primary | ckpt_best_final.pt epoch 27, n_latent 7, 16b1614a6273 | 71.7 (215/300) | 73.6 [69.7, 77.5] (368/500) | 8.18 | +1.9 |
| prosqa_hard_c1b_k4_s0 | primary | ckpt_best_final.pt epoch 30, n_latent 7, 5d4a010ca665 | 70.7 (212/300) | 71.2 [67.2, 75.2] (356/500) | 8.18 | +0.5 |
| prosqa_hard_c2_s0 | primary | ckpt_best_final.pt epoch 27, n_latent 7, e44501b349b7 | 72.3 (217/300) | 75.8 [72.0, 79.6] (379/500) | 8.20 | +3.5 |
| prosqa_hard_c2_shuffled_s0 | primary | ckpt_best_final.pt epoch 30, n_latent 7, 5f3d881a60da | 71.3 (214/300) | 72.6 [68.7, 76.5] (363/500) | 8.18 | +1.3 |
| prosqa_hard_coconut_s0 | secondary | ckpt_best.pt epoch 8, n_latent 2, 23c37716d527 | 76.7 (230/300) | 73.8 [69.9, 77.7] (369/500) | 55.26 | -2.9 |
| prosqa_hard_c1b_k8_s0 | secondary | ckpt_best.pt epoch 14, n_latent 4, 416a4dce9858 | 75.0 (225/300) | 76.8 [73.1, 80.5] (384/500) | 32.55 | +1.8 |
| prosqa_hard_c1b_k4_s0 | secondary | ckpt_best.pt epoch 18, n_latent 5, 94e040e5872f | 74.3 (223/300) | 71.2 [67.2, 75.2] (356/500) | 25.16 | -3.1 |
| prosqa_hard_c2_s0 | secondary | ckpt_best.pt epoch 9, n_latent 2, 94b26a32f079 | 81.7 (245/300) | 78.6 [75.0, 82.2] (393/500) | 56.80 | -3.1 |
| prosqa_hard_c2_shuffled_s0 | secondary | ckpt_best.pt epoch 9, n_latent 2, ce9c38ab925f | 76.7 (230/300) | 78.8 [75.2, 82.4] (394/500) | 55.31 | +2.1 |

Secondary reads (5): secondary read (3.3 item 3): all-epoch best, an earlier stage; earlier-stage checkpoint, n latents = its stage; decides nothing. Each scores the run's all-epoch best `ckpt_best.pt` at its own stage's latent count once; it never replaces the primary row.

## Notes

- Reporting only (A3.4): no verdict, kill line, selection or screen outcome is computed or changed here; every screen decision stays on valid, in the screen tables cited above.
- 1 seed (seed 0); each checkpoint scored once on test under its registered read (4.5); the primary rows are each run's selected checkpoint (3.3 item 1); among the bar's three cot trials only the bar's selected trial was read.
- 14 secondary reads (3.3 item 3): the all-epoch best of a run whose earlier stage scored higher on valid, at that stage's latent count; labelled, never a primary cell.
- The interval is 6.2's form for one accuracy (binomial SE); the test minus valid gap is descriptive: valid is the selection split and test the scoring split (1.3).
