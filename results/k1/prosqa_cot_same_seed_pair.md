# The same-seed pair on ProsQA (13.6): prosqa_cot_s0 / det1 / det2, and sigma_task's two readings

Critic entry 2026-10-21T20:21Z (a) parts (2) and (3); 1 seed per member (seed 0); test split (500 items), each checkpoint read once (4.5). Labels: pair read on two GPU types (A3.2 (ii) parity passed on both types; (iii) the type printed per member; the batch-1 audits beside each bound the eval-path flips, det1 1 of 500); d_nd is the pair's difference as read; det2's accuracy enters no cell, no seed pool and no sigma_cell; sigma_screen on GSM-50k and ProsQA-hard (A3.4) carries no determinism term: det2 changes no line of any screen table of record; the ProsQA K1 table is unchanged; the strict pair's figure is the one of record for every ProsQA verdict not yet written; rises, never falls.

| member | run | project (bucket) | resumes | selected epoch | checkpoint | test (95% interval) | read on | trained on | batch-1 audit flips | registered read |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | prosqa_cot_s0 | cloud-b | 1 | 46 | 4f8c895b3957... | 75.0 [71.2, 78.8] (375/500) | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-80GB | 0 of 500 | 2026-10-19T12:07Z |
| det1 | prosqa_cot_det1_s0 | cloud-b | 0 | 25 | 24eb701e476e... | 75.8 [72.0, 79.6] (379/500) | NVIDIA A100-SXM4-40GB | NVIDIA A100-SXM4-80GB | 1 of 500 | 2026-10-19T19:20Z |
| det2 | prosqa_cot_det2_s0 | cloud-c | 0 | 23 | f59caa4e8167... | 77.8 [74.2, 81.4] (389/500) | NVIDIA A100-SXM4-80GB | NVIDIA A100-SXM4-80GB | 0 of 500 | 2026-10-22T06:52Z |

Per-example agreement (both correct / only the first / only the second / neither; churn = examples that differ in correctness; net = the count difference):

| pair | both | only first | only second | neither | churn | net |
|---|---|---|---|---|---|---|
| s0 / det1 | 324 | 51 | 55 | 70 | 106 | -4 |
| det1 / det2 | 341 | 38 | 48 | 73 | 86 | -10 |
| s0 / det2 | 333 | 42 | 56 | 69 | 98 | -14 |

## sigma_task(ProsQA), row (a) (3)

- Strict (of record; the pair = det1, det2): d_nd = |acc(det2) - 75.8| = 2.0 (10 of 500 net) > 0.2 x 3.60 = 0.72: sigma_task = sqrt(3.60^2 + 2.0^2) = 4.12
- "One run" (the author's reading, not amended; the pair = prosqa_cot_s0, det1), printed beside: d_nd = |75.8 - 75.0| = 0.8 (4 of 500 net), sigma_task 3.69.

## Valid curves (13.2), val_acc by epoch

| epoch | s0 | det1 | det2 |
|---|---|---|---|
| 1 | 61.0 | 60.7 | 61.0 |
| 2 | 62.3 | 63.0 | 62.3 |
| 3 | 68.0 | 68.3 | 67.7 |
| 4 | 71.7 | 66.7 | 72.0 |
| 5 | 69.3 | 67.3 | 67.7 |
| 6 | 63.7 | 63.3 | 62.3 |
| 7 | 66.7 | 65.7 | 70.3 |
| 8 | 73.0 | 71.7 | 74.7 |
| 9 | 73.3 | 74.0 | 76.7 |
| 10 | 71.3 | 74.7 | 73.0 |
| 11 | 75.0 | 73.7 | 80.3 |
| 12 | 77.3 | 78.3 | 79.7 |
| 13 | 77.3 | 77.7 | 76.0 |
| 14 | 74.3 | 78.3 | 78.3 |
| 15 | 77.0 | 72.3 | 77.7 |
| 16 | 76.7 | 74.0 | 76.0 |
| 17 | 77.0 | 78.0 | 79.3 |
| 18 | 75.3 | 70.7 | 77.3 |
| 19 | 76.3 | 79.3 | 74.7 |
| 20 | 75.0 | 76.3 | 75.0 |
| 21 | 76.3 | 73.0 | 74.3 |
| 22 | 77.0 | 75.7 | 76.7 |
| 23 | 79.0 | 76.7 | 81.0 |
| 24 | 78.7 | 78.3 | 77.3 |
| 25 | 75.3 | 81.3 | 76.0 |
| 26 | 78.7 | 78.0 | 77.3 |
| 27 | 74.3 | 77.3 | 77.0 |
| 28 | 72.0 | 75.3 | 74.3 |
| 29 | 75.0 | 76.7 | 77.7 |
| 30 | 77.7 | 77.7 | 78.0 |
| 31 | 75.0 | 74.7 | 77.3 |
| 32 | 76.3 | 75.3 | 78.7 |
| 33 | 77.7 | 77.0 | 80.7 |
| 34 | 80.3 | 75.0 | 75.0 |
| 35 | 77.0 | 74.3 | 70.7 |
| 36 | 72.7 | 76.7 | 76.3 |
| 37 | 78.3 | 77.7 | 79.3 |
| 38 | 80.7 | 78.0 | 77.0 |
| 39 | 78.7 | 75.0 | 78.0 |
| 40 | 77.0 | 75.3 | 74.3 |
| 41 | 76.3 | 76.3 | 75.7 |
| 42 | 77.3 | 75.7 | 79.7 |
| 43 | 74.0 | 75.0 | 73.0 |
| 44 | 78.7 | 73.0 | 79.7 |
| 45 | 78.3 | 75.3 | 77.3 |
| 46 | 82.0 | 77.3 | 74.7 |
| 47 | 79.0 | 75.7 | 76.0 |
| 48 | 76.0 | 77.3 | 71.0 |
| 49 | 76.3 | 77.3 | 76.3 |
| 50 | 78.7 | 77.0 | 77.7 |

Status: complete.
