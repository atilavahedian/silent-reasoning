| task | arm | gates | seeds (points) | mean | SD | sigma_cell | tolerance | published | verdict | labels |
|---|---|---|---|---|---|---|---|---|---|---|
| prosqa | cot | yes | s0 75.0 (NVIDIA A100-SXM4-40GB), s1 71.2 (NVIDIA A100-SXM4-40GB), s2 78.4 (NVIDIA A100-SXM4-40GB) | 74.87 | 3.60 | 2.88 | 5.76 | 77.50 | matched |  |
| prosqa | no_cot | yes | s0 78.0 (NVIDIA A100-SXM4-40GB) | 78.00 |  | 1.00 | 2.00 | 76.70 | matched | single seed |
| prosqa | coconut | yes | s0 96.8 (NVIDIA A100-SXM4-40GB), s1 96.2 (NVIDIA A100-SXM4-40GB), s2 95.8 (NVIDIA A100-SXM4-40GB) | 96.27 | 0.50 | 1.00 | 2.00 | 97.00 | matched |  |
| prosqa | pause | yes | s0 97.0 (NVIDIA A100-SXM4-40GB) | 97.00 |  | 1.00 | 2.00 | 96.60 | matched | single seed |
| prosqa | coconut_no_curriculum | no |  |  |  |  |  |  | UNTESTED | relative to our no_cot |
| prontoqa | cot | yes |  |  |  |  |  |  | UNTESTED |  |
| prontoqa | no_cot | yes |  |  |  |  |  |  | UNTESTED |  |
| prontoqa | coconut | yes |  |  |  |  |  |  | UNTESTED |  |
| prontoqa | pause | yes |  |  |  |  |  |  | UNTESTED |  |
| prontoqa | coconut_no_curriculum | yes |  |  |  |  |  |  | UNTESTED |  |
| gsm | cot | yes | s0 40.9 (NVIDIA A100-SXM4-40GB) | 40.94 |  | 1.00 | 2.00 | 42.90 | matched | single seed; on the edge (539 would have missed); secondary oracle flagged (definition difference, 14:08Z) |
| gsm | no_cot | yes |  |  |  |  |  |  | UNTESTED |  |
| gsm | coconut | yes | s0 34.6 (NVIDIA A100-SXM4-40GB) | 34.65 |  | 1.50 | 3.00 | 34.10 | matched | single seed |
| gsm | pause | no |  |  |  |  |  |  | UNTESTED |  |
| gsm | coconut_no_curriculum | no |  |  |  |  |  |  | UNTESTED |  |

- K1 ProsQA: pass (cot, no_cot, coconut, pause)
- K1 ProntoQA: UNTESTED (cot, no_cot, coconut, pause, coconut_no_curriculum not read)
- K1 GSM8k-Aug: UNTESTED (no_cot not read) [single seed]
- K1 GSM8k-Aug (window, A1 item 1: cot and coconut): pass (cot 40.94, 540 of 1,319, matched: |40.94 - 42.9| = 1.96 <= 2.0; coconut 34.65, 457 of 1,319, matched: |34.65 - 34.1| = 0.55 <= 3.0) [single seed; no_cot UNTESTED]
- K1 on GSM8k-Aug, 1 seed: pass (cot 40.94, coconut 34.65) against 42.9 / 34.1, tolerances 2.0 / 3.0 [single seed; no_cot UNTESTED; KNOWN (reproduction); fp32-tf32 (footnote: the reference's probable strict fp32, A4; d_prec 0.0 on ProsQA does not transfer to GSM and is not claimed for it); cot trained and read on A100-40GB; coconut: cot stage on A100-40GB, coconut stage on A100-80GB, read on A100-40GB (A3.2 (i) to (iii) on the eval VM); images v20261016 / v20261019; zero resumes; audit: 0 of 1,319 differ at batch 1 on both checkpoints (7.10 item 2, A10.3; cot: one example's generation length differs, answer unchanged); cot on the edge: 540 of 1,319, abs(diff) 1.96 of 2.0, 539 would have missed; secondary oracle flagged on cot: paper-style count 30.6 vs 25.0, checked: the reference test chains count 30.1 under 2.2's definition, the paper's convention unstated (critic entry 2026-10-20T14:08Z; label-only edit, the json untouched)]
- K1 (registered, ProsQA and ProntoQA): UNTESTED; ProntoQA UNTESTED (A3.3)
- K1 for the window (A3.3) = K1 on ProsQA: pass; GSM8k-Aug: pass [1 seed]
