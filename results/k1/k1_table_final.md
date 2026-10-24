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
| gsm | cot | yes |  |  |  |  |  |  | UNTESTED |  |
| gsm | no_cot | yes |  |  |  |  |  |  | UNTESTED |  |
| gsm | coconut | yes |  |  |  |  |  |  | UNTESTED |  |
| gsm | pause | no |  |  |  |  |  |  | UNTESTED |  |
| gsm | coconut_no_curriculum | no |  |  |  |  |  |  | UNTESTED |  |

- K1 ProsQA: pass (cot, no_cot, coconut, pause) [cot and coconut three seeds; no_cot and pause single seed; audit: cot s1 3 of 500 differ at batch 1, explained: TF32 batch-shape rounding flips near-ties (top-2 margins 1e-3 to 1e-2) along the greedy chain; no cross-row dependence (same-shape swap 0 of 300 at both numerics); strict fp32 0 of 300 divergences (diag `rev8`; A14.2)]
- K1 ProntoQA: UNTESTED (cot, no_cot, coconut, pause, coconut_no_curriculum not read)
- K1 GSM8k-Aug: UNTESTED (cot, no_cot, coconut not read) [single seed]
- K1 (registered, ProsQA and ProntoQA): UNTESTED; ProntoQA UNTESTED (A3.3)
- K1 for the window (A3.3) = K1 on ProsQA: pass [three seeds cot and coconut, single seed no_cot and pause; GSM8k-Aug UNTESTED until its read, A3.3's second clause]

Labels (critic entries 2026-10-19T16:59Z; registration A12.1, A12.2; this block and the footnote were added by a label-only
commit that changes no cell of `k1_table_final.json`, whose numbers are write-once as committed at `rev2`):

- Every cell and every sentence above: KNOWN (reproduction); fp32-tf32; three seeds for cot and coconut, single seed for
  no_cot and pause; read on A100-40GB (`k1-eval-main` seed 0, `k1-eval-s12-main` seeds 1 and 2, registered reads
  2026-10-19T12:07Z and 15:40Z) of checkpoints trained on A100-80GB (A3.2 (i) to (iii) met on both eval VMs).
- Resumes per seed: cot 1 / 0 / 0, coconut 2 / 2 / 0, pause 1, no_cot 0; none fragmented (A3.6, A8.1, A8.2).
- Audit (7.10 item 2, A10.3, A12.3): batch-size-1 rescoring agrees per example on 500 of 500 for coconut s0, s1 and s2
  and for pause s0; cot s0 500 of 500; cot s2 1 of 500 (idx 366); no_cot s0 1 of 500 (idx 163); cot s1 3 of 500 (idx
  213, 364, 382), above the "at most 1" line and **explained** by the A12.3 run (critic entries 2026-10-19T18:32Z, A14.1,
  A14.2; `results/diag/diag-k1-main/`, commit `rev8`): TF32 batch-shape rounding flips near-ties (top-2 margins 1e-3
  to 1e-2) along the greedy chain; the same-shape companion swap shows 0 of 300 differences at both numerics (no row
  changes another row); strict fp32 shows 0 of 300 generation divergences. The batch-32 reads stay the numbers of
  record. The cot cell's verdict does not depend on it (71.2 or 71.4 are both inside the tolerance).
- Precision pair (A4 item 2; critic entry 2026-10-19T19:55Z): d_prec = 0.0 (ProsQA coconut seed 0, strict 96.8 against
  fp32-tf32 96.8; per-example agreement 476 of 500), **fp32-equivalent**: every cell keeps fp32-tf32 with the footnote
  below and is a reproduction under the published precision's probable setting; d_prec is reported beside sigma_task
  and never added to it.
- Pending, required before any confirmation seed, any 6.2 verdict and any WIN or tie sentence (A10.4): sigma_task's
  determinism term (the zero-resume pair, A8.3: det1 read once, det2 or the author's "one run" reading); the
  tuned-CoT bar read (3.4, A3.3). The provisional pass of 15:35Z is not reversed; the screens started on it continue.
- Diagnostic on the three coconut checkpoints (section 8; A14.3, printed as the critic's row allows, nothing more): at
  this read the three selected Coconut checkpoints' answers show no measurable dependence on their latents: scrambled
  drop 0.0 points on every seed (0 of 500 test examples change on each; paired 95% CI [0.0, 0.0]), zero-latent drop
  -0.4 / -0.2 / 0.0, step sweeps flat within 0.4 points from n = 0 to n = 8. The scrambled pass fed every example
  another example's latents (verified bit for bit on the valid split; no zero distance in 1,800 pairs per checkpoint);
  the latents are read by the model, since the next-token logits move on 300 of 300 valid examples (by up to 1.5 to
  2.3 at the first emitted position), and they decide no emitted token on any of them. A fourth checkpoint, trained
  and read at strict fp32 (the precision pair's), shows the same: scrambled drop 0.0 (0 of 500 change), zero-latent
  drop -0.2 [-1.0, 0.4] [precision pair's strict checkpoint: trained and read at strict fp32, single seed, diagnostic,
  enters no cell]. The latent vectors of different examples share a common direction (mean cosine 0.99 per rank).
  KNOWN (reproduction), fp32-tf32 (strict fp32 agrees on every count), three seeds, diagnostic, ProsQA only (GSM8k-Aug
  untested), the intervention reads on test (500) and the fed-vector and logit statements on valid (300), read on
  A100-40GB of A100-80GB checkpoints.

Footnote (A10.1, A4): our reads run at fp32-tf32 (`matmul.allow_tf32` True, `cudnn.allow_tf32` True, "high"), the numerics
each checkpoint's selection on valid used; the reference's precision is probably strict fp32 (A4). Cost accounting charges V as
the arm's head: 50,257 (cot, no_cot), 50,260 (latent arms) (A11.2).
