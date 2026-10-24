| task | arm | gates | seeds (points) | mean | SD | sigma_cell | tolerance | published | verdict | labels |
|---|---|---|---|---|---|---|---|---|---|---|
| prosqa | cot | yes | s0 75.0 (NVIDIA A100-SXM4-40GB) | 75.00 |  | 1.90 | 3.80 | 77.50 | matched | single seed |
| prosqa | no_cot | yes | s0 78.0 (NVIDIA A100-SXM4-40GB) | 78.00 |  | 1.00 | 2.00 | 76.70 | matched | single seed |
| prosqa | coconut | yes | s0 96.8 (NVIDIA A100-SXM4-40GB) | 96.80 |  | 1.00 | 2.00 | 97.00 | matched | single seed |
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

- K1 ProsQA: pass (cot, no_cot, coconut, pause) [provisional]
- K1 ProntoQA: UNTESTED (cot, no_cot, coconut, pause, coconut_no_curriculum not read)
- K1 GSM8k-Aug: UNTESTED (cot, no_cot, coconut not read) [single seed]
- K1 (registered, ProsQA and ProntoQA): UNTESTED; ProntoQA UNTESTED (A3.3)
- K1 for the window (A3.3) = K1 on ProsQA: pass [provisional; GSM8k-Aug UNTESTED until its read, A3.3's second clause]

Labels (critic entry 2026-10-19T15:35Z; registration A10.4, A11.1, A11.4; this block and the footnote were added by a
label-only commit that changes no cell of `k1_table.json`, whose numbers are write-once as committed at `rev92`):

- Every cell and every sentence above: KNOWN (reproduction); fp32-tf32; single seed; provisional; read on A100-40GB
  (`k1-eval-main`, VM `silent-eval-20261019-144240-f455`, `--registered-read 2026-10-19T12:07Z`) of checkpoints trained on
  A100-80GB (`oracle-acctb`; A3.2 (i) to (iii) met: `env_reference` ok on the eval VM, one-batch parity 0.0 / 2.5e-9 on both
  GPU types at critic entries 01:08Z and 09:29Z, `gpu_type` printed per seed).
- Resumes: cot 1, coconut 2, pause 1, no_cot 0; none fragmented (A3.6, A8.1, A8.2).
- Pending, required before any confirmation seed, any 6.2 verdict and any WIN or tie sentence (A10.4): the final K1 on seeds
  1 and 2 of cot and coconut (registered read 2026-10-19T15:40Z; `k1_table_final.{json,md}`, a separate path); the precision
  pair d_prec (A4 item 2; `prosqa_coconut_strict_s0`, read once at `--tf32 off`); the determinism pair (A8.3); the tuned-CoT
  bar read (3.4, A3.3). no_cot and pause stay single-seed cells (A3.3).
- Diagnostic on the coconut checkpoint (section 8; printed as the critic's row allows, nothing more): at this read the selected
  seed-0 Coconut checkpoint's test answers show no measurable dependence on its latents: scrambled drop 0.0 points (0 of 500
  examples change; paired 95% CI [0.0, 0.0]), zero-latent drop -0.4 [-1.0, 0.0], step sweep 97.2 / 97.0 / 97.0 / 96.8 / 96.8 /
  97.0 at n = 0 / 1 / 2 / 4 / 6 / 8. KNOWN, fp32-tf32, single seed, provisional, diagnostic.
- Audit (7.10 item 2, A10.3): rescoring at batch size 1 agrees per example on 500 of 500 for cot, coconut and pause; no_cot
  differs on 1 of 500 (idx 163), at the "at most 1" line; the batch-32 read is the number of record.

Footnote (A10.1, A4): our reads run at fp32-tf32 (`matmul.allow_tf32` True, `cudnn.allow_tf32` True, "high"), the numerics
each checkpoint's selection on valid used; the reference's precision is probably strict fp32 (A4). Cost accounting charges V as
the arm's head: 50,257 (cot, no_cot), 50,260 (latent arms) (A11.2).
