# Card C3: latent-dependence loss

Author: lead (Fable), 2026-10-18. Status: registered by the critic before its screen. Novelty label proposed:
COMBINATION (a contrastive hinge between the model's own latents and scrambled latents; the ingredients exist, the
objective as a cure for inert latents was not found in the literature). Memory kind: appended-KV scratchpad.

## Mechanism
Coconut, plus a second forward pass per training batch in which the fed-back latent vectors are replaced by a
scrambled bank: `bank[i, j] = latents[pi(i), j]` with pi a seeded derangement of the batch within the same
latent-count class, latents detached. Let CE_true and CE_scr be the per-token cross-entropy over the answer tokens
(the tokens after <|end-latent|>, i.e. the remaining steps and the answer at the current stage) under the two passes.
Loss = CE_true + beta x max(0, m - (CE_scr - CE_true)), beta = 1, m = 0.2 nat per answer token (registered, not
tuned). The hinge is satisfied only when the model's own latents make its answer at least m nats per token more
likely than another example's latents do, so the model cannot route around its latent steps. Stage 0 (no latents) has
no hinge. The scrambled pass is charged to training FLOPs (about 1.5x Coconut's per epoch; the arm carries the
"training-heavy" label, section 5.5 of the registration); inference cost is identical to Coconut.

## Three closest works
"Think Consistently, Reason Efficiently: Energy-Based Calibration for Implicit CoT" (2025: an energy term on latent
consistency); contrastive decoding and counterfactual regularisers for faithfulness in NLI and QA (2020-2024); the
scrambled-state controls of this lab (energy campaign, think-bet) turned from a test into a training signal. No work
found trains a continuous-thought model against its own scrambled thoughts.

## Prediction (before any run)
- ProsQA-hard: scrambled-latent drop rises from about 4 points (Coconut) to above 20; accuracy within sigma of Coconut
  or down by up to 3 points (if it drops more, the latents were a shortcut channel and the task is CoT-unnecessary,
  which the registration already labels).
- GSM8k-Aug 50k: scrambled drop above 15 points; accuracy +0 to +5 over Coconut.
- C2 + C3 combined is the arm most likely to pass the primary test on GSM-50k: I give it 0.3 to tie tuned CoT at
  matched FLOPs with emitted tokens under 0.2x; 0.6 that it beats Coconut by more than 2 sigma; 0.15 that it is dead
  by K2 (within the floor of its scrambled control).

## Null
Correction (critic A3.4, 2026-10-18): the originally proposed C3-placebo (hinge at a question position) has zero
hinge gradient by causality, since question positions precede the latents, so it is not a null. Registered null:
C3-foreign: the same second-pass compute, with the hinge computed between two independent scrambled banks A and B
(different seeds, A != B rowwise), while the model's own answer loss stays CE_true. Attribution: C3 minus
C3-foreign is the value of forcing dependence on the model's own latents; C3-foreign minus Coconut is the value of
the extra pass and the regularisation noise.

## Screen, kill line, fade
Screen at 124M, seed 0, bf16, both tasks, lr 1e-4, one trial; arms C3 alone and C2 + C3 (null: C3-foreign). Kill: scrambled drop under
5 points on both tasks after training (the loss failed to do its one job), or accuracy below Coconut by more than
2 sigma on both. Pass to seeds 1 and 2: scrambled drop above 10 points on at least one task, exceeding C3-foreign's drop by
sigma_screen, with accuracy within sigma of Coconut or better. Fade rule as everywhere.

## Leak and bug answers
- The bank must come from other examples only: `pi(i) != i` asserted; the blindness test from model.py applies.
- The hinge uses detached bank latents, so no gradient flows into other examples' latent computation (test).
- A degenerate solution would be to make CE_scr large by making the answer depend on garbage positions; the pause and
  zero-latent controls expose it (a model that cannot answer without its latents at all is the intended behaviour; one
  whose accuracy collapses under pause is reported as such).
- The scrambled-at-inference evaluation uses a derangement over the whole evaluation set within latent-count classes,
  seed 0, as the critic adopted; the training derangement is per batch and reseeded per epoch. The two must not share
  a seed (test asserts different permutations).
