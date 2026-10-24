# Card C2: verifiable-state supervision of latent steps

Author: lead (Fable), 2026-10-18. Status: registered by the critic before its screen. Novelty label proposed:
COMBINATION (process supervision of latents is known; the supervision source, executed ground truth rather than CoT
text, is new, and the decoder doubles as the registered readout). Memory kind: appended-KV scratchpad (Coconut base).

## Mechanism
Coconut, plus an auxiliary decoder g (2-layer MLP, hidden 768, GELU) applied during training to the fed-back latent
vector of step t, trained to reconstruct the *executed* intermediate state of step t. The decoder is dropped at
inference, so the inference model is byte-identical in shape and cost to Coconut. Total loss = CE + lambda x L_state,
lambda = 0.1 (registered, not tuned).

Targets per task (computed from data, never from the model):
- ProsQA / ProsQA-hard: the frontier F_t = the set of concepts reachable from the root in at most t "is a" hops, from
  the record's `edges`, `root`, `idx_to_symbol`. Head: multi-label sigmoid over the record's symbols (padded to the
  maximum symbol count, masked), binary cross-entropy. With c_thought = 1, latent t carries F_t.
- GSM8k-Aug: the numeric result of step t (the string after "=" in the t-th "<<a op b = c>>" step). Head: the
  result tokenised with the model's tokenizer, up to 4 tokens, 4 parallel softmax heads over the vocabulary sharing the
  MLP trunk (heads tied to the embedding matrix), cross-entropy, masked beyond the result's length. With c_thought = 2,
  the second latent of the pair carries step t.
- Steps beyond the number of latents at the current stage have no target. Stage 0 (no latents) has no state loss.

## Three closest works
SIM-CoT (Wei et al., ICLR 2026: auxiliary decoder aligns each latent to its CoT step text); CODI (Shen et al. 2025:
self-distillation to the teacher's hidden state); "What Makes Effective Supervision in Latent CoT" (Chen et al., ICML
2026: reconstruction objectives preserve I(latent; step)). All supervise with text or teacher states; none with an
executed, machine-verified state, and none reports causal load-bearing of the decoded state.

## Prediction (before any run)
- ProsQA-hard: accuracy within sigma of Coconut (the task is near the curriculum ceiling); frontier decodability (probe
  F1 on held-out) rises from about 0.5 (Coconut) to above 0.9; load-bearing fraction (F3 probe edits that move the
  answer) rises from under 0.3 to above 0.5.
- GSM8k-Aug 50k: +3 to +8 points over Coconut; scrambled-latent drop rises from under 5 points to above 10.
- C2 alone does not beat tuned CoT at matched FLOPs on GSM (prediction: still 4 to 10 points under).

## Null
C2-shuffled: the same decoder, same lambda, same compute, trained to reconstruct the state of a *different* example
(a fixed seeded derangement of the training set, same length class). This keeps the extra gradient and the extra
parameters and destroys the information; the content-matched null of the think-bet, applied to a latent.
Attribution: C2 minus C2-shuffled is the value of the correct state; C2-shuffled minus Coconut is the value of the
extra training signal regardless of content.

## Screen, kill line, fade
Screen at 124M, seed 0, bf16, both tasks, lr 1e-4, one trial. Kill: C2 within sigma of C2-shuffled on both tasks (the
state content does nothing), or C2 below Coconut by more than 2 sigma on both. Pass to seeds 1 and 2: C2 above
C2-shuffled by more than sigma on at least one task. Fade rule as everywhere.

## Leak and bug answers
- The decoder sees only the latent vector and the step index; a unit test asserts no gradient path from the target
  tensors into the language-model loss except through the latent, and that the inference graph has no decoder.
- Targets are computed by a pure function of the record (tested against hand-written cases; ProsQA frontiers checked
  by BFS; GSM results parsed from the step strings with a parser test on 1,000 records).
- Correction (critic A3.4, 2026-10-18): the last step's result equals the final answer in 95.0% of GSM8k-Aug
  training records, and the ProsQA last-step frontier contains the target. State targets are training-set
  information, as CoT's own targets are, so this is allowed; the fraction of records whose last supervised latent
  target contains the answer is reported per run, and attribution rests on C2-shuffled, not on this test.

## Readout (section 6.5 of the plan)
The trained decoder is readout R3. Its faithfulness is measured by F1 counterfactual patching and F3 probe-direction
edits on held-out examples, reported as coverage and load-bearing fraction next to any accuracy claim.
