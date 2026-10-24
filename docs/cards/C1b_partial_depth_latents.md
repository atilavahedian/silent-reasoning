# Card C1b: partial-depth latent steps ("middle-band latents")

Author: lead (Fable), 2026-10-18. Status: registered by the critic before its screen (see critic_log). Novelty label
proposed: KNOWN family, new test. Memory kind: appended-KV scratchpad (latents are positions in the cache), so the
compressed-loop bound does not apply.

## Mechanism
GPT-2 124M has L = 12 blocks. A Coconut latent step runs all 12 blocks at the latent position and feeds the last-layer
state h_12(j-1) back as the input embedding of position j. C1b runs only the band of blocks [a, b) at a latent
position: the input to block a at latent position j is h_b(j-1), the residual stream after block b at the previous
position (if j-1 is itself a latent, that is the band's own output: a recurrence through the band). Blocks below a and
at or above b do no computation at latent positions and hold no keys or values for them, so attention at those blocks
masks the latent positions out for every later token. Tokens after the latent block (the <|end-latent|> marker and the
answer) attend to the latents only inside the band. Two settings:
- k = 8: band [2, 10) (blocks 2 to 9), a latent step costs 8/12 of a pass.
- k = 4: band [4, 8), a latent step costs 4/12 of a pass.
Everything else is Coconut: same curriculum, same special tokens, same loss, same c_thought.

## Math and accounting
Per latent step, FLOPs = (b - a)/L x 2 N_ne (plus the band's attention over the prefix); no LM head. The accounting
gets a `layer_fraction` field: `flops_steps` charges 2 N_ne x layer_fraction per latent step and the attention term
only for the band's layers. Sequential passes are unchanged (one pass per latent step, shorter). Training FLOPs per
sequence fall by the same fraction at latent positions.

## Three closest works
Huginn (Geiping et al. 2025: prelude, recurrent core, coda); "Teaching Pretrained Language Models to Think Deeper
with Retrofitted Recurrence" (2025, band recurrence retrofitted on a pretrained model); Mixture-of-Recursions (Bae et
al. 2025, per-token recursion depth). None of them makes a Coconut-style fed-back latent position run a band only, with
the band's keys and values masked elsewhere, and none charges the FLOPs against a tuned CoT at matched compute.

## Prediction (before any run)
- ProsQA-hard: k = 8 within sigma of full-depth Coconut (the latents carry little, so cutting their depth costs
  little); k = 4 lower by 2 to 6 points.
- GSM8k-Aug 50k: k = 8 lower than full-depth Coconut by 1 to 4 points; k = 4 by 4 to 10.
- If k = 8 ties full depth on both tasks, that is a true 33% compute saving per latent step, the first in the project.

## Null and controls
The judge's standing controls apply (pause-same-curriculum, scrambled latents at inference, zero-latent, step sweep,
tuned CoT, plain depth). C1b's own null is full-depth Coconut at the same curriculum: the card claims a cost saving,
so the question is "does accuracy hold", and the null is the uncut model. Critic A3.4 (2026-10-18): the k = 8 band with scrambled latents is a kill line, not a diagnostic (K2 applies at the
screen to every latent arm), and the implementation must skip the non-band blocks for real, with the per-step
wall-clock measured so the layer fraction shows in time as well as FLOPs.

## Screen, kill line, fade
Screen at 124M, seed 0, bf16 autocast, ProsQA-hard and GSM8k-Aug 50k, lr 1e-4, the task's curriculum (one trial; the
tuned CoT bar has three, so the arm is the disadvantaged side). Kill: k = 8 below full-depth Coconut by more than
2 sigma on both tasks. Pass to seeds 1 and 2: k = 8 within sigma of full-depth Coconut on at least one task. Fade:
the read is at the final budget; a lead at epoch 15 that is gone at the end is HEADSTART, as everywhere.

## Leak and bug answers
- The band's keys and values must exist only for latent positions; a unit test asserts that non-band blocks never
  receive queries attending to latent positions (mask check) and that a cot arm with layer_fraction = 1 reproduces
  the reference forward exactly.
- The fed-back vector is h_b, not h_12; a test asserts the identity of the fed-back tensor with the band output.
- Accounting test: the worked example with layer_fraction 8/12 and 4/12.

## Implementation notes
HF GPT-2 exposes per-block forward; run blocks [a, b) manually at latent positions with a per-block past-key-values
slice; build a per-block 4D attention mask that drops latent columns outside the band. Position ids unchanged.
