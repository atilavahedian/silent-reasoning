# Theory memo: what a silent step can and cannot buy (Phase 0, 2026-10-17)

Lead (Fable). Five claims, each with the arithmetic or the theorem behind it and the prediction it makes for this
project. The predictions are written down here before any run; the critic may copy them into the registration.
Literature citations refer to `docs/prior_art.md` by section number.

## T1. The FLOP identity, and a head-cost confound that fades with scale

A decoder forward over one new position with context p costs about

    2 N_ne + 2 n_L d_a p      (+ 2 V d if logits are computed at that position)

where N_ne is the non-embedding parameter count, n_L the layers, d_a the attention width and V d the LM head. A written
CoT token is one such pass with the head; a Coconut latent step is one such pass without the head (the last hidden
state is fed back directly). So a latent step is not free, and at large sizes it costs the same as a token.

| Model | N_ne (approx.) | V d | token pass (no attn) | latent pass | latent / token |
|---|---|---|---|---|---|
| GPT-2 124M | 85.0M | 38.6M | 247 MFLOPs | 170 MFLOPs | 0.69 |
| GPT-2 355M | 302M | 51.5M | 707 | 604 | 0.85 |
| GPT-2 774M | 708M | 64.3M | 1,545 | 1,416 | 0.92 |
| GPT-2 1.5B | 1,475M | 80.4M | 3,111 | 2,950 | 0.95 |

Consequences.
- At matched FLOPs, a latent arm gets about 1.4 latent steps per CoT token at 124M and about 1.05 at 1.5B. This
  predicts a latent advantage that *shrinks with scale* for a reason that has nothing to do with reasoning. The Phase 3
  slope read names this confound; `accounting.py` charges the head exactly (head only where logits are produced).
- Attention cost is 2 n_L d_a p per position. For ProsQA (p about 300) it is about 2% of the pass at 124M; it matters
  for the FLOP match only at long contexts (Phase 4 traces).
- Sequential passes are the quantity the billing line follows: decode is memory-bandwidth bound, one weight read per
  sequential pass, and a latent step is the same kind of pass. "5x fewer output tokens" at equal FLOPs moves the cost
  line only if sequential passes fall too. The judge reports FLOPs (primary), sequential passes (secondary, with the
  label "billing claim only" when tokens fall and passes do not) and wall-clock at batch 1 and 32.

## T2. Bits per step: what the written symbol can carry, and what the reader can recover

What CoT writes per step. One token commits at most log2(50257) = 15.6 bits, chosen from a discrete code, into a
scratchpad that later steps can copy exactly through attention. A reasoning step of k tokens commits up to 15.6 k bits.

What Coconut writes per step. One vector in R^768 (bf16: 12,288 nominal bits; fp32 twice that). Two things bound the
*usable* capacity:
1. The reader is the next forward pass through the same frozen-shape network. A later layer recovers only what it can
   linearly or nonlinearly separate under its own noise. A robust-channel estimate is d x log2(1 + SNR) bits, and the
   empirical SNR per dimension in fine-tuned LMs is small: pretrained models resolve mixtures of token embeddings into
   one token within a few layers (prior art 2.3) and the softmax-plus-feedback dynamics amplify one component or
   contract the mixture (2.10). The practical reading: a fine-tuned GPT-2 latent behaves like a slightly noisy token
   embedding, worth tens of bits, not thousands. The bit-width ladder (quantize the fed-back vector to b bits per
   dimension, b from 1 to 16, at inference) measures this directly and is run on every arm.
2. Which kind of memory the design is. Coconut appends each latent as a new position whose keys and values persist in
   the cache at every layer: a scratchpad of continuous symbols, like CoT with a richer alphabet. Designs that refine a
   fixed set of w slots in place (the lab's energy slots and canvas, tiny recursive models, our card C1 in its in-place
   variant) are compressed loops with state w x d: the memory-budget separation (6.4) says such a loop cannot decide
   P-complete problems under logspace reductions whatever its depth, while a polynomial scratchpad can. Every Phase 2
   card states which kind it is; appended-slot designs are scratchpads and escape the bound, in-place designs do not.

What the task needs per step (the live state). ProsQA: the BFS frontier, a subset of at most about 20 concepts, under
20 bits, and the whole proof is 3 hops. GSM8k: an intermediate number (typically under 10^5, about 17 bits) plus the
operation and the operand references, about 25 to 35 bits per step, and 2 to 8 steps. Program execution: the bindings
of k live variables, roughly k x 10 to k x 20 bits, growing with program length.

Prediction P2 (registered). At 124M with single-vector latents and the usable capacity measured by the ladder, latent
arms match tuned CoT on ProsQA (live state small, search parallel), fall short on GSM8k (serial, numeric state), and
fall furthest on program output prediction; the shortfall orders by live state per step. Card C1 predicts that widening
the per-step write to w vectors closes the GSM gap with a knee near w = 2 to 4 and the code gap with a larger w that
grows with program length. If accuracy does not move with w at all, the latents are not carrying state (K2 territory)
and the width hypothesis is dead.

## T3. Parallel search is real in theory and emerges from scratch, not from fine-tuning

Zhu et al. (6.1) prove that two layers with D continuous thoughts solve directed reachability of diameter D, where
constant-depth discrete CoT needs O(n^2) steps, and the mechanism is a superposition of frontier nodes in one vector.
Xu and Sato (6.2) give the general form: latent thought parallelizes what CoT must serialize, while CoT wins where
stochastic decoding is needed (approximate counting, sampling), and loops are not universally stronger.

The empirical record says the superposition appears when the model is trained from scratch (2.3) and that fine-tuned
GPT-2 Coconut reads the answer off the question representation, with the latents carrying little (2.2, 2.4, 2.5).

Prediction P3 (registered). On the fine-tuned GPT-2 Coconut reproduced in Phase 1: (a) the logit lens on each latent
is dominated by the entity named in the CoT step it replaced, not by a mixture of frontier nodes; (b) linear probes for
frontier membership decode above chance but the load-bearing fraction (probe-direction edits that move the answer) is
under 0.3; (c) the scrambled-latent drop on ProsQA is under 5 points and the pause-same-curriculum arm is within sigma
of Coconut. The positive control, a tiny from-scratch model on synthetic reachability, shows all three the other way
(mixture in the lens, probe load-bearing fraction above 0.7, scrambled drop to near chance). If the fine-tuned model
behaves like the positive control instead, P3 is wrong and that is worth knowing.

## T4. What the scrambled control can and cannot detect

Let A be accuracy with the model's own latents, S with latents swapped for another example's (same count), P the
pause-token arm trained with the same curriculum, Z with zero latents at inference (off-distribution for a model
trained with latents), and acc(k) the step sweep.
- A minus S is the example-specific content effect: information in these vectors about this problem.
- A minus P is the feedback effect: what continuous feedback adds over the same curriculum and the same number of
  positions. This is the cleanest null for "the latent content matters", because S can hurt more than no information
  (a wrong example's latents are adversarial), which would overstate the content effect. Both are reported; P is the
  primary null, S the secondary.
- S minus Z and P minus Z are compute-and-position effects (extra passes, the curriculum's effect on the weights).
  Z is read with care because it is off-distribution.
- What scrambling cannot see: content that is example-independent (a learned constant program, a step counter). That is
  by design; such content is not reasoning about the input and the pause arm measures its worth.
- What neither null sees: the answer parked in latent 1 and copied. The step sweep (acc(k) flat from k = 1) and the
  per-step probes (answer decodable at step 1) catch it, so both are registered diagnostics, and a card that passes
  only because of it is labelled "shortcut" not "reasoning".

## T5. The only ways a silent step is cheaper than a token

1. Skip the head: 31% of a pass at 124M, 5% at 1.5B (T1). Real, already charged, fades.
2. Parallel slots: w latent positions processed in one pass (PCCoT, LOTUS; card C1). FLOPs are unchanged (FLOPs count
   positions), sequential passes fall by w, and in the bandwidth-bound decode regime wall-clock per slot falls by nearly
   w because the weights are read once per pass. This is the lever that turns a token saving into a cost saving.
3. Partial-depth loops: a latent step through the middle k of L layers costs k/L of a pass (Huginn-style prelude, core,
   coda; card C1b). A true FLOP saving per step, at the price of less computation per step.
4. A smaller thinking network feeding a larger answering network (SoftCoT family). Known, weak so far; not pursued.

Prediction P5 (registered). Under the matched-FLOPs judge, single-vector Coconut on GPT-2 does not beat tuned CoT on
any task at any size (the head saving is too small to matter and the latents carry little). If anything in Phase 2
passes the primary test it will be a design from item 2 or 3 with per-step supervision (C2) and a dependence loss (C3),
and its advantage over tuned CoT at matched FLOPs will not grow from 124M to 774M. Writing that down now is the point:
if the data say otherwise, the data win.

## Numbers used above

GPT-2 family parameter counts from the HF configs: 124M (n_layer 12, d 768), 355M (24, 1024), 774M (36, 1280), 1.5B
(48, 1600); V = 50257; N_ne = total minus wte minus wpe, computed from parameter shapes in `accounting.py` (the table
uses 12 d^2 n_L plus layer norms as the estimate; the code's exact count replaces it in the registration).
