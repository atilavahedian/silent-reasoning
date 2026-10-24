# Silent reasoning: PLAN.md (the plan, approved 2026-10-17)

Approved by Atila on 2026-10-17 from the v2 plan. Status and lab notes live in CHANGELOG.md; this file is the plan and is edited only by dated amendment at the bottom.

## 0. Context

Hinton wants models that reason in their hidden states instead of writing chain of thought (CoT). The bar: match or beat written CoT on hard verifiable tasks with at least 5x fewer output tokens at equal total compute, an advantage that holds or grows with size, and a readout of the silent steps with measured faithfulness. "Silent reasoning does not beat written reasoning" is an acceptable answer; a fake win is not.

Two things to say up front, because they shape everything below.

1. A latent step costs the same FLOPs as emitting a token: one forward pass of the whole model at one position. "5x fewer output tokens at equal compute" is a billing claim unless latent steps are fewer, or cheaper per step (several latent slots per pass, or loops over a subset of layers). The judge therefore tracks FLOPs, sequential passes and emitted tokens separately, and the headline needs all three.
2. The 2025-26 literature has already shown that Coconut's flagship win is the curriculum, not the latent content (pause tokens with the same curriculum: 96.6 vs 97.0 on ProsQA), and that its latents are mostly causally inert. This lab has seen the same pattern twice (energy slots inert; shuffled notes tied). So Phase 2 must build mechanisms whose latents are *forced* to carry state, and the readout must be judged by intervention, not decodability.

My prior before any run: P(full goal) about 0.10; P(tie with tuned CoT at equal FLOPs with 3-5x fewer emitted tokens, no growing advantage) about 0.30; P(clean negative) about 0.60. All three are publishable because nobody has the controlled, seeded, compute-matched measurement.

## 1. Phase 0 literature synthesis (done in this session; arxiv and HF are blocked here, so numbers come from search snippets and GitHub and get re-checked from the PDFs on the Mac before they enter the registration)

### 1.1 Coconut as published (Hao et al., Dec 2024, COLM 2025; code facebookresearch/coconut, MIT)
- GPT-2 124M, AdamW lr 1e-4, wd 0.01, batch 32 x 4 GPUs, fp32, 4 x A100-80GB. ProsQA/ProntoQA: c_thought 1, 5 epochs per stage, 6 latent stages, 50 epochs, optimizer reset per stage, best-validation checkpoint. GSM8k-Aug: CoT 25 epochs first, then Coconut from that checkpoint, c_thought 2, 3 stages x 3 epochs.
- Data: ProsQA ships in the repo (17,886 / 300 / 500; about 180-word questions with the graph edges included, 3 steps). ProntoQA is regenerated from asaparov/prontoqa (5-hop fictional ontology, 9,000 / 200 / 800). GSM8k-Aug is the iCoT augmented set (385k train; test = GSM8K test).
- (pub.) GSM8k: CoT 42.9, No-CoT 16.5, iCoT 30.0, pause 24.1 (corrected 2026-10-19 from the PDF's Table 1; the snippet said 16.4), Coconut 34.1. ProntoQA: 98.8 / 93.8 / 99.8 / Coconut 99.8. ProsQA: CoT 77.5, No-CoT 76.7, Coconut 97.0 +- 0.3, pause-same-curriculum 96.6 +- 0.8, Coconut without curriculum about No-CoT. Emitted tokens: ProntoQA 92.5 vs 9; ProsQA 49.4 vs 14.2.
- The official code generates at batch size 1 and rebuilds embeddings with Python loops per latent step; that is why outside groups report 36 GPU-h for ProsQA (Jin et al. 2026). The arithmetic is 2.5-3 A100-h per run with a vectorized forward and batched evaluation.

### 1.2 The critiques
- Curriculum is the mechanism on ProsQA ("The Curriculum Is the Mechanism", 2026): pause-token control with the same 7-stage curriculum, 96.6 vs 97.0, McNemar p = 0.845.
- Aswal et al. (June 2026, "Observable Patterns Are Not Explanations"): independent ProsQA retraining; BFS-like frontiers and decodable arithmetic also appear in controls without recurrence or curriculum; matched pause-token and thought-ablation controls; causal effect of a latent is graded and low-rank. This is the "parallel search also in controls" study.
- Rizvi-Martel et al. (COLM 2026): fine-tuned Coconut reads the answer off the question; superposition only when trained from scratch. Zhang et al. (Dec 2025): perturbing Coconut latents flips under 5-10% of answers vs up to 50% for CoT tokens; artifact exploitation. Dilgren and Wiegreffe (2026): latents removable on logic tasks, matter more on math. Jin et al. (July 2026): accuracy 73 to 90 while counterfactual responsiveness fell 71 to 27 along training. Backour (Sept 2026): softmax plus autoregressive feedback collapses or amplifies mixtures; superposition is not preserved in pretrained models. "Single-threaded reasoners" (ICLR 2026): soft thinking follows the argmax.

### 1.3 Methods that beat Coconut, and scale
- CODI (EMNLP 2025): self-distillation, single stage, GPT-2 GSM8k about 43.7 vs CoT 43.9 at 3.1x fewer tokens, about 5 GPU-h. SIM-CoT (ICLR 2026): auxiliary decoder per latent step; +8.2 over Coconut, +2.1 over explicit CoT on GPT-2 at 2.3x token efficiency; Coconut collapses from 3 to 5 latents (about 45 to 12.5). KaVa (ICLR 2026): KV distillation, below CoT at every size. LOTUS (June 2026): looped padded transformer with per-position supervision; states that prior latent methods trail explicit CoT past 1B and the gap widens; closes it at 3B (70.05 vs 71.5) with 6 blocks of 25 slots and 6 loops.
- RL: Latent-GRPO (Llama-1B 47.6 to 66.3 with 24 tokens; plain GRPO on latent SFT collapses), Dropout-GRPO (+1.7), SWITCH/TARPO/HyRea (hybrid routing); Ozeren and Assenmacher: RL latents still lag CoT on math. Inference-time scaling: Coconut is deterministic, dropout gives path diversity, PRM reranking barely helps.
- Recurrent depth: Saunshi et al. (k layers x L loops about kL layers; T loops simulate T CoT steps), Huginn 3.5B, Ouro 1.4B/2.6B at 7.7T tokens, Loopie MoE. Our rung 0: looped 12L x2 lost 0.038 nats to plain 24L at equal FLOPs (LM loss, from scratch, 16 epochs). Pause tokens (Goyal et al.): need pretraining with pauses; finetune-only is mild or negative.

### 1.4 Formal results
- Zhu et al. (NeurIPS 2025): 2 layers with D continuous thoughts solve reachability of diameter D; constant-depth discrete CoT needs O(n^2). Xu and Sato (ICML 2026): latent thought parallelizes; CoT wins on approximate counting and sampling; loops are not universally more expressive. Memory-budget separation (May 2026): a fixed-width recurrent state cannot decide P-complete problems under logspace reductions, polynomial CoT can; per-position loops are closer to a scratchpad. A single thought vector per step is a compressed loop. Zou et al. (2026): curriculum is provably necessary. Chen et al. (ICML 2026): outcome-only supervision gives dual collapse; reconstruction objectives preserve latent-step mutual information.

### 1.5 The open gap
1. No compute-matched, token-matched, seeded comparison of latent vs tuned CoT exists. Everything is seed 0 and methods differ in compute.
2. Theory says single-vector latents are a bounded channel; the methods that close the gap widen the per-step state or supervise it. Nobody has tested "latent matches CoT iff per-step latent state exceeds the task's live state" at matched FLOPs.
3. Faithfulness: latents are inert in fine-tuned Coconut; nobody trains for causal use of latents; readouts must be judged by intervention.
4. Scale: the one scaling statement is negative (gap grows past 1B). The lab's slope method has not been applied.
5. Code: no published latent result on program-output prediction (one CoLaR-code checkpoint, no numbers). Execution gives verifiable intermediate states for free.

## 2. What this lab already has and already tried

### 2.1 Reuse (do not rebuild)
- HF fine-tuning loops: `experiments/2026-10-13-nightshift-sleep/code/hinton_sleep/train.py` (loss on target tokens only, LoRA), `experiments/2026-10-12-think-bet/code/think/train.py` (FLOP-budgeted, bf16 autocast, compile), `common/babylm/recipes/gpt2/train.py`.
- Accounting: `think/lm.py` (FLOP formulas; needs GPT-2 n_inner fix), nightshift `gen.py:HFGenerator.stats`, `common/sprint1/scripts/infer_bench.py`.
- Statistics: `think/analyze.py` (paired bootstrap, margin, sigma floor), `common/babylm/instruments/{cells,slices}.py`, `common/sprint1/scripts/compare.py`.
- Governance: `common/sprint1/ledger/ledger.py` (hash-chained admit/close), `think/gcp/budget.py` (programme cap; copy), registration template in `experiments/2026-10-12-think-bet/governance/registration.md`, critic log format in `experiments/2026-10-09-babylm-gap-map/governance/critic_log.md`, audit format `think-bet/docs/audit_06.md`, leak tools `common/sprint1/audit/{causality,contract_audit}.py`.
- GCP: `think/gcp/launch.sh` + `vm_runner.py` plan JSONs + stage machine `pipeline_tools.py`; sprint1 `run_jobs.py` allowlist. Mac: `common/ops/pi/ops/MAC_STATE.md` (M5 Pro, 20-core GPU, 24 GiB, MLX 0.32.2, slot lock); Mac torch lock torch 2.13 / transformers 4.57.6; no MPS-vs-CUDA parity test exists (the lab did MLX-vs-torch with a float64 reference).
- Execution: `experiments/2026-09-22-next-acquisition-repair/code/hinton_next/execution.py` (bounded interpreter), `experiments/2026-09-21-c0-program-canvas/code/hinton/{tasks,sandbox}.py` (program generator with ground truth). GSM8K scorer: nightshift `general.py`.
- `tools/check_public.py`: no file over 5 MB, no credential patterns, no wandb links (the Coconut code imports wandb; we do not).
- Not in this repo: the 49% cost study, the scaling-slope study, Kaggle, "acctb/acctc". Projects named in the repo: `cloud-main`, `cloud-d`, `cloud-e`, `cloud-b`, `cloud-c`.

### 2.2 Cousins of latent reasoning already tried here (all lost or were never run)
| Attempt | Mechanism | Result | Lesson carried into this plan |
|---|---|---|---|
| C0 program canvas (09-21) | 96-slot canvas, 4 tied rounds, from scratch | plain Transformer matched it with less training; carried latent state never causally isolated | every arm gets a scrambled-state control at equal compute |
| A0 workspace (09-21) | 8 slots x 2 rounds | zero and shuffled residual solved identical tasks | the scrambled control is the first thing we look at |
| Energy campaign (09-23) | 16 slots, 4 differentiable steps on frozen 1.5B | energy = recurrent = 9B at 6/32; "iterations causally active without a net benefit"; K sweep declared, never run | the step sweep (0, 1, 2, 4, 8 latent steps) is registered and run |
| Rung 0 (10-12) | 12L looped x2 vs plain 24L | looped worse by 0.038 at equal FLOPs | the plain-depth control is mandatory |
| MoD arm B, D_gainloop, BOLD-1 V2 | adaptive depth, early exit, chunk-rate loop | all losses vs dense 2x or tuned baseline | adaptive halting is an add-on, not a headline |
| H_plancode | discrete latent plan per block | codes collapse or go unused | process supervision of the latent, not free-form latent |
| Think bet (10-12) | explicit self-written notes as training data | worse than plain data; mismatched notes tied | content-matched nulls decide attribution |
Never tried here: continuous-thought feedback, any pretrained decoder with latent steps, pause tokens, trained halting, a test-time step sweep on a verifiable task, accuracy-vs-token cost curves, latent models on math or logic, CoT-to-latent curricula, probing or faithfulness of reasoning states. That is this project's novelty inside the lab; outside it, every mechanism below is labelled with its known family.

## 3. Lab method lessons, as rules of this project
1. Port onto the strong baseline first: the bar is a *tuned* CoT fine-tune of the same base (LR x epochs grid, trial parity, best on validation), not the published CoT. ProsQA CoT 77.5 vs No-CoT 76.7 says the published CoT is under-tuned.
2. Seeds: 3 minimum, the same seeds in every arm, paired. Sigma = pooled seed std of the CoT arm with a floor (1 point on accuracy at these test sizes), rises never falls. Screen pass 1 sigma, confirmation 2 sigma and a 95% CI excluding 0, on fresh seeds. Two agreeing seeds are not a replication.
3. Nulls are audited: unit tests prove the scrambled-latent, pause-same-curriculum and no-curriculum arms cannot see the true latent content; collator label-leak test (no answer token under a latent position); `causality.py`-style batch-isolation test on the latent path.
4. Too good is a bug: any arm more than 10 points over tuned CoT at 124M on GSM8k, or above 99% on ProsQA-hard, or a null above the baseline by 2 sigma, triggers the audit (13-gram train/test overlap, rescoring from checkpoint, leak checks) before anything else runs.
5. Environment rule (gap-map row 126): Mac and GPU numbers never enter one comparison unless a same-seed cross-environment pair agrees within the floor. Phase 1 oracle runs are Mac fp32 (which is also the published precision); Phase 2 onward is GPU bf16 and re-runs its own CoT bar.
6. Head starts fade: every accuracy is a curve over training budget; the registered read is at the final budget; a lead that shrinks from 1x to 2x budget is HEADSTART.
7. Pre-register before results; commit time is the record; rows are never edited; amendments only tighten unless author-directed, and then the original verdict is still printed. Verdicts are write-once with the code commit.
8. Compute is matched in FLOPs with every pass charged (think-bet formulas extended to GPT-2); secondary accountings (sequential passes, wall-clock) are reported and a flip earns "accounting-sensitive".
9. Plain-depth control (rung 0): at fixed inference FLOPs the alternative to k latent steps is a bigger model with none. GPT-2 355M (2.9x FLOPs of 124M) is the free plain-depth arm for 124M; 774M for 355M.
10. Selection split and scoring split are disjoint; the test split is scored once per registered read; no format shopping.

## 4. The judge (critic locks it in `governance/registration.md` before any run)
- Tasks: ProsQA, ProsQA-hard (regenerated graphs, deeper than training; the released data has edges, so a generator is a day's work), ProntoQA, GSM8k-Aug (test = GSM8K test, 1,319), Phase 4 program-output prediction (our generated ladder plus CRUXEval-O 800). All exact match, machine verified.
- Arms, same base model, same seeds: No-CoT; tuned CoT; latent method; pause tokens with identical curriculum and token budget; latent without curriculum; scrambled-latent (same number of steps, latents replaced at inference by latents from another example of the same length class; and a trained-with-scrambled-feedback variant); plain-depth (bigger model, No-CoT, at the same inference FLOPs); zero-latent and step-sweep ablations at inference (0, 1, 2, 4, 8).
- Axes per example: accuracy, emitted tokens, sequential forward passes, FLOPs (prefill plus steps), wall-clock on one device.
- Primary test: accuracy(latent) >= accuracy(tuned CoT) with FLOPs(latent) <= FLOPs(CoT) and emitted tokens <= 0.2 x CoT, paired over 3 seeds, 2 sigma and CI. Secondary: full Pareto curves (CoT budgeted to k tokens; latent to k steps).
- Scale (Phase 3): gap = latent minus tuned CoT at matched FLOPs at each size; slope of the gap in log N with a seed-bootstrap CI; and the gap-map style multiplier, here "the FLOP multiplier CoT needs to match the latent arm", interpolated in log FLOPs along the budgeted-CoT curve, no extrapolation.
- Readout: every claimed win is reported with readout coverage and load-bearing fraction (section 6.5); a win whose readout decodes nothing load-bearing is labelled "unexplained" and does not count as the goal.
- Kill rules: K1 oracle mismatch (Phase 1 outside max(2 points, 2 sigma) of published after two debug rounds: stop, report, do not proceed). K2 an arm within the floor of its scrambled control is dead (its latents carry nothing). K3 an arm that loses to plain depth at equal FLOPs is dead as an efficiency claim (continues only as a labelled billing claim). K4 per size: best latent arm more than 5 points under tuned CoT at matched FLOPs after the registered budget means Phase 3 does not scale it. Fade: HEADSTART rule above.
- Labels: WIN / WIN-KNOWN / HEADSTART / NULL / LOSS / BUG / UNTESTED; novelty NEW / COMBINATION / KNOWN, defaulting to the more-known label.
- Registration also fixes: pinned model and data SHAs, plumbing permissions before signature (smoke runs capped at 2k examples and 200 steps), FLOP formulas, LR grids with trial parity, the critic's numeric predictions, per-phase caps, the report spec, and a `deviations.md`.

## 5. Phases and compute

A100-h = one A100-40GB hour; Mac-h and Kaggle-h convert to A100-h after the Phase 0 calibration.

| Phase | What | A100-h (est.) | Where | Gate |
|---|---|---|---|---|
| 0 | prior-art doc, theory memo, registration, `common/latent/` skeleton with CPU tests, Mac throughput and parity calibration | 0 (about 1 Mac-h) | here + Mac | critic signs |
| 1 | ProsQA + ProntoQA oracle: 5 arms x 3 seeds, published recipe; tuned-CoT bar (about 6 runs per task); GSM8k-Aug oracle: CoT, Coconut, No-CoT, 1 seed | ProsQA+ProntoQA about 45 (2.5-3 h per run x 15, plus tuning); GSM8k-Aug about 25 | Mac (fp32) for ProsQA/ProntoQA, Kaggle for extra seeds; GSM8k-Aug is not a Mac job (decision B) | K1 |
| 2 | cards C1-C6 screened on ProsQA-hard and a 50k GSM8k-Aug subset at 124M (1 seed, kill lines), survivors confirmed 3 seeds on full sets vs all controls | screens about 80; confirmations about 70 | GCP once allowed, else rent (ask); Kaggle covers only the smallest screens | an arm passes the primary test, or a documented tie |
| 3 | GPT-2 124M / 355M / 774M (epochs 25 / 10 / 5 on GSM8k-Aug; ProsQA-hard at all sizes); arms tuned CoT, best latent, scrambled, plain-depth; 3 seeds at 124M and 355M, 2 at 774M (labelled) | about 220 (774M about 10 h per run) | GCP after the slope study, allowed projects only | slope read |
| 4 | execution-trace dataset and ladder, 4 arms x 3 seeds at 124M and 355M, CRUXEval-O | about 50 | GCP or rent | report |
| Total | || | |

Each phase gets its own ledger line and cap, and a typed go from you with the estimate in front of you. Phase 1 is the only phase that can start on approval of this plan.

Throughput assumptions (replaced by Phase 0 measurement): vectorized Coconut on HF GPT-2 124M about 35k tokens/s fp32 and 50k bf16 on an A100; a ProsQA run is 17,886 x 50 epochs x about 300 tokens = 270M token-passes, times about 1.4 for the latent passes. Mac M5 Pro on MPS: unknown, my guess 5-15k tokens/s, so 7-20 Mac-h per ProsQA run and 1.5-3 weeks of continuous Mac time for all of Phase 1 ProsQA + ProntoQA (decision D). Kaggle: 30 GPU-h per week on T4/P100, each about 1/6 of an A100 in fp16, so about 5 A100-h per week: extra seeds, not screens.

Timeline (est.): Phase 0 two days; Phase 1 two to three weeks on the Mac (two days if rented); Phase 2 three weeks; Phase 3 two weeks; Phase 4 two weeks.

## 6. Phase details

### 6.1 Phase 0 (Fable, xhigh; no GPU)
Deliverables in ``:
- `docs/prior_art.md`: about 40 papers (sections 1.1-1.4 expanded), each with setup, numbers, what it rules out, known-family label, and whether it was read in full or from a snippet. Numbers re-verified from PDFs on the Mac.
- `docs/theory_memo.md`: T1 the FLOP identity; T2 bits per step (a CoT token commits about 15.6 bits to an unbounded scratchpad; a bf16 768-d latent is nominally 12k bits but usable capacity is bounded by what the next layer reads back, and a fixed-width state cannot do P-complete work), with the prediction that latent width must exceed the task's live state (frontier size on ProsQA, intermediate count on GSM, live variables on code); T3 superposition is real for reachability from scratch, not in fine-tuned LMs; T4 what the scrambled control can and cannot detect; T5 why parallel slots or partial-depth loops are the only ways a latent step is cheaper than a token.
- `governance/registration.md` (critic), `governance/critic_log.md`, `governance/deviations.md`, programme budget file modelled on `think/gcp/budget.py`.
- `PLAN.md` (this plan, refined) and `CHANGELOG.md` (status, next steps, every failed approach) in the experiment folder; one pointer line added to the root `AGENTS.md`.
- `common/latent/` skeleton with CPU tests (section 6.2).
- Mac calibration: GPT-2 124M forward+backward at seq 320, batch 16, fp32 and fp16 autocast on MPS, tokens/s and memory; one-batch Coconut forward parity MPS vs CPU float64. This is the only thing that "runs" in Phase 0 and it trains nothing.

### 6.2 Phase 1 reproduce (Opus 5.5 high; Fable critic audits)
`common/latent/`: `model.py` (HF `GPT2LMHeadModel` wrapper with `inputs_embeds` and KV cache; vectorized latent feedback; GPT-2 124M/355M/774M and the Qwen3 bases), `data.py` (Coconut JSON format; ProsQA from the pinned Coconut commit, hashed; ProntoQA regenerated with the pinned generator command; GSM8k-Aug from the pinned iCoT files; 13-gram train/test overlap check), `curriculum.py` (stage schedule, pad_latent_to_max, optimizer reset, bot/eot; arms cot, no_cot, coconut, pause_same_curriculum, coconut_no_curriculum), `eval.py` (batched greedy generation, per-example tokens/passes/FLOPs, official answer extraction, scrambled-latent and zero-latent and step-sweep interventions), `accounting.py`, `probe/` (6.5), `tests/` (label leak, null blindness, parity, accounting against hand-computed FLOPs). Experiment-only scripts and configs stay in the experiment folder.
Runs: day-1 smoke (ProsQA 2k subset, 1 seed, 2 arms, Mac) for plumbing and real throughput; then the oracle (ProsQA and ProntoQA, 5 arms x 3 seeds, published recipe, fp32, batch 128 via accumulation, selection on valid, scored on test); then the tuned-CoT bar; then GSM8k-Aug (decision B). The seed spread itself is a new result.
Exit: K1 passes; `CHANGELOG.md` records every deviation (transformers version, batch emulation, precision, generation batching).

### 6.3 Phase 2 invent (Fable xhigh writes the cards; Opus 5.5 xhigh implements; critic registers each card before its screen)
Each card: mechanism, prediction, known-family label with the three closest works, its own null, screen budget, kill line, fade expectation, leak answers.
- C1 Width-matched latents. Each step emits w parallel slots (w in 1, 4, 16) refined in one pass per step (Jacobi style); steps reduced so FLOPs match CoT. Prediction (T2): accuracy vs w has a knee at the task's live-state size. Family: LOTUS, PCCoT. New: the matched-FLOPs, matched-memory test and the crossover prediction.
- C1b Cheaper steps. Latent steps loop only the middle k of L layers (prelude / core / coda retrofitted onto GPT-2), so a step costs k/L of a token. Family: Huginn, retrofitted recurrence. This is the one route to a true compute saving per step; tested under the same judge.
- C2 Verifiable-state supervision. An auxiliary decoder reconstructs the *executed* intermediate state from each latent (BFS frontier, running values, variable bindings); removed at inference; it is also the readout. Family: SIM-CoT, CODI, LOTUS (they use CoT text). New: supervision from executable ground truth; faithfulness by intervention.
- C3 Latent-dependence loss (no name in the literature I found). A contrastive term: answer logits under true latents must differ from logits under scrambled latents by a margin, so the model cannot route around its own latent steps. It turns the scrambled control into a training signal and attacks the 2026 failure mode head-on. Risk: trivial use (answer parked in latent 1); mitigated by C2 and held-out-depth splits.
- C4 Adaptive number of steps with a compute penalty. Family: ACT, PonderNet, "Learning When to Stop", Tyler. Add-on.
- C5 Hybrid silent-plus-written: sparse written anchors every k latent steps. Family: SwiReasoning, TARPO, HyRea, SpiralThinker. Reported as a Pareto curve.
- C6 Latent RL on the best SFT arm, only with well-defined policy ratios (SWITCH boundary tokens or dropout-sampled latents). Low prior; runs only after a tie.
- Instrument for every card: the bit-width ladder (quantize the latent channel to b bits per step at inference) to measure information actually used per step against the 15.6 bits of a CoT token. One number for the whole project.
Screen at 124M, 1 seed, ProsQA-hard and the 50k GSM8k-Aug subset, against tuned CoT, pause-same-curriculum, scrambled, plain depth; survivors to 3 seeds on the full sets.

### 6.4 Phase 3 scale (Opus 5.5 high runs; Fable critic reads)
GPT-2 124M, 355M, 774M (1.5B and Qwen3-0.6B/1.7B only if budget and a clean GPT-2 family). Arms: tuned CoT, best latent, scrambled, plain-depth. Tasks: GSM8k-Aug and ProsQA-hard. Read: gap vs log N, slope sign and CI, and the CoT FLOP multiplier at each size. K4 per size.

### 6.5 Readout and faithfulness (requirement; built in Phase 1, used everywhere)
- R1 logit lens per latent (known to mislead; reported). R2 linear probes for task state on held-out examples (frontier membership per node, intermediate values, variable bindings). R3 the C2 decoder.
- Faithfulness, all causal: F1 counterfactual patching (swap latent step t with the latent from a minimally different problem; does the answer follow?); F2 scrambled-latent drop; F3 probe-direction edits with predicted answer change. Headline: load-bearing fraction = decoded steps whose intervention moves the answer as predicted. Positive control: a tiny from-scratch model on synthetic reachability, where superposition is known to emerge.
- The faithfulness curve is tracked along training (Jin et al.), not only at the final checkpoint.

### 6.6 Phase 4 code (Opus 5.5 xhigh builds; critic registers)
Short Python functions (3-13 lines) from templates plus the c0 generator, run in the repo's bounded interpreter; CoT = per-statement state trace; oracle states for C2 and the probes; ladder by trace length and live variables; our test split plus CRUXEval-O (expect near zero at 124M-355M; the ladder carries the measurement). Arms as in section 4 at 124M and 355M. T2's prediction (live state grows with program length) is written down before the run.

## 7. Team
- Fable 5.1 xhigh: Phase 0 synthesis and theory; Phase 2 cards; the critic (separate agent, owns registration and critic log, writes no model code, audits every claimed win with the leak checklist).
- Opus 5.5 high: `common/latent/`, data, evaluation, Phase 1 and 3 runs, Kaggle/GCP plumbing, ledger.
- Opus 5.5 xhigh: C1-C6 implementation, Phase 4 dataset, analysis and readouts.
- One objective, scope, output format and boundary per agent; parallel only on disjoint files; small questions get no agent.

## 8. Repo hygiene
- `` with README, REPORT (standard sections), PLAN.md, CHANGELOG.md, code/, configs/, results/, governance/, docs/, ARTIFACTS.md; INDEX.md row at close. Shared code in `common/latent/` with CPU-runnable tests.
- No wandb, no credentials, no checkpoints, no file over 5 MB; data downloaded by script (ProsQA is 30 MB); `python3 tools/check_public.py` before every PR.
- Pins: torch 2.7.x and transformers 4.51-4.56 for GPU (the lab's "<5" line); the existing torch 2.13 / transformers 4.57.6 lock on the Mac. Commits after each step on `claude/amazing-ramanujan-xc6ow3`; draft PR.

## 9. Decisions I need from you (the plan runs under my recommendation on a plain "go")
- A. Mac stack: torch on MPS (one codebase with the GPU phase, parity test vs CPU float64) or MLX (the lab's Mac experience, likely faster, second codebase). Recommendation: torch MPS; MLX only if calibration shows under 3k tokens/s.
- B. Recommendation: rent; it is the one published number that is not at ceiling.
- C. Which of the five GCP projects named in the repo are allowed for this project, and when the slope study finishes.
- D. Your stated preference is the Mac; I will follow it unless you say otherwise.

## 10. Verification
- Phase 0: signed registration; calibration numbers in CHANGELOG; `pytest common/latent/tests` green on CPU.
- Phase 1: oracle table (published vs ours, per arm and task, seed spread, tolerance verdict); leak audits recorded; FLOP accounting checked by hand on one example; cross-environment pair logged.
- Phase 2: each card's verdict against its null, tuned CoT, pause-same-curriculum and plain depth; bit-width ladder curve; readout coverage and load-bearing fraction.
- Phase 3: slope table and plot; the critic's read quoted verbatim.
- Phase 4: execution-verified ladder and CRUXEval-O accuracy; probe decoding of bindings.
- Throughout: CHANGELOG current at every commit; REPORT filled at close; INDEX row.

## Amendment A1 (2026-10-18, author-directed): compressed 72-hour scope, C1b added, conditional scale phase

Atila's decisions on 2026-10-18: GCP projects AcctA (`cloud-a`), AcctB (`cloud-b`) and AcctC (`cloud-c`) are allowed for this
project (the earlier "never acctb or acctc" rule is withdrawn by Atila in writing); renting outside GCP still needs a typed
yes with the estimate. The Mac path is dropped; every Phase 1 number is GPU, so the cross-environment rule applies only
between GPU types.

Scope for the first 72 hours (replaces sections 5 and 6.2-6.3 for this window; the judge in section 4 is unchanged):
1. Oracle, cut to what discriminates: ProsQA cot, no_cot, coconut, pause at seed 0, then seeds 1 and 2 for coconut and
   cot; GSM8k-Aug cot, then coconut from it, seed 0. ProntoQA dropped (at ceiling). Tuned-CoT bar: two extra LR trials
   on ProsQA. Cells with fewer than 3 seeds are labelled single-seed in the K1 table.
2. One bet, three mechanisms, screened together at 124M on ProsQA-hard and a 50k GSM8k-Aug subset, seed 0: C2
   (verifiable-state supervision), C3 (latent-dependence loss) and C1b (latent steps through the middle k of L layers,
   k in {4, 8} of 12), each alone and C2+C3 combined, against tuned CoT, pause-same-curriculum, scrambled latents and
   the 355M plain-depth control; step sweep and scrambled interventions on every checkpoint; seeds 1 and 2 on survivors.
3. Readout: probes and counterfactual patching on the survivors; load-bearing fraction reported.
4. Dropped from this window: ProntoQA, C4, C5, C6, Phase 3 scale, Phase 4 code. Each is UNTESTED, not estimated.

Conditional continuation: if the 124M read is positive under the registered primary test (any arm beats or ties
tuned CoT at matched FLOPs with emitted tokens <= 0.2x and K2/K3 not fired), the next phase is scale (GPT-2 355M and
774M, same arms, same controls, 3 seeds at 355M and 2 at 774M labelled) and then code transfer (section 6.6), under
the same registration discipline and a separate typed go with its estimate (about 270 A100-h). If the read is
negative, the project closes with the report and the INDEX row.

Compute shape and rental: decided in CHANGELOG.md on the day; the critic re-signs the plumbing permissions for the GPU
variant before the first VM.

## Amendment A2 (2026-10-18, author-directed): compute shape and window cap

"Use our existing compute, and go" (Atila, 2026-10-18). No cash rental. the critic records it. Buckets: `store://<project-
id>-bucket/silent/` in each project. Launcher: `code/launch/` (silent_launch), adapted from the nightshift launcher, hash-chained
ledger at `governance/run_log.jsonl`.

budget balances are not readable from the session (Cloud Billing API disabled on the sprint project, no billing export) and the author's statement is the
record. one hinge job only, `screen-gsm50k-accta` (C3, C3-foreign, C2+C3 on GSM-50k; 16 h; no ProsQA-hard hinge job; no east zones; no paid or out-of-pocket
charges; no other cap, quota or billing change; the existing ledger and auto-delete launcher; The critic records it.

## Amendment A3 (2026-10-19, author-directed): agent model and effort tiers; commit cadence; the GSM cot run's budget rule

Atila, 2026-10-19 (in writing, this session). Section 7 (Team) is replaced by these tiers for every agent in the
project: Fable 5.1 at high effort for the hardest thinking (theory and literature synthesis, designing mechanisms and
cards, the critic who locks rules and audits every claimed win); Opus 5.5 at extra-high effort for hard building and
analysis (mechanisms, readouts and probes, statistics, anything where a subtle bug would corrupt a result); Opus 5.5
at high effort for standard work (reproduction runs, infrastructure, VM launches, data pipelines, evaluation code,
tests); Opus 5.5 at medium effort for routine work (monitoring runs, moving files, status updates, simple fixes).
Effort scales to the task: no top model for routine work, no agent for small questions; every subagent gets one
objective, scope, output format and boundary; parallel agents only on disjoint files.

Commit and push after every step. If the main project's GSM cot job looks like it will hit the main cap or its
deadline, the run is moved to AcctA's project before it is cut (a cross-type resume under registration A7.6, labelled).
The judge, the registration and its amendments are unchanged.

## Amendment A2.2 (2026-10-21, author-directed): the closing plan

Atila, 2026-10-21 (in writing, this session), after the window closed at within a fixed compute cap (RESULTS.md `rev32`):
1. auto-shutdown on every VM): (a) det2, the second determinism run, so a task-level sigma exists; (b) Coconut without curriculum on GSM-50k and ProsQA-
   hard, seed 0, the same judge; (c) the Pareto read of the existing earlier-stage checkpoints of every latent arm (accuracy vs written tokens vs FLOPs),
   no new training. Results enter RESULTS.md as a dated section; no earlier table is edited.
2. **Close the project**: RESULTS.md marked FINAL; the standard experiments/ layout; the INDEX row; merge to main; PR #87
   closed with a one-line pointer to the report; zero VMs in every project and a closed, verified ledger.
3. **A paper draft**, nothing public: a controlled, compute-matched, seeded test of latent reasoning at 124M.
The judge, the registration and its amendments are unchanged.

det2 moves to AcctC after J2, J3 runs on AcctB; the project caps are unchanged. The critic records it.
