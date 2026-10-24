# Prior art: latent and silent reasoning (Phase 0, 2026-10-17)

Written by the operator (Fable) from the Phase 0 pull. Provenance of every number is marked: **GitHub** = read from the
official code or data; **snippet** = read from a web-search abstract or secondary summary (arxiv.org and
huggingface.co are blocked from the planning container, so no PDF was opened); **repo** = this repository. Every
snippet number is re-verified from the PDF on the Mac before it enters a registration or a report. Known-family labels
follow the lab's convention: NEW / COMBINATION / KNOWN, defaulting to the more-known label.

## 0. One-paragraph summary

Coconut (continuous chain of thought) showed in Dec 2024 that a GPT-2 fed its own last hidden state as the next input
embedding can match or beat written chain of thought on two synthetic logic tasks with far fewer emitted tokens, and
lose on GSM8k. Within eighteen months, independent groups showed that the flagship ProsQA win comes from the training
curriculum (pause tokens under the same curriculum tie it), that the latent vectors are mostly causally inert in
fine-tuned models, that "parallel search" signatures appear in controls, and that single-vector latents collapse past a
handful of steps. The methods that close the gap to explicit CoT do one of two things: supervise each latent step
against the explicit step it replaces (CODI, SIM-CoT, LOTUS) or widen the per-step latent state and loop it (LOTUS,
looped language models). The one scaling statement in print says latent methods trail explicit CoT past 1B parameters
and the gap grows. Theory agrees with the empirics: a single fixed-width recurrent state is a bounded channel that
cannot do P-complete work, while a scratchpad can; the parallel-search advantage of continuous thoughts is real for
graph reachability and emerges when trained from scratch, not when a pretrained LM is fine-tuned. Nobody has published
a compute-matched, token-matched, multi-seed comparison with scrambled-latent controls across model sizes, and nobody
has trained a model to make its latents causally necessary. That is the gap this project aims at.

## 1. The reference: Coconut

| Item | Value | Provenance |
|---|---|---|
| Paper | Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian. "Training Large Language Models to Reason in a Continuous Latent Space", arXiv 2412.06769, COLM 2025 | snippet |
| Code | facebookresearch/coconut, MIT. Files hashed in `CHANGELOG.md` (coconut.py rev394..., dataset.py 21029557..., run.py rev395...) | GitHub |
| Base model | `openai-community/gpt2` (124M) | GitHub (args/*.yaml) |
| Optimiser | AdamW, lr 1e-4, weight decay 0.01, fp32 (`bf16: False`), batch 32 per GPU on 4 A100-80GB (effective 128), gradient accumulation 1 | GitHub |
| ProsQA / ProntoQA recipe | `c_thought 1`, `epochs_per_stage 5`, `max_latent_stage 6`, `pad_latent_to_max True`, `num_epochs 50`, `reset_optimizer True`, `save_only_improve False`, seed 0 | GitHub |
| GSM8k-Aug recipe | CoT stage first (`gsm_cot.yaml`, 25 epochs, best checkpoint around 40% valid), then Coconut from that checkpoint: `c_thought 2`, `epochs_per_stage 3`, `max_latent_stage 3`, `resume 3`, 25 epochs | GitHub |
| Data: ProsQA | shipped: `data/prosqa_{train,valid,test}.json`, 30.9 MB / 0.52 MB / 0.86 MB; 17,886 / 300 / 500 records (counts: snippet, to confirm from the files); records carry `idx_to_symbol`, `edges`, `root`, `target`, `neg_target`; questions about 180 words, 3 proof steps | GitHub (sizes, fields), snippet (counts) |
| Data: ProntoQA | regenerate with asaparov/prontoqa at commit rev37: `run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1`, then `preprocessing/prontoqa.py` (steps = consecutive pairs of the chain; 9,000 / 200 / rest) | GitHub |
| Data: GSM8k-Aug | Internalize_CoT_Step_by_Step commit rev36, `data/gsm8k/{train,valid,test}.txt` (train 87.8 MB on LFS), format `question||step step ... #### answer`, converted by `preprocessing/gsm_icot.py` | GitHub |
| Evaluation | greedy, batch size 1 in the official generate; `max_new_tokens` 64 (ProsQA, ProntoQA) or 128 (GSM); answer = `text.split("#")[-1].replace(",", "").strip()`; checkpoint = best validation accuracy, then scored on test with `only_eval` | GitHub (run.py) |
| Mechanism | at each `<|latent|>` position the input embedding is replaced by the last-layer hidden state of the previous position; one extra sequential forward per latent, reusing the KV cache; the fed-back state is not detached; loss is the usual CE on the remaining steps and answer | GitHub (coconut.py) |
| Curriculum | stage k removes the first k reasoning steps and inserts k x c_thought latents between `<|start-latent|>` and `<|end-latent|>`; `pad_latent_to_max` pads short chains; `uniform_prob` mixes stages (0.3 only in analysis runs) | GitHub (dataset.py) |

Published accuracies (Table 1 of the paper; verified against the PDF text on 2026-10-19, `results/prep/coconut_pdf_excerpt.txt`, sha256 rev52...; the +- values are the paper's own standard deviations; ProntoQA pause as thought is 100.0 +- 0.1 and ProsQA 96.6 +- 0.8 in the PDF):

| Task | No-CoT | CoT | iCoT | Pause (same curriculum) | Coconut | Emitted tokens CoT vs Coconut |
|---|---|---|---|---|---|---|
| GSM8k-Aug | 16.5 +- 0.5 | 42.9 +- 0.2 | 30.0 | 24.1 +- 0.7 (pause as thought; the snippet-era 16.4 was wrong) | 34.1 +- 1.5 | 25.0 vs 8.2 |
| ProntoQA | 93.8 +- 0.7 | 98.8 +- 0.8 | 99.8 | 100.0 +- 0.1 | 99.8 +- 0.2 | 92.5 vs 9.0 |
| ProsQA | 76.7 | 77.5 | n/a | 96.6 +- 0.8 | 97.0 +- 0.3 | 49.4 vs 14.2 |

Ablations reported for ProsQA (snippet): Coconut without the curriculum is about No-CoT; "pause as thought" with the
same curriculum reaches 96.6 +- 0.8 (the paper reports this itself); mixing k latents then CoT improves with k.
Two claims the paper makes that later work contests: that the latent thoughts hold a superposition of search
frontiers (a breadth-first search), and that the latent content, rather than the curriculum, carries the win.

Compute as published: four A100-80GB per run. Outside reproductions report about 36 GPU-h for GPT-2 ProsQA Coconut and
83 GPU-h for GSM (Jin et al. 2026, snippet). Our arithmetic (`PLAN.md` section 5) says 2.5 to 3 A100-h per ProsQA run
with a vectorized forward and batched generation; the official code generates one example at a time and rebuilds the
embedding tensor with Python loops per latent step.

Label: KNOWN. The reproduction is Phase 1's oracle, not a contribution.

## 2. Critiques and mechanistic studies (2025-2026)

| # | Work | What it did | Finding that matters here | Provenance |
|---|---|---|---|---|
| 2.1 | "The Curriculum Is the Mechanism: Dissecting COCONUT's Latent Thought Gains on ProsQA" (2026; checkpoints bmarti44/coconut-curriculum-checkpoints) | retrained Coconut and a pause-token control under the same 7-stage curriculum | pause 96.6 vs Coconut 97.0, McNemar p = 0.845; the curriculum, not the continuous content, carries the ProsQA gain | snippet |
| 2.2 | Aswal, Palmeira Ferraz, Zhou, Peyrard. "Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models", arXiv 2606.12689 (June 2026) | independent ProsQA retraining of Coconut and CODI with controls lacking recurrence or curriculum; interventions; geometry | BFS-like frontiers and decodable arithmetic also appear in the controls; matched pause-token and thought-ablation controls show the advantage does not isolate continuous states; use of a latent thought is graded and concentrated in a few low-rank directions; decodability, attention or static structure alone establish nothing | snippet. This is the "parallel search also in controls" study. |
| 2.3 | Rizvi-Martel, Rabusseau, Mosbach. "The Illusion of Superposition? A Principled Analysis of Latent Thinking in Language Models", arXiv 2604.06374, COLM 2026 | logit lens and belief tracking on off-the-shelf LLMs, fine-tuned Coconut, from-scratch Coconut | LLMs resolve superposed inputs within the first few layers; fine-tuned Coconut takes the answer from the question representation and reaches similar accuracy without using its latents; only from-scratch Coconut shows superposition | snippet |
| 2.4 | Zhang et al. (SJTU). "Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought", arXiv 2512.21711 (Dec 2025) | steering and swapping latent tokens; shortcut tests on MMLU, HotpotQA | perturbation success for Coconut tokens under 5-10% (near 0 in some models) vs up to 50% for CoT tokens; Coconut exploits dataset artefacts | snippet |
| 2.5 | Dilgren, Wiegreffe. "Are Latent Reasoning Models Easily Interpretable?", arXiv 2604.04902 (2026; released GPT-2 ProsQA/ProntoQA Coconut checkpoints connordilgren/*) | forward-chaining decoding of latents | latents often unnecessary: on logical datasets the same answers come out with no latents; a trace is recoverable in 93% of correct ProsQA instances under the loosest threshold and less often when wrong; latents matter more on math | snippet |
| 2.6 | Jin, Yang, Wang. "Final Checkpoints Are Not Enough: Analyzing Latent Reasoning Faithfulness Along Training Trajectories", arXiv 2607.06648 (July 2026) | counterfactual responsiveness and activation interventions along training | GPT-2 Coconut accuracy 73.1 to 90.1 while counterfactual responsiveness fell 71.3 to 27.4; faithfulness and accuracy diverge; final-checkpoint evaluation hides it | snippet |
| 2.7 | Liang, Pan. "Do Latent-CoT Models Think Step-by-Step? A Mechanistic Study on Sequential Reasoning Tasks", arXiv 2602.00449 | CODI on polynomial-iteration tasks with known intermediate states; logit lens, probes, patching | bridge states form for low hop counts; as depth grows the model stops sustaining a full latent rollout and compresses or loses the latent signature | snippet |
| 2.8 | Li et al. (HITSZ). "Dynamics Within Latent Chain-of-Thought: An Empirical Study of Causal Structure", arXiv 2602.08783, ICML 2026 | step-wise do-interventions on Coconut and CODI (GPT-2, Llama-1B, Qwen3-4B) | latent steps behave as staged functionality with non-local routing, not uniform extra depth; early output bias vs later commitment | snippet |
| 2.9 | "LLMs are Single-threaded Reasoners: Demystifying the Working Mechanism of Soft Thinking", arXiv 2508.03440, ICLR 2026 | soft-thinking analysis | the model follows the argmax token of the soft input (the "greedy pitfall"); vanilla soft thinking underperforms discrete sampling; Gumbel-softmax stochasticity helps | snippet |
| 2.10 | Backour. "The Dynamics of Continuous Mixture Collapse in Language Models", arXiv 2609.02049 (Sept 2026) | theory and measurement of mixture preservation | transformers distort mixture geometry and training amplifies it; softmax readout plus autoregressive feedback either amplifies one component or contracts mixtures to indistinguishability; pretrained rollouts sit on the amplifying side | snippet |
| 2.11 | Zou, Xiong, Liu. "Capabilities and Fundamental Limits of Latent Chain-of-Thought", arXiv 2602.01148 (2026) | theory plus GPT-2 experiments | an exploration-execution trade-off indexed by decisional certainty; curriculum is necessary because direct training fails from distribution mismatch; reproduces ProsQA 97.0 vs GSM 34.1 | snippet |
| 2.12 | "Understanding Latent Reasoning in Large Language Models", arXiv 2602.22441 (2026) | zero-latent-step ablations across methods grouped by supervision strength | a model that keeps its accuracy at zero latent steps is using shortcuts | snippet |

Lab reading: items 2.1 to 2.6 are the strongest reason the judge in `PLAN.md` section 4 requires the pause-same-
curriculum control, the scrambled-latent control, the zero-latent and step-sweep ablations, and an intervention-based
faithfulness score. They are also the reason Phase 2 card C3 (latent-dependence loss) exists.

## 3. Methods that improved on Coconut

| # | Work | Mechanism | Numbers | Family / what it tells us | Provenance |
|---|---|---|---|---|---|
| 3.1 | Shen et al. "CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation", arXiv 2502.21074, EMNLP 2025; code zhenyi4/codi; weights zen-E/CODI-gpt2 | one-stage self-distillation: a teacher pass on explicit CoT and a student pass with latents share weights; the student's hidden state at the answer-cue position is regressed to the teacher's | GPT-2 GSM8k-Aug about 43.7 vs CoT 43.9 (matches explicit CoT, 3.1x compression); "surpasses Coconut by over 20%" on GPT-2 and Llama-1B; about 5 GPU-h | supervision of the latent beats outcome-only; the single stage removes Coconut's curriculum cost | snippet (numbers), GitHub (code, args) |
| 3.2 | Wei et al. "SIM-CoT: Supervised Implicit Chain-of-Thought", arXiv 2509.20317, ICLR 2026; code InternLM/SIM-CoT | auxiliary decoder aligns each latent token with its explicit step during training; removed at inference | +8.2 over Coconut on GPT-2, +2.1 over explicit CoT at 2.3x token efficiency (snippet; one source reads the CODI gain as +4.3, another as +0.6, so verify); Coconut collapses from 3 to 5 latents (about 45 to 12.5); stable at 8 to 16 latents | step-level supervision cures the latent collapse | snippet |
| 3.3 | Kuzina et al. (Qualcomm). "KaVa: Latent Reasoning via Compressed KV-Cache Distillation", arXiv 2510.02312, ICLR 2026 | the student's latents are aligned to a compressed KV cache of a teacher that reasons in text | Qwen2.5-0.5B: CoT 50.6, CODI 37.5, KaVa 46.9; Llama-1B: 61.6 / 55.6 / 56.5; Llama-3B: 73.2 / 61.0 / 65.7; robust to natural-language traces (GSM-Aug-NL 44.4 vs CODI 20.2) | below full CoT at every size | snippet |
| 3.4 | Fan, Svete, Lee. "Bridging the Gap Between Latent and Explicit Reasoning with Looped Transformers" (LOTUS), arXiv 2606.31779 (June 2026); code yingfan-bot/lotus | looped padded transformer: K = 6 blocks of c = 25 padded slots (13 on GPT-2), R = 6 loops, each latent position supervised on a gold CoT-step token; latent blocks run in parallel | states that prior latent methods trail explicit CoT beyond 1B and the gap widens; LOTUS on Llama-3.2-3B 70.05 (70.58 with CODI) vs explicit CoT 71.5; thought-phase latency down 2.5x to 6.9x; compares against Coconut, CODI, SIM-CoT, PCCoT, KaVa | wide per-step state plus per-position supervision is what closes the gap at 3B; GitHub repo publishes stage-1 CoT checkpoints for GPT-2, Llama-1B and 3B | snippet (numbers), GitHub (README) |
| 3.5 | Tan et al. "Think Silently, Think Fast: Dynamic Latent Compression of LLM Reasoning Chains" (CoLaR), arXiv 2505.16552, NeurIPS 2025 | merges consecutive token embeddings by a sampled compression factor; a latent head predicts the next compressed embedding; speed is promptable; GRPO stage | reasoning chain 53.3% shorter for a 4.8-point drop vs CoT; +14.1 over baselines at comparable compression; RL on MATH +5.36 at 82.8% shorter | compression trades accuracy for length; the Pareto curve is the result | snippet |
| 3.6 | Teng et al. "Parallel Continuous Chain-of-Thought with Jacobi Iteration" (PCCoT), arXiv 2506.18582, EMNLP 2025 | all latent slots updated in parallel by Jacobi iterations instead of sequentially | GPT-2 small LoRA on GSM8k-Aug, 3 runs: comparable or better accuracy at about half the training and inference time; more iterations do not always help | parallel latents are cheaper per step; motivates card C1 | snippet |
| 3.7 | Suleymanzade et al. "MUX: Continuous Reasoning via Multiplexed Tokens", arXiv 2607.18264 (NeurIPS 2026) | each latent is a weighted superposition of a span of discrete reasoning subwords; geometric-decay weights give lossless multiplexing | outperforms latent baselines across 32 settings on four LMs; argues globally supervised models can score high without using the latents | local lossless targets prevent latent collapse | snippet |
| 3.8 | Cheng, Van Durme. "Compressed Chain of Thought" (CCoT), arXiv 2412.13171; LightThinker (EMNLP 2025); SoftCoT and SoftCoT++ (Xu et al. 2025); Soft Thinking (2025); "Soft Tokens, Hard Truths" (2025) | dense or soft intermediate representations, mostly with frozen backbones or soft-token mixtures | gains are modest and inconsistent; soft tokens follow the argmax (2.9) | COMBINATION family; not adopted | snippet (titles and claims from the Awesome-Latent-CoT index) |

## 4. Reinforcement learning on latents and hybrid routing

| # | Work | Finding | Provenance |
|---|---|---|---|
| 4.1 | Deng et al. "Latent-GRPO: Group Relative Policy Optimization for Latent Reasoning", arXiv 2604.27998 (AAAI 2026) | Llama-3.2-1B-Instruct on GSM8k-Aug: latent SFT 47.55 to 66.29 pass@1 with 22 to 24 tokens; explicit GRPO 62.26 with 111 tokens; Soft-GRPO 48.34; standard GRPO on the latent SFT model collapses; +4.27 over explicit GRPO on hard tasks with 3-4x shorter chains | snippet |
| 4.2 | Jung. "Dropout-GRPO: Variational Stochasticity for Continuous Latent Reasoning", arXiv 2606.10184 | one Bernoulli dropout mask per rollout makes GRPO well defined over deterministic latents; Coconut GSM8K 27.29 to 29.01 | snippet |
| 4.3 | Yang et al. "Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy RL" (SWITCH), arXiv 2606.13106 | discrete boundary tokens make the policy ratio well defined; the latent computation concentrates at a single transition on entry | snippet |
| 4.4 | Ozeren, Assenmacher. "Reinforcement Learning for Latent-Space Thinking in LLMs", arXiv 2512.11816 (Nov 2025) | Coconut is highly sensitive to design choices; RL-trained latent models still lag language-space CoT on math | snippet |
| 4.5 | TARPO (arXiv 2606.05859), HyRea ("Learning to Reason over Continuous Tokens with RL", ICLR 2026), SwiReasoning (arXiv 2510.05069), SpiralThinker, "Learning When to Stop: Adaptive Latent Reasoning via RL" (2025), Tyler (2026), LEPO, SofT-GRPO, RLTT ("Prioritize the Process", arXiv 2602.10520) | token-wise or turn-wise switching between latent and explicit modes, learned halting, trajectory rewards for looped models; all report token reductions at held or improved accuracy on math benchmarks with Qwen2.5 1.5B to 8B | snippet |
| 4.6 | Wang et al. "Towards Inference-time Scaling for Continuous Space Reasoning", arXiv 2510.12167, ACL Findings 2026; GTS (arXiv 2602.14077) | Coconut's latent phase is deterministic so sampling yields one path; dropout in the latent phase gives pass@N gains comparable to discrete methods; PRM/ORM reranking gives only marginal gains in continuous space | snippet |

Lab reading: RL on latents works only when the policy ratio is defined on something discrete or stochastic; it is a
Phase 2 stretch (card C6), not a headline. Hybrid routing is a KNOWN family (card C5) and is reported as a Pareto
curve.

## 5. Recurrent depth and looped models

| # | Work | Finding | Provenance |
|---|---|---|---|
| 5.1 | Saunshi et al. "Reasoning with Latent Thoughts: On the Power of Looped Transformers", ICLR 2025, arXiv 2502.17416 | k layers looped L times nearly match a kL-layer model on addition, p-hop induction and math, and beat the k-layer model; T loops simulate T CoT steps; scaling follows effective depth; inductive bias toward reasoning even at worse perplexity | snippet |
| 5.2 | Geiping et al. "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach" (Huginn), arXiv 2502.05171, NeurIPS 2025 | 3.5B prelude/core/coda model trained on 800B tokens; accuracy rises with test-time recurrence up to a 50B-equivalent load; GSM8K keeps using more recurrence than easier tasks | snippet |
| 5.3 | Zhu et al. "Scaling Latent Reasoning via Looped Language Models" (Ouro), arXiv 2510.25741 | 1.4B and 2.6B looped models pretrained on 7.7T tokens with entropy-regularised depth allocation; with 4 recurrent steps they match 4B and 8B baselines; the gain is knowledge manipulation, not storage | snippet |
| 5.4 | Gao et al. "Loop the Loopies!", arXiv 2607.16051 (July 2026) | looped MoE at 20B-A2B and 6B-A0.6B; beats compute-matched vanilla baselines in ablations; olympiad claims unverified | snippet |
| 5.5 | Goyal et al. "Think before you speak: Training Language Models With Pause Tokens", arXiv 2310.02226, ICLR 2024 | gains on 8 of 9 tasks only when pauses are used in pretraining and fine-tuning; fine-tuning-only gains are mild and sometimes negative | snippet |
| 5.6 | This repo, data-wall rung 0 (2026-10-12) | 12L looped x2 vs plain 24L at equal FLOPs on BabyLM 10M, 16 epochs, from scratch, LM loss: looped worse by 0.038 nats, also worse than the 12L control; "Sharing weights across depth costs more than the extra passes buy" | repo |
| 5.7 | Also indexed: LoopFormer (elastic depth), Mixture-of-Recursions, Hierarchical Reasoning Model, Tiny Recursive Models, PonderLM, "Teaching Pretrained Language Models to Think Deeper with Retrofitted Recurrence", "Memory-Efficient Looped Transformer", "Fixed-Point Reasoners" | the family is large and active; retrofitted recurrence on a pretrained model is the known-family label for card C1b | snippet (index) |

## 6. Formal results

| # | Work | Result | Consequence for the design | Provenance |
|---|---|---|---|---|
| 6.1 | Zhu, Hao, Hu, Jiao, Russell, Tian. "Reasoning by Superposition: A Theoretical Perspective on Chain of Continuous Thought", arXiv 2505.12514, NeurIPS 2025 | a two-layer transformer with D continuous thoughts solves directed-graph reachability where D is the diameter; constant-depth discrete CoT needs O(n^2) steps; the superposition of frontiers emerges in training without supervision | the parallel-search advantage is real for reachability; the positive control for our readout is a tiny from-scratch model on synthetic reachability | snippet |
| 6.2 | Xu, Sato. "A Formal Comparison Between Chain of Thought and Latent Thought", arXiv 2509.25239 (v2 Jan 2026), ICML 2026 | latent thought admits parallel computation that sequential CoT cannot; CoT wins on approximate counting and sampling via stochastic decoding; looped transformers are not universally more expressive than CoT | pick tasks on both sides: reachability (parallel) and arithmetic chains (serial) | snippet |
| 6.3 | Merrill, Sabharwal (2025) and "To CoT or To Loop?" (arXiv 2505.19245) | separation favouring looped transformers in the logarithmic regime; depth that grows with input adds power | log-depth loops matter; a constant number of latents does not | snippet |
| 6.4 | "Chain-of-Thought and Compressed Looped Transformers: A Memory-Budget Separation", arXiv 2605.30757 (May 2026) | a compressed loop (fixed-size recurrent state) cannot decide P-complete problems under logspace reductions under a standard assumption; polynomial CoT can; loops that keep state at every position are closer to scratchpads; pointer-chasing and associative-recall sweeps show the crossover at the working-memory demand | a single thought vector per step is a compressed loop; width must exceed the task's live state (card C1, theory memo T2) | snippet |
| 6.5 | Chen et al. "What Makes Effective Supervision in Latent Chain-of-Thought: An Information-Theoretic Analysis", arXiv 2606.20075, ICML 2026 | outcome supervision gives "dual collapse" (gradient attenuation, representational drift); reconstruction objectives maximise a lower bound on I(latent; step); a Unified Latent Probe measures it | the C2 decoder is a reconstruction objective on executable states; the probe doubles as the readout | snippet |
| 6.6 | Zou et al. (2.11) | curriculum provably necessary; exploration vs execution | keep the curriculum; measure the "symbolic index" if cheap | snippet |

## 7. Faithfulness and readouts

| # | Work | Method and finding | Provenance |
|---|---|---|---|
| 7.1 | Ramjee. "Ulterior Motives: Detecting Misaligned Reasoning in Continuous Thought Models", arXiv 2604.23460 (2026) | MoralChain benchmark (12k scenarios); a continuous-thought model with a dual-trigger backdoor reasons in a misaligned way in latent space while outputs stay aligned; linear probes separate the states; misalignment shows in early latent tokens | snippet |
| 7.2 | Aswal et al. (2.2), Jin et al. (2.6), Dilgren and Wiegreffe (2.5), Li et al. (2.8) | the field's own conclusion: decodability is not evidence; use matched controls and causal interventions; track along training | snippet |
| 7.3 | Chen et al. (6.5) | mutual-information probe between latent trajectory and explicit steps | snippet |

Lab reading: our readout (`PLAN.md` 6.5) reports decodability (R1-R3) and causal load-bearing fraction (F1-F3)
separately, tracked along training.

## 8. Code reasoning assets

| # | Asset | Notes | Provenance |
|---|---|---|---|
| 8.1 | CRUXEval (Gu et al., ICML 2024; facebookresearch/cruxeval) | 800 Python functions of 3 to 13 lines with input-output pairs; CRUXEval-O is output prediction; GPT-4 with CoT about 81% output, Code Llama 34B 46%; fine-tuning plateaus under 70% | GitHub (README), snippet (numbers) |
| 8.2 | CodeI/O (Li et al., ICML 2025 oral, arXiv 2502.07316) | input-output prediction in natural language as a reasoning-pattern source; code hkust-nlp/CodeIO (README not reachable from here) | snippet |
| 8.3 | CoLaR-code checkpoints (rjz123/colar-code-cruxreal-l1b) | latent reasoning for code output prediction on a three-tier ladder built from CRUXEval, LiveCodeBench, MBPP; no published numbers found | snippet |
| 8.4 | This repo | bounded Python interpreter (`experiments/2026-09-22-next-acquisition-repair/code/hinton_next/execution.py`), program generator with ground truth (`experiments/2026-09-21-c0-program-canvas/code/hinton/{tasks,sandbox}.py`), 2,423 real executions (private release) | repo |

## 9. Pretraining-time variants (out of scope, recorded)

CoCoMix ("LLM Pretraining with Continuous Concepts", 2025), "Pretraining LLM with Latent Thoughts in Continuous Space",
"Pretraining with Token-Level Adaptive Latent CoT", PonderLM, Ouro. All require pretraining budgets far beyond this
project; recorded so Phase 3 can say what it does not test.

## 10. What is open (restated for the registration)

1. Compute-matched, token-matched, 3-seed comparison of a latent method against a *tuned* CoT of the same base, with
   scrambled-latent, pause-same-curriculum, no-curriculum and plain-depth controls. Not in print.
2. The working-memory hypothesis: accuracy vs per-step latent width at matched FLOPs, with the crossover predicted from
   the task's live state. Not in print.
3. Training for causal use of latents (a latent-dependence objective). Not found.
4. The scaling slope of the matched-FLOPs gap across 124M, 355M, 774M with seeds. Not in print; the one statement is
   negative.
5. Latent reasoning on program-output prediction with executable intermediate states as supervision and as readout
   targets. Not in print.

## 11. Index of everything else screened

From the EIT-NLP/Awesome-Latent-CoT index (289 arXiv links, read as titles only): the continuous-token family (about 90
papers), the looped/recursive family (about 55), implicit-CoT distillation (iCoT, "Distilling System 2 into System 1",
LoRi) and the analysis section. Titles are listed in the index itself; nothing there changes the gap statement above.
