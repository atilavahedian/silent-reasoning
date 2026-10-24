# Silent reasoning: rules of the experiment (pre-registration), Phase 1 and the standing judge

Critic (Fable 5.1, separate agent; writes no model code), written **Sat 17 Oct 2026, 20:34 UTC**, before any run of
this project: no smoke run, no training and no calibration had happened when this file was written. When the rules
were fixed (20:34 UTC) the experiment folder held only PLAN.md and `common/latent/` held INTERFACE.md and a fake
tokenizer; by the time the file was saved the operator had added untracked code, configs, a data manifest for ProsQA and
GSM8k-Aug and a CHANGELOG, none of it run. The critic read that code only after sections 0-14 were written; the
working-tree check is amendment A1. Sections are final once written;
the signature line closes the registration. **This file may only be tightened, never loosened.** Changes are dated
below the line and do not apply to numbers already seen. The author (Atila) outranks this file: a author-directed
loosening is recorded below the line as such, with what had been seen when it arrived, and the original verdict is
still computed and printed next to the new one. Verdicts are write-once with the code commit that computed them.

The plan is `PLAN.md` (approved 2026-10-17). Where the plan and this file
differ, this file binds for anything that decides a verdict. **Author's brief, verbatim from PLAN.md section 0:**

> Hinton wants models that reason in their hidden states instead of writing chain of thought (CoT). The bar: match or
> beat written CoT on hard verifiable tasks with at least 5x fewer output tokens at equal total compute, an advantage
> that holds or grows with size, and a readout of the silent steps with measured faithfulness. "Silent reasoning does
> not beat written reasoning" is an acceptable answer; a fake win is not.

**Scope labels (binding).**

- **Phase 1 = reproduction oracle.** The published Coconut recipe (Hao et al., facebookresearch/coconut) on GPT-2
  124M in fp32, effective batch 128 via gradient accumulation, plus the tuned-CoT bar. Novelty label for every Phase 1
  number: **KNOWN (reproduction)**. The seeded, compute-accounted table is a new measurement of a known method; it is
  never presented as a method. No Phase 1 number can be a WIN; at most WIN-KNOWN.
- **The standing judge (sections 7 to 10)** applies to Phases 2 to 4. Each Phase 2 card gets its own registration row
  in `critic_log.md` (section 7.11) before its screen; a screen run before that row is a label, never a screen.
- **Snippet rule.** Every published number quoted here comes from PLAN.md section 1, which took them from search
  snippets and GitHub, not PDFs. Each is marked "(snippet; verify from PDF before the read)". The PDF-verified values
  replace them at the manifest commit (section 1.4), in a critic-log row, before any oracle run starts. After the first
  oracle run starts, the targets are frozen.

## 0. Plumbing permissions (in force now, until the data manifest is signed)

Allowed before the manifest signature (section 1.4) and the first oracle run:

1. Downloads of exactly the items of section 1, each with its SHA-256 written to `configs/data_manifest.json`.
2. Smoke runs capped at **2,000 training examples** and **200 optimizer steps** per configuration, scored only on at
   most **100 validation examples**, for pipeline checks, throughput and memory. The plan's day-1 smoke (ProsQA 2k
   subset, 1 seed, 2 arms, Mac) fits this cap. Smoke numbers are throughput, memory and "the pipeline runs"; two smoke
   arms are never compared on accuracy.
3. The Mac calibration of plan section 6.1 (forward+backward at seq 320, batch 16, fp32 and fp16 autocast on MPS;
   one-batch Coconut forward parity MPS vs CPU float64 on real GPT-2 weights). It trains nothing.
4. CPU unit tests under `common/latent/tests/` (random tiny configs, the fake tokenizer).
5. The untrained base model's generation accuracy on at most 100 validation examples per task (a chance-floor sanity
   number; recorded).
6. Reading smoke generations by eye to fix stop, length and extraction compliance against the reference code.

Forbidden before the signature, and afterwards except at the registered points:

1. **Scoring any checkpoint on any test split**, by anyone, for any reason, before the registered read (section 4.5).
   The test files are downloaded and hashed; they are opened only by `eval.py` at a registered read.
2. Any prompt, format or recipe change after seeing an accuracy number (smoke, validation or otherwise). A fix for a
   pipeline bug found by the parity tests is allowed and is logged in CHANGELOG.md with the test that found it.
3. Any training run on more than 2,000 examples or more than 200 steps.
4. Any arm comparison on accuracy, including smoke arms.
5. Any run on GCP, Kaggle or rented hardware (section 12: blocked until the author's typed yes).

## 1. Models, data, splits and access

### 1.1 Model

- `openai-community/gpt2` (GPT-2 124M: n_layer 12, d 768, n_head 12, head_dim 64, n_inner 3072, V 50,257). The HF
  revision is pinned once resolved on the Mac (arxiv and HF are blocked from the operator container) and frozen at the
  manifest commit, before any training run. Every run's config records the revision.
- Special tokens, added in this order exactly as the reference does: `<|start-latent|>`, `<|end-latent|>`,
  `<|latent|>`; embeddings resized to 50,260; the three new rows (input embedding and the tied head) are initialised
  **from the embedding of the token `<<`** (reference `run.py`: `target_id = tokenizer.convert_tokens_to_ids("<<")`).
  INTERFACE.md says "from the eos token"; the reference wins (section 14, D1).
- The reference resizes embeddings only for the latent arms (`coconut`), not for `cot` or `no_cot`. If the operator
  resizes for every arm, the operator shows on the smoke set that a resized and an unresized `cot` model agree in
  validation loss to 1e-4 and logs the deviation in CHANGELOG.md; otherwise `cot` and `no_cot` are not resized.
- Phase 3 sizes: `openai-community/gpt2-medium` (355M) and `gpt2-large` (774M), same pinning rule. Non-embedding
  parameter counts the accounting test must reproduce from parameter shapes: **85,056,000 / 302,311,424 /
  708,390,400** (everything except `wte` and `wpe`; the tied head is charged separately as V d).

### 1.2 Data (all under `data/`, gitignored; every file hashed in `configs/data_manifest.json`)

| task | source | train / valid / test | notes |
|---|---|---|---|
| ProsQA | `data/prosqa_{train,valid,test}.json` from facebookresearch/coconut at a commit SHA frozen at the manifest commit | 17,886 / 300 / 500 | ships in the repo; about 180-word questions with the graph edges included; 3 reasoning steps |
| ProntoQA | regenerated with asaparov/prontoqa at commit `f0145b867b3c106285ec9ea1941a3f6eb7c6162d`, the exact README command `python run_experiment.py --model-name json --model-size dummy --ordering random --num-trials 10000 --few-shot-examples 0 --ontology fictional --min-hops 5 --max-hops 5 --hops-skip 1`, converted with the reference `preprocessing/prontoqa.py` (`data[:9000]`, `[9000:9200]`, `[9200:]`) | 9,000 / 200 / 800 | the generator uses `--ordering random` with no seed in the command, so the regenerated set is **not byte-identical to the paper's**; the Python version, the generator's seed if it exposes one, and the raw `5hop_0shot_random.json` hash are recorded; the generator is run twice and both hashes recorded; the first is kept. Every ProntoQA oracle cell carries the label "regenerated data" |
| GSM8k-Aug | `train.txt`, `valid.txt`, `test.txt` from da03/Internalize_CoT_Step_by_Step at commit `e06a32ee5e4cd117171daeb4755d2a97ece62761` (`data/gsm8k/`), converted with the reference `preprocessing/gsm_icot.py` semantics (question = text before `||`; steps = text between `||` and `##`, split on spaces; answer = text after the last `##`, stripped) | about 385k / (count recorded in the manifest) / 1,319 | test = the GSM8K test set |

- Tokenisation is exactly the reference `tokenize_sample`: question + `"\n"` (`add_special_tokens=True`), each step +
  `"\n"`, `"### "` + answer + `[eos]`. The reference's own assertion (the concatenation equals tokenising the complete
  string) is kept as a test on record 0 of every file.
- `max_size` truncation is never used outside smoke runs.

### 1.3 Splits and reads

- **Selection split = valid. Scoring split = test.** Every decision (checkpoint choice, LR choice, kill lines at
  screens, debugging) uses valid. Test is scored **once per registered read per checkpoint** (section 4.5), by
  `eval.py`, writing a per-example file whose SHA-256 is logged. A second test read of the same checkpoint happens
  only under the audit of section 7.10 (rescoring), and that read is compared to the first, never substituted for it.
- **13-gram overlap** (whitespace-token 13-grams of question + steps + answer) of train against test and of train
  against valid is computed by `common/latent/contamination.py` and reported, per task, **before the read**. ProsQA and
  ProntoQA are generated data with shared vocabulary and templates, so some overlap is expected; the report gives the
  rate, the count of test examples sharing a 13-gram with any train example, and the rate for the GSM8k-Aug train
  (385k augmented) against the GSM8K test. A GSM8k-Aug rate above 1% of test examples blocks the GSM read until
  explained (the augmented set must not contain test problems).
- No other project's protected data is read. No test example is ever printed in a log, a report or a commit.

### 1.4 The data manifest and its signature

- `configs/data_manifest.json` lists, for every file of section 1.2 and for the model weights: path, SHA-256, size,
  source URL or command, source commit, the local date, and the record count. It also records the Coconut repo commit,
  the HF revision, torch and transformers versions, and the Python version used for ProntoQA generation.
- The manifest is **committed before any training run** that is not a smoke run. Its commit SHA is logged in
  `critic_log.md` by the operator; the critic's acceptance row is the **signature** that ends section 0's regime.
- Anything built before the signature is kept only if its hash equals a rebuild under the registered rule.

## 2. Metric and cost axes

### 2.1 Primary metric

**Exact-match accuracy** on the test split, in points (0 to 100), with the reference extraction and normalisation:

- prediction: `tokenizer.decode(prompt_ids + generated_ids, skip_special_tokens=True).split("#")[-1].replace(",", "").strip()`
  (the reference decodes the whole sequence including the prompt; an output with no `#` therefore yields a tail of the
  question and scores wrong, exactly as in the reference);
- gold: `answer.replace(",", "").strip()`;
- correct iff the two strings are equal. No numeric parsing, no case folding, no format shopping.

Generation: batched greedy decoding, per-example stop at eos, **`max_new_tokens` = 64 for GSM8k-Aug and 128 for ProsQA
and ProntoQA** (reference `run.py`: `64 if "gsm" in val_path else 128`; the brief given to the critic had these
inverted; the reference code binds). The reference generates at batch size 1 without a KV cache; ours is batched with a
cache and must be token-identical (section 14, D3).

### 2.2 Cost axes, per example (every eval writes all of them per example)

- `emitted_tokens`: generated vocabulary tokens, **excluding** the terminating eos and excluding latent positions and
  the two marker tokens (which are prompt positions). The paper-style count `emitted_tokens + latent_steps` is also
  reported, because the published token counts appear to include the continuous thoughts (ProsQA 14.2 for Coconut
  with 6 thoughts and a short answer) (snippet; verify from PDF before the read).
- `latent_steps`: sequential single-position passes whose output is fed back (Coconut-style). `latent_positions`: the
  number of non-vocabulary positions in the sequence (equals `latent_steps` for Coconut, L for pause, 0 for cot).
- `sequential_passes` = 1 + `latent_steps` + `emitted_tokens` (the prefill counts one; the pass that produces the eos
  is the last emitted token's pass). Pause tokens add no pass.
- `flops`: by section 5's formulas and the position table of section 5.3.
- `wall_ms`: wall-clock of the example's generation at batch size 1 on one named device, median of 3 repeats on a
  fixed subset of 200 test examples (seeded, `default_rng(2)`), reported with device, precision, torch version; never
  compared across devices.
- Training FLOPs per run (section 5.2) and the per-epoch validation curve.

## 3. Phase 1 arms, recipe and the tuned-CoT bar

### 3.1 Arms (per task, seeds 0, 1, 2; the same seeds in every arm; all arms start from the same base weights)

| arm | construction (section 1.2 tokenisation, reference `dataset.py` semantics) | Why |
|---|---|---|
| `no_cot` | question + `### answer` + eos; loss on the answer tokens; no markers | the floor: what the question alone gives. Published ProsQA No-CoT 76.7 vs CoT 77.5 (snippet; verify from PDF before the read) says this floor is close to CoT on ProsQA |
| `cot` | question + all steps + `### answer` + eos; loss on steps and answer; no markers; published recipe | the published baseline the oracle must match; it is **not** the bar for Phase 2 (that is the tuned bar, 3.4) |
| `coconut` | staged curriculum: at stage s the first k = min(s, len(steps)) steps are replaced by `<|start-latent|>` + n_latent x `<|latent|>` + `<|end-latent|>`, n_latent = c_thought x min(s, max_latent_stage) with `pad_latent_to_max` (so n_latent is padded to the stage count when the example has fewer steps); continuous feedback (last-layer hidden state at position j-1 becomes the input embedding at latent position j); labels -100 on question, markers and latents | the method under reproduction |
| `pause` | identical curriculum, identical token layout and token budget as `coconut` at every stage, **no feedback**: latent positions keep the learned `<|latent|>` embedding; one forward pass; same optimizer reset schedule as `coconut` | the published same-curriculum control (96.6 vs 97.0 on ProsQA (snippet; verify from PDF before the read)). It separates the curriculum from the latent content. Note the paper also has a "pause token" row without the curriculum; our arm is the same-curriculum one |
| `coconut_no_curriculum` | `coconut` with the stage fixed at `max_latent_stage` from epoch 0 (no step token is ever a target), same epochs | the curriculum-necessity control (published: about No-CoT (snippet; verify from PDF before the read)) |

- GSM8k-Aug runs `cot`, `coconut` (initialised from the `cot` run's selected checkpoint, as published) and `no_cot`,
  **seed 0 only, labelled "single seed"**, and only after the author's typed yes (section 12).
- `uniform_prob` = 0 in every arm (every reference yaml).

### 3.2 Recipe, per task, from the reference yamls (binding for the oracle; any difference is a deviation logged in CHANGELOG.md before the run)

| setting | ProsQA (`prosqa_coconut.yaml`) | ProntoQA (`prontoqa_coconut.yaml`) | GSM8k-Aug cot (`gsm_cot.yaml`) | GSM8k-Aug coconut (`gsm_coconut.yaml`) |
|---|---|---|---|---|
| optimizer | AdamW, lr 1e-4, weight decay 0.01, torch defaults otherwise (betas 0.9/0.999, eps 1e-8), **no LR schedule, no warmup, no gradient clipping** (none in `run.py`) | same | same | same |
| effective batch | 128 (reference: 32 per GPU x 4 GPUs); ours: micro-batch x grad_accum = 128, micro-batch recorded; loss = mean over micro-batches of the per-micro-batch token-mean CE | 128 | 128 | 128 |
| epochs | 50 | 50 | 25 | epochs 3 to 24 (resume 3 from the cot checkpoint; 22 epochs) |
| c_thought | 1 | 1 | 0 | 2 |
| epochs_per_stage | 5 | 5 | n/a | 3 |
| max_latent_stage | 6 | 6 | 0 | 3 |
| pad_latent_to_max | True | True | True | True |
| stage at epoch e | min(e // 5, 6): stages 0..6; epochs 30-49 are stage 6 (6 latents) | same | 0 | min(e // 3, 3): epochs 9-24 are stage 3 (6 latents) |
| optimizer reset | **every epoch** (reference `run.py` rebuilds AdamW inside the epoch loop whenever `reset_optimizer` is True; it is not "on stage change") | every epoch | none (`reset_optimizer: False`) | every epoch |
| precision | fp32 end to end (published) | fp32 | fp32 | fp32 |
| checkpoints | every epoch | every epoch | on validation improvement (`save_only_improve: True`) | every epoch |
| validation | generation accuracy on valid every epoch, with the current stage's latent count | same | same | same |
| max_new_tokens | 128 | 128 | 64 | 64 |

- `cot` and `no_cot` on ProsQA and ProntoQA: the reference repo ships no yaml for them. Registered: the same
  optimizer, batch, 50 epochs, no optimizer reset (as `gsm_cot.yaml`), `save_only_improve` semantics, best-valid
  checkpoint. The PDF appendix is checked for the baselines' epoch count at the manifest commit; if it differs, the
  PDF value is registered by a critic-log row before the first run.
- Seeds set torch, numpy and random; data order is seeded per epoch. The reference's exact order (DistributedSampler
  over 4 shards plus a global shuffle) cannot be reproduced; this is a known, unavoidable deviation and is logged once.
- Effective batch: 128 by accumulation is a registered deviation from 4 x 32 with DDP gradient averaging (the loss
  weighting differs only when the micro-batch is not 32). Logged once in CHANGELOG.md with the micro-batch used.

### 3.3 Checkpoint rule (selection on valid)

1. The selected checkpoint of a run is the one with the **highest validation generation accuracy among the
   checkpoints of the final stage** (ProsQA/ProntoQA latent arms: epochs 30-49, 6 latents; GSM coconut: epochs 9-24,
   6 latents; `cot` and `no_cot`: all epochs). Ties go to the **earliest** epoch (the reference's strict `>`).
2. The valid accuracy of every epoch is logged (`log.jsonl`), so the selection is reproducible from the log.
3. If a checkpoint from an earlier stage has a higher valid accuracy than the selected one, that is reported
   ("earlier stage better on valid") and the earlier checkpoint is scored on test **once**, as a registered secondary
   read labelled "earlier-stage checkpoint, n latents = its stage". It never replaces the primary cell.
4. The test read of a latent arm uses the final stage's latent count (6). The step sweep (section 8) is the only other
   latent count scored on test, and it is registered here.
5. GSM `coconut` is initialised from the GSM `cot` run's selected checkpoint (the README says "around 40%" valid). The
   cot checkpoint used is named by epoch and hash in the coconut run's config.

### 3.4 The tuned-CoT bar (the bar for every Phase 2 comparison; Phase 1 measures it on ProsQA and ProntoQA)

- **Grid (6 trials per task, selection on seed 0, valid accuracy, best-valid checkpoint per trial):**
  lr in {3e-5, 1e-4, 3e-4} x schedule in {constant (the reference), linear decay to 0 at epoch 50}; 50 epochs each;
  AdamW wd 0.01; effective batch 128; per-epoch validation; `cot` data. The constant/1e-4 trial **is** the oracle
  `cot` seed-0 run (shared, not rerun). *Why not "epochs in {25, 50}": with a constant LR and per-epoch best-valid
  selection, a 25-epoch run is the first half of the 50-epoch run with the same seed, so an epochs axis is free and
  decorative. The decay schedule makes the second axis real; the per-epoch selection already covers every shorter
  constant-LR run.*
- **Pick:** the trial with the highest best-valid accuracy on seed 0; ties go to the smaller LR, then to constant.
  If the winner sits at an LR edge (3e-5 or 3e-4), one more trial a factor of 3 beyond it, same schedule, joins the
  grid (once per task; then 7 trials, and trial parity below becomes 7).
- **Bar value:** seeds 1 and 2 are run at the winning setting; each seed selects its own best-valid checkpoint; the bar
  is the 3-seed mean test accuracy (seeds 0, 1, 2). Selection on seed 0 inflates the bar slightly, which works
  against the latent arms: the conservative direction.
- **Trial parity (binding in Phase 2):** any latent arm compared against the tuned bar gets **the same number of
  tuning trials (6, or 7 with the edge trial)** on the same selection split, logged per arm, and no more. A card that
  reports a latent arm tuned on more trials than the bar is UNTESTED against the bar until the bar gets the same
  number. Phase 2 (GPU, bf16 autocast) **re-runs the bar** with this grid on its own hardware; the Mac bar never
  enters a Phase 2 comparison (section 7.8).

### 3.5 Phase 1 novelty and labels

Every Phase 1 cell: KNOWN (reproduction). The tuned-CoT bar: KNOWN (a tuned baseline). The seed spreads, the
per-example cost tables and the cross-environment pair: new measurements, reported as such, never as a method.

## 4. The oracle test (kill rule K1)

### 4.1 Tolerance

A published cell is **matched** iff our 3-seed mean lies within **max(2.0 points, 2 sigma_cell)** of the published
value, where

- sigma_cell = max(1.0, pooled seed SD) in points, pooled over (a) our three seeds of that arm on that task (ddof 1,
  df 2) and (b) the published spread of that cell when the PDF reports one (the paper reports means over 3 runs with a
  spread for its main cells (snippet; verify from PDF before the read); if the PDF gives a spread, it enters the pool
  with df 2; if not, only (a) enters);
- with the floor at 1.0, 2 sigma_cell >= 2.0 always, so the tolerance is 2 sigma_cell and the "2.0 points" clause bites
  only at the floor;
- sigma_cell is frozen at the read; later seeds do not change a K1 verdict (write-once).
- Single-seed cells (GSM8k-Aug): sigma_cell = max(1.0, published spread if reported); tolerance = max(2.0,
  2 sigma_cell); the verdict carries "single seed".

Arithmetic the reader should know: with a 500-example test set the binomial SE alone is 1.9 points at 77% and 0.8
points at 97%; with 800 examples, 0.85 at 94% and 0.35 at 99%; with 1,319 examples, 1.4 at 43%. The published
value is itself a draw (one seed or a 3-seed mean), so the difference between our mean and theirs has SD about
1.15 sigma for a seed-0 publication; a 2 sigma tolerance then has a false-miss rate of roughly 8% per cell under a
perfect reproduction, and about one in two across the ten gating cells. A miss therefore starts a debug round, not a
stop (4.4); the critic reads the pattern across cells, not one cell alone.

### 4.2 Oracle targets (PLAN.md section 1.1; every value: snippet; verify from PDF before the read)

| task | cell | published | gates K1 | strength as an oracle |
|---|---|---|---|---|
| ProsQA | coconut | 97.0, spread 0.3 | yes | strong (the headline cell) |
| ProsQA | pause (same curriculum) | 96.6, spread 0.8 | yes | strong |
| ProsQA | cot | 77.5 | yes | medium (seed spread will be about 2 points) |
| ProsQA | no_cot | 76.7 | yes | medium |
| ProsQA | coconut_no_curriculum | "about No-CoT": matched iff within the tolerance of **our own** no_cot mean | yes | medium (a relative cell) |
| ProntoQA | coconut | 99.8 | yes | **weak: at ceiling.** A tolerance of 2 points around 99.8 is [97.8, 100]; it can detect a broken pipeline but not a faithful one |
| ProntoQA | cot | 98.8 | yes | weak: at ceiling |
| ProntoQA | no_cot | 93.8 | yes | **the informative ProntoQA cell** |
| ProntoQA | pause, coconut_no_curriculum | not in PLAN.md; frozen from the PDF at the manifest commit, else "no published target" | only if a PDF value exists | pause will be at ceiling; no_curriculum is informative if the PDF has it |
| GSM8k-Aug | cot | 42.9 | yes (single seed) | strong |
| GSM8k-Aug | coconut | 34.1 | yes (single seed) | strong, and known to be fragile outside the original group |
| GSM8k-Aug | no_cot | 16.5 | yes (single seed) | strong |
| GSM8k-Aug | pause | 16.4 (this is likely the no-curriculum "pause token" row, not our arm; verify) | no arm run | none |
| emitted tokens | ProsQA CoT 49.4 vs Coconut 14.2; ProntoQA 92.5 vs 9 (paper-style count) | flag only | our paper-style count within 20% of published, else a data/format check before the read continues | secondary oracle |

ProntoQA as a whole is a weak oracle because three of its published cells sit at or above 98.8. A ProntoQA pass is
"consistent with the paper", not "reproduces the paper"; the ProsQA table and the ProntoQA no_cot cell carry K1.

### 4.3 K1 verdict

- **K1 passes on a task** iff every gating cell of that task is matched.
- **K1 passes for Phase 1** iff it passes on ProsQA and on ProntoQA. GSM8k-Aug, if run, must also pass for Phase 2 to
  use GSM8k-Aug (section 7.11); if GSM8k-Aug is not run it is UNTESTED and no Phase 2 GSM screen may start.
- The verdict is computed by one script from the per-example files of the read, printed per cell with our three
  values, mean, SD, sigma_cell, tolerance, published value and "matched / missed", and committed write-once.

### 4.4 What happens on a miss

1. The miss is logged in `critic_log.md` (cell, our values, published value, sigma_cell, tolerance) within the day.
2. **Debug round 1** (documented in CHANGELOG.md as a dated entry with: hypothesis written before any change; the
   evidence; the change set as a commit SHA; the rerun). The round runs the audit checklist of section 7.10 on the
   missed cell's checkpoints, the reference-parity test on real weights for one batch, and a line-by-line diff of
   `curriculum.py`/`data.py`/`model.py` against `dataset.py`/`coconut.py`/`run.py`. If a cause is found and fixed,
   **all three seeds** of every arm whose code path changed are rerun (not only the missed cell), and the task is read
   again (new checkpoints, one read each).
3. **Debug round 2**, same format. The recommended round-2 instrument is one run of the **unmodified reference code**
   (wandb stubbed, single GPU with `batch_size_training 32` and `gradient_accumulation_steps 4`) on a rented GPU,
   about 3 A100-h, which needs the author's typed yes; its number is a second oracle for our pipeline.
4. **After two rounds the task stops.** The report prints the oracle table, "K1 fails on <task>: <cells>", and the two
   rounds. No Phase 2 run on that task. A recipe change that moves our number toward the published one without a
   traced cause is tuning to the oracle and is not a pass; it is reported as "matched after untraced change" and K1
   still fails.
5. A seed-0 **gross miss** (more than 2 x tolerance from the published value on `cot` or `coconut`) may start debug
   round 1 before the other seeds run, to save Mac time; it counts as round 1.
6. The author may direct a continuation after a failed K1. It is recorded below the line; every later number on that
   task carries "unverified reproduction".

### 4.5 The registered read

- A task is read when its five arms x three seeds (and the determinism rerun, 6.4) have finished: one invocation of
  the read script over the 15 (16) selected checkpoints, each scored once on test, per-example files hashed, the
  cross-environment pair (7.8) not required for the K1 read.
- The step sweep and the zero-latent and scrambled interventions on the `coconut` checkpoints (section 8) are part of
  the same read (they score the same checkpoints under registered interventions; each intervention is one read per
  checkpoint).
- The tuned-bar read (3.4) is a separate read after the oracle read.

## 5. Compute convention

### 5.1 Formulas (verbatim from `common/latent/INTERFACE.md`; the code implements exactly these, so the two cannot drift)

With N_ne = non-embedding parameters (everything except `wte` and `wpe`; GPT-2's tied head is charged separately as
V d), V = vocab size (**50,260 after the resize, for every arm**), d = n_embd, n_L = n_layer, d_a = n_head x head_dim:

```
flops_train_per_sequence(dims, T)                  = 3 * [2 (N_ne + V d) T + 2 n_L d_a T^2]
flops_prefill(dims, p, head_positions)             = 2 N_ne p + 2 V d head_positions + 2 n_L d_a p^2
flops_steps(dims, p, n_steps, head_steps)          = 2 N_ne n + 2 V d head_steps + 2 n_L d_a [(p+n)^2 - p^2]
   (n_steps sequential single-position passes after a prefix of p; a latent step has head_steps contribution 0,
    an emitted token has 1)
sequential_passes                                  = 1 + latent_steps + emitted_tokens
```

`model_dims` computes N_ne, V, d, n_L, d_a from parameter shapes, not from formulas, and the test checks it against
torch's parameter counts (section 1.1 numbers). T is the number of **real (non-pad) positions** of the sequence,
latent positions included; padding is a wall-clock cost, not a FLOP.

Worked example the accounting test must reproduce for GPT-2 124M dims (from `GPT2Config()` defaults, V = 50,260,
no download): ProsQA `coconut` with p = 251 (250 question tokens + `<|start-latent|>`), L = 6 latents, E' = 8 answer
tokens: prefill (head_positions 0) = 2 x 85,056,000 x 251 + 2 x 12 x 768 x 251^2 = 43,859,346,432; steps with
n = L + 1 + E' = 15 and head_steps = 1 + E' = 9: 2 x 85,056,000 x 15 + 2 x 38,599,680 x 9 + 2 x 12 x 768 x
(266^2 - 251^2) = 3,389,414,400; total **47,248,760,832 FLOPs**, sequential passes **15**, emitted tokens 8, latent
steps 6.

### 5.2 What is charged and what is not

- **Charged to a run (training):** every forward pass of every training sequence of every epoch at 3x forward (the
  backward), latent positions included (each position is computed once; the latent passes are single positions
  with a KV cache). The GSM `coconut` arm is charged its own epochs plus the `cot` run it starts from.
- **Charged to an example (inference):** prefill plus every sequential pass, latent steps and emitted tokens alike,
  including the pass that produces the eos. The head is charged where logits are needed (section 5.3), **not at latent
  positions**.
- **Not charged:** GPT-2 pretraining (an inherited cost shared by every arm, named as such in every table); the
  per-epoch validation generation and the test read (evaluation); smoke runs; data generation; the 13-gram check;
  checkpoint I/O; the reference's redundant logits at latent positions and its no-cache regeneration (these appear in
  the secondary accounting (c)).

### 5.3 Position table (binding for `inference_record`; |q| = question tokens including the trailing newline)

| arm / intervention | prefill p (head positions) | positions after prefill n (head steps) | latent_steps | emitted_tokens | sequential_passes |
|---|---|---|---|---|---|
| `cot`, `no_cot` | \|q\| (1) | E' (E') | 0 | E' | 1 + E' |
| `coconut` with L latents (also `coconut_no_curriculum`, scrambled, step sweep at L) | \|q\| + 1 (0) | L + 1 + E' (1 + E') | L | E' | 1 + L + E' |
| zero-latent (L = 0) | \|q\| + 1 (0) | 1 + E' (1 + E') | 0 | E' | 1 + E' |
| `pause` with L pause tokens | \|q\| + L + 2 (1) | E' (E') | 0 (latent_positions = L) | E' | 1 + E' |

E' = generated vocabulary tokens before the eos (the `### answer` tokens for latent arms; CoT steps plus the answer
for `cot`); the eos pass is inside the E' head steps because the pass at the last emitted position produces it.

### 5.4 The head-cost confound (named here so no later read over-interprets it)

A latent step skips the LM head; an emitted token does not. Per position, excluding attention: a CoT token costs
2 N_ne + 2 V d, a latent step 2 N_ne. With the section 1.1 counts and V d from the tied head: **124M: 170.1 vs
247.3 MFLOPs, ratio 0.69; 355M: 604.6 vs 707.5, ratio 0.85; 774M: 1,416.8 vs 1,545.4, ratio 0.92; GPT-2 XL 1.5B:
about 0.95.** At matched FLOPs a latent arm therefore gets about 1.45 steps per CoT token at 124M and about 1.05 at
1.5B. This alone predicts a latent advantage that shrinks with size. Rules:

- (a) the head is charged exactly as 5.1 says (head_steps 0 for a latent step, 1 for an emitted token); the saving is
  real compute and is not hidden;
- (b) every matched-FLOPs comparison is printed **side by side with a matched-sequential-passes comparison** (same
  curves read at equal passes), at every size;
- (c) the Phase 3 slope read (7.4) names this confound: a gap that shrinks under matched FLOPs but not under matched
  passes is labelled **"head-cost artefact"**, and a "shrinks" verdict is reported under both accountings.

### 5.5 Secondary accountings and the "accounting-sensitive" label

Reported beside the primary FLOP accounting for every comparison of sections 7.2 to 7.4:

- (a) **sequential passes** (the cost line that matters for decode-bound serving; see 7.2);
- (b) **wall-clock** at batch size 1 on one named device (2.2);
- (c) **reference-cost FLOPs**: the head charged at every position and no KV cache (every pass recomputes the whole
  prefix), which is what the reference code actually computes;
- (d) **training FLOPs** per arm (5.2) with the ratio to the `cot` arm; an arm above 1.25x the tuned bar's training
  FLOPs at the same epochs carries **"training-heavy"**.

If any of (a) to (c) flips any primary comparison, the FLOP verdict stands and the result is labelled
**accounting-sensitive**, with the flipped accounting named.

## 6. Seeds, sigma and decisions

### 6.1 Sigma

- **sigma_task** = max(1.0 point, the seed SD (ddof 1) of the **tuned-CoT bar's** test accuracy over seeds 0, 1, 2 on
  that task). Until the tuned bar exists (Phase 1 before the bar read), sigma_task uses the oracle `cot` arm's seed SD
  with the same floor. In Phase 2 onward, sigma_task comes from the GPU tuned-bar seeds (7.8).
- **Floor 1.0 point.** *Why: at 2 degrees of freedom s/sigma ranges 0.16 to 1.92 (95%), and 1.0 point is the test-set
  granularity scale (0.2 points per example on ProsQA, 0.08 on GSM).*
- **Rises, never falls:** more bar seeds can raise sigma_task; nothing lowers it. A new sigma applies only to verdicts
  not yet written.
- sigma_cell (K1) is section 4.1's quantity and is separate.

### 6.2 Paired differences and the interval

For arms A and B on one task with the same seeds s in S (|S| = n = 3 unless stated):

- Delta = mean over s of [acc_A(s) - acc_B(s)], on the test split, paired per example.
- SE^2 = Var_ex + sigma_task^2 (1/n_A + 1/n_B), where Var_ex is the variance of Delta over **10,000 paired example
  bootstrap draws** (numpy `default_rng(0)`, test examples resampled with replacement, the same draw applied to every
  arm and seed; `summarize` in `eval.py`). The 95% interval is Delta +- 1.96 SE.
- **Beats:** Delta > 2 sigma_task and Delta - 1.96 SE > 0.
- **Loses:** Delta < -2 sigma_task and Delta + 1.96 SE < 0.
- **Tie (non-inferior, "documented tie"):** not beats, and the lower bound Delta - 1.96 SE > -2 sigma_task (the arm
  is shown not to lose by more than 2 sigma). Otherwise **unresolved**. Arithmetic: with sigma_task = 2 and two arms
  disagreeing on 15% of 500 examples, SE is about 2.4 points, so a tie needs Delta above about +0.7 and a win needs
  Delta above about +4.7; a tie is a demonstrated bound, not an absence of evidence.
- *Why both the point bar and the interval: the point bar stops a 1-sigma lead with a lucky interval; the interval
  stops a 2-sigma point lead on a noisy test set.*

### 6.3 Screen and confirmation

- **Screen (1 seed, seed 0):** a Phase 2 arm advances iff, against the GPU tuned bar's seed 0, acc(arm) >= acc(bar) -
  1 sigma_task, **and** it is ahead of its scrambled-at-inference null and of pause-same-curriculum by 1 sigma_task,
  **and** it meets the FLOP and token legs of 7.2 on seed 0. A screen filters; it is never a verdict.
- **Confirmation on fresh seeds:** seeds **1, 2, 3** of the arm and of each control it is judged against; the tuned bar
  adds seed 3. The paired test (6.2) uses seeds 1-3 only; seed 0 is reported and does not enter. *Why: the screen
  selected among cards on seed 0 (winner's curse, sprint protocol section 3).*
- **Two agreeing seeds are not a replication.** No verdict is written with fewer than three seeds per arm; two seeds
  give "single/partial seed" labels only. 774M in Phase 3 runs 2 seeds by plan and is labelled "2 seeds" in every
  table and verdict sentence.

### 6.4 Determinism check and same-seed reruns

- Before the ProsQA read, the ProsQA `cot` seed 0 run is **rerun once on the Mac** with the same code, config, seed
  and data (the determinism rerun). d_nd = |acc_test(rerun) - acc_test(original)| at the selected checkpoints; the
  per-epoch valid curves are also compared. If d_nd > 0.2 sigma_task, sigma_task := sqrt(sigma_task^2 + d_nd^2) for
  every later verdict on the Mac (the nondeterminism is added in quadrature). The original run is the one that enters
  the oracle table; the rerun is reported beside it.
- **Same-seed reruns measure nondeterminism, not seed variance.** They never enter a seed pool or a sigma; they are
  reported in their own table ("same-seed pairs").
- The same check is run once per new environment (first GPU type, first precision) with that environment's `cot`
  seed 0.

## 7. The standing judge (Phases 2 to 4; binding)

### 7.1 Arms and axes

Same base model, same seeds, every arm: `no_cot`; tuned CoT (3.4, re-tuned per environment); the latent method; pause
tokens with identical curriculum and token budget; latent without curriculum; scrambled-latent (at inference, and the
trained-with-scrambled-feedback variant); plain-depth (bigger model, `no_cot`, section 7.7); zero-latent and the step
sweep {0, 1, 2, 4, 8} at inference. Axes per example: section 2.2. Tasks: ProsQA, ProsQA-hard (generator validated
against the released data's statistics; its registration row precedes its first use), ProntoQA, GSM8k-Aug (test =
GSM8K test, 1,319), Phase 4 program-output prediction (our ladder plus CRUXEval-O 800). Phase 2 screens use a
**50,000-example GSM8k-Aug subset = the first 50,000 of a seeded permutation (seed 0) of the train set**, hashed in the
manifest, the same subset for every card and for the screen-time bar.

### 7.2 The primary claim test (one task, one size, one latent arm vs the tuned-CoT bar)

All three legs, paired over the same 3 seeds (6.3 confirmation seeds), at the registered training budget:

1. **Accuracy leg:** by 6.2 against the tuned bar: "beats" = WIN leg; "tie" = TIE leg; "loses" or "unresolved" = fail.
2. **FLOP leg:** mean per-example inference FLOPs (5.1, 5.3) of the latent arm <= that of the tuned bar, on every
   seed, with the ratio and its example-bootstrap interval printed.
3. **Token leg:** mean emitted tokens of the latent arm <= 0.2 x the tuned bar's, on every seed.

Secondary criterion, printed next to leg 3 at all times: **sequential passes <= 0.2 x the tuned bar's**. *Why: output
tokens are billed several times input tokens because each decode step is a sequential, memory-bandwidth-bound pass,
and a latent step is the same kind of pass. "5x fewer output tokens" moves the cost line only if sequential passes
also fall.* An arm that meets leg 3 but not the passes criterion carries the label **"billing claim only"** in every
verdict sentence and table that names it, and it cannot be the author's full goal (section 0's bar) on its own.

- **Primary pass** = WIN leg and legs 2 and 3. **Documented tie** = TIE leg and legs 2 and 3 (plan section 5's Phase 2
  gate accepts either). Anything else is no pass.
- A task where `no_cot` or pause-same-curriculum is itself non-inferior to the tuned bar (6.2 tie) at that size is
  labelled **"CoT-unnecessary at this size"**; a latent pass on it is reported, labelled WIN-KNOWN (curriculum or
  question shortcut), and does not count toward the goal. *Why: on ProsQA the published pause-same-curriculum control
  already beats CoT by about 19 points; the task, not the latent, is the mechanism.*
- A pass must also survive K2, K3 (7.5), the HEADSTART rule (7.6), the readout requirement (7.9) and the audit (7.10)
  to be labelled WIN.

### 7.3 Pareto curves (secondary; reported, never decide)

- Budgeted CoT: the tuned-CoT checkpoint generating at most k CoT tokens, then forced `### ` and the answer decoded,
  for k in {0, 4, 8, 16, 32, full} on GSM8k-Aug and {0, 8, 16, 32, 48, full} on ProsQA/ProntoQA (full = no cap).
  A CoT trained on shortened chains is a different arm and is UNTESTED unless registered.
- Latent: the latent checkpoint at n_latent in {0, 1, 2, 4, 8} (8 exceeds the training count of 6 and is
  out-of-distribution; labelled).
- Both plotted as accuracy vs mean inference FLOPs, vs mean sequential passes and vs mean emitted tokens, with
  example-bootstrap bands, per seed and pooled.
- **The FLOP multiplier**: the FLOPs at which the budgeted-CoT curve reaches the latent arm's accuracy, interpolated
  linearly in log FLOPs between adjacent k points, divided by the latent arm's FLOPs. **No extrapolation**: if the
  latent accuracy is above full CoT, the multiplier is reported as "> full CoT (undefined)"; if below the k = 0 point,
  "< k = 0".

### 7.4 The scale test (Phase 3)

- Sizes 124M, 355M, 774M; 3 seeds at 124M and 355M, 2 at 774M (labelled "2 seeds"). Budgets per plan (GSM8k-Aug
  epochs 25 / 10 / 5; ProsQA-hard at all sizes), registered per size before that size runs.
- gap_N = acc(best latent arm) - acc(tuned bar) at size N **at matched FLOPs**, where the tuned bar's accuracy at the
  latent arm's FLOPs is read on the budgeted-CoT curve (7.3) by log-FLOP interpolation; if the latent arm costs more
  than full CoT, the comparison is against full CoT and labelled "latent costs more than CoT". gap_N is also computed
  at matched sequential passes (5.4 b).
- Slope = least-squares slope of gap_N against log N over the sizes run, with a 95% interval from 10,000 bootstrap
  draws that resample seeds within each (arm, size) with replacement and test examples as in 6.2.
- **Holds** iff gap_774M >= gap_124M - sigma_pooled (sigma_pooled = the pooled bar seed SD over sizes, floor 1.0) and
  the latent arm is non-inferior (6.2 tie or better) at every size. **Grows** iff also gap_774M - gap_124M > 2
  sigma_pooled with the slope interval excluding 0. **Shrinks** iff gap_774M < gap_124M - sigma_pooled. The verdict
  sentence prints both accountings (5.4 c); "shrinks under FLOPs, holds under passes" = "head-cost artefact".
- **K4 (per size):** if the best latent arm is more than 5 points under the tuned bar at matched FLOPs after the
  registered budget at a size, Phase 3 does not scale it to the next size; the slope is read over the sizes run and
  labelled "stopped at <size> by K4".
- The gap-map style multiplier (7.3) is reported at each size; no extrapolation beyond the sizes run.

### 7.5 Kill rules K2 and K3 (K1 is section 4; K4 is 7.4)

- **K2 (latents carry nothing):** Delta_scr = mean over seeds of [acc(arm, own latents) - acc(arm, scrambled at
  inference)] on test, paired. **K2 fires iff Delta_scr < sigma_task or its 95% example-bootstrap interval includes
  0.** A fired K2 kills the arm as a latent-reasoning claim; it continues only as a labelled curriculum or token-budget
  effect ("latents inert"). The zero-latent drop and the pause comparison are printed beside it: an arm within
  sigma_task of pause-same-curriculum is additionally labelled **CURRICULUM-ONLY**.
- **K3 (loses to plain depth at equal FLOPs):** with A_pd(F) from 7.7, **K3 fires iff acc(latent) <
  A_pd(FLOPs(latent)) - sigma_task** (3-seed means). A fired K3 kills the efficiency claim; the arm continues only as
  a **labelled billing claim** (fewer emitted tokens at more or equal total compute than a plain model of the same
  accuracy). An arm within sigma_task of A_pd is "no better than plain depth" and cannot be labelled WIN on
  efficiency.
- Kill rules are evaluated at the confirmation seeds; at the screen they are applied on seed 0 with the 1-sigma bar
  (6.3) as kill lines.

### 7.6 Fade: the HEADSTART rule

- Every accuracy is a **curve over training budget**: the per-epoch valid accuracy of every run is saved (CSV + PNG);
  the registered read is at the final budget B of that phase and size.
- Free instrument: the operator over the tuned bar at 0.5 B and at B from the valid curves of the same runs. If the operator at
  B is smaller than at 0.5 B by more than sigma_task, the arm is flagged "fading".
- **For a WIN label**, the arm and the tuned bar are also run at **2 B** on the confirmation seeds (1-3). If the operator
  at 2 B is smaller than the operator at B by more than sigma_task, the label is **HEADSTART**, not WIN. Without the 2 B
  run the label is at most "WIN (1x budget; fade UNTESTED)", which is reported as HEADSTART-pending and never as WIN.
- A documented tie (7.2) is reported with its 0.5 B / B trend; no 2 B run is required for a tie.

### 7.7 The plain-depth control

- Plain depth at size N' > N is the `no_cot` arm of the bigger GPT-2 (355M for 124M claims; 774M for 355M claims),
  3 seeds, same data and epochs as the size-N `no_cot`, at the per-example FLOPs of its own prefill plus answer.
- A_pd(F) = the `no_cot` accuracy at inference FLOPs F, **interpolated linearly in log F between the `no_cot` points of
  the sizes run** (124M, 355M; 774M in Phase 3). No extrapolation: for F below the 124M `no_cot` point (impossible for
  a latent arm) or above the largest point, the nearest point is used and the comparison is labelled "nearest size".
- Arithmetic the reader should know: on ProsQA (about 250 prompt tokens) six latent steps add about 2% to the
  `no_cot` FLOPs and CoT adds about 20%, while 355M `no_cot` costs 2.9x. So for FLOP-matched latent arms A_pd will sit
  close to the 124M `no_cot` point and K3 reduces to "does the latent arm beat `no_cot` at its own size"; the direct
  355M comparison bites only for arms that loop or take many steps. Both are printed.

### 7.8 The environment rule (gap-map critic log row 126, adopted)

- The torch build, device type and precision are part of the recipe. Every score record carries torch, transformers,
  device (MPS / CUDA GPU type / Kaggle GPU type), precision and commit.
- **Mac and GPU numbers never enter one comparison** (one table cell, one Delta, one sigma pool) unless a same-seed
  cross-environment pair agrees within the floor: the ProsQA `cot` seed 0 and `coconut` seed 0 runs, trained with the
  identical config in fp32 on both environments, must satisfy |acc_Mac - acc_GPU| <= sigma_task on test for **both**
  arms. A second pair (fp32 vs bf16 autocast on the same GPU, same seed) is required before any fp32 number is
  compared with a bf16 number. A failed pair voids nothing already written; it forbids the mixed comparison.
- Phase 1 oracle runs are Mac MPS fp32 (the published precision). Phase 2 onward is GPU with fp32 master weights and
  bf16 autocast, and **re-runs its own tuned-CoT bar and its own `no_cot`** on that hardware. Kaggle (T4/P100, no
  bf16) is a third environment: its runs join a seed pool only after their own pair; otherwise they are "Kaggle,
  unpaired" labels, which means extra seeds run on Kaggle cannot be pooled with Mac or GCP seeds of the same arm.
- Determinism (6.4) is checked once per environment.

### 7.9 The readout requirement

- Every claimed win (primary pass or documented tie) is reported with **readout coverage** and **load-bearing
  fraction**, computed on held-out test examples with probes trained on disjoint examples (plan 6.5):
  - coverage = the fraction of latent steps (over examples and step indices) at which a registered readout (R2 linear
    probe for the registered task state, or the R3 decoder) decodes that state above a shuffled-label chance baseline
    by 2 sigma (binomial over examples);
  - load-bearing fraction = among decoded steps, the fraction at which the registered intervention (F1 counterfactual
    patch from a minimally different problem, or F3 probe-direction edit) moves the answer in the predicted direction
    with a paired example-bootstrap interval excluding 0.
- **Labels:** "explained" iff coverage >= 0.5 and load-bearing fraction >= 0.5; "partly explained" iff both >= 0.25;
  otherwise **"unexplained"**, which does not count toward the goal however large the win. R1 logit lens is reported
  and never counts.
- **Positive control:** the tiny from-scratch model on synthetic reachability must reach coverage and load-bearing
  fraction >= 0.5 under the same probes and interventions; if it does not, the readout instrument is BUG/UNTESTED and
  no "explained" label can be given to anything.
- The faithfulness curve (coverage and load-bearing fraction at each stage boundary along training) is reported.
- The bit-width ladder (latent channel quantised to b bits per step at inference, b in {1, 2, 4, 8, 16, full}) is an
  instrument reported for every latent arm; the bits at which accuracy first drops by sigma_task is "bits used per
  step", compared with the 15.6 bits of a CoT token. It never decides.

### 7.10 Too-good triggers and the audit checklist

In Phase 1 the checklist runs **unconditionally before the K1 read**. In Phases 2 to 4 any of these blocks every
announcement and every further run on that task until the checklist is done and logged:

- any arm more than **10 points over the tuned bar** at 124M on GSM8k-Aug;
- any arm **above 99.0%** on ProsQA-hard;
- a null (scrambled-at-inference, scrambled-in-training, zero-latent) above the tuned bar by 2 sigma_task, or above
  the arm it is a null of by 1 sigma_task;
- a `no_cot` arm above its published value by more than the K1 tolerance (the "reads the answer off the question"
  symptom);
- any seed more than 3 sigma_task from the other seeds of its arm.

The checklist (every item logged with the commit and the per-example file hashes):

1. **13-gram overlap** train vs test for that task (1.3), recomputed on the files actually used.
2. **Rescoring from checkpoint**: `eval.py` rerun on the saved selected checkpoint at batch size 1 and in a different
   example order; the per-example correctness must equal the registered read's file except for at most 1 example in
   500 (MPS nondeterminism); more is a BUG until explained.
3. **Label-leak test** (`tests/`): for every arm, stage and record, labels are -100 at and before the `<|end-latent|>`
   position (latent arms) or at and before the last question token (cot/no_cot); no latent position carries a label;
   generation prompts contain no step or answer token.
4. **Null-blindness test** (`tests/`): under mode "scrambled" the fed-back vectors at every latent position of example i
   equal `scramble_bank[i]` and the output is unchanged when example i's own hidden states are replaced by random
   vectors; pi(i) != i for every i; under mode "pause" the forward equals the plain HF forward on the same input_ids;
   `coconut_no_curriculum` never emits a step token in any epoch.
5. **Batch-isolation test** (`causality.py`-style, on the latent path, train and eval mode, B = 1 vs B = 2 and 8):
   example i's logits agree within 1e-4 (fp32) whatever its batch companions; row j changes nothing in row i.
6. **Counts match:** parameter count and FLOPs per the accounting test (5.1); the mechanism is active (latent vectors
   are not constant across examples; feedback hooks called the registered number of times).
7. The critic reads the diff of the run's commit against the last audited commit.

### 7.11 Per-card registration (Phase 2) and the mechanism label

Before a card's screen, the critic logs an acceptance row that names: the mechanism; its prediction with a number; the
known-family label with the three closest works (ID, what was read, overlap, difference), defaulting to the more-known
label; its own null (same compute, tokens, tuning budget and code path, with the mechanism's information removed, not
switched off); the screen budget B and the kill line in points on valid at a stated budget point; the fade expectation;
and the leak answers (how it passes 7.10 items 3 to 5). Every card also states which kind of latent state it builds:
**"appended-KV scratchpad"** (Coconut-style: latents are appended positions in the KV cache, a growing store of
continuous symbols) or **"fixed-width in-place state"** (slot refinement or looping that does not grow the cache).
*Why: the memory-budget separation of PLAN.md 1.4 binds fixed-width designs, not appended-KV ones; the theory memo's
T2 prediction and the bit-width ladder read differently for the two kinds.* A card without this label is not accepted.
A result from a run that started before the acceptance row is a label (UNTESTED), never a screen.

## 8. Null constructions (exact; each has a unit test under 7.10)

| null | construction | what it destroys | what it attributes |
|---|---|---|---|
| **scrambled-at-inference** | For each eval batch, examples are grouped by latent count (one class in Phase 1: all have 6); within a class a seeded derangement pi (pi(i) != i; batches of size 1 are merged with the previous batch so every class has >= 2 members) is drawn; a first pass with own feedback records every example's latent vectors; the example is then run with the vector at **every** latent position j taken from `bank[i, j] = latents[pi(i), j]`, never its own, never a mixture; the prompt, markers, count and positions are unchanged | the example-specific content of the latent channel, while keeping count, positions, markers, compute and the distribution of real latent vectors | Delta_scr = the part of the accuracy that depends on what the latents carry (K2). Near zero means the answer is read from the question and the latents are inert. It cannot detect "the model only needs some in-distribution vector to switch into answer mode"; that is by design (content vs form) |
| **scrambled-in-training** | trained like `coconut`, but at every latent position of every training batch the fed-back vector comes from a derangement of the batch within the same latent-count class (own hidden states never fed back); evaluated with scrambled feedback too | the possibility of learning a latent channel at all, keeping curriculum, tokens, passes and compute | `coconut` minus this arm = the value of a learnable channel; this arm minus pause = the effect of a real-latent-shaped input vs a fixed embedding |
| **pause-same-curriculum** | section 3.1 `pause` | the information channel and the sequential passes, keeping curriculum and token budget | curriculum vs content (the published 96.6 vs 97.0 control) |
| **no-curriculum** | section 3.1 `coconut_no_curriculum` | the staged supervision, keeping the mechanism | curriculum necessity (published: about No-CoT) |
| **zero-latent** | the `coconut` checkpoint evaluated with n_latent = 0 (question + `<|start-latent|>` + `<|end-latent|>` + answer) | all latent compute at inference | what survives with no silent steps; against `no_cot` it separates a training-induced gain from an inference-time latent gain ("reads the answer off the question") |
| **step sweep {0, 1, 2, 4, 8}** | the `coconut` checkpoint evaluated at each n_latent; 8 > 6 is out-of-distribution and labelled | nothing; a dose-response | accuracy vs inference-time silent compute; monotonicity; whether more steps help (the energy campaign's K sweep, finally run) |

Scrambled and step-sweep reads on a checkpoint are registered reads (4.5): each is scored once on test.

## 9. Run order and adaptive rules (Phase 1)

1. Smoke and calibration under section 0; `pytest common/latent/tests` green on CPU; CHANGELOG.md holds the
   calibration and parity numbers (section 14).
2. Data manifest committed and signed (1.4). From here, section 0's cap is lifted for registered runs only.
3. **ProsQA**, Mac fp32: `cot` s0, `coconut` s0 (gross-miss check, 4.4 item 5), then `no_cot` s0, `pause` s0,
   `coconut_no_curriculum` s0; then every arm s1; then every arm s2; then the determinism rerun of `cot` s0 (6.4).
   Then the audit checklist (7.10) and the ProsQA read (4.5), including the scrambled, zero-latent and step-sweep
   reads on the three `coconut` checkpoints.
4. **ProntoQA**, same order, same read.
5. **Tuned-CoT bar** on ProsQA, then ProntoQA: the five extra seed-0 trials, the pick, seeds 1 and 2, the bar read.
6. **GSM8k-Aug** (blocked until the author's typed yes with the estimate in front of them): `cot` s0, then `coconut`
   s0 from its selected checkpoint, then `no_cot` s0; audit; read.
7. **Cross-environment pair** (7.8): when the first GPU is available, ProsQA `cot` s0 and `coconut` s0 in fp32 on it,
   then the fp32 vs bf16 pair; logged before any Phase 2 number is compared with a Phase 1 number.
8. No run order change moves a read earlier. A Mac run that exceeds 2x its calibrated estimate is investigated and
   logged, not killed. A preempted or crashed run is restarted from scratch with the same config and seed; its partial
   log is kept and enters no number.

## 10. Diagnostics (reported, never decide)

Per task and arm: valid-accuracy and train-loss curves per epoch with stage boundaries; the distribution of emitted
tokens and of sequential passes; the paper-style token count; the "earlier stage better on valid" flag; the untrained
base model's accuracy on 100 valid examples; the 13-gram rates; the step-sweep and zero-latent curves; the scrambled
drop per seed; the same-seed pair table; the cross-environment pair table; 20 seeded test predictions per arm printed
as quoted data **by index only, without the example text** (section 1.3 forbids printing test examples).

## 11. Labels and the critic's predictions

**Verdict labels:** WIN / WIN-KNOWN / HEADSTART / NULL / LOSS / BUG / UNTESTED, with the sprint protocol's meanings
(a tie by 6.2 is reported as "documented tie", a NULL that is attributable). **Novelty labels:** NEW / COMBINATION /
KNOWN, defaulting to the more-known label. Anything not run is UNTESTED, never estimated.

**Critic's numeric predictions, made before any run (3-seed means in points; spread = seed SD):**

| task | arm | predicted mean | predicted spread | note |
|---|---|---|---|---|
| ProsQA | cot (published recipe) | 78 | 2.0 | published 77.5 |
| ProsQA | no_cot | 77 | 1.5 | published 76.7 |
| ProsQA | coconut | 95.5 | 1.5 | published 97.0; batch emulation and MPS cost a little; 80% interval [92, 98] |
| ProsQA | pause | 95 | 1.5 | published 96.6 |
| ProsQA | coconut_no_curriculum | 77 | 2.0 | "about No-CoT" |
| ProntoQA | cot | 98.5 | 0.7 | published 98.8 |
| ProntoQA | no_cot | 93.5 | 1.0 | published 93.8 |
| ProntoQA | coconut | 99.5 | 0.4 | published 99.8 |
| ProntoQA | pause | 99.5 | 0.4 | no PLAN.md target |
| ProntoQA | coconut_no_curriculum | 60 | 8 | low confidence; no PLAN.md target |
| GSM8k-Aug (seed 0) | cot | 42 | n/a | published 42.9 |
| GSM8k-Aug (seed 0) | coconut | 32 | n/a | published 34.1; the fragile cell; 80% interval [27, 36] |
| GSM8k-Aug (seed 0) | no_cot | 16 | n/a | published 16.5 |

- **Tuned-CoT bar on ProsQA:** 84, spread 3. Prior that it exceeds 77.5 substantially: P(bar > 81.5, i.e. 2 sigma above
  the published CoT) = 0.60; P(bar > 90) = 0.25. *Why: CoT 77.5 against No-CoT 76.7 says the published CoT barely used
  its chain; a higher LR or a decay schedule should buy several points, but greedy CoT on ProsQA also fails by
  committing to a wrong branch, so a ceiling in the low 80s is plausible.* On ProntoQA the bar is at ceiling: 99.0,
  spread 0.5.
- **Scrambled-latent drop for Coconut on ProsQA** (own minus scrambled-at-inference, 3-seed mean): **4 points**, 80%
  interval [1, 20]. The literature says small (perturbations flip under 5-10%; graded, low-rank causal effect).
  P(K2 fires on ProsQA Coconut, i.e. drop < sigma_task or interval includes 0) = 0.35. Zero-latent drop: 8 points,
  80% interval [2, 25].
- **P(K1 passes):** ProsQA 0.70; ProntoQA 0.80; GSM8k-Aug 0.50 (conditional on the author funding it; single seed;
  the Coconut cell is the risk). P(ProsQA and ProntoQA both pass) = 0.60; P(all three) = 0.30.
- **Determinism:** P(d_nd > 0.2 sigma_task on the Mac) = 0.8; the nondeterminism will be added.
- **Mac throughput:** fp32 MPS at seq 320, batch 16: 6k tokens/s (80% interval [3k, 12k]); a ProsQA coconut run
  10-15 Mac-h; Phase 1 ProsQA + ProntoQA + bars about 350 Mac-h.

## 12. Budget and priority

- if the calibration implies more, the operator puts the number in front of the author before the oracle starts (decision D is
  the author's). The Mac is shared (MAC_STATE.md: Codex's guarded runs stop when another training process starts); the slot lock
  is respected and every stop is an operational incident in `deviations.md`.
- a plain "go" on the plan is the Phase 1 Mac work only (plan section 5).
- Each phase gets its own ledger line and a typed go. No stage starts if committed compute plus its worst case exceeds the phase cap. The 2 B fade
  runs (7.6) and the seed-3 confirmations (6.3) are inside the Phase 2 cap; the operator re-estimates from measured throughput before Phase 2 and
  cuts from the bottom of the priority list, never from the controls.
- **Priority = run order (section 9).** Anything not run is UNTESTED, never estimated.
- The programme budget file (plan 6.1, modelled on `think/gcp/budget.py`) holds every ledger line; the cost ledger is
  hash-chained as in `common/sprint1/ledger/ledger.py`.

## 13. Report spec (committed under `results/` and the experiment RESULTS.md)

1. **Oracle table:** per task and arm: published value (PDF-verified, with the snippet value if it differed), our three
   seeds, mean, SD, sigma_cell, tolerance, verdict per cell, environment, commit; the ProntoQA "regenerated data" and
   GSM "single seed" labels; the emitted-token secondary cells.
2. **Curves over epochs:** valid accuracy and train loss per run, CSV (`results/curves/<task>_<arm>_s<seed>.csv`) and
   PNG with stage boundaries; the determinism pair overlaid.
3. **Per-example cost tables:** `results/cost/<task>_<arm>_s<seed>.csv` with idx, correct, emitted_tokens,
   latent_steps, latent_positions, sequential_passes, flops, wall_ms (no example text), and the summary table of
   means, medians and intervals per arm, with the paper-style count and the secondary accountings of 5.5.
4. **The tuned-bar table:** every trial, its best-valid epoch and accuracy, the pick, the three seeds, the bar value.
5. **Interventions:** scrambled, zero-latent and step-sweep accuracies per seed with intervals; K2 verdict on Coconut
   (diagnostic in Phase 1).
6. **The cross-environment pair** and the same-seed pair tables.
7. **Every run kept:** `results/runs_index.csv` lists smoke, calibration, oracle, bar, debug-round, failed and
   restarted runs with commit, config hash, data manifest hash, device, precision, torch/transformers, seconds, FLOPs.
8. **Provenance:** model revision, data SHA-256s, generator command and commits, the registration commit, every
   verdict's code commit; `deviations.md`; the audit checklist outcomes; the 13-gram rates.
9. **Predictions vs outcomes** (section 11), printed without editing the predictions.
10. **Verdict sentences, verbatim:** "K1 ProsQA: pass/fail (<cells>)"; "K1 ProntoQA: pass/fail (<cells>)"; "K1
    GSM8k-Aug: pass/fail/UNTESTED (single seed)"; "Tuned-CoT bar ProsQA: <value> +- <SD> (seeds 0-2)"; "Scrambled
    drop, Coconut ProsQA: <value> [<interval>]; K2 would fire: yes/no". Each sentence carries its labels (KNOWN,
    environment, single seed, regenerated data, accounting-sensitive, UNTESTED).

## 14. Defects the operator must fix before the first real (non-smoke) run

Found while reading INTERFACE.md, the plan and the reference files. Each fix is a commit named in CHANGELOG.md; the
critic re-reads the diff and logs acceptance before the manifest signature.

- **D1. Special-token initialisation.** INTERFACE.md (tokenizer protocol) initialises the three new rows from the
  eos embedding. The reference `run.py` copies the embedding of `<<` (`target_id = convert_tokens_to_ids("<<")`) into
  the input embedding and the tied head. The reference binds for the oracle; fix INTERFACE.md and `load_base`.
- **D2. Optimizer reset schedule.** INTERFACE.md `train.py` says "reset AdamW on a stage change if reset_optimizer".
  The reference rebuilds AdamW **at the start of every epoch** whenever `reset_optimizer` is True (inside the epoch
  loop, unconditionally), and the Coconut yamls set it True for every latent arm. Fix the contract and the code:
  latent arms (coconut, pause, coconut_no_curriculum, GSM coconut) reset every epoch; cot/no_cot do not. Document the
  paper's wording ("when switching stages") vs the code's behaviour in CHANGELOG.md; the code produced the numbers.
- **D3. `max_new_tokens` and the generation contract.** The reference uses 64 for GSM and 128 for ProsQA/ProntoQA
  (`"gsm" in val_path`); INTERFACE.md's `evaluate` defaults to 64, and the brief handed to the critic had the values
  swapped. Register per task (section 2.1) and make `evaluate` require the value explicitly. The batched generation
  must decode `prompt + generated` with `skip_special_tokens=True` as the reference does, stop per example at eos, and
  pass a parity test: batched output == the reference's per-example `generate` token-for-token on 100 valid examples
  with real GPT-2 weights (CPU fp32), for `cot`, `no_cot` and `coconut` prompts.
- **D4. Reference-parity test on the forward.** A test must port `coconut.py`'s forward verbatim as the oracle and
  assert `LatentLM.forward(mode="coconut")` is `allclose` (atol 1e-5 fp32) on a random tiny config **and** on one real
  GPT-2 124M batch (CPU), including the loss, with the reference `MyCollator` padding. It must pass before the
  manifest signature; the one-batch MPS vs CPU float64 parity number goes in CHANGELOG.md.
- **D5. Embedding resize for cot/no_cot.** The reference resizes only for latent arms; INTERFACE.md's `load_base`
  resizes always. Either do as the reference, or show the 1e-4 valid-loss agreement on smoke and log it (1.1).
- **D6. Checkpoint rule not in the contract.** INTERFACE.md's `train.py` writes `ckpt_best.pt` without saying best
  among what. Implement section 3.3 (final-stage best valid, ties to the earliest epoch; all-epoch valid accuracies
  in `log.jsonl`; the "earlier stage better" flag; for GSM `cot`, `save_only_improve` semantics).
- **D7. Position accounting not pinned.** INTERFACE.md's `inference_record(dims, p, latent_steps, emitted_tokens)`
  does not say where the `<|end-latent|>` position and the head positions go. Implement the table of 5.3 and the
  worked example of 5.1 as a test (expected 47,248,760,832 FLOPs and 15 passes), plus N_ne = 85,056,000 /
  302,311,424 / 708,390,400 from parameter shapes (`GPT2Config` defaults for 124M; the medium/large configs).
- **D8. Mac calibration and parity numbers must be in CHANGELOG.md before the oracle starts.** CHANGELOG.md does not
  exist yet. Required entries: tokens/s and peak memory for fp32 and fp16 autocast at seq 320, batch 16 (forward +
  backward), the MPS vs CPU float64 one-batch parity (max abs and relative error of logits and loss), the chosen
  micro-batch and grad_accum for 128, the torch/transformers versions on the Mac, and the implied Mac-h per run.
- **D9. The data manifest must be committed before any training.** `configs/` is empty. Section 1.4 lists the fields.
  The Coconut repo commit SHA is not in the plan; freeze it at the manifest commit. Record the ProntoQA generator's
  nondeterminism (two runs, two hashes).
- **D10. Tests directory is empty** (only `fake_tokenizer.py`). Required green on CPU before the signature: label
  leak, null blindness (scrambled, pause, no-curriculum), batch isolation on the latent path, batched-vs-per-example
  generation, collator parity vs `MyCollator`, dataset-builder parity vs `get_cot_latent_dataset` /
  `get_question_latent_dataset` for every stage and arm (including `pad_latent_to_max` padding when len(steps) <
  stage), accounting (D7), tokenisation assertion (1.2).
- **D11. Scrambled-bank construction is under-specified in INTERFACE.md** ("bank = latents of a derangement of the
  batch within the same latent-count class"). Implement section 8 exactly: a first own-feedback pass records all
  latents; `bank[i, j] = latents[pi(i), j]` for every j; pi seeded (`default_rng(3)` per batch index); classes with one
  member merged with the previous batch; assert pi(i) != i.
- **D12. Loss weighting under accumulation** (3.2): record the micro-batch; if it is not 32, log the weighting
  difference once as a deviation.
- **D13. `uniform_prob` consumes a random draw per example per epoch in the reference even at 0.** Our data order
  will not match the reference's in any case (3.2); log once, do not try to match.
- **D14. The plan's `think/lm.py` n_inner issue** (plan 2.1) must not be inherited: `model_dims` reads shapes, and the
  D7 numbers are the check.
- **D15. GSM `coconut` initialisation.** The config must name the `cot` checkpoint by epoch and hash, start at epoch
  3 (stage 1), and the selection pool is epochs 9-24 (3.3). The reference's `resume` semantics (skip epochs, keep the
  stage arithmetic) must be mirrored, not re-invented.

Registered by the critic, 2026-10-17 20:34 UTC, before any run. The commit SHA of this file is appended by the operator in
the next `critic_log.md` row; the commit time is the record.

---
*Amendments (tighten only, unless author-directed and then recorded as such with the original verdict still printed),
dated below this line.*

**A1 (2026-10-17 20:45 UTC, before any run; the critic's check of the operator's working tree against sections 0-14; tighten or
clarify only).** Seen when written: untracked code under `common/latent/` (`data.py`, `accounting.py`, `model.py`,
`train.py`, `eval.py`, `contamination.py`, `prosqa_gen.py`, tests with ports of the reference `coconut.py` and
`dataset.py`), the experiment's `code/` (`fetch_data.py`, `run_train.py`, `calibrate_mac.py`), ten configs,
`configs/data_manifest.json` (ProsQA at Coconut commit `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`; GSM8k-Aug at the
registered iCoT commit; 385,620 / 500 / 1,319 records), data files under the gitignored `data/`, and a CHANGELOG.md
with no calibration numbers. **No accuracy number of any kind exists; `results/` is empty; nothing is committed.**

- **Resolved in the working tree (acceptance pending the commit and a green test run):** D1 (`init_latent_embeddings`
  copies `<<`; falls back to eos only when no `<<` token exists, which is never the case for GPT-2); D2 (`train.py`
  rebuilds AdamW every epoch when `reset_optimizer`, `optimizer_reset="epoch"`); D3's per-task rule
  (`default_max_new_tokens`: 64 if "gsm" in val_path else 128; tests for it and for batch-one parity with the
  reference `generate`); D5 (`load_base(resize=False)` for cot/no_cot, as the reference); D11's construction (a first
  clean pass, `bank = latents[donor]` at every position, Sattolo derangement, `pi(i) != i` asserted); most of D10
  (label-mask layout, scrambled blindness, pause equals the plain LM, batched equals per-example, collator and
  stage-dataset parity against the reference port per stage and arm, N_ne = 85,056,000 from `GPT2Config()`,
  configs mirror the yamls); D13 and D12 are logged in the config notes (micro-batch 32 x 4).
- **Still open, with the specifics now known:**
  - **D7:** `accounting.inference_record` charges `head_positions=0` for the whole prefix (for `coconut` the prefix
    includes `<|end-latent|>`), so the head evaluation that produces the **first** token is never charged: heads = E'
    instead of E' + 1. Fix: `flops_prefill(dims, p, head_positions=1)` inside `inference_record`. With that one
    change the code equals section 5.3 exactly (the attention and N_ne terms already agree: total positions are the
    same). The difference is one 2 V d = 77.2 MFLOPs per example at 124M, 0.16% of the worked example; immaterial to
    any verdict, but the formulas in 5.1 and the code must be identical. Add the worked example (47,248,760,832 FLOPs,
    15 passes, from p = 251, L = 6, E' = 8 on `GPT2Config()` dims with V = 50,260) as a test, and the 355M / 774M
    N_ne values.
  - **D6:** `train.py` keeps `ckpt_last.pt` (overwritten) and `ckpt_best.pt` = best valid accuracy over **all**
    epochs (strict `>`). Section 3.3 selects among **final-stage** checkpoints. Fix: track and save
    `ckpt_best_final.pt` = best valid accuracy among epochs whose stage equals `max_latent_stage` (strict `>`, ties to
    the earliest), which is the selected checkpoint for every latent arm; keep `ckpt_best.pt` (all-epoch best) for
    the "earlier stage better on valid" flag and the labelled secondary read of 3.3 item 3. For cot/no_cot the two
    coincide. The reference's per-epoch validation at the stage's latent count is mirrored, as registered.
  - **D4:** `calibrate_mac.py` checks forward parity "on a tiny random model". Section 14 requires the one-batch
    parity on the **real GPT-2 124M weights** with one real ProsQA training batch (stage 6, batch 16), MPS fp32 vs
    CPU float64: max abs and relative error of the logits and of the loss, into CHANGELOG.md before the oracle. The
    tiny-model check stays as a test.
  - **D8:** CHANGELOG.md exists; the calibration numbers do not yet. Unchanged requirement.
  - **D9:** the manifest lacks ProntoQA (the generator has not run; the two-hash nondeterminism record and the
    "regenerated data" label stand), the HF revision of `openai-community/gpt2`, the torch/transformers/Python
    versions, and the Mac's values of the same. The Coconut commit is now known and is frozen when the manifest is
    committed: `27273cb8cca4bb763c041a63b036d0c3b7cbbb48`. The CHANGELOG's content hashes of the reference files are
    accepted as the pin of the reference code.
  - **D10 (remaining):** an explicit train-mode batch-isolation test on the latent path (B = 1 vs B = 2 and 8: loss
    and logits of example i unchanged by its companions, fp32 atol 1e-4), in addition to the eval-mode
    batched-equals-per-example test; the D7 worked-example test.
  - **Read script:** `evaluate` still defaults `max_new_tokens=64`; the registered-read script must pass the per-task
    value explicitly and record it in the per-example file header, so a ProsQA read can never run at 64.
  - **`optimizer_reset`:** every Phase 1 config must carry the default `"epoch"`; a run with `"stage"` is not an oracle
    run and enters no oracle cell (it may be a labelled diagnostic in Phase 2).
  - **Checkpoint files** are state dicts under the gitignored `runs/`; the selected checkpoint of every oracle run is
    kept until the report is closed (the rescoring audit needs it). About 0.5 GB per checkpoint, 2 per run.
- **Clarifications adopted from the code (each at least as strict as section 8 or 1.3):**
  - **Scrambled-at-inference** draws the derangement over the **whole evaluation set** within each latent-count class
    (`scramble_donors`, `random.Random(scramble_seed)`, Sattolo single cycle, so no fixed points), not per batch;
    `scramble_seed` = 0 for the registered read. Section 8's per-batch wording is superseded by this; everything else
    in section 8 stands. A class with one member raises instead of merging; in Phase 1 every example has 6 latents,
    so this cannot occur; a Phase 2 card whose latent counts vary registers its own handling.
  - **13-gram overlap** (`contamination.py`): train side = question + steps per record, test side = question, spans
    never cross a record, reported as the fraction of test items sharing any span; items shorter than 13 words count
    in the denominator. Accepted as the primary rate of 1.3. In addition the operator reports the rate with the test side
    = question + steps + answer, as a second number. The GSM8k-Aug 1% block of 1.3 applies to the primary rate.
- **Nothing in A1 changes a verdict rule, a tolerance, a seed rule or a registered comparison.**

**A2 (2026-10-17 21:29 UTC, before any run; manifest signature, corrections and rulings on the operator's report of the committed tree;
tighten or correct only).** Verified by the critic against the branch `claude/amazing-ramanujan-xc6ow3` (registration
commit `rev373`, lead's SHA row `rev374`, INTERFACE.md v2 `rev375`, code `rev376` and `rev35`, INTERFACE.md v3
`rev377`, CHANGELOG `rev40`), the committed `configs/data_manifest.json` at `rev35`, an independent run of the
test suite (891 passed, 24.7 s, CPU) and an independent recomputation of the 13-gram rates and of the step and
duplicate counts from the files under `data/`. Seen when written: no accuracy number of any kind; `results/` empty;
no run on any device.

1. **Manifest signature.** `configs/data_manifest.json` at `rev35` is **signed as the data pin**: ProsQA 17,886 / 300
   / 500 from Coconut commit `27273cb8cca4bb763c041a63b036d0c3b7cbbb48` (resolved by `git ls-remote`, files fetched
   from URLs pinned to it); GSM8k-Aug 385,620 / 500 / 1,319 from iCoT `rev36`; ProntoQA 9,000 / 200 / 800 from
   generator `rev37` with the exact README command, raw file sha256 `rev38...`; SHA-256, bytes and record counts
   for every file; 13-gram rates recorded in the manifest; build environment Python 3.13.16, torch 2.14.1, transformers
   4.57.6. **Conditions, both before the first oracle run:** (a) the **pin row**: a manifest update commit with the HF
   revision and weights sha256 of `openai-community/gpt2` and the Mac's Python/torch/transformers (the 355M and 774M
   revisions may be pinned at the start of Phase 3, before their first run), logged by the critic; (b) the **PDF row**:
   the section 4.2 targets verified from the Coconut PDF (arXiv 2412.06769) and the same-curriculum pause source, with
   published spreads where the paper gives them, logged by the critic. Until both rows exist, section 0's caps
   (2,000 examples, 200 steps, 100 validation examples, no test read) remain in force for every training process; the
   calibration, the smoke runs and the `--real` parity run may proceed now. The signature does not move any read.
2. **Corrections to section 1.2 (text only; no rule changes).** (a) ProsQA proofs are **3 to 6 steps** (train 39.9 /
   42.7 / 14.2 / 3.1%; valid 41.3 / 44.3 / 12.0 / 2.3%; test 40.4 / 43.4 / 13.8 / 2.4%; critic's count), not 3. Under
   the reference semantics (training n_latent = stage for stage <= max_latent_stage, the first `stage` steps removed,
   `pad_latent_to_max` acting only past the cap) every ProsQA record has all its steps removed from stage 6 on, so the
   ProsQA and ProntoQA final regime of 3.3 (epochs 30-49) is unchanged. (b) ProntoQA's generator is **seeded by default**
   (`run_experiment.py` seeds `random` and `numpy` with the argparse default `--seed 62471893`); two full runs were
   byte-identical (sha256 `b8d969259bceb08594dfa5c3d131b8e2f1a0556622cc8ee5723aa27ca68bff5b`, 10,000 records), so the
   two-hash record of 1.2 is satisfied by two identical hashes. The label **"regenerated data" stays**: the paper's files
   were never released; if the authors used the same commit and default seed the sets may be identical, which cannot
   be checked. ProntoQA records have 6 steps each (critic's count).
3. **Tightening of 3.2 and 3.3 for GSM8k-Aug `coconut` (new defect D16).** Under the reference semantics, GSM epochs
   9-11 (stage 3) still keep every step beyond the third as a language target, and epochs 12-24 (stage >= 4, pad)
   remove every step. The paper's GSM Coconut token count (about 2 emitted tokens beside 6 thoughts (snippet; verify
   from PDF before the read)) is the all-removed regime. **Selection pool for GSM `coconut` = epochs 12-24 (0-indexed;
   checkpoints 13-25)**, 6 latents, every step removed; epochs 9-11 are "stage 3 (partial language)" and, if one of
   them has the higher valid accuracy, it is the labelled secondary read of 3.3 item 3. General rule: the final
   regime of a latent arm is the set of epochs at which every training record has all its language steps removed,
   `stage >= min(max_latent_stage + 1, max_steps_train)`; ProsQA and ProntoQA: stage >= 6, epochs 30-49, unchanged.
   **D16:** `train.is_final_stage` currently uses `stage >= max_latent_stage`, which admits GSM epochs 9-11; change it
   to the rule above (or a config field set to 4 for GSM) with a test, before the GSM run. `coconut_no_curriculum`
   pinned at `max_latent_stage + 1` is this regime from epoch 0 and is consistent with it.
4. **13-gram overlap: ruled, with the critic's own numbers.** Primary (test side = question) / secondary (question +
   steps + answer): ProsQA test 127/500 = 25.4% / 447/500 = 89.4%, valid 72/300 = 24.0% / 254/300; ProntoQA test
   781/800 = 97.6% / 800/800, valid 198/200 = 99.0% / 200/200; GSM8k-Aug test **7/1,319 = 0.53%** / 15/1,319 = 1.14%;
   GSM valid 7/500 = 1.4% / 2.2% (lead's numbers; the critic's recomputation of the valid pair is logged in its own
   row). **Exact duplicate questions train vs valid and train vs test: 0 in all six pairs** (critic's count; the operator
   had stated ProsQA and GSM, ProntoQA is now also 0). Rulings: (a) the logic-task rates are template overlap over a
   tiny vocabulary, not contamination; they are reported with that sentence and the duplicate counts, and no read is
   blocked by them; (b) the GSM test rate is under the 1% block of 1.3, so the GSM read is not blocked; the 7 test items
   are listed **by idx** with the idx of a matching train record and the shared span's length in words (never the
   text), and the GSM read also reports accuracy on the 1,312 non-overlapping items as a slice (no verdict depends on
   the slice; a flip is reported); (c) **the 1% rule is a rule on the scoring split and does not touch valid**; valid
   is the selection split, its overlap is the same for every arm, and its 7 items are listed by idx and recorded as a
   known bias in `deviations.md` (direction: valid accuracy slightly optimistic, equally for all arms); (d) the
   secondary rates are reported beside the primary ones in every data table.
5. **Reference-forced semantics and implementation deviations: accepted, logged before any run** (CHANGELOG 21:40 and
   22:10 UTC entries). Training n_latent = stage whenever stage <= max_latent_stage (pad matters only past the cap;
   generation uses min(stage, cap)); `uniform_prob` draws from [0, len(steps)] per record (immaterial in Phase 1:
   every oracle config has 0); `coconut_no_curriculum` at cap + 1; generation stops at the first eos even as the first
   token (the reference appends it and continues): **the number of examples whose first generated token is eos is
   reported per read**, and bounds any difference from the reference; scrambled mode runs as one pass but is charged
   L latent steps and the latent-arm position table, so its wall-clock is not comparable and is not reported as a cost
   axis; autocast with fp32 master weights for bf16/fp16 (fp32 in Phase 1 anyway); `ckpt_last/best/best_final` instead
   of per-epoch files, both best trackers starting at -1; seeded per-epoch data order; loss as the micro-batch mean
   (D12); the model raising on misaligned latents. None of these touches a tolerance, a seed rule or a comparison.
6. **Defect status (section 14 and A1) after rev35, verified by the critic in the code and tests:** closed D1, D2,
   D3, D5, D6 (`ckpt_best_final.pt`, strict >, subject to D16), D7 (`inference_record` per 5.3; the worked example
   47,248,760,832 FLOPs / 15 passes and N_ne 85,056,000 / 302,311,424 / 708,390,400 are tests), D10 (including
   `test_train_mode_batch_isolation_same_shape_bit_identical` and `test_batch_isolation_alone_vs_with_companions`),
   D11, D12, D13, D14, D15 (subject to D16). **Open:** D4's number (`calibrate_mac.py --real` on the Mac), D8's numbers,
   D9's pin row, the PDF row, D16. The first oracle run needs all five closed and logged.
7. **Nothing in A2 loosens a rule.** The GSM selection pool is narrower; the reporting is larger; the signature adds
   two conditions before the first oracle run.

**A3 (2026-10-18 22:31 UTC, before the first VM; GPU variant of the plumbing permissions, author-directed scope of the 72-hour
window, card registrations, budget and resume rules).** Authority: PLAN.md Amendment A1 (2026-10-18, author, in writing:
projects AcctA `cloud-a`, AcctB `cloud-b`, AcctC `cloud-c` and the main project's A100-40GB allowed; Mac path dropped; 72-hour
scope; C1b added; Author's words of record, verbatim from CHANGELOG.md: "use our existing compute, and go", then "be as fast as
you can ... I need a significant result". Seen when written: no accuracy number of any kind; `results/` empty; no VM of this
project has run; the cards exist at `docs/cards/` (commit `rev41`); `code/launch/`, `code/make_screen_data.py` and
`governance/run_log.jsonl` do not yet exist. Where this amendment loosens a registered rule it does so only as the author's A1
directs, and the registered rule's verdict is still printed (header clause).

**A3.1 Plumbing permissions, GPU variant (replaces section 0's Mac wording; section 0's caps stay in force until the
oracle-start conditions below are met).** Allowed now, before the pin row and the PDF row, on one **prep VM** (main
project, A100-40GB spot or a CPU VM for the non-GPU items), each VM a ledger line:
- fetch the data by `code/fetch_data.py` and verify every SHA-256 against the committed manifest (a mismatch stops
  everything until explained);
- resolve and pin the HF revision and weights sha256 of `openai-community/gpt2` (355M and 774M may wait for a scale
  phase), write them and the VM's Python/torch/CUDA/cuDNN/driver/transformers versions into the manifest, commit; the
  critic's acceptance of that commit is the **pin row**;
- fetch the Coconut PDF (arXiv 2412.06769) and the same-curriculum pause source, record their sha256 and URLs, extract
  the text of the results table(s) and the training details and commit the excerpt as `results/prep/coconut_pdf_excerpt.txt`
  (text only, small) with the sha256 of the PDF it came from; the critic's row that freezes the 4.2 targets from that
  file is the **PDF row**;
- the GPU calibration, which **replaces D4 and D8**: tokens/s and peak memory at batch 32 x 4 accumulation, seq 320,
  in fp32 (TF32 off, A3.2) and bf16 autocast; the one-batch parity on real GPT-2 124M weights with one real ProsQA
  batch (16 examples, stages 3 and 6): CUDA fp32 (TF32 off) vs CPU float64, max abs and relative error of logits and
  loss, written to `results/calibration_cuda.json` and `results/parity_real_cuda.json` and into CHANGELOG.md;
- `pytest common/latent/tests` on the VM image (the number of passed tests and the image id recorded);
- the smoke config (`configs/smoke_prosqa.json`: 2,000 examples, 2 epochs, 100 validation examples) for `coconut` and
  `cot`, including **one forced resume** from `ckpt_last.pt` in the bucket (stop the process after epoch 1, restart,
  check the run continues at epoch 2 with the logged resume), and the bucket round trip of `log.jsonl` and the three
  checkpoint files;
- reading the smoke generations by eye for answer-shaped outputs under the reference extraction (section 0 item 6).
Forbidden on the prep VM and everywhere until the oracle-start conditions are met: any run beyond the smoke caps, any
screen, any test-split read, any accuracy comparison.

**Oracle start.** The oracle may start on the GPUs as soon as **all** of these are logged: the pin row; the PDF row;
the calibration numbers in CHANGELOG.md; the smoke **completed end to end** (train, per-epoch validation generation,
checkpoints to the bucket, one resume from the bucket, `log.jsonl` rows) with answer-shaped outputs on inspection;
the suite green on the image; `env.json` recorded. **The smoke's accuracy is noise by construction (2,000 examples,
2 epochs) and is not a gate; the gate is completion.** Yes: once those rows exist the oracle starts even if the smoke's
accuracy is 0.

**A3.2 Precision and the environment rule between GPU types.**
- Oracle precision on GPU: **fp32 end to end with TF32 disabled**: `torch.backends.cuda.matmul.allow_tf32 = False`,
  `torch.backends.cudnn.allow_tf32 = False`, `torch.set_float32_matmul_precision("highest")`, set in `train.py` before
  the model is built, asserted at the start of every epoch, and the three values written into every `log.jsonl` row
  and `env.json` (**new defect D18**: `train.py` currently sets only `cudnn.deterministic` and `cudnn.benchmark`). Micro-
  batch 32 x grad_accum 4 = 128 on one GPU, as the configs say. Screens and confirmations: fp32 master weights with
  bf16 autocast, the screen bar in the same precision (7.8 unchanged).
- Environment rule, GPU types: every Phase 1 number is now GPU, so 7.8's Mac-vs-GPU pair is void and 6.4's
  determinism rerun moves to the GPU (ProsQA `cot` seed 0 rerun once on the same GPU type, before the ProsQA read; it
  may run on any project in parallel and is not on the critical path; if it has not finished when the other runs
  have, the read waits for it, because sigma_task needs it). Between **A100-80GB (a2-ultragpu-1g, budget projects)
  and A100-40GB (a2-highgpu-1g, main project)** a same-seed training pair is **not required**, under three conditions,
  all testable: (i) one software environment on every VM (same image; Python, torch, CUDA, cuDNN, driver, transformers
  recorded in each run's `env.json` and equal to the prep VM's; a run whose `env.json` differs is "env-mismatched" and
  joins no pooled cell until a same-seed pair passes); (ii) the one-batch parity of A3.1 is run on **both** GPU types
  from the same checkpoint and agrees: loss within 1e-5 relative, gradient norm within 1e-4 relative, in fp32 with
  TF32 off; (iii) the GPU type is recorded per run and printed per seed in every table. If (ii) fails, a same-seed
  ProsQA `cot` seed-0 pair across the two types must agree within sigma_task before any cell pools seeds from both.
  The fp32-vs-bf16 pair of 7.8 stays required before any fp32 oracle number is compared with a bf16 screen number
  (not needed inside the window: the screens compare against their own bf16 bar).

**A3.3 Scope of the window (author-directed, PLAN.md A1 item 1; the registered rules' verdicts are still printed).**
- **ProntoQA: UNTESTED** in the window (dropped by the author as at-ceiling). Section 4.3's "K1 passes for Phase 1 iff
  ProsQA and ProntoQA" therefore cannot pass as registered; it is printed as "K1 (registered): ProntoQA UNTESTED".
  **K1 for the window = K1 on ProsQA**, plus GSM8k-Aug single seed when the GSM chain finishes.
- ProsQA arms in the window: `cot` and `coconut` at seeds 0, 1, 2; `no_cot` and `pause` at **seed 0 only**, labelled
  "single seed" in the K1 table, with sigma_cell = max(1.0, published spread) as 4.1 says for single-seed cells;
  `coconut_no_curriculum` is **UNTESTED** in the window (not in A1's list). sigma_task on ProsQA = max(1.0, seed SD of
  the three GPU `cot` seeds), plus the determinism term of 6.4.
- **Provisional and final K1.** A provisional K1 on ProsQA seed 0 (`cot`, `coconut`, `no_cot`, `pause`; single-seed
  tolerance) may be read as soon as those four runs finish, and may gate the **start** of the screens (speed). The
  final K1 (3 seeds of `cot` and `coconut`) is required before any confirmation seed runs and before any WIN or tie
  sentence is written. A provisional pass that the final K1 reverses stops the screens' confirmations and is logged.
- **Tuned-CoT bar in the window:** two extra LR trials on ProsQA, lr 3e-5 and 3e-4, constant LR, 50 epochs, seed 0,
  best-valid checkpoint, alongside the oracle `cot` at 1e-4: a 3-trial LR grid. The schedule axis of 3.4 is **UNTESTED**.
  The bar value = the best trial's seed 0, labelled **"bar: single seed, LR-tuned (3 trials)"**; if the best LR is not
  1e-4 and time permits, seeds 1 and 2 at that LR make it a 3-seed bar (then unlabelled). Trial parity for Phase 2 is
  therefore **3**.
- **Screen bars (Phase 2, per screen dataset):** `cot` at lr {3e-5, 1e-4, 3e-4}, seed 0, best on valid; latent arms
  **one trial at 1e-4** (the disadvantaged side; conservative for the arm). Every screen sentence carries "bar
  LR-tuned 3 trials; arm 1 trial". A latent arm may be re-screened at another LR only by a logged row, and then the
  bar's count rises to match before any comparison.
- Section 12's Mac budget (400 Mac-h) is void. GSM8k-Aug runs in the window on budget (A1 and A3.5), seed 0.
- Everything else in the registration stands: the position table, the K1 tolerance, K2-K4, HEADSTART, the readout
  requirement, the audit checklist, the per-card rows (A3.4).

**A3.4 Cards registered before their screens (section 7.11): C1b, C2, C3 and the combined arm C2+C3.** Common to all:
memory-kind label **appended-KV scratchpad** (Coconut base; C1b's band recurrence still appends positions to the
band's cache), so the fixed-width separation of PLAN.md 1.4 does not bind them; novelty label **KNOWN (pending)**
for each, because the three closest works were read from snippets (7.11: the more-known label until the critic checks
the full texts; the prep VM may fetch them; the operator may request COMBINATION for C2 and C3 in a row with the full-text
readings); screen at 124M, seed 0, bf16 autocast, lr 1e-4, one trial, on ProsQA-hard and GSM-50k (A3.4.5); the
standing controls on each screen dataset: tuned bar (3 LRs), `no_cot`, full-depth `coconut`, pause-same-curriculum,
scrambled-at-inference, zero-latent, step sweep {0, 1, 2, 4, 8}, plain depth = GPT-2 355M `no_cot` (same data and
epochs); **K2 at screen applies to every latent arm** (an arm within 1 sigma_screen of its scrambled control, or whose
scrambled interval includes 0, is killed as a latent claim whatever its accuracy); the too-good triggers of 7.10
(ProsQA-hard above 99.0%; more than 10 points over the bar on GSM-50k) block announcement; the 0.5 B / B fade
instrument of 7.6 is printed. **Screen decisions use valid only** (ProsQA-hard valid 300; GSM valid 500); the screen
checkpoints' test read happens once at the end of the screen round, for reporting, with no decision attached.
**sigma_screen(task)** = max(1.0 point, 100 x sqrt(p (1 - p) / n_valid) at the bar's valid accuracy p, and the ProsQA
`cot` 3-seed SD once it exists). Kill lines are evaluated at the end of the registered screen budget on valid; an arm
killed on one task and passing on the other is confirmed on the passing task only. **Confirmation in the window:**
seeds 1 and 2 on survivors (A1), seed 3 if time permits; the verdict is written on seeds 1-3 when seed 3 ran (clean,
6.3), otherwise on seeds 0-2 **labelled "selection seed included"**, with the seeds 1-2 estimate printed beside it (two
seeds, no verdict); the 2 B fade run of 7.6 is UNTESTED in the window, so no label above "WIN (1x budget; fade
UNTESTED)" can be written. Predictions for every card are the critic's, made before any run.

- **C1b (partial-depth latents; band [2, 10) for k = 8 and [4, 8) for k = 4 of L = 12).** Family KNOWN (Huginn,
  retrofitted band recurrence, Mixture-of-Recursions). Claim: a cost saving per latent step with accuracy held.
  Accounting: `layer_fraction` = (b - a) / L applied to the 2 N_ne term and the attention term of the band's layers
  only at latent positions; head 0; sequential passes unchanged; the worked example of 5.1 at 8/12 and 4/12 is a test;
  the **implementation must skip the non-band blocks** (not merely mask them) and the measured wall-clock per latent
  step must show the fraction, else the saving is "nominal" and K3 is judged on measured cost. Null: full-depth
  `coconut` at the same curriculum (the uncut model), plus K2 (scrambled band) as above. Kill: k = 8 below full-depth
  `coconut` by more than 2 sigma_screen on both tasks, or K2 fires on k = 8 on both tasks. Pass to confirmation: k = 8
  within sigma_screen of full-depth `coconut` on at least one task **and** K2 not fired there; k = 4 is screened and
  judged by the same lines independently. Critic's predictions: ProsQA-hard k = 8 minus full depth = -1 point (80%
  interval [-4, +2]), k = 4: -5 [-10, 0]; GSM-50k k = 8: -2 [-5, +1], k = 4: -6 [-12, -1]; P(k = 8 passes its screen
  on at least one task) = 0.55; P(K2 fires on k = 8 at screen, same as on full-depth Coconut) = 0.4; P(the measured
  wall-clock saving per step is within 10% of 8/12) = 0.5.
- **C2 (verifiable-state supervision; decoder g = 2-layer MLP 768 GELU on the fed-back latent, lambda = 0.1 fixed,
  dropped at inference; targets: ProsQA frontier F_t from the record's edges, GSM step-t result; steps beyond the
  stage's latents have no target).** Family KNOWN (pending; proposed COMBINATION: SIM-CoT, CODI, Chen et al. 2026).
  **Correction to the card's leak answer, from the data:** the last step's result **equals the final answer in 95.0% of
  GSM8k-Aug train records** (critic's count: 366,447 of 385,620; 80.6% of records have 3 or fewer steps, so with 3
  stages x c_thought 2 the last supervised latent carries the answer for most of them), and the ProsQA frontier at the
  last step contains the target concept. So state targets do contain the answer in most records. This is allowed: it
  is training-set information, exactly as the CoT arm's targets are; but the card's sentence "no answer token enters a
  state target" is false and is replaced by: targets are a pure function of the record's steps and edges, computed
  from the training split only, never from valid or test; the decoder is absent from the inference graph (test); the
  fraction of records whose last state target equals the answer is reported per task. Attribution then rests on the
  null: **C2-shuffled** (same decoder, lambda, compute; targets from a fixed seeded derangement of the training set
  within the same step-count class), which receives the same kind of answer-shaped signal with the wrong content.
  Kill: C2 within sigma_screen of C2-shuffled on both tasks, or below full-depth `coconut` by more than 2 sigma_screen
  on both, or K2 fires on both. Pass: C2 above C2-shuffled by more than sigma_screen on at least one task with K2 not
  fired there. Readout: the decoder is R3; coverage and load-bearing fraction per 7.9 on held-out test examples, with
  F1 patching and F3 edits; the positive control of 7.9 is still required before any "explained" label. Critic's
  predictions: ProsQA-hard C2 minus `coconut` = 0 [-3, +2], C2 minus C2-shuffled = +1 [-2, +4]; GSM-50k C2 minus
  `coconut` = +2 [-2, +6], C2 minus C2-shuffled = +2 [-1, +5]; scrambled drop of C2 on GSM-50k 8 points [2, 20];
  P(C2 passes its screen) = 0.45; P(frontier probe F1 > 0.9 on ProsQA-hard held-out) = 0.7; P(load-bearing fraction
  > 0.5) = 0.3; P(C2 alone within a documented tie of the bar at matched FLOPs on GSM-50k) = 0.05.
- **C3 (latent-dependence hinge; second pass with a scrambled bank, loss = CE_true + beta max(0, m - (CE_scr -
  CE_true)), beta = 1, m = 0.2 nat per answer token, banks detached; training-heavy label; inference cost = Coconut).**
  Family KNOWN (pending; proposed COMBINATION). **Correction to the card's null:** "C3-placebo", the hinge applied to a
  random non-answer position of the question, has **zero hinge gradient**: every question position precedes the latents,
  so by causality its cross-entropy is identical under true and scrambled latents, CE_scr - CE_true = 0 there, and the
  hinge is the constant m. It is Coconut plus a wasted pass, not a content-matched null, and its stated attribution
  ("the gradient of the hinge is noise of the same magnitude") does not hold. **Registered null: C3-foreign**: the same
  hinge between two independent scrambled banks A and B, loss = CE_true + beta max(0, m - (CE_B - CE_A)), same beta,
  m, detachment and per-batch derangements (A and B drawn with different seeds, A != B rowwise asserted). It keeps the
  extra passes, the hinge magnitude and the gradient into the latent path, and destroys the own-vs-foreign information
  (A and B are exchangeable, so the expected push has no direction). It costs a third pass per batch (the null is more
  training-heavy than the arm, which is the conservative side). The standing scrambled-in-training arm is the second
  reference. Attribution: C3 minus C3-foreign = the value of forcing dependence on the model's own latents; C3-foreign
  minus `coconut` = the effect of the extra passes and the hinge as a regulariser. The operator may propose the sign-flip
  variant (hinge direction randomised per example) instead, by a row before the screen; one null, not both. Kill: the
  scrambled-at-inference drop of C3 under 5 points on both tasks after training, or accuracy below full-depth `coconut`
  by more than 2 sigma_screen on both, or (new) C3 within sigma_screen of C3-foreign on the scrambled drop on both tasks.
  Pass: scrambled drop above 10 points on at least one task with accuracy within sigma_screen of `coconut` or better
  there, and the drop exceeding C3-foreign's by more than sigma_screen. The card's degenerate-solution note stands: a
  model whose pause or zero-latent accuracy collapses is reported as such, not killed. The training derangement (per
  batch, reseeded per epoch) and the evaluation derangement (whole set, seed 0) must use different seeds (test).
  Critic's predictions: ProsQA-hard scrambled drop of C3 above 20 points: P = 0.5, accuracy change -2 [-8, +1];
  GSM-50k drop above 15: P = 0.45, accuracy +1 [-4, +4]; P(C3 passes its screen) = 0.45; P(C3-foreign's drop also
  exceeds 10 on GSM-50k, i.e. the hinge works as a regulariser without content) = 0.25.
- **C2+C3 (both mechanisms on one Coconut model).** Registered as an arm with the two single-mechanism arms and
  `coconut` as its attribution set; no combined null at screen (the interaction's attribution is UNTESTED at screen
  and, if the arm survives, gets its null at confirmation by a row). Kill and pass: C3's lines (the dependence claim)
  and C2's lines against C2-shuffled both apply. Critic's predictions on GSM-50k: C2+C3 minus `coconut` = +3 [-2, +9];
  P(beats `coconut` by 2 sigma_screen) = 0.4; P(documented tie or better against the bar at matched FLOPs with emitted
  tokens <= 0.2x) = 0.15 (the operator gives 0.3); P(dead by K2) = 0.2; P(the sequential-passes criterion of 7.2 also
  holds, so no "billing claim only" label) = 0.1 on GSM-50k (CoT emits about 25 tokens there; 6 latents plus about 2
  answer tokens is 9 passes vs 26).
- **A3.4.5 Screen data (definitions registered now; the hashes are frozen at the commit of `code/make_screen_data.py`
  that writes them into the manifest, before any screen run, and the critic logs that commit).**
  - **ProsQA-hard:** `latent.prosqa_gen.generate` records with proof depth drawn **uniformly from {5, 6, 7} per
    record**, distractor edges = **2 x the generator's default draw** for that record (the default is
    `clip(round(gauss(37 - (depth - 3), 6)), 15, 63) - depth` non-proof edges; the doubled value is passed as
    `n_distractor_edges`, and a record the generator cannot place is redrawn, with the redraw count reported), one
    seeded stream (seed 0) producing train, valid, test in that order: **10,000 / 300 / 500**; no duplicate question
    across splits (asserted); realised means of nodes, edges, words and depth per split recorded in the manifest;
    13-gram rates vs its own train and vs the ProsQA train reported. Recipe for the latent arms: c_thought 1,
    epochs_per_stage 3, max_latent_stage 7, 30 epochs, lr 1e-4, AdamW wd 0.01, batch 128, reset every epoch, bf16
    autocast; final regime by D16: stage >= min(8, 7) = 7, **selection pool epochs 21-29** (0-indexed), evaluated with 7
    latents; `cot` and `no_cot` 30 epochs, no reset; plain depth 355M `no_cot` 30 epochs on the same files.
  - **GSM-50k:** the training subset = the first 50,000 records of a **seed-0 permutation of `gsm_train.json` in file
    order** (the permutation's RNG, numpy `default_rng(0)` or `random.Random(0)`, is named in `make_screen_data.py`,
    and the sha256 of the sorted index list is in the manifest); valid = the full 500, test = the full 1,319 as in the
    oracle; `cot` 25 epochs at each of the three LRs (bar) and `no_cot` 25 epochs; every latent arm starts from the
    **lr 1e-4 `cot` run's `ckpt_best_final.pt`** (the reference recipe; never from the best-LR trial), `resume 3`,
    c_thought 2, epochs_per_stage 3, max_latent_stage 3, 25 epochs, reset every epoch, **selection pool epochs 12-24**
    (A2 item 3); plain depth 355M `no_cot` 25 epochs on the same 50,000. The subset is a subset of the oracle train, so
    its 13-gram overlap with the GSM test is at most the oracle's 7 items; reported.
  - Predictions for the screen datasets (critic, before any run): ProsQA-hard bar (best of 3 LRs, seed 0) 60 [40, 80];
    `no_cot` 55; full-depth `coconut` 70 [50, 90]; pause within 3 of `coconut`; P(ProsQA-hard is labelled
    "CoT-unnecessary" because pause or `no_cot` ties the bar) = 0.5. GSM-50k bar 30 [24, 36]; `no_cot` 12; `coconut`
    from `cot` 22 [15, 30]; P(K2 fires on full-depth `coconut` on GSM-50k) = 0.4. P(at least one card passes its
    screen on at least one task) = 0.6; P(the 124M read is positive under A1's definition within the window) = 0.15.

**A3.5 Budget for the window (ceilings, not permissions; budget are compute at the same rates).**
- **Ledger:** `governance/run_log.jsonl`, the nightshift format (`experiments/2026-10-13-nightshift-sleep/governance/cost_run_log.jsonl`:
  hash-chained `admit` / `close` events with id, project, machine, gpu, provisioning, `expected_usd_per_hour`, `close_usd_per_hour`,
  `max_hours`, `cap_usd`, `project_cap_usd`, code commit, `prev_sha256`, `sha256`). **Rates as nightshift `cost.py`:** the ceiling per
  hour is the on-demand price of the whole machine x 1.10; admissions reserve ceiling x (max run hours + the teardown allowance); a
  closed spot job is charged min(ceiling, recorded spot price x 1.25) per measured hour until reconciled with the bill. With nightshift's
  `prices.json` (retrieved 13 Oct 2026; Setup, idle, failures, preemptions and the prep VM all count. No VM is admitted if committed
  compute plus its worst case exceeds the cap.
- The author's later words "use our existing compute, and go" and "be as fast as you can ... I need a significant result" contain no number; The
  per-project caps **AcctA 290 / AcctB 230 / AcctC 220 / main 45** are admitted as per-project ceilings (they sum to 785; the window total binds
  first). **No cash rental** (A1). The priority list of section 12 is replaced for the window by CHANGELOG's launch order: prep, oracle (GSM
  chain on AcctA; ProsQA seed 0 on AcctB; seeds 1-2 on AcctC; the two LR trials on main), screens, confirmations, readout, report; the controls
  are never cut before the arms.
- The critic's estimate (arithmetic, not measured; the operator re-estimates from the calibration before the first admission):
  oracle ProsQA 8 runs x about 2.5-3 A100-h = 20-24; GSM chain 2 runs about 25; two LR trials about 6; determinism rerun about
  3; screens about 12 arms x 2 tasks x 0.5-0.9 A100-h = 12-22 plus the 355M plain-depth arms about 6; confirmations about
  20-40; what is not run is UNTESTED, never estimated.

**A3.6 Preemption, resume and sequential jobs (new rule; new defects D17 and D19).**
- Global GPU quota is 1 per budget project, so jobs run sequentially per project in the launch order; a queue change
  never moves a read earlier.
- A preempted VM resumes its run from the bucket's `ckpt_last.pt`. **A resumed run is the same run, not a rerun**,
  iff all of: the same config hash and seed; the resume starts at an epoch boundary from its own `ckpt_last.pt`
  (sha256 checked against `ckpt_state.json`; a partial epoch is redone from the boundary); the data order and dropout
  stream of every epoch are functions of (seed, epoch) alone (as `train.py` does); the optimizer state is restored for
  arms without a per-epoch reset (**D17:** `cot` and `no_cot` run with `reset_optimizer: False`, and `_save` writes the
  model state only, so a resume today gives them a fresh AdamW, a recipe change; fix: save `optimizer.state_dict()`
  beside the model in `ckpt_last.pt` and restore it; latent arms rebuild AdamW every epoch, so a boundary resume is
  exact for them); the best trackers and `log.jsonl` continue from `ckpt_state.json`; and every resume is logged with
  the epoch and the VM id (**D19:** the resume list and count per run, printed in `runs_index.csv`; `ckpt_last.pt`,
  the two best files, `ckpt_state.json` and `log.jsonl` synced to the bucket after every epoch). A resumed run is
  never a same-seed pair and never enters the determinism check. A run resumed more than 3 times is flagged
  "fragmented" (reported, not void). GPU nondeterminism makes a resumed run a fresh draw of the same seed, which is
  inside what 6.4's determinism term measures.
- Defects open at A3: D4 and D8 (now the GPU calibration of A3.1), D9 pin row, PDF row, D17, D18, D19, and the
  make_screen_data hash commit. The oracle needs D18 closed (TF32 flags asserted) and D17 closed before the first
  `cot` run; D19 before the first spot VM.

**Nothing in A3 is a critic loosening.** The loosenings (ProntoQA and `coconut_no_curriculum` UNTESTED, single-seed
`no_cot` and `pause`, a 3-trial LR-only bar, the selection seed inside a window confirmation, no 2 B fade run) are the
author's A1 and are labelled wherever they touch a number; the registered verdicts are printed beside them.

**A4 (2026-10-18 23:27 UTC, before the oracle launch; no calibration number, no accuracy number; `train.py` implements D18 as
strict fp32 (TF32 off, "highest"). The operator's arithmetic, checked by the critic: GPT-2 124M training costs about 0.8 GFLOPs per
token-pass (6 (N_ne + V d) plus attention at seq 320); In the pinned torch build `torch.backends.cuda.matmul.allow_tf32` defaults
to False and `float32_matmul_precision` to "highest", and the reference sets neither, so **the published numbers were most likely
produced with strict fp32 matmuls** (the cuDNN TF32 default is True but GPT-2 has no convolutions).

1. **Accepted: oracle precision = fp32 storage and accumulation with TF32 tensor-core matmuls ON, label "fp32-tf32"**
   (`torch.backends.cuda.matmul.allow_tf32 = True`, `torch.backends.cudnn.allow_tf32 = True`,
   `torch.set_float32_matmul_precision("high")`), for every oracle run, the determinism rerun, the two LR trials and
   the GSM chain, and for each run's own evaluation and generation. It is a **registered deviation from the
   reference's probable strict fp32**: every oracle cell carries "fp32-tf32" and the oracle table's footnote states
   the reference's probable precision. *Why accepted: TF32 keeps fp32 storage and fp32 accumulation and rounds only the
   matmul inputs to a 10-bit mantissa; the operator's alternative within the window (one seed per ProsQA arm and no bar) would
   remove the seed structure that section 6 and method lesson 2 rest on, which is the worse loss.*
2. **Condition (tightening in exchange): a same-seed precision pair, measured, not assumed.** ProsQA `coconut` seed 0
   is run twice on the same GPU type with the identical config: once fp32-tf32 (the oracle run itself) and once strict
   fp32 (TF32 off, "highest"), the strict run labelled "precision pair" and entering no cell. d_prec = |acc_test(strict)
   - acc_test(tf32)| at each run's own selected checkpoint; the valid curves are overlaid in the report.
   - d_prec <= sigma_task: the fp32-tf32 cells are **"fp32-equivalent"** and the K1 verdicts are final.
   - d_prec > sigma_task: every oracle cell keeps "fp32-tf32, precision pair failed"; K1 verdicts stay provisional; a
     K1 miss on any cell makes a strict-fp32 rerun of that cell the first item of debug round 1 (4.4), and a K1 pass is
     reported as "pass under fp32-tf32" and cannot be called a reproduction of the published precision.
   - the operator schedules it on whichever project is free). The **provisional** K1 of A3.3 may be read and the screens may start
     before it finishes; the **final** K1, and therefore any confirmation seed and any WIN or tie sentence, waits for it. The pair is
     logged like the determinism rerun (6.4) and, with it, separates nondeterminism (same precision, same seed: d_nd) from precision
     (d_prec); d_prec is not added to sigma.
3. **Accepted as a diagnostic, not as the test:** the prep VM's one-batch parity reports strict fp32 vs fp32-tf32
   (and vs CPU float64) relative differences of loss and gradient norm on one real ProsQA batch, into CHANGELOG.md.
   The operator's point (c), "the 3-seed spread covers the rest", is **not accepted as a reason**: a seed spread is not
   evidence about a precision change; item 2 is.
4. **D18 restated:** the three flags are set explicitly at process start, asserted at the start of every epoch and
   written into every `log.jsonl` row and `env.json`. Registered values: oracle, determinism rerun, LR trials and GSM
   chain: `matmul.allow_tf32 True`, `cudnn.allow_tf32 True`, `float32_matmul_precision "high"`; the precision pair's
   strict run: `False`, `False`, `"highest"`; Phase 2 screens and confirmations: bf16 autocast with fp32 master weights
   on top of the oracle's TF32 setting (the bf16 matmuls are bf16 regardless), bar and arms alike. A run whose logged
   flags differ from its registered values is "precision-mismatched" and enters no cell. The label "fp32" in section
   3.2 and A3.2 now reads "fp32-tf32" for every oracle run; A3.2's "TF32 disabled" is superseded by this amendment for
   the oracle and kept for the strict run of the pair.
5. Nothing else changes: 7.8's fp32-vs-bf16 pair is still required before an oracle number is compared with a screen
   number (the strict-vs-TF32 pair does not substitute for it); the K1 tolerance, sigma rules, caps and launch order
   stand. The operator re-estimates the window from the measured fp32-tf32 tokens/s before the first oracle admission.

**A5 (2026-10-18 23:34 UTC, before any screen run; screen-data hashes frozen, two definitional deviations registered, card
corrections from the implementation, D17-D19 closed).** Seen when written: no accuracy number; the prep VM running;
commit `rev43` (mechanisms, screen data, configs, run_eval, read_oracle) and `rev378` (changelog); the working
tree holding the operator's uncommitted A4 edits. Everything below was verified by the critic against the committed
manifest, the local files and the code, not taken from the message.

1. **Screen-data hashes frozen at `rev43`** (`configs/data_manifest.json`, "screen" block; the local files hash
   identically): `prosqa_hard_train` `65165e1dc841faea13b0036d461c4280e59f3072e6aede76e9f733a0eed58ce7` (10,000);
   `prosqa_hard_valid` `2d7123f2cc5e97639cf8cce6eb466b61ec8db09126502c46ddf20b5317075cae` (300); `prosqa_hard_test`
   `85a26a32ae6f01bc0961451221292f218e87dd7ed1098426387dd6c5c28aafbf` (500); `gsm50k_train`
   `00480524d05e67582346a08d78fc42826f306a8db74fa042a07f21d6594d3a86` (50,000; the records equal
   `gsm_train.json[perm[i]]` for `perm = numpy.random.default_rng(0).permutation(385,620)`, in permutation order,
   checked record by record; sorted-index sha256 under the manifest's newline rule
   `1bba60897ef06bbbbc1082d89619e15266faae3f800196c8ae10da7ed8218de8`, recomputed). Generator and script sha256 are in
   the manifest. Critic's recount: depth 5 / 6 / 7 = 3,397 / 3,323 / 3,280 (valid 101 / 93 / 106; test 163 / 170 /
   167); 300 words per question; 61.7 edges and 22.8 symbols per record; duplicate questions within and across splits
   0; GSM-50k 2.59 steps per record. 13-gram rates (critic's recomputation, equal to the manifest): ProsQA-hard vs its
   own train valid 111/300 = 37.0%, test 180/500 = 36.0% (secondary 290/300, 489/500); vs the ProsQA train, test
   197/500 = 39.4%; **GSM-50k vs the GSM test 3/1,319 = 0.23%** (secondary 6), under the 1% block; the 3 items are
   listed by idx as 1.3 and A2 require. The template-overlap sentence of A2 item 4 applies to ProsQA-hard.
2. **Deviation registered: the context cap.** At 2x the default distractor draw, ProsQA-hard questions average about
   600 GPT-2 tokens and 16 draws exceeded GPT-2's 1,024 positions under the curriculum plus 128 new tokens. Rule, as
   implemented and frozen with the hashes: a record whose question + "\n" exceeds **887 GPT-2 tokens** (= 1,024 - 2
   markers - 7 latents - 128 new tokens; the repo's vendored GPT-2 BPE at
   `common/babylm/recipes/sprint1_port/gpt2_tokenizer`, hashed in the manifest) is redrawn at the same depth (81 length
   redraws, 3 unplaceable). Critic's check with that tokenizer: the longest question is exactly 887 tokens, none over;
   the longest CoT training sequence is 970 tokens. The cap truncates the long tail of distractor counts (requested
   mean 55.65, realised 55.7: negligible) and binds any future regeneration of ProsQA-hard identically.
3. **Clarification, not a deviation: "twice the distractors".** A3.4.5 said 2 x the generator's default draw, which is
   what was built. Relative to the released ProsQA train (critic's count: 36.2 edges, about 32.4 of them off the proof
   path, 180 words) ProsQA-hard has 1.70x the edges (61.7), 1.72x the distractor edges (55.7) and 1.67x the words
   (300). **Every table names it "ProsQA-hard: depth 5-7, 1.7x edges, 300-word questions (released: depth 3-6, 36
   edges, 180 words)"**, so nobody reads it as "2x harder".
4. **C3 cost corrected (card text superseded):** the hinge's gradient flows through the model's processing of the
   bank, so each extra pass is a forward and a backward: **C3 = 2.0x Coconut's training FLOPs per batch, C3-foreign =
   3.0x**, not 1.5x; `flops_extra_passes` is logged per epoch and charged to training FLOPs; both arms carry
   "training-heavy" (5.5 d); the null heavier than the arm stays the conservative side.
5. **C2 head corrected (card text superseded):** the decoder's inputs are the fed-back latent, a learned step-index
   embedding and, for ProsQA, the **detached** mean token embeddings of the record's symbol names as the scoring basis
   (multi-label sigmoid over the record's symbols, the root's slot masked); the card's "sees only the latent vector and
   the step index" is replaced by this sentence. Accepted: every symbol appears in the question, so no information
   beyond the input enters, and the detachment leaves no gradient path (unit-tested). GSM results longer than 4 tokens
   are masked (518 of 114,956 supervised steps, 0.45%; reported). **The "last supervised target contains the answer"
   fraction is 1.00 on ProsQA-hard** (every proof depth <= max_latent_stage 7) **and 0.770 on GSM-50k** at 3 latent steps
   (consistent with the critic's 80.6% x 95.0% on the full set); each C2 run writes it to its `config.json`, and the
   report prints it beside every C2 number. Consequence (reporting requirement, no rule change): on ProsQA-hard the
   decoder is trained to emit the answer symbol at the last step by construction, so **R3 coverage is reported for
   steps 1 to depth - 1 separately from the final step**, and "explained" (7.9) is judged on the F1/F3 interventions as
   registered, never on the decoder's own last-step accuracy.
6. **D17, D18, D19 closed in `train.py` at `rev43`** (critic read the code): `optimizer_last.pth` saved beside the
   latest checkpoint and restored, resume bit-identical for `cot`; `REGISTERED_FLAGS` for "fp32" (strict: False /
   False / "highest") and "fp32-tf32" (True / True / "high"), set at start and asserted every epoch, logged per row (A4
   item 4 implemented); resumes listed per run with the VM, "fragmented" beyond 3, `runs_index.csv` row, `ckpt_last.pt`
   sha256 checked against `ckpt_state.json`.
7. **Test suite.** The operator reports 1030 passed, 1 skipped at `rev43`. The critic's run on the working tree, which
   already holds the uncommitted A4 edits (oracle configs switched to "fp32-tf32"; a new `prosqa_coconut_strict.json`
   whose `name` field `prosqa_coconut_s0_strict` differs from its filename), gave 1029 passed, **1 failed**
   (`test_experiment_configs_load_and_mirror_the_reference`, on that name field), 1 skipped; consistent with the
   committed count, and in uncommitted work. **No oracle or screen admission cites a commit whose suite is not green;
   the launch commit's suite result is logged with its first admission.**
8. **Screen launch conditions, restated:** hashes frozen (this amendment); the suite green at the launch commit; the
   Fable adversarial review of the mechanisms (`docs/review_phase2_code.md`) delivered and its residual-risk list
   logged by the critic with any rule it forces; the provisional K1 of A3.3; the screen bar's three `cot` trials
   admitted no later than the first latent screen arm on the same dataset.

**A6 (2026-10-19 00:11 UTC, before any screen admission; the residual risks of the adversarial code review made rules: K2 independence for
hinge-trained arms, the C1b wall-clock reference, checkpoint-start pins, the reference-cost column, the accessor freeze, numerics and
hinge diagnostics, screen-data labels, screen admission).** Authority: A3.4 and A5 item 8 (the review's residual-risk list is logged
with any rule it forces before any screen admission). Source: `docs/review_phase2_code.md` at commit `rev44` (verdict "safe to
screen" with the twelve items of its section 5; its F1 fix and 14 tests in the same commit) and the operator's follow-up `rev45`
(launcher pins, accessor freeze). Seen when written: no accuracy number of any kind; `results/` holds no read; the second prep VM
admitted and running (nothing from it is read here); the critic's rows of 2026-10-19T00:11Z (the review acknowledged; the twelve
dispositions). Checked by the critic at `rev45`: the 14 review tests (14 passed, 3.70 s), the full suite (1044 passed, 1 skipped,
72.5 s), the launcher tests (40 passed), the F1 diff (four lines in `LatentLM.__init__`), and every claim below at the cited code.
Every rule here defines how a registered number is read, adds a label, a reference, a refusal, a column or a freeze; none moves a
kill line, a seed rule, a tolerance or a cap.

**A6.1 K2 is not independent for arms trained with the dependence hinge (C3, C2+C3; review F3).**
- Fact (`aux_losses.hinge`, `train._aux_step`): the hinge maximises the cross-entropy of the answer under foreign latents minus
  under the model's own latents, which is the quantity K2's Delta_scr measures at inference. A model satisfies it with a
  fingerprint (one latent carries a check of the question; the answer head emits garbage on mismatch) and no reasoning state;
  such a model has a large scrambled drop, and the pause-mode and zero-latent reads of that checkpoint feed a third distribution
  the check never saw, so their outcome is arbitrary.
- Rules, for every arm whose config has `dep_loss: true` (C3, C2+C3, and any later card that trains against scrambled latents):
  1. The K2 cell is printed **"K2 not independent"** at the screen and at confirmation. A fired K2 still kills (the hinge failed at
     its one job; the card's kill line). A non-fired K2 is not evidence that the latents carry state.
  2. The C3 minus C3-foreign difference on the scrambled drop is reported as **"dependence on own latents (content or
     fingerprint)"**.
  3. The pause-mode and zero-latent reads of such a checkpoint are diagnostics (section 10) and never decide for these arms. The
     comparison against the separately trained pause-same-curriculum arm (CURRICULUM-ONLY, 7.5) is unaffected.
  4. The non-circular reads are: (A) the **accuracy leg** (accuracy vs full-depth `coconut`, vs C3-foreign, vs the bar), as the
     card's lines require; (B) a **directional intervention read**. On ProsQA-hard: the F1 counterfactual patch of the probe
     toolkit (`probe.tasks.PAIR_RULES["prosqa"]`: `target_swap`, the default, and `root_swap`, both printed), scored as 7.9's
     load-bearing fraction (the patched answer moves to the pair's predicted answer; paired example-bootstrap interval excluding
     0), with the **garbage fraction** (patched answers that are neither the own nor the predicted answer) printed beside it. On
     GSM-50k, where `PAIR_RULES["gsm"]` is empty: the **donor-answer read**, under the registered scrambled derangement (whole
     set, seed 0): among examples whose donor's last supervised step result equals the donor's answer (the 0.770 class of A5
     item 5), the rate at which the generated answer equals the donor's answer, against the same rate under own latents, paired
     bootstrap interval excluding 0. The operator may register a minimal-pair rule for GSM by a row before the read; it then replaces
     the donor-answer read (one directional read per task). At the screen the directional read uses the valid split and the
     screen checkpoint; at confirmation the test split, as 7.9 says.
  5. A large scrambled drop whose directional read has an interval including 0, or whose garbage fraction exceeds its directional
     fraction, is labelled **"drop without direction (fingerprint-consistent)"**, and K2 counts as fired for that arm's
     latent-reasoning claim on that task.
  6. If the directional read is absent at the screen read, the outcome is written "drop leg not independent; directional read
     pending"; the card's pass and kill lines apply as registered, and: (a) the directional read on the screen checkpoint is
     admitted (an eval job, in the ledger) no later than the arm's first confirmation seed on that task; (b) no K2 sentence, no
     "latents carry state" wording and no verdict naming the arm is written until the directional read exists on the checkpoint
     it refers to; (c) if it is never produced for a task, the arm's K2 on that task is **"not independent; direction UNTESTED"**
     and the arm cannot be the positive read of A1 on that task.
  7. The readout labels of 7.9 are unchanged and still need the positive control; the directional read of this section answers
     K2 independence, a separate question.

**A6.2 The C1b wall-clock fraction is measured against the band path at full depth, never against the HF path (review F2).**
- Fact (`model.py`): the HF path's `_pass` runs `lm_head` on every row of every pass, latent columns included; the band path heads
  only rows without a latent at that column. A full-depth `coconut` checkpoint's latent pass on the HF path therefore carries a
  [B, 768] x [768, 50,260] head matmul (about 31% of a single-column pass in FLOPs at 124M) plus HF's mask and cache overhead,
  which the band path lacks; band ms over HF ms is biased below (b minus a) over L for reasons unrelated to the skipped blocks.
  The band (0, L) forward equals the reference forward (`test_band.py::test_full_band_is_the_reference_coconut`: logits, loss,
  embeddings and every parameter gradient).
- Rules:
  1. **The denominator** is the task's full-depth `coconut` screen checkpoint run through the band path with `latent_band
     [0, 12]`: `run_eval.py --config configs/screen/<task>_coconut.json --checkpoint <its ckpt_best_final.pt> --set
     latent_band=[0,12] --time-steps` (plus the same `--split` and `--batch-size` as the C1b arm's timed read), on the same GPU
     type. The HF-path time of the `coconut` checkpoint is that arm's own 5.5 (b) wall-clock and is never the denominator.
  2. **The quantity**: f_meas = `latent_ms_mean`(C1b checkpoint, its band) over `latent_ms_mean`(`coconut` checkpoint, band
     [0, 12]) from `summarize_timing` (kind "latent"), with the "latent_last", "token" and "prefill" means printed beside, per
     arm and per k. Batch size 1 is the registered wall-clock axis (5.5 b) and gives the fraction of record; the eval batch size
     (32) is printed beside it. Both carry the device, the torch build and the number of timed batches.
  3. **Like-for-like check**, logged with the fraction: the accuracy of the band [0, 12] run beside the `coconut` arm's own
     HF-path read on the same split (a difference of more than 1 example in 500 goes into deviations.md with the operator's
     explanation; the timing is still reported), and a one-batch strict-fp32 (TF32 off) maximum absolute difference of the
     first-token logits between the HF path and band [0, 12] on the real checkpoint (expected at the 1e-4 level or below; larger
     is a BUG until explained).
  4. **A3.4's "nominal" rule gets its number**: the saving shows iff f_meas <= (b minus a) over L + 0.10 (k = 8: f_meas <= 0.77;
     k = 4: f_meas <= 0.43). Otherwise the saving is "nominal" and K3 and 7.2 use `layer_fraction = min(1, f_meas)` in
     `inference_record` for that arm (recomputed from the per-example counts; the primary FLOPs at (b minus a) over L are printed
     beside it, labelled nominal). f_meas above 1 means the band costs more than full depth and the arm cannot be labelled a
     saving at all.
  5. The CHANGELOG 23:40 UTC CPU ratios (0.56x for k = 8, 0.36x for k = 4) are superseded as a measurement and enter no table.

**A6.3 Every checkpoint start is pinned and verified; the launcher refuses an unpinned start (review F4; D15 is a refusal, not a
warning).**
- Fact: the nine GSM latent screen configs (`gsm50k_{coconut,c1b_k4,c1b_k8,c2,c2_shuffled,c3,c3_foreign,c2c3,pause}.json`) carry
  `load_model_path: runs/gsm50k_cot_lr1e-4/ckpt_best_final.pt` and `load_model_sha256: null`; `train.py` raises on a mismatch and
  only warns on null.
- Rules (what must hold; `rev45` is the operator's implementation and was read):
  1. A train run whose effective config (the file plus the launcher's `set`) carries a `load_model_path` and no
     `load_model_sha256` is refused by the launcher at validation: no VM, no process. The `train.py` warning is not a control.
  2. Every GSM-50k latent screen arm starts from the lr 1e-4 `cot` screen run's `ckpt_best_final.pt` (A3.4.5; never the best-LR
     trial, never a rerun or a resumed copy), with `load_model_epoch` = that checkpoint's epoch and `load_model_sha256` = its
     sha256 as the source run recorded it (RUN_DONE and SYNC_MANIFEST); `train.py` verifies the hash at start (the existing raise)
     and writes both into the run's `config.json`.
  3. A start across jobs (the source run on another VM or project) states the expected sha256 in the job spec (`init_sha256`
     beside `init_from_run`), and the launcher verifies the downloaded checkpoint against it before the run starts; a mismatch or a
     missing checkpoint is a refusal, logged, never a warning.
  4. The critic logs the start checkpoint's sha256 and epoch with the admission of the first GSM latent arm; at the screen read
     every GSM latent arm's `config.json` is checked against that row; a mismatch makes the run "speculative, unused"
     (deviations.md standing rule) and it enters no screen cell.
  5. The same rules bind any later start from a checkpoint (confirmation seeds, a scale phase).

**A6.4 The secondary reference-cost accounting (c) of 5.5 exists before the first comparison table (review F7).**
- Fact: no function in `common/latent` or `code/` computes 5.5 (c); `inference_record` charges the registered 5.3 table from the
  counts alone, and the logits the code computes at every prefill position and at latent columns are the reference's redundant
  head, uncharged in the primary accounting as 5.2 says.
- Rules. "Exists" means all three of:
  1. A function in `common/latent/accounting.py` (named in INTERFACE.md) implementing the literal 5.5 (c): with p_0 the arm's 5.3
     prefill length and K its 5.3 sequential passes, pass k (k = 0 to K minus 1) processes the whole prefix of length p_0 + k with
     the head at every position, so reference_cost = the sum over k of `flops_prefill(dims, p_0 + k, head_positions = p_0 + k)`;
     a formula of (|q|, L, E') and the arm as 5.3 lists them, with no layer fraction (the reference has no band).
  2. A test reproducing the critic's hand example on the 5.1 worked example (GPT-2 124M dims, V = 50,260, p_0 = 251, L = 6,
     E' = 8, K = 15, lengths 251 to 265): **975,503,738,880 FLOPs**, 20.65x the primary 47,248,760,832.
  3. A "reference-cost FLOPs" column in `summarize`'s output and in every comparison table of 7.2 to 7.4 and every screen table,
     with the accounting-sensitive check of 5.5 applied (a flip under (c) labels the comparison; the FLOP verdict stands).
- No table is written before the function and the test exist; a table written without the column is reissued.

**A6.5 The probe accessors are frozen in INTERFACE.md (review F8).** `LatentLM.embedding` (the wrapped model's input
embeddings), `LatentLM.backbone` (the transformer body; with `lm_head` it equals the causal LM's forward) and `LatentLM.lm_head`
are frozen as stable accessors at the INTERFACE.md v4 line "Frozen accessors (review F8, 2026-10-18 ...)" committed in
`rev45`, together with the sentence that `LatentLM.__init__` inherits the base model's mode (review F1). A change to any of the
three after that commit (semantics, return type, what they exclude from `state_dict`) is a **deviation**: a deviations.md entry
and a critic entry before the change is used, and every readout number computed across the change (R1 to R3, F1 to F3, coverage,
load-bearing fraction) is recomputed before it is printed. A native `latent_override` hook, if added, is a new INTERFACE.md line
with its own tests and never a change to the three.

**A6.6 How `dep_gap` and a stage-7 numerics event are read and reported (review F6; residual risk 9).**
- `dep_gap` (`train_dep_gap` in `log.jsonl`: the per-epoch mean over micro-batches of the mean over valid rows of CE_alt minus
  CE_ref, detached), with `train_dep_active` (the fraction of rows inside the margin), `train_dep_hinge` and `train_dep_rows`, is
  a **training diagnostic**: measured in train mode between a multi-pass true forward and a single-pass scrambled forward, under
  different dropout draws and, with bf16 autocast, different kernel shapes, so it carries symmetric noise (about 1e-2 nat of
  numerical noise plus dropout) against m = 0.2 nat; it widens the hinge's effective margin and biases nothing. Rules: it is
  printed per epoch for C3, C3-foreign and C2+C3 with the sentence "measured under training noise; not the dependence read";
  C3-foreign's per-epoch absolute dep_gap (two foreign banks, expected near 0) is the noise floor printed beside C3's; a C3
  dep_gap under twice that floor is reported "hinge not separating in training" (diagnostic); **the dependence read is the
  scrambled-at-inference drop on valid in eval mode and nothing else**; no dep_gap number enters a kill, a pass or a verdict.
- Numerics at stage 7 (ProsQA-hard latent arms, 7 latents; the band recurrence feeds the raw residual stream h_b, `ln_f` applied
  only when b = 12): `train_loss` in `log.jsonl` is the epoch mean of the micro-batch losses, so one non-finite micro-batch makes
  it non-finite. Rules: (i) a run with a non-finite `train_loss` at epoch e is **"numerics-failed at epoch e, stage s"**;
  checkpoints saved after e are not selectable; the arm-task cell is UNTESTED at the screen unless a fix is registered by a row
  before a rerun (a change to the mechanism is a card revision; a change of precision makes the run "precision-mismatched" under
  A4 item 4 and it enters no cell); (ii) a finite rise of `train_loss` or a fall of `val_acc` at a stage boundary is a curve
  feature, printed with the stage boundaries in the fade table (7.6) for every latent arm beside full-depth `coconut`'s, and
  never a verdict input: the final-stage selection rule (A3.4.5, epochs 21 to 29) already reads recovery; (iii) the report prints,
  for every C1b arm, the per-epoch `train_loss` and `val_acc` at stages 6 and 7 beside `coconut`'s, so a band-specific
  instability is visible.

**A6.7 Standing labels on the screen-data reads (residual risks 6 to 8), with the review's numbers.**
1. ProsQA-hard depth and distractors: the generator's default draw gives fewer distractors to deeper proofs (600-record replica
   with every draw logged: depth 5 requested 59.4 and accepted 59.3 distractor edges; depth 6 56.7 and 56.5; depth 7 51.9 and
   51.9; the frozen files: requested mean 55.65, realised 55.66, realised equal to requested in every accepted record), so **a
   per-depth slice is confounded with distractor count by construction**: every per-depth number is printed "depth d (fewer
   distractors at greater depth, by construction)" and is a diagnostic (section 10), never a decision.
2. ProsQA-hard 13-gram overlap vs its own train: 37.0% valid, 36.0% test (A5 item 1; template sentences over a tiny vocabulary);
   duplicate questions 0 and duplicate graphs 0 across splits, with names and ignoring names. Printed with every ProsQA-hard table
   as the template-overlap sentence of A2 item 4; **never read as leakage**.
3. ProsQA-hard name triples: 163 of 500 test records share a (root, target, neg) name triple with train; every test (target, neg)
   pair occurs in train, and the train majority over that pair predicts the test answer 219 of 500 = 43.8%, so names carry no
   shortcut. Printed once per report; no read depends on it.
4. C2 on ProsQA-hard: the decoder is trained to emit the answer symbol at the last step by construction (the last supervised
   target contains the answer: 1.00 on ProsQA-hard, 0.770 on GSM-50k; A5 item 5), so R3 coverage for the final step is not
   evidence and is reported separately from steps 1 to depth minus 1 (A5 item 5, unchanged). C2-shuffled's donor frontier
   contains the record's target name **34.8%** of the time, symmetric between target and negative target (the donor is
   independent of which option is the answer), so the null has no answer signal; on GSM-50k the donor shares the answer 430 of
   50,000 = 0.86% (chance). Both sentences accompany every C2-shuffled number.
5. C2-shuffled is a fixed random-target memorisation pressure: one seeded derangement per run within step-count classes
   (ProsQA-hard classes 3,397 / 3,323 / 3,280; GSM-50k classes down to 3 members at 10 steps; no class merged), drawn before the
   epoch loop, so its state loss is not expected to fall. Every C2 minus C2-shuffled sentence carries **"the gap includes
   learnable vs unlearnable auxiliary target"**; the C2-shuffled `train_state_loss` curve is printed beside C2's; the kill and
   pass lines of A3.4 and its attribution sentences are unchanged.

**A6.8 Screen admission.** The first screen admission, and every later one, cites by critic-row timestamp or commit the conditions of A5
item 8 and of this amendment as the critic's checklist row of 2026-10-19T00:11Z enumerates them: the review acknowledged and its twelve
dispositions logged; A6 committed (the operator's SHA row); the frozen screen hashes of A5 item 1 verified on the VM; the pin row and the
PDF row; the calibration numbers; the provisional K1 of A3.3; the suite green at the launch commit with the count logged; the three `cot`
bar trials of a dataset admitted no later than its first latent arm; for a GSM latent arm, the start checkpoint's sha256 and epoch
(A6.3); the directional, timed and reference-cost reads of A6.1, A6.2 and A6.4 scheduled. **Nothing in A6 loosens a registered rule**: it
adds labels, a reference for one measurement, a refusal where a warning stood, a column, a freeze and reading rules; the registered
verdicts, kill lines, seeds, tolerances and caps are as A3 to A5 left them.

**A7 (2026-10-19 01:10 UTC, before the first oracle admission; the prep VM's outputs accepted: the base-model pin in force, the 4.2
targets frozen from the PDF with the paper's spreads, the calibration record; the coconut forced-resume smoke relocated to the first
80 GB job behind a memory probe; the window's allocation re-registered; cross-GPU-type rules; cap, suite and inspection conditions).**
Authority: A3.1 (the pin row and the PDF row are the critic's acceptance rows), A3.2 (ii), A3.5 (allocation and caps), A3.6
(resume), A4 items 2, 3 and 5, A6.3. Source: commit `rev50` (`results/prep/`, `results/calibration_cuda.json`,
`results/parity_real_cuda.json`, `configs/reference_env.json`, the manifest pin), the prep job's bucket output
(`silent/prep/prep-main/`: the smoke `log.jsonl` rows, `pytest_cpu.log`, the PDF and its full text), and the critic's own counts
over the data files. Seen when written: no accuracy number except the smoke's `val_acc` (0.16, 0.21, 0.19 on 100 examples after 2
epochs on 2,000), which A3.1 declares noise by construction; `results/` holds no read; the operator's new job specs exist only in the
uncommitted working tree. Critic entries: 2026-10-19T01:06Z to 01:10Z.

**A7.1 The pin (A2 item 1a).** `openai-community/gpt2` hub revision `607a30d783dfa663caf39e06633721c8d4cfcd7e` (resolved from
`main` on 2026-10-19T00:18:58Z), `model.safetensors` sha256 `248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707`,
the six companion files hashed in `configs/data_manifest.json` `models[0].files_sha256`; the data hashes of the manifest are
unchanged since the signature of 2026-10-17 (checked block by block between `rev51` and `rev50`). Reference environment
(`configs/reference_env.json`; `gpu_environment` in the manifest): Python 3.10.12, torch 2.14.1 (CUDA 13.0 build), cuDNN 9.24
(92400), driver 580.178.04, transformers 4.57.6, tokenizers 0.22.2, image `common-cu129-ubuntu-2204-nvidia-580-v20261016`.
Rules: every VM fetches the base model by this revision, never by a branch name; a downloaded weights hash other than the pinned
one stops the job before any run; a run whose `env.json` differs from the reference in Python, torch, CUDA, cuDNN, driver,
transformers or image is "env-mismatched" (A3.2 (i)) and joins no pooled cell until a same-seed pair passes. The 355M and 774M
entries stay unpinned; no run uses them in the window.

**A7.2 The 4.2 targets, frozen from the PDF (supersedes the 4.2 table and its "snippet; verify" clause).** Source: arXiv
2412.06769, PDF sha256 `27db5069d4361a607f02133ab4da4d9dae601bfbff0677191b3ba7febe2f987c`, Table 1 (excerpt
`results/prep/coconut_pdf_excerpt.txt`, page 9). Accuracy (%) +- as printed, emitted tokens in brackets:

| task | our arm | paper row | published | gates K1 |
|---|---|---|---|---|
| ProsQA | coconut | Coconut | 97.0 +- 0.3 (14.2) | yes |
| ProsQA | pause | pause as thought | 96.6 +- 0.8 (8.2) | yes |
| ProsQA | cot | CoT | 77.5 +- 1.9 (49.4) | yes |
| ProsQA | no_cot | No-CoT | 76.7 +- 1.0 (8.2) | yes |
| ProsQA | coconut_no_curriculum | w/o curriculum | 76.1 +- 0.2 (14.2) | yes: within tolerance of our own no_cot mean (as registered) and within tolerance of 76.1 (added) |
| ProntoQA | coconut | Coconut | 99.8 +- 0.2 (9.0) | yes (at ceiling) |
| ProntoQA | cot | CoT | 98.8 +- 0.8 (92.5) | yes (at ceiling) |
| ProntoQA | no_cot | No-CoT | 93.8 +- 0.7 (3.0) | yes |
| ProntoQA | pause | pause as thought | 100.0 +- 0.1 (3.0) | if the arm runs |
| ProntoQA | coconut_no_curriculum | w/o curriculum | 52.4 +- 0.4 (9.0) | if the arm runs, as the ProsQA rule |
| GSM8k-Aug | cot | CoT | 42.9 +- 0.2 (25.0) | yes (single seed) |
| GSM8k-Aug | coconut | Coconut | 34.1 +- 1.5 (8.2) | yes (single seed) |
| GSM8k-Aug | no_cot | No-CoT | 16.5 +- 0.5 (2.2) | yes (single seed) |
| GSM8k-Aug | pause | pause as thought | 24.1 +- 0.7 (2.2) | no arm run |
| GSM8k-Aug | coconut_no_curriculum | w/o curriculum | 14.4 +- 0.8 (8.2) | no arm run |

Not our arms, recorded for the report's footnotes: iCoT 30.0 / 99.8 +- 0.3 / 98.2 +- 0.3; Pause Token (no curriculum) 16.4 +- 1.8
/ 77.7 +- 21.0 / 75.9 +- 0.7; w/o thought 21.6 +- 0.5 / 99.9 +- 0.1 / 95.5 +- 1.1. Correction of record: the plan's GSM8k "pause
16.4" was the Pause Token baseline; our `pause` arm is "pause as thought" (24.1 +- 0.7 on GSM8k). The operator corrects `PLAN.md`
line 21, `docs/prior_art.md` line 46, `code/read_oracle.py` `SNIPPET_DEFAULTS` and `governance/oracle_published.template.json`
by a commit named in CHANGELOG; the registered read uses a `--published` file with `source: "pdf"`, the PDF sha256,
`critic_log_row: "2026-10-19T01:07Z"` and per cell `pdf_table: "Table 1"` and the paper row name above. Emitted tokens (secondary
oracle, flag only, 4.2 last row): ProsQA CoT 49.4 vs Coconut 14.2; ProntoQA 92.5 vs 9.0; GSM8k 25.0 vs 8.2. Spreads: the paper
does not say how many runs its +- summarises or which statistic it is (the full text was searched); 4.1's reading (a 3-run SD,
df 2) stands, every tolerance line carries "+- as printed, statistic unstated", and the floor of 1.0 stays. Three-seed cells:
sigma_cell = max(1.0, sqrt((s_ours^2 + s_pub^2) / 2)), frozen at the read. Single-seed cells and the provisional K1 (A3.3):
sigma_cell = max(1.0, s_pub), tolerance = max(2.0, 2 sigma_cell): ProsQA coconut 2.0, pause 2.0, cot 3.8, no_cot 2.0; GSM cot
2.0, coconut 3.0, no_cot 2.0; ProntoQA 2.0 each.

**A7.3 The calibration record (A3.1; A4 items 3 and 5).** Accepted as written in the 01:08Z row: the bench's "fp32" rows are
strict fp32 (no TF32 flag set; the pinned torch defaults to off); the oracle precision fp32-tf32 is measured through the smoke
rows only (45,055 and 44,660 padded tokens/s for `cot`, 47,835 for `coconut` stage 0; validation 7.5 s, 11.6 s and 21.1 s per
100 examples at 128 new tokens; flags logged True / True / "high" in every row). A4 item 5 is satisfied for every ProsQA-shaped
run; for the GSM runs (mean padded length 127 against ProsQA's 537) the rate is unmeasured and the caps are provisional until the
first GSM epoch row; an overrun is a resume (A3.6), never a recipe change. The strict pair's hours are estimated at the strict
bench rate (19,535 tokens/s): 8.7 h, cap 12 h or more. A4 item 3's one-batch strict-vs-TF32 diagnostic (loss and gradient norm,
relative) is outstanding and due in CHANGELOG before the provisional K1 read. The one-batch parity on real weights (loss relative
2.4e-7 against CPU float64) is the 40 GB record; A3.2 (ii)'s both-types check is A7.4.

**A7.4 Plumbing re-signed: the coconut smoke, the memory probe, the parity reference (A3.1 smoke item, A3 items 4 and 5).** The
prep's `cot` forced-resume smoke passed with the optimizer state restored (D17 exercised); its `coconut` smoke resumed correctly
and died of memory at stage 1 on the 40 GB card: ProsQA batches at batch 32 pad to a worst case of 702 positions at stage 1 (the
collator's bound; observed 698; GSM `cot` 255; GSM `coconut` 406 at stage 1). Rules: (1) `oracle-acctb` runs, before any
training run and in this order, a memory probe (one forward + backward + optimizer step at batch 32, fp32, seq >= 704, 6 latents)
and the `coconut` forced-resume smoke (`configs/smoke_prosqa.json` unchanged) and writes the parity reference from the smoke's
checkpoint; a failure of either ends the job with no training run; the receipt steps are the record of A3.1's "smoke completed
end to end" for `coconut`. (2) No other oracle job is admitted before that receipt shows both steps ok and the reference
uploaded; each later admission cites it. (3) Memory probes on every oracle job: ProsQA jobs seq >= 704 with 6 latents;
`oracle-main-gsm` seq >= 256; the GSM coconut job seq >= 416 with 6 latents (or 704). (4) `parity_first: true` on every job of
every project other than AcctB, the pair and the GSM coconut job included; the main project's GSM job is the 40 GB side of A3.2
(ii), and until its parity step passes (loss 1e-5, gradient norm 1e-4 relative, strict fp32, same checkpoint) no number is pooled
or handed across the two GPU types. (5) A job spec whose probe shape or cap is below these figures is not admitted.

**A7.5 Allocation and caps (replaces A3.5's launch-order sentence "GSM chain on AcctA; ProsQA seed 0 on AcctB; seeds 1-2 on AcctC; the two LR trials
on main").** AcctB (A100-80GB): probe, smoke, reference, then ProsQA `cot`, `coconut`, `pause`, `no_cot` at seed 0 (estimates 3.4 / 5.0 / 3.3 / 3.2
h; cap 20 h). AcctC (A100-80GB): ProsQA `coconut` seeds 1 and 2, `cot` seeds 1 and 2 (cap 22.5 h). AcctA (A100-80GB), in sequence: the two ProsQA LR
trials (lr 3e-5, 3e-4; cap 9.5 h); the strict precision pair (cap 12 h or more); `gsm_coconut_s0` from the main project's `gsm_cot_s0` checkpoint
(estimate 17.6 h at the ProsQA-shaped rate, provisional; cap 23.5 h), the pair and the GSM coconut job in either order. Main (A100-40GB):
`gsm_cot_s0` (estimate 7.8 h, provisional; cap 8.5 h accepted at 1.1x the estimate with headroom kept for a resume admission; Recipe unchanged
everywhere: batch 32 x grad_accum 4, lr 1e-4, fp32-tf32 (strict for the pair), resume per A3.6. Caps are 1.3x the estimate plus 0.5 h unless a
project cap binds (main), in which case the ledger's admissible hours and the resume rule apply; a cut run is one run across admissions (A3.6) and
enters no cell until it has reached its registered epochs. the per-project ceilings hold (AcctA 251 of 290; AcctB 112 of 230; AcctC 126 of 220; main
37 of 45).

**A7.6 Across GPU types.** (1) The GSM chain crosses types: `cot` on A100-40GB, `coconut` on A100-80GB from its checkpoint.
Allowed iff the main job's parity step passed (A7.4 (4)) and both runs' `env.json` equal the reference (A3.2 (i)); the GSM coconut
cell carries the label "cot stage on A100-40GB, coconut stage on A100-80GB" in every table; the start is pinned by `init_from_run`
+ `init_sha256` in a job written only once the hash exists, and the critic logs the sha256 and epoch in a row cited by that
admission (A6.3 rules 3 and 4). (2) A resume across GPU types is allowed only after that parity pass, is labelled "resumed across
GPU types", is never a same-seed pair and never enters the determinism check; otherwise A3.6 applies unchanged.

**A7.7 Suite, the VM's test failure, the generations inspection.** (1) The VM's suite on the image: 1 failed, 1028 passed, 2
skipped; the failure is a test-fixture path (the FakeTokenizer fallback where the tarball carries no tokenizer directory), fixed
in the test at `rev50` and verified by the critic on both branches; training and the C2 targets take the tokenizer from the
pinned snapshot, so no VM re-run is required and the image count is the record. (2) The full suite at `rev50` is not green
(`test_experiment_configs_load_and_mirror_the_reference` on `configs/reference_env.json`); under A5 item 7 no admission cites that
commit; the launch commit's suite is green with its count logged in the first admission. (3) A3.1's "answer-shaped outputs on
inspection": no generation text was saved; the operator evaluates the bucket's `smoke_prosqa_cot_gpu/ckpt_best.pt` on a few valid
examples with `code/run_eval.py` and quotes two generations in CHANGELOG before the second oracle admission; AcctB may be
admitted on the rows' indirect evidence (`val_acc` 0.16 to 0.21, mean emitted length 47 to 52 tokens against the paper's 49.4)
because its own smoke precedes its runs.

**A7.8 What each launch command cites.** As the critic's row of 2026-10-19T01:10Z lists per job: the rows 2026-10-18T22:31Z,
2026-10-18T23:27Z, 2026-10-19T01:06Z, 01:07Z, 01:08Z, 01:09Z, 01:10Z and the operator's SHA row for A7, with the launch commit, its
suite count, the tarball sha256 and the ledger id; every job after AcctB also cites AcctB's receipt steps and the generations
line; the main GSM job also states its cap arithmetic; the pair its strict-rate cap; the GSM coconut job the hash row. An
admission missing an item is not an oracle admission and its runs are "speculative, unused".

**Nothing in A7 loosens a registered rule**: the pin and the targets become fixed values; the smoke's relocation cannot be
bypassed by a training run; a cell gains a second condition; caps and probe shapes rise; the cross-type handoff gains a parity
condition and a label; the registered verdicts, kill lines, seeds, tolerances and the window cap are as A3 to A6 left them.

**A8 (2026-10-19, critic, additive): the resume point as A3.6 is read, the row of record, the determinism pair after a
resume, the supervisor's admissions.** Written before any read (no accuracy read of any run exists; `results/` holds no
read). Authority: A3.6, 3.2, 3.3, 6.4, A3.2, A7.5, A7.8; critic_log rows of 2026-10-19T03:00Z.

**A8.1 Resume point for `save_only_improve` arms (A3.6 read, not changed).** For `cot` and `no_cot`, which 3.2 binds to
`save_only_improve: True`, `train.py` writes no `ckpt_last.pt`; the resume point A3.6 calls `ckpt_last.pt` is the file
named by `ckpt_state.json`'s `latest_path`, the latest improving epoch's `ckpt_best.pt`, with `optimizer_last.pth` and the
trackers written at that same epoch and sha-checked on resume (train.py 383-403, 460-465, 559-575 at `rev57`). Epochs
after that boundary are re-run with their own (seed, epoch) data order and dropout stream (478-481). A run resumed this way
is the same run under A3.6; latent arms (`ckpt_last.pt` every epoch) resume exactly. Every resume stays logged with the
epoch and the VM (`ckpt_state.json` `resumes`, `runs_index_row.csv`); "fragmented" stays at more than 3 resumes.

**A8.2 The row of record.** `log.jsonl` keeps both rows of a re-run epoch. For each (run, epoch) the row with the largest
`resumes` is the row of record; earlier rows of that epoch are "superseded by resume": kept verbatim, printed in the run's
appendix table under that label with their count beside the resume count, and entering no number: not the selection, the
trackers, the K1 curve, the "earlier stage better on valid" check (3.3 item 3), the HEADSTART read (7.6) or any plot. Every
reader of `log.jsonl` dedupes by this rule. The selection is the one `train.py` made (`ckpt_best.pt`, `ckpt_best_final.pt`,
the `best` tuples of `ckpt_state.json`); at the read the critic checks it equals the argmax over rows of record with ties
to the earliest epoch; a mismatch is a BUG trigger under 7.10, never a re-selection. (A superseded row can never have been a
best, since an improving epoch is itself the resume point; the rule fixes what curves and checks print.)

**A8.3 The determinism pair after a resume (A3.6 and A3.2 as written).** A resumed run never enters the determinism check
(A3.6). `prosqa_cot_s0` on AcctB has one resume and is therefore not the original of 6.4's pair; the ProsQA read waits for
the determinism term (A3.2). The GPU determinism pair is two zero-resume runs of `configs/prosqa_cot.json`, seed 0,
lr 1e-4, fp32-tf32, on A100-80GB; a member that is preempted is restarted from scratch as a new run (its partial log kept,
entering no number). They may run on any project with that GPU type, in parallel, off the critical path (A3.2); the operator
schedules them. A author-directed amendment may admit a once-resumed, optimizer-exact `cot` run as the original; if written,
both readings are printed. Until a zero-resume pair exists, no ProsQA verdict is written.

**A8.4 Supervisor admissions (`code/launch/silent_launch/supervise.py`, `rev57`).** (1) An automatic relaunch of a cut or
preempted job under the critic entries of its last admission is an admission within A3.1 and A7.8 iff (a) the job spec minus
`_launch` equals the last receipt's spec and the tarball commit differs from the last admission's in nothing under
`common/latent/` (tests excluded) or the run's configs; a difference makes the resumed run "code-changed across resume",
which enters no cell before a deviations.md entry and a critic entry; the supervisor refuses and alerts on either (the operator's
change), and until then the operator verifies by hand before any commit while jobs are open and the critic diffs the receipts
at the next row; (b) the VM was preempted (gone while the receipt said running) or cut by its deadline with runs left; a
receipt saying the training process itself failed, including a resume sha mismatch, is an ALERT with no automatic relaunch
until the operator rules (statuses added to the stop set); (c) the job's run has not been handed to another project under A7.6
(before a handoff the operator takes the origin project out of the supervisor's reach); (d) the east zones (us-east4-c,
us-east5-a, factor 1.2) are tried only after a capacity refusal in both central zones and listed after them. (2) The one
automatic successor, oracle-accta to pair-strict-accta under oracle-accta's rows, is an admission within A7.8: the row
of 2026-10-19T01:10Z named those rows and a 12 h strict-rate cap, which `configs/jobs/pair-strict-accta.json` carries;
it launches only when both LR trials have `RUN_DONE.json`; its receipt shows `precision_diag`, whose numbers enter CHANGELOG
before the provisional K1 read. Jobs needing a new row (oracle-accta-gsm, the screens) are never launched by the supervisor.
(3) Every supervisor admission is a ledger record with job, rows, zone, factor and cap; the operator's next CHANGELOG entry
lists every supervisor action; every relaunch is an incident line in deviations.md with the receipt status that caused it.
(4) The east factor scales the ceiling and close rate by 1.2 whenever the east zones are listed, even when the VM lands in
us-central1 (seq 24): accepted as the stricter side; never lowered after the fact except by `reconcile` with a billed
figure; cap arithmetic uses the ledger's figures. A run that lands in us-east4/5 is the same machine, GPU type and image;
A3.2 applies unchanged. (5) The supervisor grants only the instance-scoped `roles/compute.instanceAdmin.v1` to the VM's own
default compute account on the VM it launched, re-attempts a failed grant on later passes (the operator's change) or the operator
deletes that VM by hand at job end; commits only `governance/run_log.jsonl`; never writes critic_log.md or registration.md;
deletes no VM it did not launch except a TERMINATED or STOPPED `silent=1` VM of ours.

**Nothing in A8 loosens a registered rule**: A8.1 names the file A3.6 meant for arms the registration binds to
`save_only_improve`; A8.2 removes superseded rows from every number; A8.3 restates A3.6's exclusion and its cost; A8.4
adds conditions to relaunches and the successor launch. Pre-commit; the commit SHA is appended by the operator in the next row.

**A9 (2026-10-19, critic, additive): the GSM coconut start pinned, the copy of the start checkpoint, and the supervisor
successor launch of `oracle-accta-gsm` (critic entries 2026-10-19T09:29Z; before the admission).**

**A9.1 The pinned start (A6.3 rule 4 applied to the oracle; A7.6 (1); D15).** The GSM coconut oracle run starts from
`gsm_cot_s0` (main project, A100-40GB, job `oracle-main-gsm`, attempt `silent-train-20261019-014019-f8e7`, 25 epochs,
zero resumes, selected by 3.3 as the unique best valid among all epochs) at its `ckpt_best_final.pt`, sha256
`6d798e8a3c2f27b3fcccf988b740a00c4f377aa3c94de984e21b64dbaa900200`, epoch 11 (1-based; `epoch_index` 10), as
`RUN_DONE.json`, `ckpt_state.json` (`best.final` [0.45, 11]) and `SYNC_MANIFEST.json` record it. Every GSM coconut oracle
run's `config.json` must carry `load_model_path` runs/gsm_cot_s0/ckpt_best_final.pt, `load_model_sha256` equal to that hash
and `load_model_epoch` 11; the critic checks it at the GSM read; a mismatch or a null epoch makes the run "speculative,
unused" (deviations.md standing rule) and it enters no cell. The main job's parity step passed (loss_rel_diff 0.0,
grad_norm_rel_diff 2.48e-9, same checkpoint, strict fp32): the 40 GB side of A3.2 (ii) and A7.4 (4) is met, and the GSM
coconut cell carries "cot stage on A100-40GB, coconut stage on A100-80GB" in every table (A7.6 (1)); its second condition,
the coconut run's `env.json` equal to the reference, is read from that run's receipt. The selection pool of the coconut
run is `epoch_index` 12-24 (`epoch` 13-25; D16, `train.is_final_stage` with `max_steps_train`); an `epoch_index` 9-11
checkpoint with the higher valid accuracy is 3.3 item 3's labelled secondary read.

**A9.2 The copy of the start checkpoint across buckets (A6.3 rule 3).** The launcher's `init_pin` reads the source run from
the launching project's own bucket. The operator copies exactly three objects from the main bucket to AcctA's
`runs/gsm_cot_s0/`: `ckpt_best_final.pt`, `ckpt_state.json` (sha256 11e40fdbf428619fa401b455e17d27ab0eded74e00265713d503d94557bd782c)
and `RUN_DONE.json` (sha256 e21f0a3fd20364c55bbbe1d8a4f898f0191a2a8022fca238e5d40d67124e3fb0 as fetched); no `log.jsonl`,
`runs_index_row.csv` or `optimizer_last.pth`. After the copy, `gsutil stat` of source and destination shows equal crc32c (and
md5 where present) for each object, and the next CHANGELOG entry quotes the three pairs beside the sha256s above, before the
job is registered as a successor or launched by hand. The receipt's `init` block and the run's `config.json` must show the
values of A9.1. The main bucket's `runs/gsm_cot_s0/` is the record of the cot run; the AcctA copy is an initialisation
source and is never a AcctA run in any table, index or listing.

**A9.3 The supervisor successor launch (A8.4 (2) read with the row that now exists).** A8.4 (2) excluded `oracle-accta-gsm` from
`SUCCESSORS` because its A6.3 rule-4 row did not exist; that row is 2026-10-19T09:29Z. The successor launch `pair-strict-accta` ->
`oracle-accta-gsm` by the supervisor is an admission within A7.8 iff all of: (a) the `SUCCESSORS` entry carries `rows` = the
predecessor admission's critic entries plus `2026-10-19T09:29Z`, and `code` "HEAD"; (b) HEAD at the launch has `common/latent/`
(tests excluded), `configs/gsm_coconut.json` and `configs/jobs/oracle-accta-gsm.json` (sha256
c27d34926972e2cecacd92df515e17d69e95ac9f09a8700466d7cff143a7665f) equal to commit `rev76`'s in every key but the config's `_note`
(which the operator corrects to D16's pool), where `common/latent/` and the GSM configs equal `rev61`'s (the main admission's
tarball), verified by the critic at `rev76`; a change in any of them needs a new critic entry before the launch, and the operator
verifies by hand before every commit while the successor is queued (A8.4 (1)(a)'s interim rule); (c) A9.2's copy is in AcctA's
bucket and quoted in CHANGELOG before the `SUCCESSORS` entry is committed (a pin refusal on the VM is a receipt status `failed`, a
STOP with an ALERT, never a relaunch); (d) the launch waits for `prosqa_coconut_strict_s0`'s `RUN_DONE.json` (coded), and the pair's
`precision_diag` numbers enter CHANGELOG before the provisional K1 read, unchanged; the east factor 1.2 whenever the east zones are
listed) and the admission record carries job, rows, zone, factor and cap (A8.4 (3)); (f) the experimenter's next CHANGELOG entry lists the
launch and quotes the receipt's `init` block. If any condition is not met, the job is launched by hand with the same citations (the
eight rows of A7.8, AcctB's receipt steps and the generations line, the main job's parity pass, the rows of 09:29Z, the launch
commit with its suite count, the tarball sha256 and the ledger id).

**Nothing in A9 loosens a registered rule**: A9.1 fixes the start checkpoint as values every GSM coconut `config.json` must
carry; A9.2 adds a verification to a copy no rule covered; A9.3 lets the supervisor make exactly the admission a human would
make with the same rows, under conditions a human launch does not carry; the recipe, the selection pool, the labels, the
tolerances, the caps and the read split are as A3 to A8 left them. Pre-commit; the commit SHA is appended by the operator in
the next row.

**A10 (2026-10-19, critic, additive): the registered provisional K1 read on ProsQA seed 0 (A3.3; 4.5 and section 8), the
read's numerics, write-once in the bucket, checkpoint pins, the 7.10 rescoring, the screens' bf16 flags.** Written before any
test read of any run (`results/` holds calibration, parity and prep only); the valid accuracies in the run records are
selection diagnostics. Authority: 1.3, 3.3, 4.1, 4.5, 5.3, 5.5, 7.10, section 8, A3.2, A3.3, A4, A6.4, A6.8, A7.2, A7.6, A8.1
to A8.3; critic entries 2026-10-19T12:07Z.

**A10.1 The read's numerics.** A4 item 1 registers fp32-tf32 "for every oracle run ... and for each run's own evaluation and
generation", and A4 item 4 makes a logged flag set that differs from the registered values "precision-mismatched". The
registered read of an oracle checkpoint is that checkpoint's evaluation: it runs at **fp32-tf32** (`matmul.allow_tf32 True`,
`cudnn.allow_tf32 True`, "high"), the numerics its selection on valid used; `run_eval.py` is invoked with **`--tf32 on`**,
and the launcher passes the flag from the job's `tf32` field, never a constant. The strict pair's checkpoint
(`prosqa_coconut_strict_s0`, A4 item 2) is read with `--tf32 off`. A `summary.json` whose `numerics` block differs from the
checkpoint's registered flags is "precision-mismatched" and enters no cell, and its one read is spent. The K1 table's
footnote states the reference's probable strict fp32 (A4).

**A10.2 Write-once in the bucket and checkpoint pins.** `run_eval.py`'s write-once check sees only the VM's local output
directory; a relaunched eval job would rsync a second read over `evals/<job>/<name>/summary.json`. Rule: the launcher never
runs an eval entry whose `summary.json` already exists in the job's bucket (it records the entry as `already_read` and
moves on), and never overwrites one; an audit rescoring (7.10 item 2; `--audit-rescore`) is written beside it under
`<name>_audit`. Every test eval entry carries `checkpoint_sha256`, the selected checkpoint's hash as the critic's read row
lists it; the launcher verifies the downloaded file before scoring (a mismatch is `checkpoint_mismatch`, no read); the
summary's `checkpoint_sha256`, `split_sha256` (`data/prosqa_test.json`
82ddc0afe6ee30eb31bb16fa208e293e55278a20fa987098d48519cae7e12680, 500 records) and `registered_read` are checked by the
critic at the verdict row. The code tarball carries `governance/critic_log.md` so that `run_eval.py`'s registered-read check
runs against the committed log on the VM.

**A10.3 The 7.10 checklist before the provisional K1 verdict.** 7.10 runs unconditionally in Phase 1. Items 1, 3 to 7 are met
from the manifest's 13-gram record on the hash-verified files (critic entry 2026-10-17T21:29Z), the suite green at the launch
commit with its count, and the critic's code-equality rows (2026-10-19T03:00Z; the tarball commits of the runs' attempts).
Item 2 (rescoring from the saved checkpoint at batch size 1) is run on the four `*_test_none` checkpoints in the same job,
after their registered reads, as `--audit-rescore --batch-size 1` entries written under `<name>_audit`; at batch size 1 the
example order is immaterial (each example is its own batch), so the order clause is satisfied by construction; per-example
correctness may differ from the registered read's file in at most 1 example in 500, else BUG until explained. The
provisional K1 verdict row waits for the four rescorings. The rescoring is 7.10's second scoring under 1.3 and consumes no
registered read.

**A10.4 The provisional K1: what it is and what it gates.** Cells and single-seed tolerances as A7.2: coconut 97.0 +- 2.0,
pause 96.6 +- 2.0, cot 77.5 +- 3.8, no_cot 76.7 +- 2.0 (sigma_cell = max(1.0, s_pub), frozen at the read). Secondary oracle:
the paper-style count (`emitted_tokens_mean` + `latent_steps_mean`) within 20% of 14.2 / 8.2 / 49.4 / 8.2 (coconut / pause /
cot / no_cot); a flag withholds the verdict pending a data and format check, never a recipe change. The zero-latent read
and the step sweep's n = 0 read build the same prompt (question, `<|start-latent|>`, `<|end-latent|>`) and are one number;
their equality is a free determinism check of the eval path and a difference is a BUG trigger. n = 8 carries "out of
distribution". Labels on every cell and sentence: KNOWN (reproduction); fp32-tf32; single seed; provisional; "read on
A100-40GB, trained on A100-80GB" (A3.2 (i) to (iii) met: `env_reference` on the eval VM, parity both ways, `gpu_type`
printed per seed); the run's resume count (A3.6, A8.1); precision pair pending (d_prec, A4 item 2); determinism pair pending
(A8.3). The provisional K1 gates the **start of the screens** (A3.3; A6.8 item 6) and nothing else; the final K1 (seeds 1
and 2 of `cot` and `coconut`, three-seed sigma_cell), the strict pair's d_prec and the zero-resume determinism pair
(sigma_task) are required before any confirmation seed, any 6.2 verdict and any WIN or tie sentence, as A3.3, A4 and A8.3
say. The K1 table of 13.1 needs no reference-cost column; the per-example cost table beside it (13.3) carries
`reference_flops` (A6.4 items 1 and 2 exist: `accounting.reference_cost_flops`, test value 975,503,738,880) and the
paper-style count; the comparison tables of 7.2 to 7.4 and every screen table carry the column (A6.4 item 3 unchanged).

**A10.5 The screens' registered flags (A4 item 4 as written).** The screens and confirmations run bf16 autocast "on top of
the oracle's TF32 setting": the registered flag set for `precision: bf16` is `matmul.allow_tf32 True`, `cudnn.allow_tf32
True`, "high", set at process start, asserted per epoch and logged on every row like the oracle's. `train.py` at HEAD sets
no flag for bf16 (torch's defaults are logged), so a screen run at HEAD would be "precision-mismatched" and enter no cell;
the operator adds the bf16 entry to `REGISTERED_FLAGS` (with its test) in the screens' launch commit. A9.3 (b) and A8.4 (1)(a)
are read with that change: HEAD's `common/latent/` may differ from `rev76`'s (and from the open admissions' tarballs) by
exactly that entry and its test, verified by the critic at the commit; any other difference needs a new row.

**Nothing in A10 loosens a registered rule**: A10.1 names the numerics A4 already registered for a checkpoint's evaluation;
A10.2 moves the write-once rule to where the files live and pins the checkpoints read; A10.3 schedules 7.10 item 2 as 1.3
allows; A10.4 fixes labels and gates as A3.3, A4 and A8.3 left them; A10.5 names A4 item 4's flag set for bf16 and admits one
diff under A9.3 (b). Pre-commit; the commit SHA is appended by the operator in the next row.

**A11 (2026-10-19, critic, additive): the ProsQA seed-0 oracle read recorded; the 5.1 vocabulary clause read with D5; the
conditions before the "inert" sentence; a label-only commit to the write-once table.** Written after the read of A10 (critic entries
2026-10-19T15:35Z); every number it cites is in `results/evals/k1-eval-main/` and `results/k1/` at commit `rev92`. Authority: 1.3,
4.3, 4.5, 5.1, 7.5, 7.9, 7.10, section 8, 13.5, 13.10, A3.3, A10; defect D5 as accepted at A2.

**A11.1 The read of record.** The ProsQA oracle read of 4.5 for seed 0 is `k1-eval-main` (VM `silent-eval-20261019-144240-f455`,
`--registered-read 2026-10-19T12:07Z`, code commit `rev110`, tarball sha256 `rev379...`), committed write-once as
`6d7af88413aaa87d12751eb25be453a6efe67726`: the four test cells (cot 75.0, no_cot 78.0, coconut 96.8, pause 97.0; examples.csv
sha256 rev380..., rev381..., rev382..., rev383...), the seven intervention reads on the coconut checkpoint (scrambled,
zero_latent, n = 0, 1, 2, 4, 8) and the four `_audit` rescorings. `results/k1/k1_table.json` is the verdict of 4.3 for these
cells and is never recomputed; the checkpoints rev87..., rev90..., rev88..., rev89... have spent their test reads
and intervention reads (1.3, 4.5). The final K1 (seeds 1 and 2 of `cot` and `coconut`), the tuned-bar read (3.4, A3.3) and the
strict pair's read are separate reads, each needing its own `--registered-read` row and its own verdict row. The window sentence
is printed as "K1 for the window (A3.3) = K1 on ProsQA: pass [provisional; GSM8k-Aug UNTESTED until its read]".

**A11.2 The vocabulary term of 5.1.** 5.1 says "V = 50,260 after the resize, for every arm"; D5 (accepted at A2; CHANGELOG
2026-10-17 21:40Z) made the code resize only the latent arms, as the reference does, and `model_dims` reads V from the head's
rows as the model is. Accounting as implemented and as the registered files carry it: V = 50,260 for `coconut`,
`coconut_no_curriculum`, scrambled and `pause`; V = 50,257 for `cot` and `no_cot`. The difference is 2 x 3 x 768 = 4,608 FLOPs
per head evaluation (2.2e-6 of a ProsQA `cot` example) and changes no comparison; the files stand; every cost table footnotes
"V as the arm's head: 50,257 (cot, no_cot), 50,260 (latent arms)". The worked example of 5.1 (V = 50,260) is unchanged.

**A11.3 The "inert" sentence.** "Coconut's latents are inert on ProsQA in our reproduction", or any sentence that attributes
inertness to Coconut rather than to a named checkpoint at a named read, may be printed only when all of: (1) 7.10 items 4 and 5
run with the real checkpoint on a GPU on the valid split (the fed vector's distance to the example's own latent at every rank,
> 0 for every example; the fraction of examples whose first-answer-token logits change; batch isolation at B = 1 against 32),
logged in CHANGELOG with the commit and hashes; (2) three seeds' registered scrambled, zero-latent and sweep reads with each
seed's paired drop and interval, and the three-seed Delta_scr sentence of 13.10 with sigma_task (7.5 read as the Phase 1
diagnostic of 13.5); (3) the F1 counterfactual patch of 7.9 on the three checkpoints with its load-bearing fraction; (4) the
bit-width ladder of 7.9 on the three checkpoints; (5) the labels KNOWN (reproduces the published critiques), fp32-tf32, three
seeds, ProsQA only, GSM8k-Aug untested. Until then the only printable sentence is the diagnostic of the critic's row (no
measurable dependence at this read, with its intervals and labels), and the words "mechanism", "reproduces" and "K2 would fire"
do not appear outside CHANGELOG.

**A11.4 Labels on the write-once table; the summary's commit.** The numbers of `results/k1/k1_table.{json,md}` are write-once
(4.3). One commit that adds the A10.4 label block, the A10.1 fp32 footnote and the bracketed sentence qualifiers to `k1_table.md`,
changing no cell of `k1_table.json`, is allowed, named in CHANGELOG and diffed by the critic. From the next eval job on,
`run_eval.py` writes the launcher's code commit into `summary.json` (`commit` was null on this read because the VM holds no
`.git`; the receipt's `code_commit` binds for this read).

**Nothing in A11 loosens a registered rule**: A11.1 records a read 4.5 already defines; A11.2 names which V the accepted D5 made
the code charge; A11.3 adds conditions before a sentence; A11.4 allows a label-only commit and a provenance field. Pre-commit;
the commit SHA is appended by the operator in the next row.

**A12 (2026-10-19, critic, additive): the ProsQA seeds 1-2 oracle read recorded; the label-only commit for the final table;
7.10 item 2 above the "at most 1" line; the three-seed diagnostic sentence.** Written after the read of the 15:40Z row (critic entries
2026-10-19T16:59Z, three); every number it cites is in `results/evals/k1-eval-s12-main/` and `results/k1/` at commit `rev2`.
Authority: 1.3, 4.3, 4.5, 6.1, 7.5, 7.10, section 8, 13.5, 13.10, A3.3, A8.3, A10, A11.

**A12.1 The read of record.** The ProsQA oracle read of 4.5 for seeds 1 and 2 of `cot` and `coconut` is `k1-eval-s12-main` (VM
`silent-eval-20261019-155235-f43e`, `--registered-read 2026-10-19T15:40Z`, code commit `rev106`, tarball sha256 `rev384...`),
committed write-once as `20fd1b9510bbabe8077c7ce6ce7184b8fda750a6`: the four test cells (cot s1 71.2, cot s2 78.4, coconut s1 96.2,
coconut s2 95.8; examples.csv sha256 rev385..., rev386..., rev113..., rev114...), the fourteen intervention reads on the two
coconut checkpoints and the four `_audit` rescorings. `results/k1/k1_table_final.json` is the verdict of 4.3 for the three-seed
cells `cot` (74.87, SD 3.60, sigma_cell 2.88, tolerance 5.76 against 77.5: matched) and `coconut` (96.27, SD 0.50, sigma_cell 1.0,
tolerance 2.0 against 97.0: matched) and for the single-seed cells as `k1_table.json` fixed them, and is never recomputed; sigma_cell
of both cells is frozen (4.1); the checkpoints rev102..., rev103..., rev104..., rev105... have spent their test reads and
intervention reads (1.3, 4.5). The window sentence is printed as "K1 for the window (A3.3) = K1 on ProsQA: pass [three seeds cot and
coconut, single seed no_cot and pause; GSM8k-Aug UNTESTED until its read]". The provisional K1 of A10.4 and A11.1 is not reversed.

**A12.2 Labels on the final table.** As A11.4 for the provisional table: one commit that adds the A10.4 label block (with the
three-seed and single-seed labels per cell, the resumes per seed and the audit label of A12.3), the A10.1 fp32 footnote and the
bracketed sentence qualifiers to `results/k1/k1_table_final.md`, changing no cell of `k1_table_final.json`, is allowed, named in
CHANGELOG and diffed by the critic. The numbers of both files are write-once (4.3).

**A12.3 7.10 item 2 above the line.** When the batch-size-1 rescoring of a registered read differs from it in more than 1 example
in 500: the registered read (batch 32, the read of record) stays the number of record and is never re-read (1.3, 4.5); the cell's
verdict is computed from it as 4.1 says; the cell and every sentence that depends on it carry "audit: k of 500 differ at batch 1,
BUG until explained (7.10 item 2)" and do not travel outside CHANGELOG and the critic log (RESULTS.md, INDEX.md, a PR, a message to
the author, a card) until the explanation row. The explanation is 7.10 item 5 on that arm's generation path with the real
checkpoint on a GPU on the **valid** split (no test read is spent): (i) at `--tf32 off`, the prompt-position logits of every example
at B = 1 against B = 32 with two different companion sets agree within 1e-4 and the count of diverging greedy generations is logged;
(ii) at the read's numerics, the same counts and, per diverging example, the first divergent token's position and top-2 logit margin
under both batch sizes; results in CHANGELOG with the commit and the per-example file hashes, then a critic entry. Outcome A (the
fp32-strict agreement holds; the divergences at the read's numerics have margins at TF32 rounding scale and depend on the batch's
padding length, not on the companions' content): the label becomes "audit: k of 500 differ at batch 1, explained: <cause>" and the
verdict travels. Outcome B (a divergence survives `--tf32 off`, or row j changes row i beyond 1e-4): an eval-path BUG; the verdict is
withdrawn by a critic entry, every batch-32 read of record is suspect, and 4.4 round 1 starts on the eval path; what a re-read then
needs is ruled by the critic at that row. At this amendment the item is open on `prosqa_cot_s1` (3 of 500: idx 213, 364, 382) and
closed on `prosqa_cot_s2` (1 of 500), `prosqa_coconut_s1` and `_s2` (0 of 500) and the four seed-0 cells (A10.3, 15:35Z).

**A12.4 The three-seed diagnostic sentence and the 13.10 sentence.** The only sentence about the latents that may be printed
outside CHANGELOG, with the labels KNOWN, fp32-tf32, three seeds, diagnostic, read on A100-40GB of A100-80GB checkpoints, ProsQA only,
is the critic's three-seed sentence of the row of 2026-10-19T16:59Z (no measurable dependence at this read: scrambled drop 0.0 on every seed,
zero-latent -0.4 / -0.2 / 0.0, sweeps flat within 0.4 points), beside or instead of the seed-0 sentence of 15:35Z. The 13.10 sentence
"Scrambled drop, Coconut ProsQA: <value> [<interval>]; K2 would fire: yes/no" is written at the critic entry that first has sigma_task
(A3.3, 6.1, A8.3), with the three-seed value and interval of that row (0.0 [0.0, 0.0] at this read), as the Phase 1 diagnostic of
13.5 and never as an arm kill; A11.3 is unchanged: "K2 would fire", "inert", "mechanism" and "reproduces" stay inside CHANGELOG until
A11.3 (1) to (5) are met, and this read meets (2) in numbers but not in sigma_task.

**Nothing in A12 loosens a registered rule**: A12.1 records a read 4.5 already defines; A12.2 extends A11.4's label-only commit to
the final table; A12.3 adds a label, a holding rule and an explanation requirement where 7.10 says "BUG until explained"; A12.4 names
which sentence may be printed and when 13.10's is written. Pre-commit; the commit SHA is appended by the operator in the next row.

**A13 (2026-10-19, critic, additive): the gpt2-medium pin in force; the 355M plain-depth job's conditions; det1 recorded and the
determinism pair's requirements.** Written after the pin-only prep `prep-pin-medium-main` (`silent-prep-20261019-164733-2c65`) and
det1 (`silent-train-20261019-142142-432b`) finished (critic entries 2026-10-19T17:40Z, three); every number it cites is in the fetched outputs
named there. Authority: 3.2, 3.3, 4.5, 6.1, 6.4, 7.7, 13.2, 13.6, A2 item 1a, A3.1, A3.2, A3.4, A3.5, A3.6, A7.1, A8.2 to A8.4, A10.

**A13.1 The pin (A2 item 1a; the A7.1 pattern).** `openai-community/gpt2-medium` hub revision `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`
(resolved from `main` on 2026-10-19T16:54:07Z on the prep VM), `model.safetensors` sha256
`fc5a354a19255ad494f3d71549390baca1ccf61d1d822b9408971705c687c9cd`, the seven files hashed in `pin.json` and, by `apply_prep_pin.py`,
in `configs/data_manifest.json` `models[1].files_sha256`. Rules as A7.1: every VM fetches the model by this revision, never by a
branch name; a downloaded weights hash other than the pinned one stops the job before any run; the launcher refuses a run whose
`model_id` has no pin. The pin commit changes the manifest's `models[1]` entry and nothing else (the critic diffs it at the SHA
row); `pin.json` and the job's final receipt are committed under `results/prep/`. A7.1's "no run uses them in the window" is read
as: 355M is used in the window by the 7.7 plain-depth control of the screen set (A3.4) only; 774M stays unpinned and unused. The
reference environment of A7.1 is unchanged; a VM image of the same family with a later patch date is not an environment
mismatch when every compared component (Python, the pinned packages, driver, torch, CUDA, cuDNN, transformers) is equal; the
image id is recorded per run and printed as a label where two members of a pair differ in it.

**A13.2 The plain-depth job `screen-plain-depth-355m-acctc` (7.7; A3.4).** Two runs, `gsm50k_plain_depth_355m_s0` and
`prosqa_hard_plain_depth_355m_s0`, whose configs differ from the 124M `no_cot` screen configs in `model_id`, `name`, `save_dir`
and the note only (same data, epochs, lr, batch, seed, precision, selection pool). Before the first run the launcher's memory
probe runs at the 355M model with batch 32, the longest padded `no_cot` micro-batch of the two datasets and 0 latents, its peak
in the receipt; if it fails at micro-batch 32, micro-batch 16 x grad_accum 8 (effective 128 unchanged) is allowed as a registered
deviation of 3.2's micro-batch clause, logged before the run and printed on every row and K3 line that uses the control; nothing
else of the recipe changes. The cap is provisional: the estimates file's 124M-measured rates times the parameter factor 2.8538
(1.661 h + 3.551 h) under the 1.3x + 0.5 h rule give 7.5 h; the admission line says "hours provisional"; after epoch 1 of the first
run the measured epoch seconds and the projected total are written into CHANGELOG; a job that reaches its cap is resumed by a
new admission (A3.6) whose cap is re-derived from the measurement; a running admission's cap is never raised (A3.5). Until both
runs have a selected checkpoint and a valid read, every screen K3 line uses the 124M `no_cot` point alone and carries "nearest
size"; the control is a screen number ("plain depth 355M, 1 seed") and a confirmation K3 needs its three seeds (7.7).

**A13.3 det1 and the determinism pair (6.4; A3.2; A8.3; 13.6).** `prosqa_cot_det1_s0` (attempt `silent-train-20261019-142142-432b`,
A100-80GB, fp32-tf32, zero resumes, code `rev128` whose training path equals `rev61`'s; selected epoch 25, valid 0.8133,
`ckpt_best_final.pt` sha256 `24eb701e476e59a0fbcee5865a5fb8fb0e4769390f2238907cd7191d0e304481`) is the first member of the GPU
determinism pair. The pair is complete when det2 (`determinism-acctc`, `prosqa_cot_det2_s0`, restart policy, same GPU type, the
same override set, `common/latent/` training path byte-identical to `rev128`'s, both receipts' environments equal as A13.1 reads
them) has a selected checkpoint, and both selected checkpoints have one registered test read under one critic entry (main,
fp32-tf32, pinned hashes, write-once, the batch-1 audit beside each); the two test accuracies enter no cell and no seed pool.
d_nd = the absolute difference of those two test accuracies; with the seed component of sigma_task on ProsQA now 3.60 (16:59Z),
a d_nd above 0.72 points (4 or more of 500 examples) enters in quadrature for every ProsQA verdict not yet written. The oracle
`prosqa_cot_s0` stays the K1 table's seed 0. A author-directed amendment may admit `prosqa_cot_s0` (one exact resume) as the
original, making the pair (`prosqa_cot_s0`, det1) with d_nd against the K1 read 75.0 and no second read of that checkpoint; if
written, both readings are printed. Under either reading det1's checkpoint needs one registered test read, registered by its own
row. The valid curves of det1 and `prosqa_cot_s0` may be compared and printed now as a diagnostic (6.4), never as d_nd.

**Nothing in A13 loosens a registered rule**: A13.1 records a pin A3.1 already provides for and reads A7.1's sentence against
the earlier A3.4; A13.2 adds a memory probe, a provisional-cap procedure and a labelled micro-batch fallback inside 3.2's own
clause; A13.3 records a run and names what 6.4 and A8.3 already require. Pre-commit; the commit SHA is appended by the operator in the
next row.

**A14 (2026-10-19, critic, additive): 7.10 item 5 read on a GPU; the cot s1 audit explained; A11.3 (1) met; the conditions on the
patch and the ladder.** Written after the diagnostic read `diag-k1-main` (`silent-eval-20261019-173657-9cfe`, valid split, no test
read spent; results write-once at `rev8`; critic entries 2026-10-19T18:32Z, three). Authority: 1.3, 4.5, 7.5, 7.9, 7.10, 7.11, section 8, 13.5,
A10, A11.3, A12.3.

**A14.1 7.10 item 5 on a GPU.** Item 5's criterion ("example i's logits agree within 1e-4 (fp32) whatever its batch companions;
row j changes nothing in row i") is a companion test and is met by the same-shape swap (row i and the tensor shapes fixed, every
companion replaced by an example from outside the batch): the differences must be at most 1e-4 at the registered numerics and at
strict fp32. The comparison of B = 1 against B = batch changes the padded shape, and on A100 under cuBLAS the logits then differ at
strict fp32 by up to a few 1e-3 (a kernel and reduction-order effect, not a dependence on the companions); it is reported with its
maximum per quantity and the count of greedy generations that diverge, and a count of 0 at strict fp32 is the shape pass. A
diverging generation at strict fp32, or a swap difference above 1e-4 at either numerics, is A12.3's outcome B. The diag tool
(`common/latent/diag.py`, `code/run_diag.py`; `rev132`) is the registered instrument for items 4 and 5 with a real checkpoint;
it reads the valid split only.

**A14.2 The cot s1 audit and later audits.** The `prosqa_cot_s1` audit (3 of 500 at batch 1, 16:59Z) is explained: TF32
batch-shape rounding flips near-ties (top-2 margins 1e-3 to 1e-2 at the divergent token) along the greedy chain; the same-shape
swap is 0 of 300 at both numerics; strict fp32 diverges on 0 of 300. The batch-32 read stays the number of record; the cell's
label is "audit: 3 of 500 differ at batch 1, explained: TF32 near-tie flips (diag `rev8`)"; the final-K1 sentences travel
without "BUG until explained". Later audits: on a `cot` or `no_cot` checkpoint read at fp32-tf32, an audit that differs on at most
10 of 500 with a net accuracy change of at most 0.4 points carries "explained: TF32 near-tie flips (18:35Z diagnostic)" without a
new run; a larger difference, or any difference on a latent-feedback arm (0 of 1,000 at this read), needs its own `cot_batch` or
`latent_path` run before the cell's verdict travels. The "at most 1 in 500" line of item 2 stays as the trigger.

**A14.3 A11.3 (1) met; the sentence; the patch and the ladder.** On the three Coconut checkpoints the scrambled pass fed the
donors' latents bit for bit (zero distances 0 of 1,800 pairs per checkpoint; no self-donor; every prompt column fed once), the
path is deterministic at fixed shape (own pass repeated: 0.0), the last-prompt-position logits change on 300 of 300 valid
examples under the scrambled latents and no emitted token changes on any; batch isolation on the latent path passes as A14.1.
A11.3 (1) is met. The sentence of the row of 2026-10-19T18:32Z (no measurable dependence on the latents; the scrambled pass fed other
examples' latents; the latents are read, since the logits move on every example, and decide no emitted token; the latent vectors
share a common direction, mean cosine 0.99 per rank) may be printed with its labels (KNOWN, fp32-tf32, three seeds, diagnostic,
ProsQA only, test for the intervention reads and valid for the fed-vector and logit statements); A11.3's forbidden words stay
forbidden until (2) sigma_task, (3) and (4). The F1 counterfactual patch (A11.3 (3)) and the bit-width ladder (A11.3 (4), 7.9) run
on the valid split only (the test split is never scored under an intervention outside section 8's list), each after a design
written in CHANGELOG and accepted by a critic entry naming the construction, the control (for the patch: the per-rank mean latent
of the valid set fed to every example), the prediction with a number and the paired bootstrap; both through `eval.evaluate` with
the `Recorder`'s fed-vector check; write-once with the receipt; their outputs enter no cell or verdict, and "bits used per step"
is printed only when sigma_task exists.

**Nothing in A14 loosens a registered rule**: A14.1 reads item 5's own words on the hardware it now runs on and keeps the 1e-4 bar
where the companions are the variable; A14.2 closes one audit with the evidence 7.10 asks for and bounds what later audits may
borrow from it; A14.3 records a condition met and adds conditions before two instruments. Pre-commit; the commit SHA is appended by
the operator in the next row.

**A15 (2026-10-20, critic, additive): A6.2 rule 3's logits clause is function equality at float64; rule 4's lines at four
decimals; `latent_last` in rule 2.** Written after the two float64 records and their strict-fp32 GPU records: GSM-50k
(checkpoint rev180; strict-fp32 on A100 `results/diag/diag-band-gsm50k-main/gsm50k_coconut_s0_valid_band_logits`, commit
`rev15`; float64 on CPU `results/diag/diag-band-gsm50k-cpu/gsm50k_coconut_s0_valid_band_logits_cpu_float64`, commit `rev16`)
and ProsQA-hard (checkpoint rev241; strict-fp32 on A100 `results/diag/diag-band-prosqa-hard-main/prosqa_hard_coconut_s0_valid_band_logits`,
commit `rev23`; float64 on the A100-40GB (device cuda) `results/diag/diag-band-prosqa-hard-main/prosqa_hard_coconut_s0_valid_band_logits_cuda_float64`,
commit `rev22`, receipt `silent-eval-20261020-224808-0209`). Critic entries: 2026-10-20T12:40Z (row 4 and rulings (a), (d));
14:54Z row 1 (parts (4) and (5): the criterion); 15:26Z rows 1, 2, 3 and 4 (row 4: this text); 21:03Z row 3; 21:33Z (1) and (2)
(the device clause); 22:36Z (1) (check (4) is not a clause; the complete-record rule); 23:12Z (three rows: the ProsQA-hard record
read and the slots filled). Authority: A6.2 rules 2 to 4, A9.3 (b), A14.1, the tighten-only rule of this file (the original
verdict stays printed).

**A15.1 Rule 3's logits clause.** In A6.2 rule 3 the clause after "and" ("and a one-batch strict-fp32 (TF32 off) maximum absolute
difference of the first-token logits between the HF path and band [0, 12] on the real checkpoint (expected at the 1e-4 level or
below; larger is a BUG until explained)") is replaced by: "and, on the real checkpoint, the first-answer-token logits of the HF
path against band [0, L] (the PathRecorder's quantity: `run_diag.py --mode band_logits`, `latent.diag.band_logits`; the
registered instrument of A14.1). **The line is function equality at float64 on a CPU, or on the read's GPU type when the CPU
available cannot hold the shapes (the record names the device, and the dtype assertions, the read-back flags and the 1e-8 line
are the same on either)** (`--dtype float64 --strict-only`; weights and every module output asserted float64 during the checks,
the tool refusing otherwise; TF32 off, 'highest', no autocast, read back at the start and the end): max abs difference <= 1e-8 on
every example of the split's first file-order batch at B = batch and at B = 1, the fed latents <= 1e-8, 0 first-argmax
differences, 0 diverging greedy generations; and the shape comparator, the band path at B = 1 against B = batch on the same
examples plus the idx of any like-for-like deviation, <= 1e-8 likewise, with its margins. A difference above 1e-8 at float64 on
any example, or an argmax difference, is a difference of construction in the band path: BUG naming the band path (rule 1's
denominator); no f_meas is cited from it until the fix passes this same test and A9.3 (b) names the change; the timings already
in a table stay with the label, never edited. **Printed beside, no line:** the same comparison at strict fp32 on the read's GPU
type with its maximum, the count over 1e-4 and its shape comparator from check (3) (A14.1's allowance on A100: up to a few 1e-3,
a kernel and reduction-order effect), the CPU fp32 figures of the same comparisons where a CPU record exists, and the
registered-numerics figure with its diverging count (12:40Z (2)). The record names the device, torch and transformers builds,
the attention implementation the loader resolved, the checkpoint and split hashes, the commit and the row cited, and is written
once under `results/diag/` with its per-example CSV. The same-shape companion swap and the repeat (A14.1's item 5 test, check
(4)) are not clauses of this line; a float64 record that does not carry them names, in `checks_not_run`, the strict-fp32 record
whose check (4) is of record on that checkpoint and why; a record is read only when complete (`complete` true, or every
requested check present); a partial file is never a record. A GPU float64 record carries its device in every label built from
it, and no record is printed as "on CPU" unless its device is cpu. **The 1e-4 verdicts of records `rev15` (GSM-50k: BUG at
strict fp32 on A100, 32 of 32 over 1e-4, max 6.4e-4) and `rev23` (ProsQA-hard: BUG at strict fp32 on A100, 32 of 32 over
1e-4, max 7.6e-4) stay printed unedited beside their explanations (float64 record `rev16` on CPU: 2.3e-13 at B = 32, 3.4e-13
at B = 1, shape comparator 6.8e-13, CPU fp32 1.2e-4 beside; float64 record `rev22` on the A100-40GB: 1.7e-12 at B = 32,
1.4e-12 at B = 1, shape comparator 1.1e-12, no CPU fp32 figure, the CPU available not holding the shapes); 1e-4 at fp32 was not
a line two kernel paths meet on either device and is not reused as a line.** When the pinned snapshot is unavailable to the
machine running this check, the base may be built from the pinned config and tokenizer files (every file present hashed equal
to the manifest's pin; config.json and vocab.json required) with every state-dict tensor from the checkpoint, verified shape and
bit for bit after loading (`--base-config`; the record's `base` block and `checkpoint_covers`), labelled as the 2026-10-20T15:26Z
row 2 ruling fixed; this applies to this diagnostic only and to no training run or eval read (A2 item 1a, A7.1 and A13.1
unchanged)."

**A15.2 Rule 4's lines at four decimals (12:40Z (a)).** Rule 4's line is (b minus a) over L + 0.10 read at its exact value and at
its printed value, and the line of record is the stricter of the two: k = 8: 0.7667 (printed 0.77); k = 4: 0.43 printed (exact
0.4333), the k = 4 verdicts already given (GSM-50k 0.4326, ProsQA-hard 0.4391: nominal) read at 0.43. Both values are printed on
every f_meas line.

**A15.3 `latent_last` in rule 2.** Rule 2's "printed beside" names "latent_last"; the timed reads of record do not hold it
(12:40Z (d): printed as "not recorded"; the fraction is unchanged). Once the field exists in `summarize_timing`'s record it is a
required field of every later timed read and the table prints it; the reads already of record are not rerun for it.

**Nothing in A15 loosens a registered rule**: A15.1 moves the line from 1e-4 at fp32, which no pair of kernel paths met on
three records, to 1e-8 at float64, four orders below the figures measured, keeps every fp32 verdict printed as it was, and
keeps 7.10 item 5's companion test (A14.1) where the companions are the variable; A15.2 reads rule 4 at the stricter of its two
values; A15.3 names a field rule 2 already lists. Pre-commit; the commit SHA is appended by the operator in the next row.
