# common/latent: interface contract (v3, 2026-10-17; v2 corrected the critic's D1-D3, D5-D7; v3 records the fields the code added)

Importable as `latent` with `PYTHONPATH=common`. Python >= 3.10. Dependencies: torch, transformers (GPU line: >=4.46,<5;
Mac lock: 4.57.6), numpy. No wandb, no network access inside library code (downloads live in scripts). Every module
runs on cpu, cuda and mps. Tests under `common/latent/tests/` run on CPU in under two minutes, download nothing, and use
random tiny GPT-2 configs plus the `FakeTokenizer` in `tests/fake_tokenizer.py`.

Reference implementation being mirrored: facebookresearch/coconut (MIT), files saved at
`<scratch>` (coconut.py, run.py,
dataset.py, utils.py, args/*.yaml, preprocessing/*). Where this contract and the reference disagree, the reference wins
for anything that affects the Phase 1 oracle numbers; log the disagreement in CHANGELOG.md.

## Tokenizer protocol (duck-typed; HF tokenizers satisfy it)
`encode(text, add_special_tokens: bool) -> list[int]`, `decode(ids) -> str`, `eos_token_id`, `pad_token_id`,
`add_tokens([...])`, `convert_tokens_to_ids(tok)`. Special tokens, added in this order exactly as Coconut does:
`"<|start-latent|>"`, `"<|end-latent|>"`, `"<|latent|>"`. For latent arms only (coconut, pause, coconut_no_curriculum) the
input embeddings are resized and the three new rows (input and tied head) are copied from the row of the token `<<`,
exactly as Coconut run.py does; the cot and no_cot arms do not resize (registration D1, D5).

## data.py (owner: infra-2)
```python
@dataclass
class Example:            # one Coconut-format record
    question: str; answer: str; steps: list[str]; idx: int
    meta: dict            # ProsQA extras: idx_to_symbol, edges, root, target, neg_target; else {}

def load_examples(path: str, max_size: int | None = None) -> list[Example]
def tokenize_examples(examples, tokenizer) -> list[dict]
    # dict keys: question_tokenized, steps_tokenized (list[list[int]]), answer_tokenized, idx
    # exactly Coconut's tokenize_sample: question+"\n" (add_special_tokens=True), each step+"\n", "### "+answer+[eos]
def build_stage_dataset(tok_examples, *, stage: int, arm: str, c_thought: int, max_latent_stage: int,
                        pad_latent_to_max: bool, uniform_prob: float, latent_id: int, start_id: int, end_id: int,
                        rng: random.Random, for_generation: bool = False) -> list[dict]
    # arm in {"cot", "no_cot", "coconut", "pause", "coconut_no_curriculum"}.
    # Mirrors Coconut get_cot_latent_dataset / get_question_latent_dataset / no-cot path:
    #   cot: question + all steps + answer, no markers.
    #   no_cot: question + answer, no markers.
    #   coconut / pause: question + <start> + n_latent x <latent> + <end> + remaining steps + answer, where the first
    #     k = min(stage, len(steps)) steps are removed and n_latent = k*c_thought (pad_latent_to_max pads n_latent to
    #     min(stage, max_latent_stage)*c_thought when len(steps) < stage, exactly as the reference does).
    #   coconut_no_curriculum: like coconut with stage fixed at max_latent_stage from epoch 0.
    #   uniform_prob: with that probability the record is drawn from a uniformly random stage in [0, max_latent_stage].
    #   for_generation=True: question + markers + latents only (no steps/answer), as get_question_latent_dataset.
    # Each dict: input_ids, labels (-100 everywhere except steps+answer tokens; also -100 on <latent> positions, as
    # the reference), attention_mask, position_ids (plain arange; the collator rewrites), idx.
class Collator:          # mirrors MyCollator: right padding, left pad so the earliest <latent> aligns across the batch,
                         # position_ids 0 on the left pad then arange, labels -100 on pads, pad_token_id on inputs.
    def __init__(self, pad_token_id: int, latent_id: int | None, label_pad: int = -100)
    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]   # input_ids, labels, attention_mask, position_ids, idx
def answer_extract(text: str) -> str      # Coconut: text.split("#")[-1].replace(",", "").strip()
def answer_normalize(ans: str) -> str     # Coconut: ans.replace(",", "").strip()
```
Also in data.py or sibling modules owned by infra-2: `prosqa_gen.py` (ProsQA-format generator with controllable depth
and distractors, validated against the released data's statistics), `contamination.py` (13-gram overlap train vs test,
reusing the ruler experiment's check if one exists), and the download/preprocess scripts under
`code/fetch_data.py` (ProsQA from the Coconut repo; GSM8k-Aug from the iCoT
commit e06a32ee5e4cd117171daeb4755d2a97ece62761 converted with the reference gsm_icot.py semantics; ProntoQA: the exact
generator command and the reference prontoqa.py conversion). Every downloaded file gets a SHA-256 line in
`configs/data_manifest.json`. Data goes to `data/` (gitignored), never to git.

## accounting.py (owner: infra-2)
Registered convention (think-bet registration section 6, extended to GPT-2). With N_ne = non-embedding parameters
(everything except wte and wpe; GPT-2's tied head is charged separately as Vd), V = vocab size, d = n_embd,
n_L = n_layer, d_a = n_head * head_dim:
```python
def model_dims(config_or_model) -> Dims        # N_ne, V, d, n_L, d_a (computed from parameter shapes, not formulas)
def flops_train_per_sequence(dims, T: int) -> float      # 3 * [2 (N_ne + V d) T + 2 n_L d_a T^2]
def flops_prefill(dims, p: int, head_positions: int = 1) -> float   # 2 N_ne p + 2 V d head_positions + 2 n_L d_a p^2
def flops_steps(dims, p: int, n_steps: int, head_steps: int) -> float
    # n_steps sequential single-position passes after a prefix of p: 2 N_ne n + 2 V d head_steps + 2 n_L d_a [(p+n)^2 - p^2]
    # a latent step has head_steps contribution 0 (no logits needed); an emitted token has 1.
@dataclass
class InferenceRecord: prefill_tokens: int; latent_steps: int; emitted_tokens: int; sequential_passes: int; flops: float
def inference_record(dims, arm, q_len, latent_steps, emitted_tokens) -> InferenceRecord
    # Binding position table (registration 5.3; |q| = question tokens incl. the trailing newline, E' = emitted tokens
    # before eos; the eos pass is inside the E' head steps):
    #   cot, no_cot:        prefill p = |q| with 1 head position; n = E' steps with E' head steps; passes 1 + E'
    #   coconut (L latents, also scrambled / no_curriculum / step sweep): p = |q| + 1 with 0 head positions;
    #                       n = L + 1 + E' steps with 1 + E' head steps; latent_steps L; passes 1 + L + E'
    #   zero-latent (L = 0): p = |q| + 1 (0); n = 1 + E' (1 + E'); passes 1 + E'
    #   pause (L pauses):    p = |q| + L + 2 with 1 head position; n = E' (E'); latent_steps 0; passes 1 + E'
    # Worked example (124M dims from GPT2Config defaults, V = 50,260): coconut, |q| = 250, L = 6, E' = 8 gives
    # 47,248,760,832 FLOPs and 15 sequential passes; this is a test.
```
Tests check the formulas against a hand computation on a 2-layer toy and against torch's parameter counts.

## model.py (owner: infra-1)
```python
class LatentLM(torch.nn.Module):
    def __init__(self, base_causallm, latent_id, start_id, end_id, eos_id)
    def forward(self, input_ids, attention_mask, labels, position_ids, *, mode: str = "coconut",
                scramble_bank: torch.Tensor | None = None, return_latents: bool = False) -> Outputs
        # Outputs(loss, logits, inputs_embeds, latents)  latents: list per example of [n_latent, d] tensors (or None)
        # mode "coconut": reference semantics (last-layer hidden state at position j-1 becomes the input embedding at
        #   latent position j; KV cache reused; one sequential pass per latent; logits concatenated; CE on shifted labels).
        #   Must be numerically equal to the reference forward (tests port the reference and assert allclose).
        #   Vectorized: no per-token Python loops over the embedding tensor.
        # mode "pause": no feedback; latent positions keep the learned <|latent|> embedding (one pass, same tokens).
        # mode "scrambled": the fed-back vector at latent j of example i is taken from scramble_bank[i, j] (built by
        #   eval from other examples), never from example i's own hidden states. Blindness is unit-tested.
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, position_ids, *, max_new_tokens: int, mode: str = "coconut",
                 scramble_bank=None, return_latents: bool = False) -> GenOutputs
        # Batched greedy decoding with KV cache; per-example eos stop; GenOutputs(tokens: list[list[int]],
        # emitted_tokens: list[int], latent_steps: list[int], sequential_passes: list[int], latents).
        # Right-padded batches: tests assert batched output == per-example output.
def load_base(model_id_or_path, dtype, device, resize=True) -> (model, tokenizer)   # HF GPT-2 family + the Qwen3
    # bases; adds the three special tokens; resize=False for cot/no_cot (no embedding resize, as the reference)
```

## train.py (owner: infra-1)
```python
@dataclass
class TrainConfig:  # mirrors the reference yaml fields plus ours
    name: str; arm: str; model_id: str; load_model_path: str | None; train_path: str; val_path: str
    c_thought: int; epochs_per_stage: int; max_latent_stage: int; pad_latent_to_max: bool; uniform_prob: float
    num_epochs: int; lr: float; weight_decay: float; batch_size: int; grad_accum: int; seed: int; resume: int
    reset_optimizer: bool; save_only_improve: bool; precision: str  # "fp32" | "bf16" | "fp16"
    device: str; save_dir: str; max_new_tokens: int; eval_batch_size: int; max_train_examples: int | None
    max_val_examples: int | None; optimizer_reset: str  # "epoch" (reference) for latent arms
    auto_resume: bool; log_every: int; load_model_epoch: int | None; load_model_sha256: str | None  # D15
def train(cfg: TrainConfig) -> dict   # runs the epoch loop: scheduled_stage = 0 for cot/no_cot else epoch // epochs_per_stage
    # (capped at max_latent_stage by the dataset builder); rebuild the stage dataset each epoch; when reset_optimizer
    # is True (all latent-arm reference yamls) AdamW is rebuilt at the start of EVERY epoch, as reference run.py does
    # (cot and no_cot set it False); validation loss and generation accuracy every epoch via eval.py; writes
    # save_dir/log.jsonl (one row per epoch: stage, train_loss, val_loss, val_acc, tokens, flops, seconds), config.json,
    # ckpt_best_final.pt (best valid accuracy among FINAL-stage epochs, strict >, ties to the earliest epoch: the
    # selected checkpoint, registration 3.3), ckpt_best.pt (all-epoch best, for the labelled secondary read) and
    # ckpt_last.pt (state_dict only; `runs/` is gitignored); log.jsonl records the stage of every epoch. GSM coconut
    # starts from the named cot checkpoint with `resume: 3` semantics (registration D15). Seeds torch, numpy, random;
    # data order is seeded per epoch. Effective batch = batch_size * grad_accum (the reference uses 32 x 4 GPUs = 128).
```

## eval.py (owner: infra-1)
```python
@dataclass
class ExampleResult: idx: int; correct: bool; cot_match: bool; pred: str; gold: str; text: str; prefill_tokens: int;
                     latent_steps: int; emitted_tokens: int; sequential_passes: int; flops: float; latents: list | None
def evaluate(model, tokenizer, examples, *, arm: str, n_latent: int, batch_size: int, device: str,
             intervention: str | None = None, max_new_tokens: int, return_latents: bool = False) -> list[ExampleResult]
    # max_new_tokens has no default: the reference uses 128 for ProsQA and ProntoQA and 64 for GSM (`"gsm" in val_path`);
    # every config states it and the read script passes it explicitly (registration D3).
    # Builds generation prompts with build_stage_dataset(for_generation=True, stage=n_latent-equivalent), runs
    # LatentLM.generate, extracts the answer with answer_extract, computes accounting via accounting.inference_record.
    # intervention: None | "scrambled" (bank = latents of a derangement of the batch within the same latent-count class)
    #   | "zero_latent" (n_latent forced to 0). The step sweep is n_latent in {0, 1, 2, 4, 8}.
def summarize(results) -> dict   # accuracy, mean/median of each cost axis, bootstrap 95% CI over examples (10k draws, seeded)
```

## Shared conventions
- `runs/`, `data/`, `artifacts/` are gitignored at the repo root; nothing over 5 MB is ever committed.
- Logging is JSONL plus stdout; no wandb.
- Every public function has a docstring that names the reference function it mirrors.

## v4 additions (2026-10-18, Phase 2 cards C1b, C2, C3; registration A3.4, A3.2, A3.6). Every default leaves v3 unchanged.
- `model.LatentLM(..., latent_band=(a, b))` (C1b): at latent positions only blocks [a, b) run (non-band blocks are
  skipped, their KV columns are zeros masked out for every query); the fed-back vector is h_b(j-1) (HF
  hidden_states[b]; with a = 0 it replaces the token embedding, so (0, L) is the full-depth forward). GPT-2 family only.
  `lm.layer_fraction` = (b - a) / L. `generate(..., time_steps=True)` fills `GenOutputs.timing` (wall-clock per prefill,
  latent and token pass; `summarize_timing`). Logits at latent positions are zeros on the band path (labels -100).
- `accounting.flops_steps(..., layer_fraction=1.0)`, `inference_record(..., layer_fraction=1.0)` (latent steps only),
  `flops_train_per_sequence(dims, T, latent_positions=0, layer_fraction=1.0)` (2 N_ne term at latent positions only).
- `eval.evaluate(..., timing_out=None)`: a list receives one timing dict per batch; a band LatentLM is charged its
  layer fraction.
- `state_targets.py` (C2, pure functions of a training record): ProsQA frontiers F_t (BFS, root excluded), GSM step
  results ("<<expr=c>>", <= 4 tokens), `supervised_steps` (latent t * c_thought - 1 carries step t), `target_stats`
  (fraction of records whose last supervised target contains the answer), `shuffled_donors` (C2-shuffled).
- `aux_losses.py` (training only; never imported by model.py / eval.py): `StateDecoder` (2-layer MLP, GELU, hidden
  768, readout through the detached embedding matrix), `state_loss`, the C3 hinge (`per_example_ce`, `hinge`,
  `batch_derangement`, `build_bank`, `train_derangement_seed` != `EVAL_SCRAMBLE_SEED` = 0).
- `train.TrainConfig` fields `latent_band`, `state_sup`, `state_lambda` (0.1), `state_null` ("none" | "shuffled"),
  `dep_loss`, `dep_margin` (0.2), `dep_beta` (1.0), `dep_null` ("none" | "foreign"); D18 `set_numerics` /
  `numerics_state` (TF32 off for fp32, logged in every row); D17 `optimizer_last.pth` beside the latest checkpoint for
  arms without a per-epoch reset; D19 resume records in ckpt_state.json and log.jsonl, `runs_index_row.csv`.
  Files beside the checkpoints: `state_decoder_{last,best,best_final}.pth` (C2), `state_targets.json`.
- Frozen accessors (review F8, 2026-10-18; a change after this line is a deviation): `LatentLM.embedding` (the wrapped model's input embeddings, `get_input_embeddings()`), `LatentLM.backbone` (the transformer body, `base_causallm.base_model`; with `lm_head` it equals the causal LM's forward: hidden_states[-1] then logits), `LatentLM.lm_head` (`get_output_embeddings()`, tied to the input embeddings in GPT-2). Properties, not modules of the wrapper: excluded from its state_dict, used by probe/patch.py and probe/linear.py. `LatentLM.__init__` inherits the base model's train/eval mode (review F1).
- `accounting.reference_cost_flops(dims, p0, passes)` and `InferenceRecord.reference_flops` / `ExampleResult.reference_flops` (registration 5.5 (c), amendment A6.4, 2026-10-19): the reference code's own cost, the head at every position and no KV cache, pass k over the prefix p0 + k; `summarize` reports `reference_flops_mean` and friends through `COST_AXES`; run_eval.py writes the `reference_flops` CSV column. Secondary accounting only, never a verdict.
