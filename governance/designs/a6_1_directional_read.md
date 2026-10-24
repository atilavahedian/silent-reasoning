# Design: the A6.1 directional read on GSM-50k (the donor-answer read) for C3, C3-foreign and C2+C3

Status: **design note, pre-build** (critic entry 2026-10-20T17:12Z part (5): "the donor-answer read enters the code only
after a design row in CHANGELOG and a critic entry that names the class rule, the own-latents baseline, the paired
bootstrap and a numerical prediction (the A14.3 pattern)"; the operator's SHA row of 17:12Z: "a design note goes to
CHANGELOG and the critic before any code"). Written 2026-10-20 while `screen-gsm50k-accta` (attempt
`silent-train-20261020-170444-6ece`, AcctA, us-central1-c, deadline 2026-10-21T09:04:44Z) is training; no number of
any of the three arms exists. Nothing here is code, a launch, or an edit to a registered file. Every "registration
fixes" sentence cites its section; every "proposal" is marked as such and is the critic's to accept, narrow or refuse by
a row. Written for an Opus 5.5 (xhigh) build agent and for the critic's registering row.

Scope: GSM-50k only, **valid split only** (`data/gsm_valid.json`, 500 records, manifest sha256
`1833a3479186e4e6c0183aad32fb7ee343479c958f2a1b57631ec22f5ee3aa7f`, the `screen/gsm50k/valid` entry and the oracle's
`tasks/gsm/files[3]`), the three selected screen checkpoints (`ckpt_best_final.pt` of `gsm50k_c3_s0`,
`gsm50k_c3_foreign_s0`, `gsm50k_c2c3_s0`; selection pool epochs 12-24, A3.4.5; identified by RUN_DONE hash, A10.2 and
17:12Z part (4)), seed 0. ProsQA-hard's directional read (the F1 patch, A6.1 rule 4 (B) first clause) is **UNTESTED**
under A2.1 (no ProsQA-hard hinge job; 17:12Z part (8)) and is not designed here.

---

## 1. What the registration requires, and what "donor-answer read" means operationally

### 1.1 The rule, in the registration's words

A6.1 (K2 is not independent for arms trained with the dependence hinge: C3, C2+C3; review F3) states the fact and the
rules. Fact: "the hinge maximises the cross-entropy of the answer under foreign latents minus under the model's own
latents, which is the quantity K2's Delta_scr measures at inference. A model satisfies it with a fingerprint (one
latent carries a check of the question; the answer head emits garbage on mismatch) and no reasoning state". Rules, for
every arm whose config has `dep_loss: true` (all three arms here: `configs/screen/gsm50k_{c3,c3_foreign,c2c3}.json`
carry `dep_loss: true`; C3-foreign carries `dep_null: foreign`, C2+C3 `state_sup: true`; 17:12Z part (3)):

- Rule 1: the K2 cell is printed **"K2 not independent"** at the screen and at confirmation; a fired K2 still kills; a
  non-fired K2 is not evidence that the latents carry state.
- Rule 2: the C3 minus C3-foreign difference on the scrambled drop is reported as **"dependence on own latents (content
  or fingerprint)"**.
- Rule 3: the pause-mode and zero-latent reads of such a checkpoint are diagnostics (section 10) and never decide.
- **Rule 4**: "The non-circular reads are: (A) the **accuracy leg** (accuracy vs full-depth `coconut`, vs C3-foreign, vs
  the bar), as the card's lines require; (B) a **directional intervention read**. [...] On GSM-50k, where
  `PAIR_RULES["gsm"]` is empty: the **donor-answer read**, under the registered scrambled derangement (whole set, seed
  0): among examples whose donor's last supervised step result equals the donor's answer (the 0.770 class of A5 item
  5), the rate at which the generated answer equals the donor's answer, against the same rate under own latents, paired
  bootstrap interval excluding 0. The operator may register a minimal-pair rule for GSM by a row before the read; it then
  replaces the donor-answer read (one directional read per task). At the screen the directional read uses the valid
  split and the screen checkpoint; at confirmation the test split, as 7.9 says."
- Rule 5: "A large scrambled drop whose directional read has an interval including 0, or whose garbage fraction exceeds
  its directional fraction, is labelled **"drop without direction (fingerprint-consistent)"**, and K2 counts as fired
  for that arm's latent-reasoning claim on that task."
- Rule 6: "If the directional read is absent at the screen read, the outcome is written "drop leg not independent;
  directional read pending"; the card's pass and kill lines apply as registered, and: (a) the directional read on the
  screen checkpoint is admitted (an eval job, in the ledger) no later than the arm's first confirmation seed on that
  task; (b) no K2 sentence, no "latents carry state" wording and no verdict naming the arm is written until the
  directional read exists on the checkpoint it refers to; (c) if it is never produced for a task, the arm's K2 on that
  task is **"not independent; direction UNTESTED"** and the arm cannot be the positive read of A1 on that task."
- Rule 7: the readout labels of 7.9 are unchanged and still need the positive control; "the directional read of this
  section answers K2 independence, a separate question."

Critic entry 2026-10-20T17:12Z part (5) applies rule 6 from the start to the three arms ("the job does not carry it ...
the read does not exist in the code"), restates rule 4 (B) with "the garbage fraction printed beside it", and names
what the registering row must contain (the class rule, the own-latents baseline, the paired bootstrap, a numerical
prediction). Part (6) registers the follow-up eval job in form (24 reads: `none`, `scrambled`, `zero_latent`,
`n_latent` in {0, 1, 2, 4, 8} per arm; no timed read; valid only; bf16 autocast over TF32 on; "the donor-answer entry
only if its design row exists by then"). Part (7): every number is a screen number, 1 seed, valid; A11.3's forbidden
words stay forbidden. Part (9): the fourth GSM-50k table (the third table's reissue, citing `rev12`) closes the
round for the three arms; the single test read waits for it.

### 1.2 The operational meaning (what is fed, what is predicted, what counts as the direction)

**Fixed by the registration** (section 8 "scrambled-at-inference"; A6.1 rule 4 (B); A3.4's test clause on seeds;
INTERFACE.md `model.py` mode "scrambled" and `eval.py`):

1. **The pass.** The registered scrambled-at-inference construction, exactly as `latent.eval.evaluate(intervention=
   "scrambled", scramble_seed=0)` runs it on the whole valid set: a first pass with own feedback records every
   example's six fed-back latent vectors (final regime of the GSM-50k recipe: `c_thought` 2 x `max_latent_stage` 3 =
   6 latents for every example, `pad_latent_to_max` true, so there is **one latent-count class of 500**); then
   `scramble_donors({idx: 6}, seed=0)` draws one Sattolo derangement pi over idx 0..499 (pi(i) != i for every i); the
   example i is regenerated with the vector at **every** latent rank j taken from `bank[i, j] = latents[pi(i), j]`, its
   donor's own latents bit for bit, never its own, never a mixture; prompt, markers, count and positions unchanged;
   greedy decoding, `max_new_tokens` 64, eval batch 32, file order. The seed is `EVAL_SCRAMBLE_SEED` = 0 and never a
   training derangement's seed (`aux_losses.train_derangement_seed`; `test_derangements_and_seeds`).
2. **The donor.** pi(i) is the donor of i. The donor's record is the valid record at idx pi(i): its question is not
   shown to the model; only its six latent vectors (computed by the same checkpoint on the donor's own prompt) are fed.
3. **What is predicted.** The generated answer of example i under the donor's latents, `pred_scr(i)` =
   `answer_extract` of the decoded text, compared with `answer_normalize` of a gold string (the reference's scoring,
   `eval.py`). Two golds matter: the example's own answer `gold(i)` and the donor's answer `gold(pi(i))`.
4. **The class.** "Among examples whose donor's last supervised step result equals the donor's answer": the donor's
   supervised steps at the final regime are steps 1..min(n_steps(donor), 3) (`state_targets.supervised_steps` with
   n_latent 6, c_thought 2: latent t x 2 - 1 carries step t; three supervised steps), so the "last supervised step" is
   step `min(n_steps(donor), 3)`, and the class test is `state_targets.answer_in_last_target(record_targets(donor),
   donor.answer, n_latent_steps=3)` for the GSM kind: the text after the last "=" of that step's `<<expr=c>>`,
   normalised, equals the donor's normalised answer. This is a pure function of the donor's valid record (steps and
   answer), computed on the valid split, never on test. It is the construction behind A5 item 5's "0.770 on GSM-50k at
   3 latent steps" (that figure is the **training** subset's; the valid split's count is given in 1.3).
5. **The direction.** The donor-answer rate under scrambled latents, R_dir = the fraction of class members with
   `pred_scr(i) == gold(pi(i))`, against the same rate under own latents, R_own = the fraction with
   `pred_own(i) == gold(pi(i))` where `pred_own(i)` is the generated answer of the own-feedback pass (the coincidence
   baseline: how often the model's own answer happens to equal a stranger's answer). The directional read is **present**
   iff the paired difference Delta_dir = R_dir - R_own has a paired example-bootstrap interval excluding 0 (and
   Delta_dir > 0; a negative interval is "absent", not a direction). Rule 5 adds the garbage fraction: predictions under
   the donor's latents that are **neither** the own answer nor the donor's answer.
6. **The null.** The registration fixes the null of the directional read as **the own-latents pass of the same
   checkpoint on the same examples** ("against the same rate under own latents"), paired per example. It does **not**
   use A14.3's per-rank mean-latent control (that control is registered for the F1 counterfactual patch and the
   bit-width ladder on the Coconut oracle checkpoints, "the per-rank mean latent of the valid set fed to every
   example"; not for this read), and it does not use the foreign or training banks (those are training constructions,
   `aux_losses.build_bank`, never imported by `eval.py`). The C3-foreign arm is the null **arm** (A3.4: the same hinge
   between two foreign banks, no own-vs-foreign direction), so its own directional read is the reference line printed
   beside C3's (proposal 1.4 (e)); the C3-foreign **config** is not a null of this read.

**Proposals** (marked; none changes the quantities above):

- (a) **Ambiguous pairs.** A class member whose own answer equals its donor's answer (`gold(i) == gold(pi(i))`) can
  carry no direction (the same prediction counts for both golds). Proposal: exclude them from the scored set, print
  their count and idx; the registration's class wording does not mention them. Under seed 0 there are 6 (1.3).
- (b) **The garbage fraction on GSM.** Rule 4 (B)'s ProsQA clause defines garbage as "patched answers that are neither
  the own nor the predicted answer"; for GSM the 17:12Z row says "the garbage fraction printed beside it". Read
  literally on GSM (G_scr = the fraction of class members whose scrambled prediction is neither gold), the ordinary
  error rate of a 10-30% accurate model (every wrong answer is "neither") makes G_scr exceed any plausible R_dir for
  every arm, hinge or not, so rule 5's second clause would fire on GSM whatever the latents do. Proposal: print G_scr
  and, beside it, the **excess garbage** Delta_g = G_scr - G_own (G_own = the same fraction under own latents, paired,
  same bootstrap draws), and read rule 5's "garbage fraction exceeds its directional fraction" on GSM as **Delta_g >
  Delta_dir** (the scramble-induced rise in neither-answer predictions against the scramble-induced rise in
  donor-answer predictions). Both comparisons are printed; which one labels is the critic's (open question 6.3).
- (c) **Scored set.** The statistic is over the class minus the ambiguous pairs; the complement (donors with 4 or more
  steps, whose step-3 result is an intermediate value) is printed as a diagnostic with the same quantities, labelled
  "complement class (donor's step-3 result is not its answer); diagnostic".
- (d) **Secondary diagnostics, decide nothing** (section 10): the rate at which `pred_scr(i)` equals **any** step result
  of the donor (an intermediate-state read), the rate at which it equals any step result of the own record, and the
  fraction of class members whose donor's answer also occurs among the own record's step results (a confound: 14 of
  295 under seed 0, printed and excluded from nothing).
- (e) C3 minus C3-foreign on Delta_dir, with an interval from the same bootstrap draws over the common scored set, as a
  diagnostic beside rule 2's drop difference.

### 1.3 The class on the registered valid file under the registered derangement (computed, not estimated)

Computed from `data/gsm_valid.json` (hash verified against the manifest) with a pure-Python copy of
`latent.eval._derangement` and `scramble_donors` (random.Random(0), Sattolo, one class of 500); the build's data-pin
test (section 2.6, T8) asserts these numbers:

- valid step counts: 1: 32, 2: 155, 3: 140, 4: 91, 5: 53, 6: 18, 7: 8, 8: 3 (327 of 500 have at most 3 steps);
- records whose own step min(n_steps, 3) result equals the own answer: **295 of 500 (59.0%)**; the training subset's
  0.770 (A5 item 5) does not transfer because the valid split has more long problems (173 of 500 with 4 or more steps);
- the donor-answer class under seed 0, C = {i : donor pi(i)'s step min(n, 3) result equals pi(i)'s answer}: **|C| =
  295** (donor step counts in C: 1: 14, 2: 146, 3: 133, 4: 2);
- ambiguous pairs in C (own gold equals donor gold): **6**, idx 15, 124, 125, 175, 470, 477 (9 over all 500);
- **scored set: 289**;
- 180 distinct answers in valid; the chance that two independent golds agree, sum of p^2 = 0.0134, so the coincidence
  baseline R_own is expected near 0.01 (a wrong own answer that happens to be the donor's);
- resolution: at 289 examples a directional rate 3 points above baseline (about 9 examples) has a bootstrap SE near
  0.010 and a lower bound above 0; the read resolves roughly 9 or more directional examples.

The first ten class members and their donors (idx, donor): (0, 431), (2, 459), (3, 201), (6, 419), (8, 366), (9, 108),
(10, 196), (11, 326), (12, 118), (13, 153): the build's record must show exactly these (the derangement is seeded and
the idx set is the file order, `load_examples`: idx = position).

---

## 2. The instrument

### 2.1 Choice: a new `run_diag.py` mode, `--mode donor_answer`, not a `run_eval` intervention and not a new tool

- Not a `run_eval.py` intervention: the directional read is a **scoring** of the registered scrambled pass, not a new
  intervention; the registered per-example file (13.3, `CSV_FIELDS` in `run_eval.py`) carries no prediction strings
  (section 1.3: no example text), so the registered `scrambled` read cannot be scored for the donor's answer after the
  fact, and adding columns to the registered read's file would touch the write-once read format (A10.2) for all nine
  arms. `run_eval` stays as it is.
- Not a new tool: A14.3 requires the fed-vector evidence "through `eval.evaluate` with the `Recorder`'s fed-vector
  check"; `latent.diag.scramble` already runs `evaluate(None)` and `evaluate("scrambled")` under a `Recorder` and
  records, per example, the donor, `fed_equals_donor` (bit for bit), `donor_is_self`, the per-rank L2 and cosine of fed
  vs own, the first-answer-token logits and argmax under both passes, `correct_own`, `correct_scr`, the
  prompt-column coverage, and the `bug` list (zero distance, self donor, a non-donor vector, a column not fed exactly
  once, wrong pass counts). The donor-answer read is that function plus a scorer over the `ExampleResult` objects it
  already holds in memory (`pred`, `gold` strings) and the valid records' steps. The launcher already admits diag
  entries (`jobs.py` `EVAL_TOOLS` ("eval", "diag"); `DIAG_FORBIDDEN`: no intervention, n_latent, audit_rescore,
  registered_read, time_steps, set; `DIAG_NUMERICS`: no precision or tf32 on the entry), runs `run_diag.py --mode
  <mode> --split valid --tf32 both`, writes the record write-once under the job's bucket and `results/`, and the
  12:40Z row registered `band_logits` this way ("a diagnostic ... which the operator admits by citing this row"). The
  donor-answer read is registered the same way: by a critic entry the code names (a `DONOR_ROW` constant, as
  `BAND_ROW` = "2026-10-20T12:40Z"), refused without it.
- A6.1 rule 6 (a) calls it "an eval job, in the ledger": a diag entry inside an eval job is that. this runs on valid,
  `--split` anything else refused before the model loads), and its `registered_read` field stays null with
  `critic_row` naming the registering row (the `band_logits` convention).

### 2.2 Inputs

| input | value (fixed) | source |
|---|---|---|
| checkpoint | `ckpt_best_final.pt` of each of `gsm50k_c3_s0`, `gsm50k_c3_foreign_s0`, `gsm50k_c2c3_s0` in AcctA's bucket (`silent/runs/<run>/`), `checkpoint_sha256` = RUN_DONE's `checkpoints.ckpt_best_final.pt` = `ckpt_state` `best.final`, epoch in the pool 12-24 (17:12Z (4); A8.2; A10.2: a mismatch is no read) | the critic's read row for the three arms |
| config | `configs/screen/gsm50k_{c3,c3_foreign,c2c3}.json` unchanged (`arm` coconut, `c_thought` 2, `max_latent_stage` 3, `max_new_tokens` 64, `eval_batch_size` 32, `precision` bf16) | A3.4.5; 17:12Z (3) |
| split | valid only: `data/gsm_valid.json`, 500, manifest hash `rev184...`; `split_sha256` written and checked | 1.3; A3.4; rule 4's last sentence |
| n_latent | 6 (the final regime, `final_latents(cfg)`); no sweep, no zero-latent variant of this read | 3.3 item 4; A3.4.5 |
| derangement | `scramble_donors(seed=0)`, whole set, one class; the record's `donor` column must equal it (test) | section 8; A3.4's seed clause |
| batch size | the config's `eval_batch_size` 32, file order (the registered reads' batches; the same shapes as the `none` and `scrambled` reads, so the identification check in 2.4 expects 0 differences on a latent-feedback arm, A14.2) | 10:51Z (2) |
| numerics | **record of record: the checkpoint's registered set, bf16 autocast over TF32 on** (`REGISTERED_FLAGS["bf16"]`: matmul.allow_tf32 True, cudnn.allow_tf32 True, "high"; `autocast_ctx("bf16")`), A10.1 read for a screen checkpoint, A10.5; the strict-fp32 set (TF32 off, no autocast) is run **beside** it as a diagnostic of numerics sensitivity (the diag tool's `--tf32 both` convention, A14.1), never as the record of record; `--tf32 off` alone is refused for this mode | A10.1; A10.5; A14.1; 12:40Z |
| class rule | `answer_in_last_target(record_targets(donor), donor.answer, 3)`, GSM kind; the tokenizer is the config's (the pinned GPT-2) and enters only `record_targets`' token lists, which the rule does not read | A6.1 rule 4 (B); A5 item 5; `state_targets.py` |
| bootstrap | 10,000 paired example draws over the scored set, numpy `default_rng(0)`, the same draws for every quantity | 6.2; read_screen's K2 construction (10:51Z (2)) |
| GPU | A100-40GB on main (the project of every eval so far) or A100-80GB on AcctA (section 3); the record carries `gpu_type`, torch, transformers, the commit and the label "read on A100-40GB, trained on A100-80GB" where it applies | A3.2; A10.4 |
| code | `common/latent/diag.py` (+ the scorer), `code/run_diag.py` (+ the mode), `code/launch/silent_launch/{jobs,make_jobs}.py` (the entry), `code/read_screen.py` (the fourth table's consumer); the eval path (`model.py`, `eval.py`, `data.py`, `accounting.py`, `run_eval.py`) unchanged since `rev166` (git diff empty, named in CHANGELOG, as 10:51Z (5) requires) | A6.5; 10:51Z (5); 12:40Z |

The mode's refusals (each a test): a split other than valid; a non-feedback arm (`LATENT_FEEDBACK_ARMS`); a
`latent_band` config; an existing `<out>/summary.json` (write-once); a `critic_log.md` without the registering row;
`--tf32 off`; a `dep_loss: false` config is **allowed** (the read is defined for any latent-feedback checkpoint; open
question 6.8 asks whether `gsm50k_coconut_s0` is run as a reference) and the record then carries "reference arm (no
hinge); diagnostic".

### 2.3 Computation (one entry = one checkpoint)

1. Load the model from the checkpoint (hash verified by the launcher against the entry's `checkpoint_sha256` before the
   run; `run_diag` writes `checkpoint_sha256` and `split_sha256` into the record).
2. For each numerics set (registered, then strict): `latent.diag.scramble(...)` unchanged: `evaluate(None)` and
   `evaluate("scrambled", scramble_seed=0)` under Recorders; keep its per-example rows, per-rank rows and `bug` list.
3. Scorer (new, pure, in `diag.py`: `donor_answer_rows(examples, own_results, scr_results, donors, tokenizer,
   n_latent_steps=3)` and `donor_answer_summary(rows, n_boot, seed)`), per example i with donor j = donors[i]:
   - `in_class` = answer_in_last_target(record_targets(ex_j), ex_j.answer, 3) (donor record only);
   - `ambiguous` = normalize(gold_i) == normalize(gold_j); `scored` = in_class and not ambiguous;
   - `own_n_steps`, `donor_n_steps`, `donor_last_sup_step` = min(donor_n_steps, 3);
   - `correct_own` = pred_own == gold_i; `correct_scr` = pred_scr == gold_i (own answer survives the scramble);
   - `d_own` = pred_own == gold_j; `d_scr` = pred_scr == gold_j (the donor-answer events);
   - `g_own` = not correct_own and not d_own; `g_scr` = not correct_scr and not d_scr (neither gold);
   - secondary: `scr_eq_any_donor_step`, `scr_eq_any_own_step`, `donor_gold_in_own_steps`;
   - the inherited columns from `scramble`: donor, donor_is_self, fed_equals_donor, fed_equals_collected, fed_min_l2,
     fed_max_cos, first_maxabs_own_scr, first_argmax_own, first_argmax_scr, stream_div_own_scr, emitted counts.
   Predictions are compared as strings after `answer_normalize`; **no pred, gold or text string is written** (1.3; the
   convention of every per-example file here, valid included).
4. Summary over the scored set S (|S| = 289 expected): R_dir = mean d_scr, R_own = mean d_own, Delta_dir = R_dir -
   R_own; G_scr = mean g_scr, G_own = mean g_own, Delta_g = G_scr - G_own; acc_own and acc_scr over all 500 (these must
   equal the registered `none` and `scrambled` reads' accuracies on the same checkpoint); counts beside every fraction.
   Bootstrap: 10,000 draws of |S| indices with replacement, `default_rng(0)`, one index matrix applied to d_scr, d_own,
   g_scr, g_own (paired across quantities and across the two passes); Var_ex = the variance of the draw means of
   Delta_dir (and of Delta_g); interval = Delta +- 1.96 sqrt(Var_ex) (6.2's form as read_screen prints K2: "Delta +-
   1.96 sqrt(Var_ex), Var_ex over 10,000 paired example draws, default_rng(0)"), with the percentile [2.5, 97.5]
   interval printed beside. **No sigma_task or sigma_screen term**: two interventions on one checkpoint, one seed (the
   K2 pattern of 7.5 at the screen); open question 6.4.
5. Identification check (proposal, inherited from the 10:51Z identification rule): if the registered `none` and
   `scrambled` reads of the same checkpoint exist in the job's bucket (`<job>/<run>_valid_none/examples.csv`,
   `<run>_valid_scrambled/examples.csv`), compare `correct_own` and `correct_scr` with their `correct` columns example
   by example; counts written; **0 of 500 expected on a latent-feedback arm at the same numerics and shapes (A14.2)**;
   more than 1 in 500 on either sets `held: identification check failed` (A12.3's holding pattern) and the record
   lifts nothing until the critic explains it by a row. When the reads are not present (the diag entry runs first, or
   in a separate job), the check is `not run` and read_screen performs it from the committed files at table time.
6. Outcome field (the tool's words; the label of record is the critic's and read_screen prints only labels verbatim
   from its row, as the third table's tool refuses otherwise):
   - `BUG` if `scramble`'s bug list is non-empty (no statistic is printed as a result; the numbers stay in the file
     marked `bug: true`);
   - `held: identification check failed` as in 5;
   - `directional read present` iff Delta_dir > 0 and Delta_dir - 1.96 sqrt(Var_ex) > 0;
   - `directional read absent (interval includes 0)` otherwise (a negative Delta_dir with an interval excluding 0 is
     also "absent", printed with its sign);
   - the garbage clause, printed as two booleans: `garbage_literal_exceeds` = G_scr > R_dir and
     `garbage_excess_exceeds` = Delta_g > Delta_dir; rule 5's label is applied by read_screen with the scrambled drop
     and the reading of the clause the critic registers (6.3).

### 2.4 Outputs

`<out>/examples.csv` (one row per numerics set and example; ids, booleans and numbers only): `tf32`, `idx`, `donor`,
`donor_is_self`, `fed_equals_donor`, `fed_equals_collected`, `fed_min_l2`, `fed_max_cos`, `in_class`, `ambiguous`,
`scored`, `own_n_steps`, `donor_n_steps`, `donor_last_sup_step`, `correct_own`, `correct_scr`, `d_own`, `d_scr`,
`g_own`, `g_scr`, `scr_eq_any_donor_step`, `scr_eq_any_own_step`, `donor_gold_in_own_steps`, `first_maxabs_own_scr`,
`first_argmax_own`, `first_argmax_scr`, `stream_div_own_scr`, `emitted_own`, `emitted_scr`.

`<out>/fed_vectors.csv`: as the `latent_path` mode writes it (tf32, idx, rank, donor, l2, cos, own_norm, fed_norm):
3,000 rows per numerics set (500 x 6).

`<out>/summary.json`: the `run_diag` common block (tool, mode "donor_answer", `answers` = "A6.1 rule 4 (B), the
donor-answer read on GSM-50k (critic entries 2026-10-20T17:12Z part (5) and <the registering row>)", task, run, arm, seed,
config, overrides, checkpoint, checkpoint_sha256, split, split_file, split_sha256, n 500, batch_size 32,
max_new_tokens 64, n_latent 6, scramble_seed 0, tol, n_boot 10000, boot_seed 0, tf32_modes, numerics per set,
precision, device, gpu_type, torch, transformers, commit, registered_read null, critic_row, bug, flags, files,
examples_csv_sha256, seconds, utc) and, per numerics set, `results[<set>].donor_answer`:
`{class_rule, n, n_class, n_ambiguous, ambiguous_idx, n_scored, k_dir, r_dir, k_own, r_own, delta_dir, lo, hi,
pct_lo, pct_hi, k_g_scr, g_scr, k_g_own, g_own, delta_g, lo_g, hi_g, garbage_literal_exceeds, garbage_excess_exceeds,
acc_own, acc_scr, drop (acc_own - acc_scr, printed for context only; the registered drop is the eval reads'),
identification: {none_diff, scrambled_diff, entries, status}, complement: {the same quantities over the complement
class}, secondary: {any_donor_step_rate, any_own_step_rate, donor_gold_in_own_steps}, outcome}` plus
`results[<set>].scramble` (the fed-vector block as `latent_path` writes it) and, for the strict set, `strict_vs_
registered: {delta_dir_strict, outcome_strict, examples_whose_d_scr_differs}`.

Default `--out`: `results/<run>/valid_diag_donor_answer/`; in the job: `results/evals/<job>/<run>_valid_donor_answer/`
beside the 24 reads, with the receipt (A10.2 write-once in the bucket and in git).

The **pass/fail line** the record prints (one line per checkpoint, the registered set):
`directional read (A6.1 rule 4 (B), donor-answer, GSM-50k valid, seed 0, <run> <epoch> <sha8>): R_dir <k_dir> of
<n_scored> = <r_dir> vs R_own <k_own> of <n_scored> = <r_own>; Delta_dir <+x.xxx> [<lo>, <hi>] (6.2 form, 10,000 paired
draws, default_rng(0); percentile [..]): present | absent; garbage G_scr <..> (G_own <..>, excess <+..> [..]);
identification 0/0 of 500 vs the none / scrambled reads; fed vectors: 0 zero distances of 3,000, 0 self donors,
columns fed once; bf16 autocast over TF32 on; A100-40GB (trained A100-80GB); screen number, 1 seed, valid, no test read;
ProsQA-hard UNTESTED (A2.1)`.

Labels on every number (17:12Z (7); A3.3; A10.4): "screen number: 1 seed, valid; bar LR-tuned 3 trials; arm 1 trial;
diagnostic entry (no 4.5 read spent); bf16 autocast over TF32 on; read on A100-40GB, trained on A100-80GB" (or the
AcctA GPU if run there); the A11.3 words do not appear in the record.

### 2.5 The prediction (the A14.3 pattern asks for one with a number)

The operator's, for the critic to replace or adopt in the row (A3.4: predictions are the critic's, before any run): R_own
= 0.010 [0.003, 0.025] on every arm (coincidence). C3: Delta_dir = +0.02 [-0.01, +0.12], P(present) = 0.25; C2+C3:
+0.04 [-0.01, +0.20], P(present) = 0.35; C3-foreign: +0.00 [-0.01, +0.04], P(present) = 0.10. P(at least one hinge arm
is "drop without direction (fingerprint-consistent)" under reading 6.3's recommendation) = 0.6. Excess garbage Delta_g
on C3 = +0.15 [0, +0.5] (the hinge trained the head to miss under foreign latents).

### 2.6 Tests the build must carry (CPU, tiny random GPT-2 config, `FakeTokenizer`; under two minutes with the suite)

- **T1 hand example (scorer).** Eight synthetic GSM records with chosen steps and answers, a fixed donor map, injected
  `ExampleResult` preds: the class (hand: which donors have step min(n, 3) equal to their answer), the ambiguous set,
  k_dir, k_own, k_g_scr, k_g_own, Delta_dir and Delta_g equal hand-computed values; the bootstrap interval equals an
  independent implementation in the test (same `default_rng(0)` draws) to 1e-12; a 1-step and a 4-step donor are
  included (step 1 and step 3 are the last supervised steps), and a step not of the `<<expr=c>>` form puts its donor
  out of the class.
- **T2 class rule.** `answer_in_last_target` at `n_latent_steps` 3 on 1-, 3- and 4-step records; the rule reads the
  donor record only (permuting every example's own answer leaves `in_class` unchanged; permuting donors' answers
  changes it as computed by hand).
- **T3 null blindness** (the standing list: "nulls that read neighbouring rows or that silently rerun the candidate";
  7.10 item 4). (a) The record's `donor` column equals `scramble_donors({idx: 6}, 0)` and every `fed_equals_donor` is
  true on the tiny model (the Recorder sees the donor's latents bit for bit at every rank); (b) a monkeypatched bank
  that feeds an example its own latents, or a non-donor vector, or a self donor, makes `bug` non-empty and the outcome
  `BUG` with no pass/fail line (test_diag's `test_scramble_distances_are_positive_and_identity_donors_are_bug`
  extended to the mode); (c) the scorer receives no model and no bank: with the same `ExampleResult` lists and two
  different donor maps it returns two different hand-checked statistics (the statistic is a function of the fed
  donors and nothing else).
- **T4 batch isolation** (7.10 item 5; A14.1). At fp32 on CPU, the mode's per-example booleans (`correct_own`,
  `correct_scr`, `d_scr`, `d_own`, `g_scr`) at B = 1 equal those at B = 4 and under the same-shape companion swap of
  `diag.isolation` (row i fixed, every companion from outside the batch); the first-answer-token logits agree within
  `TOL` 1e-4 (the existing isolation test's criterion, asserted on the donor-answer columns too).
- **T5 identification.** The mode's `correct_own` and `correct_scr` equal `evaluate(None)` and
  `evaluate("scrambled", scramble_seed=0)` run outside any Recorder on the tiny model; the identification check
  against two written `examples.csv` files reports 0 differences on identical files and k on a copy with k flipped
  rows, and sets `held` above 1.
- **T6 refusals and write-once.** Each refusal of 2.2; an existing `summary.json` refuses; the record's
  `registered_read` is null and `critic_row` names the registering row; a `critic_log.md` without the row refuses
  before the model loads.
- **T7 determinism.** Two runs on the tiny model at fp32 write byte-identical `examples.csv` and equal statistics.
- **T8 data pin** (skipped when `data/gsm_valid.json` is absent; runs where the data is). On the manifest-hashed valid
  file: |C| = 295, ambiguous idx [15, 124, 125, 175, 470, 477], scored 289, the first ten (idx, donor) pairs of 1.3.
- **T9 launcher.** `jobs.py` validation accepts a diag entry with `mode: donor_answer` and the `DIAG_FORBIDDEN` /
  `DIAG_NUMERICS` sets unchanged; `make_jobs --screen-eval --task gsm50k-b` (the `SCREEN_GSM_B` extension of 17:12Z
  (6)) emits, with `--donor-row <ts>`, three diag entries named `<run>_valid_donor_answer` with `est_hours` from the
  measured rate (section 3) beside the 24 reads, refuses without the row, and the job JSON regenerates byte for byte;
  the cap follows the 1.3x + 0.5 h rule up to the half hour.
- **T10 read_screen.** From fixtures of the five outcomes (present, absent, fingerprint-consistent under each reading
  of the garbage clause, BUG, held), the fourth table's C3 and C2+C3 lines carry the registered text (section 4); the
  rule-6 label is lifted only by a record with `bug` false, identification passed and the registering row named; the
  tool's forbidden-label refusal covers the new lines (every label verbatim from the critic's row); the first three
  tables regenerate byte for byte.

---

## 3. The job

### 3.1 Entries and hours

Rate of record: `screen-eval-gsm50k-main`'s receipt (`silent-eval-20261020-112122-f7da`, A100-40GB, main): in-VM
2026-10-20T11:27:07Z to 11:49:43Z = **1,356 s / 45 entries = 30.1 s per 500-item entry** (bf16 autocast over TF32 on,
batch 32, `max_new_tokens` 64; the rate `make_jobs --screen-eval --task prosqa_hard` already uses, scaled). A `none`
read is one generation pass over 500 examples; a `scrambled` read is two; the donor-answer mode is three passes per
numerics set (own, then the scrambled call's own and scrambled passes) under Recorder hooks, two sets: **6 passes, so
6 x 30.1 s = 180.6 s per entry** (conservative: the 30.1 s average includes the three batch-1 timed reads, so a batched
pass is faster than 30.1 s; the hinge arms' inference cost equals Coconut's, A3.4).

Option A (recommended): the three donor-answer entries **inside the follow-up eval job of 17:12Z (6)** for the three arms (24 reads + 3 diag entries = 27
entries): 24 x 30.1 + 3 x 180.6 = 1,264 s = 0.351 h; cap = ceil((1.3 x 0.351 + 0.5) x 2) / 2 = **1.0 h**;

Option B: a separate three-entry job (if the eval job is admitted before the registering row exists): 541.8 s = 0.15 h; cap 1.0 h (the rule's 0.70 h
rounded up to the half hour); the operator names the project in the admission row (17:12Z (6)).

Numerics in the job: the job's `precision` field carries the TF32 flags (`tf32` true), the config the autocast; the
diag entry sets no numerics (`DIAG_NUMERICS`), `run_diag` sets both sets per entry.

Timing: `6ece`'s three runs need RUN_DONE.json (deadline 2026-10-21T09:04:44Z; expected 05:00Z to 09:00Z); the critic's
read row for the three arms (the 10:51Z pattern: hashes pinned to RUN_DONE, epochs in the pool) and the registering row
for this read; then the job spec (`make_jobs`), the admission citing both rows, the run, the fetch, the commit of
`results/evals/<job>/` write-once, then the fourth table.

### 3.2 What the registered-read row (the critic's, after the job) must contain, per arm

The identification of the checkpoint (run, epoch, sha256, agreement with the table's selection); the `none` and
`scrambled` accuracies and Delta_scr with its interval (from the eval reads; the K2-not-independent cell); the
donor-answer record's commit and the hashes of its three files; n_class, n_ambiguous, n_scored (295 / 6 / 289 expected
and asserted); R_dir and R_own with counts; Delta_dir with the 6.2-form and the percentile intervals; G_scr, G_own,
Delta_g with its interval; the two garbage booleans and which reading labels; the identification counts against the
`none` and `scrambled` reads (0 / 0 expected); the fed-vector block (0 zero distances of 3,000, 0 self donors, every
column fed once, bit-for-bit donor latents); the strict-set outcome beside the registered one; the outcome word and
the A6.1 label verbatim as section 4 lists them; the C3 minus C3-foreign Delta_dir difference (diagnostic); the
numerics, GPU, torch and transformers; the labels of 2.4; the complement-class and secondary diagnostics "decide
nothing"; the sentence that no test read exists and none is registered by the row.

---

## 4. How the result enters the fourth GSM-50k table (17:12Z parts (5), (6), (9))

The fourth table is `read_screen.py`'s reissue of the third (`rev12`) with the three arms' run rows, cost rows, K2,
7.10, 7.2, 5.5 (b), K3 and card lines under the third table's refusal (only those lines and the notes they touch may
change). The donor-answer record enters through the job's eval directory (`--eval-dir`, each entry hashed into the
provenance as `load_evals` does; a diag entry is recognised by `tool: run_diag.py`, `mode: donor_answer`, attached to
its run as `eval_stats["directional"]`). Per hinge arm, the C3 (or C2+C3) card line and the K2 line then read, by
outcome (texts are the registration's; the exact strings are fixed in the critic's row and the tool refuses any label
not verbatim from it):

- **Record absent, BUG, or held:** unchanged from rule 6: "drop leg not independent; directional read pending" (plus
  "(BUG: <reason>)" or "(held: identification <k> of 500, A12.3)"); no K2 sentence, no "latents carry state" wording,
  no verdict naming the arm; rule 6 (a) to (c) stand.
- **Present** (Delta_dir > 0, lower bound > 0, and the registered garbage comparison not exceeded): the pending label
  is lifted and replaced by "drop leg not independent (A6.1 rule 1); directional read present: donor-answer rate
  <r_dir> vs <r_own> under own latents, Delta_dir <+x> [<lo>, <hi>] (1 seed, valid)". The K2 line may now be written in
  the registered form: "K2 not independent (A6.1): Delta_scr = ... [...] against sigma_screen 3.60: fired / not fired"
  with rule 2's "dependence on own latents (content or fingerprint)" C3 minus C3-foreign line beside it, and the
  directional sentence as the non-circular evidence rule 4 (B) names. The card's pass line (A3.4 C3: drop above 10 on
  GSM-50k with accuracy within sigma_screen of `coconut` or better, and the drop exceeding C3-foreign's by more than
  sigma_screen) is applied as registered; the kill clauses needing "both tasks" cannot be met with ProsQA-hard
  UNTESTED (17:12Z (8)); C2+C3's C2-side line against C2-shuffled applies too. Still **no verdict**: a screen filters
  (6.3), 1 seed, valid; "latents carry state" remains a sentence about this checkpoint at this read with its labels,
  not a claim about the card (rule 1; rule 7's positive control is still required for any readout label).
- **Absent** (interval includes 0, or Delta_dir <= 0) **or the registered garbage comparison exceeded**, together
  with a large scrambled drop (the card's own line: drop above 10 points; a drop under 5 fires the card's kill part
  on this task by itself): the label **"drop without direction (fingerprint-consistent)"** and "K2 counts as fired
  for the arm's latent-reasoning claim on GSM-50k" (rule 5); the arm is killed as a latent claim at the screen on
  this task (A3.4's "every latent arm" rule read with A6.1 rules 1, 5 and 6), continues only as a labelled effect,
  and cannot be A1's positive read on GSM-50k. The K2 line is then written as "K2 not independent (A6.1); counts as
  fired (rule 5): drop without direction (fingerprint-consistent)". With a small drop and an absent direction, the
  K2 line is "K2 not independent; fired on the drop leg" and the directional sentence is printed as "absent" beside it.
- **C3-foreign** gets the same lines; its directional outcome is the reference for C3's (expected absent); the C3
  minus C3-foreign Delta_dir difference is printed as a diagnostic beside rule 2's drop difference.

Every line keeps "screen numbers (1 seed); valid only; no test read; bar LR-tuned 3 trials; arm 1 trial"; the A11.3
words do not appear in the table; the fourth table's `diff_vs_third_table` lists every changed path for the critic's
diff; the first three tables regenerate byte for byte.

---

## 5. What this read does not decide

- **No test read.** The test split is never scored under this read (A14.3's rule: no intervention outside section
  8's list on test; rule 4's last sentence puts the screen's directional read on valid). The single test read of the
  nine GSM-50k screen checkpoints (A3.4: once, for reporting, no decision) waits for the fourth table (17:12Z (9)).
- **One seed, valid.** Nothing here is a 6.2 verdict, a 7.2 claim, a tie, a WIN or a LOSS; a screen filters (6.3); a
  confirmation's directional read is a separate read on test at confirmation seeds (rule 4), outside this window's
  compute (A3.5; no admission exists).
- **ProsQA-hard UNTESTED** for C3 and C2+C3 under A2.1 (no hinge job on that task; the F1 patch directional read with
  `target_swap` and `root_swap` is UNTESTED for both arms; every "both tasks" kill clause of A3.4 C3 is unmeetable as
  written; 17:12Z (8)).
- **K2 stays "not independent"** (rule 1) whatever the outcome; the read answers whether the drop has direction, not
  whether K2 is independent.
- **No readout label** (7.9 "explained" / "partly explained" / "unexplained") follows from it (rule 7; the positive
  control is still required); R3 coverage for C2+C3 is a separate report item (A5 item 5; A6.7 (4)).
- **No bits-per-step, no F1 patch, no ladder** on these checkpoints (A14.3's conditions; sigma_task).
- The secondary and complement diagnostics decide nothing (section 10).
- The strict-fp32 set beside the record of record decides nothing; a disagreement between the sets on the outcome is
  printed and goes to the critic (A14.1's pattern), never resolved by the tool.

---

## 6. Open questions for the critic (each with the operator's recommendation)

1. **The class count on valid.** Rule 4 (B) names "the 0.770 class of A5 item 5"; on the registered valid file under
   the seed-0 derangement the class is 295 of 500 (59.0%; 1.3) because valid has more long problems. Recommendation:
   the row registers the **rule** (donor's step min(n_steps, 3) result equals the donor's answer, `answer_in_last_
   target` with `n_latent_steps` 3) and the valid **count 295** as the number of record, with the training figure
   0.770 printed beside as A5 item 5's; T8 pins it.
2. **Ambiguous pairs.** Recommendation: exclude the 6 pairs whose own answer equals the donor's (idx listed) from the
   scored set (289), print them; they cannot carry a direction and keeping them inflates R_dir and R_own alike.
3. **The garbage clause on GSM.** Literal reading (G_scr > R_dir) fires for every arm at GSM accuracies (the ordinary
   error rate counts as "neither"). Recommendation: register the **excess garbage** Delta_g = G_scr - G_own against
   Delta_dir as rule 5's comparison on GSM, both readings printed, the literal one labelled "literal reading: fires at
   any GSM accuracy under 50%". If the critic keeps the literal reading, rule 5's second clause is expected to fire on
   all three arms and the design still prints everything.
4. **Interval form.** Recommendation: 6.2's Delta +- 1.96 sqrt(Var_ex) over 10,000 paired draws (`default_rng(0)`),
   percentile beside, **no sigma term** (one checkpoint, two interventions: the K2 construction of 7.5 at the screen),
   "excluding 0" judged on the 6.2-form interval.
5. not a `run_eval` intervention, not a new tool.
6. **Numerics.** Recommendation: the record of record at the checkpoint's registered set (bf16 autocast over TF32 on,
   A10.5), the strict set beside as a diagnostic (`--tf32 both`), `--tf32 off` alone refused.
7. else Option B (own job, main after the first closes at actual, else AcctA after `6ece` closes under the A2.1 stop rule).
8. **A reference arm.** Should `gsm50k_coconut_s0` (sha `rev180...`, the C3 attribution set's base, K2 not fired at
   the screen) carry the same read as a diagnostic reference (what a non-hinge checkpoint's donor-answer rate is)?
   Cost one entry, 180.6 s, 0.05 h, inside the same cap. Recommendation: **yes, coconut only**, labelled "reference arm
   (no hinge); diagnostic; decides nothing", by the same row; C2 and C2-shuffled not (scope).
9. **Identification check.** Recommendation: the diag record's `correct_own` / `correct_scr` against the `none` /
   `scrambled` reads' `correct` columns, 0 of 500 expected (A14.2's latent-path record), more than 1 holds the record
   (A12.3's pattern), the check done in the tool when the reads are present and by read_screen at table time
   otherwise.
10. **The prediction.** Recommendation: the row carries the critic's numbers (A3.4's rule); the operator's in 2.5 are
    printed beside them as the operator's, never as the registered prediction.
11. **A minimal-pair rule for GSM** (rule 4 (B) allows the operator to register one, replacing the donor-answer read).
    Recommendation: **not in this window**; the donor-answer read is the registered default, needs no new data, and a
    pair rule would need its own design, generator and tests.

---

## CHANGELOG draft (to be pasted by the operator under a dated UTC heading; not applied here)

## 2026-10-20 <HH:MM> UTC: design note for the A6.1 directional read on GSM-50k (the donor-answer read), before any code

`governance/designs/a6_1_directional_read.md` (`<SHA>`) designs A6.1 rule 4 (B)'s donor-answer read for the three hinge arms of
`screen-gsm50k-accta` (`6ece`, training; no number exists), as the 17:12Z rows require before any code. Instrument: a `run_diag.py
--mode donor_answer` entry (`latent.diag.scramble` unchanged, under the Recorder's fed-vector checks, plus a pure scorer),
registered by a critic entry the code names, valid only, no 4.5 read spent; not a `run_eval` intervention (the registered per-
example file carries no prediction strings) and not a new tool. The pass: the registered scrambled-at-inference construction (whole
set, `scramble_donors` seed 0, one class of 500 at 6 latents, the donor's latents bit for bit at every rank). The class: donors
whose step min(n_steps, 3) result equals their answer (`answer_in_last_target`, `n_latent_steps` 3), computed on the manifest-hashed
valid file: **295 of 500** under seed 0 (the training figure 0.770 of A5 item 5 does not transfer: valid has 173 records with 4 or
more steps), 6 ambiguous pairs (own answer equals the donor's; idx 15, 124, 125, 175, 470, 477) proposed excluded, **289 scored**.
Statistic: R_dir (scrambled prediction equals the donor's answer) minus R_own (the own-pass prediction equals the donor's answer,
the coincidence baseline, expected near 0.01), paired, 10,000 example draws `default_rng(0)`, 6.2's form without a sigma term,
percentile beside; present iff the lower bound is above 0; the garbage fraction printed as G_scr and as the excess Delta_g = G_scr -
G_own (the literal reading fires at any GSM accuracy under 50%; the critic chooses the reading of rule 5). Record: `examples.csv`
(ids, booleans, numbers; no strings), `fed_vectors.csv`, `summary.json` with the pass/fail line; identification against the `none` /
`scrambled` reads (0 of 500 expected). Tests: a hand example, the class rule, null blindness (donor map and bit-for-bit fed vectors;
BUG on a self donor), batch isolation on the donor-answer columns, identification, refusals and write-once, determinism, the data
pin (295 / 6 / 289), the launcher entry, the fourth table's lines. Outcomes and the fourth table's lines per outcome, what the read
does not decide (no test read; 1 seed; ProsQA-hard UNTESTED under A2.1; K2 stays "not independent"; no readout label), and eleven
open questions with recommendations (the valid class count, ambiguous pairs, the garbage reading, the interval form, the instrument,
numerics, placement, a coconut reference entry, identification, the prediction, no minimal-pair rule) are in the note. Next: the
critic's registering row; then the build (Opus, from the note); then the job spec once `6ece` has RUN_DONE.json for all three runs
and the read row exists.
