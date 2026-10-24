# Silent Reasoning at 124M: A Pre-registered, Cost-Accounted Test of Continuous Latent Reasoning Against Tuned Chain of Thought

Atila Vahedian, Independent researcher

arXiv: <to be added>

Code, protocol and result records behind the paper. The paper itself is `paper/silent_reasoning_124m.pdf`.
Code is Apache 2.0 (`LICENSE`); the data CSVs, result records and curves are CC BY 4.0.
No model checkpoint is included.

## Layout

| path | what it holds |
|---|---|
| `common/latent/` | the library: GPT-2 latent-feedback model (Coconut, C1b, C2, C3), training loop, data, evaluation, FLOP accounting, diagnostics, tests |
| `code/` | entry points: `fetch_data.py`, `make_screen_data.py`, `run_train.py`, `run_eval.py`, `run_diag.py`, and the readers that wrote every result table (`read_oracle.py`, `read_screen.py`, `read_test.py`, `read_closing.py`) |
| `configs/` | one JSON config per arm (K1 oracle cells at the top level, screen arms under `configs/screen/`), the data manifest and the reference environment |
| `PLAN.md`, `governance/registration.md` | the pre-registered protocol and its dated amendments |
| `governance/critic_log.md`, `governance/deviations.md`, `governance/designs/` | the independent reviewer's rulings, registered predictions and every deviation, in order |
| `governance/run_log.jsonl` | the run ledger: every admission and close of a GPU job (price fields removed; see below) |
| `docs/cards/` | the registered card of each new method (C1b, C2, C3) with its pass line |
| `results/` | the per-example result files (`results/evals/*/*/examples.csv` + `summary.json`), diagnostics, and the reader outputs every table is copied from |
| `data/` | the CSV behind every table and figure in the paper |
| `curves/` | per-epoch validation curves of every screen run (CSV and PNG) |
| `RESULTS.md` | the full results record |

## Tables and figures

Every number in each CSV is checked against its record by `python3 tools/check_data.py data/<file>.csv`.
The CSVs are extracts of the reader outputs in `results/`; the reader outputs were written by the commands in
`code/read_*.py` from the training records, which stayed in the private run store. From this repository the
one command per CSV re-derives and checks it against its record:

| paper | CSV | record | command |
|---|---|---|---|
| Table 1 (arms and cards) | none (protocol) | `docs/cards/`, `governance/registration.md` | |
| Table 2 (K1 reproduction) | `data/table1_k1_reproduction.csv` | `results/k1/k1_table_final.json`, `k1_table_window.json` | `python3 tools/check_data.py data/table1_k1_reproduction.csv` |
| Table 3a, 3b (same-seed pair) | `data/table1b_same_seed_pair.csv`, `data/table1b_same_seed_pair_agreement.csv`, `data/table1b_sigma_task_prosqa.csv` | `results/k1/prosqa_cot_same_seed_pair.json` | `python3 tools/check_data.py data/table1b_*.csv` |
| Figure 1 | `data/figure1_same_seed_pair_curves.csv` | `results/k1/prosqa_cot_same_seed_pair.json` | `python3 tools/check_data.py data/figure1_same_seed_pair_curves.csv` |
| Table 4 (GSM-50k screen) | `data/table2_gsm50k_screen.csv` | `results/screens/gsm50k/gsm50k_screen_with_no_curriculum.json` | `python3 tools/check_data.py data/table2_gsm50k_screen.csv` |
| Table 5 (ProsQA-hard screen) | `data/table3_prosqa_hard_screen.csv` | `results/screens/prosqa_hard/prosqa_hard_screen_with_no_curriculum.json` | `python3 tools/check_data.py data/table3_prosqa_hard_screen.csv` |
| Figure 2 (screens) | `data/table2_gsm50k_screen.csv`, `data/table3_prosqa_hard_screen.csv` | as Tables 4 and 5 | `python3 paper/figures/make_figures.py` |
| Table 6 (donor-answer read) | `data/table4_directional_read.csv` | `results/screens/gsm50k/gsm50k_screen_with_hinge.json` | `python3 tools/check_data.py data/table4_directional_read.csv` |
| Table 7 (cost per example) | `data/table5_cost_per_example.csv` | both `*_screen_with_no_curriculum.json` | `python3 tools/check_data.py data/table5_cost_per_example.csv` |
| Table 8 (C1b wall-clock) | `data/table8_c1b_wallclock.csv` | both `*_screen_with_no_curriculum.json` | `python3 tools/check_data.py data/table8_c1b_wallclock.csv` |
| Figure 3, Table 9 (Pareto, valid) | `data/figure2a_pareto_points_valid.csv` | `results/screens/pareto_read.json` | `python3 tools/check_data.py data/figure2a_pareto_points_valid.csv` |
| Figure 4, Table 10 (Pareto, test) | `data/figure2b_pareto_points_test.csv` | `results/screens/pareto_read.json` | `python3 tools/check_data.py data/figure2b_pareto_points_test.csv` |
| Table 11 (test read) | `data/table6_test_read.csv` | `results/screens/test_read.json` | `python3 tools/check_data.py data/table6_test_read.csv` |
| Table 12 (predictions) | none | `governance/critic_log.md` (each prediction's entry) | |
| step sweeps (Sections 4.3, 5.2, 5.4, 7) | `data/table7_step_sweep.csv` | the screen tables and `results/evals/k1-eval-*/*/summary.json` | `python3 tools/check_data.py data/table7_step_sweep.csv` |
| numbers that appear only in the text | `data/table9_text_numbers.csv` | the record named in each row's `source` column | |

All figures: `python3 paper/figures/make_figures.py` (PDFs from `data/`). Curves: `python3 curves/make_curves.py`.

## Rerun one arm end to end

```bash
pip install -r requirements.txt
python3 code/fetch_data.py --only prosqa                       # ProsQA from facebookresearch/coconut into data/
PYTHONPATH=common python3 code/run_train.py configs/prosqa_cot.json      # the CoT run Coconut starts from
PYTHONPATH=common python3 code/run_train.py configs/prosqa_coconut.json  # Coconut, 6 latent stages
PYTHONPATH=common python3 code/run_eval.py --config configs/prosqa_coconut.json \
    --checkpoint runs/prosqa_coconut/ckpt_best_final.pt --split test --intervention none
```

The screen arms use `python3 code/make_screen_data.py` for ProsQA-hard and GSM-50k, then the same two commands with a
config from `configs/screen/`. Tests: `PYTHONPATH=common python3 -m pytest common/latent/tests -q`
(a few tests load code from git history and skip in this single-commit repository). Runs used one A100 (40GB or 80GB).

## What is not here

- Checkpoints (none is released) and the per-run training logs, which stayed in the private run store; `curves/` holds
  the per-epoch validation reads that selection used.
- The cloud launcher, which is tied to private accounts. `code/read_test.py` imports its job constants, so it is
  kept as the record of how Table 11 was read but does not run on its own.
- Ledger: the run ledger keeps every job's GPU, times and status; its price and cap fields (and the hash chain
  over them) are removed, and so are price figures in the governance text.
- Cloud account and bucket names are replaced by neutral labels (`acctA`, `bucket-a`, ...), and private commit
  hashes by labels (`rev12`); the same label always means the same original.
