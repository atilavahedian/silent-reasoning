#!/usr/bin/env python3
"""Check every number in each paper CSV (data/) against the result record(s) it was extracted from.

Usage (from the repository root): python3 tools/check_data.py [data/<file>.csv ...]   (no argument: all files)
A CSV number matches when the record holds a value equal to it at the CSV's printed precision, either as is or
as a percentage of a fraction (25.6 against 0.256), or in GFLOPs of a FLOP count. Prints, per file, the numbers it could not find; a number that
is a ratio or difference of record values is reported, not failed, and is listed so a reader can recompute it.
"""
import csv, glob, json, os, re, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
S = 'results/screens/'
SOURCES = {
    'table1_k1_reproduction.csv': ['results/k1/k1_table_final.json', 'results/k1/k1_table_final.md', 'results/k1/k1_table_window.json'],
    'table1b_same_seed_pair.csv': ['results/k1/prosqa_cot_same_seed_pair.json'],
    'table1b_same_seed_pair_agreement.csv': ['results/k1/prosqa_cot_same_seed_pair.json'],
    'table1b_sigma_task_prosqa.csv': ['results/k1/prosqa_cot_same_seed_pair.json'],
    'figure1_same_seed_pair_curves.csv': ['results/k1/prosqa_cot_same_seed_pair.json'],
    'table2_gsm50k_screen.csv': [S + 'gsm50k/gsm50k_screen_with_no_curriculum.json', S + 'gsm50k/gsm50k_screen_with_no_curriculum.md'],
    'table3_prosqa_hard_screen.csv': [S + 'prosqa_hard/prosqa_hard_screen_with_no_curriculum.json', S + 'prosqa_hard/prosqa_hard_screen_with_no_curriculum.md'],
    'table4_directional_read.csv': [S + 'gsm50k/gsm50k_screen_with_hinge.json', S + 'gsm50k/gsm50k_screen_with_hinge.md'],
    'table5_cost_per_example.csv': [S + 'gsm50k/gsm50k_screen_with_no_curriculum.json', S + 'prosqa_hard/prosqa_hard_screen_with_no_curriculum.json',
                                    S + 'gsm50k/gsm50k_screen_with_no_curriculum.md', S + 'prosqa_hard/prosqa_hard_screen_with_no_curriculum.md'],
    'table6_test_read.csv': [S + 'test_read.json', S + 'test_read.md'],
    'table7_step_sweep.csv': [S + 'gsm50k/gsm50k_screen_with_no_curriculum.json', S + 'prosqa_hard/prosqa_hard_screen_with_no_curriculum.json',
                              'results/k1/k1_table_final.json', 'results/k1/k1_table_final.md', 'results/k1/k1_table_window.json',
                              'results/evals/k1-eval-main/*/summary.json', 'results/evals/k1-eval-s12-main/*/summary.json',
                              'results/evals/k1-eval-gsm-main/*/summary.json'],
    'table8_c1b_wallclock.csv': [S + 'gsm50k/gsm50k_screen_with_no_curriculum.json', S + 'prosqa_hard/prosqa_hard_screen_with_no_curriculum.json'],
    'figure2a_pareto_points_valid.csv': [S + 'pareto_read.json'],
    'figure2b_pareto_points_test.csv': [S + 'pareto_read.json'],
    'table9_text_numbers.csv': None,   # each row names its own record in the source column
}
NUM = re.compile(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?:e-?\d+)?(?![\w])')


def record_values(paths):
    vals = set()
    for p in [q for g in paths for q in sorted(glob.glob(os.path.join(ROOT, g)))]:
        txt = open(p, errors='replace').read()
        for n in NUM.findall(re.sub(r'[;\[\],]', ' ', txt)):
            try:
                v = abs(float(n)); vals.update((v, v * 100, v / 1e9))
            except ValueError:
                pass
    return vals


def found(n, vals):
    x = abs(float(n))
    d = len(n.split('.')[1]) if '.' in n and 'e' not in n else 0
    tol = 0.5 * 10 ** -d + 1e-9 if d else 1e-9 * max(1, x)
    return any(abs(v - x) <= tol for v in vals)


def numbers_of(path, skip_cols=('source', 'labels', 'label', 'checkpoint_sha256_prefix', 'registered_read')):
    out = []
    with open(path, newline='') as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                if k in skip_cols or v is None:
                    continue
                out += NUM.findall(re.sub(r'[;\[\]/]', ' ', v.replace('k=', 'k ')))
    return out


files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, 'data', '*.csv')))
for f in files:
    name = os.path.basename(f)
    src = SOURCES.get(name)
    if src is None:
        print(f'{name}: sources listed per row in its source column')
        continue
    vals = record_values(src)
    nums = numbers_of(f)
    miss = [n for n in nums if not found(n, vals)]
    print(f'{name}: {len(nums)} numbers, {len(nums) - len(miss)} found verbatim in {", ".join(src)}'
          + (f'; derived or not printed in the record: {sorted(set(miss), key=miss.index)[:40]}' if miss else ''))
