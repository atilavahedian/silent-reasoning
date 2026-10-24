#!/usr/bin/env python3
"""Per-epoch validation curves of every screen run and of the ProsQA same-seed pair, from the result records.

Writes curves/screen_valid_curves.csv, curves/screen_<task>.png (accuracy and mean emitted tokens per epoch) and
curves/same_seed_pair.png. Run from the repository root: python3 curves/make_curves.py
The training-loss logs (log.jsonl per run) stayed in the private run store and are not in this repository; these
curves are the per-epoch validation reads that selection used.
"""
import csv, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'curves')
SCREENS = {'gsm50k': 'results/screens/gsm50k/gsm50k_screen_with_no_curriculum.json',
           'prosqa_hard': 'results/screens/prosqa_hard/prosqa_hard_screen_with_no_curriculum.json'}

rows = []
for task, rel in SCREENS.items():
    d = json.load(open(os.path.join(ROOT, rel)))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for r in d['runs']:
        c = r.get('curve') or []
        for p in c:
            rows.append({'task': task, 'run': r['run'], 'epoch': p['epoch'], 'stage': p.get('stage'),
                         'val_acc': p['val_acc'], 'val_emitted_mean': p.get('val_emitted_mean'), 'source': rel})
        if c:
            ep = [p['epoch'] for p in c]
            a1.plot(ep, [100 * p['val_acc'] for p in c], lw=1, label=r['run'].replace(task + '_', ''))
            a2.plot(ep, [p.get('val_emitted_mean') for p in c], lw=1)
    a1.set(title=f'{task}: valid accuracy per epoch', xlabel='epoch', ylabel='accuracy (points)')
    a2.set(title=f'{task}: mean emitted tokens per epoch', xlabel='epoch', ylabel='tokens', yscale='log')
    a1.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f'screen_{task}.png'), dpi=130); plt.close(fig)

with open(os.path.join(OUT, 'screen_valid_curves.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

pair = os.path.join(ROOT, 'data', 'figure1_same_seed_pair_curves.csv')
with open(pair) as fh:
    pr = list(csv.DictReader(fh))
fig, ax = plt.subplots(figsize=(6, 4))
for col in [k for k in pr[0] if k.endswith('_val_acc')]:
    pts = [(int(r['epoch']), float(r[col])) for r in pr if r[col]]
    ax.plot([e for e, _ in pts], [100 * v for _, v in pts], lw=1, label=col.replace('_val_acc', ''))
ax.set(title='ProsQA cot, same seed: valid accuracy per epoch', xlabel='epoch', ylabel='accuracy (points)')
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(OUT, 'same_seed_pair.png'), dpi=130)
print(len(rows), 'curve points')
