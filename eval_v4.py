#!/usr/bin/env python3
"""Quick evaluation of the best v4 checkpoint on held-out test proteins."""
import sys, os, json
sys.path.insert(0, '.')
import numpy as np
from src import utils, model as md, evaluate as ev

CKPT     = 'checkpoints/best_pdb_v4.pt'
CROP_LEN = 60

TEST_IDS = [
    ('1crn', 'A'),  # crambin
    ('1vii', 'A'),  # villin headpiece
    ('1lyz', 'A'),  # lysozyme
    ('1trz', 'A'),  # insulin
    ('1cbn', 'A'),  # crambin crystal form B
]

print('Loading model from', CKPT)
model = md.load_model('transformer', CROP_LEN, CKPT)
n = sum(p.numel() for p in model.parameters())
print(f'Params: {n:,}')
print()

all_results = {}
for pid, chain in TEST_IDS:
    try:
        path   = utils.fetch_pdb(pid)
        seq    = utils.pdb_sequence(path, chain=chain, max_residues=CROP_LEN)
        coords = utils.pdb_ca_coords(path, chain=chain, max_residues=CROP_LEN)
        N = min(len(seq), len(coords))
        seq, coords = seq[:N], coords[:N]
        print(f'Evaluating {pid} ({N} aa)...', flush=True)
        metrics = ev.evaluate_model(model, seq, coords)
        all_results[pid] = metrics
        for k, v in metrics.items():
            print(f'  {k}: {v:.4f}')
        print()
    except Exception as e:
        print(f'  ERROR {pid}: {e}')
        print()

os.makedirs('results', exist_ok=True)
out = 'results/eval_v4_realdata.json'
with open(out, 'w') as f:
    json.dump(all_results, f, indent=2)

print('=' * 50)
print('SUMMARY (mean over test proteins):')
for metric in ['rmsd_aligned', 'local_lDDT', 'contact_f1', 'long_range_precision_L5', 'tm_proxy']:
    vals = [all_results[p][metric] for p in all_results if metric in all_results.get(p, {})]
    if vals:
        print(f'  {metric:32s}: {np.mean(vals):.4f}')

import shutil
shutil.copy(CKPT, 'model_v4.pt')
print()
print(f'Saved: model_v4.pt  |  {out}')
