#!/usr/bin/env python3
"""eval_v5.py — Evaluate v5 model on all 7 test proteins."""
import sys, os, json
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md
from src.evaluate import evaluate_model

PROTEINS = [
    ('1crn', 'A'), ('1vii', 'A'), ('1lyz', 'A'), ('1trz', 'A'),
    ('1aho', 'A'), ('2ptl', 'A'), ('1tig', 'A'),
]
CROP_LEN = 60
MODEL_PATH = 'model_v5.pt'

print('Loading v5 model...')
model = md.load_model('transformer', seq_len=60, path=MODEL_PATH)

results = {}
for pid, chain in PROTEINS:
    print(f'Evaluating {pid}...', end=' ', flush=True)
    try:
        path   = utils.fetch_pdb(pid)
        seq    = utils.pdb_sequence(path, chain=chain, max_residues=CROP_LEN)
        coords = utils.pdb_ca_coords(path, chain=chain, max_residues=CROP_LEN)
        N = min(len(seq), len(coords))
        r = evaluate_model(model, seq[:N], coords[:N])
        results[pid] = r
        ldt = r['local_lDDT']
        f1  = r['contact_f1']
        lr  = r['long_range_precision_L5']
        tm  = r['tm_proxy']
        print(f'lDDT={ldt:.1f}  F1={f1:.3f}  LR={lr:.3f}  TM={tm:.3f}')
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f'ERROR: {e}')

print()
lrs  = [results[p]['long_range_precision_L5'] for p in results]
f1s  = [results[p]['contact_f1'] for p in results]
ldts = [results[p]['local_lDDT'] for p in results]
tms  = [results[p]['tm_proxy'] for p in results]
print(f'MEAN  lDDT={np.mean(ldts):.1f}  F1={np.mean(f1s):.3f}  LR={np.mean(lrs):.3f}  TM={np.mean(tms):.3f}')

os.makedirs('results', exist_ok=True)
with open('results/eval_v5.json', 'w') as f:
    json.dump(results, f, indent=2)
print('Saved results/eval_v5.json')
