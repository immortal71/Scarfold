#!/usr/bin/env python3
"""eval_new_proteins.py — Evaluate v4 model on 1AHO, 2PTL, 1TIG for Table 4.3."""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md
from src.evaluate import evaluate_model

PROTEINS = [
    ('1aho', 'A', 'alpha+beta'),
    ('2ptl', 'A', 'alpha+beta'),
    ('1tig', 'A', 'alpha+beta'),
]
MODEL_PATH = 'checkpoints/best_pdb_v4.pt'
CROP_LEN = 60   # model was trained on L=60 crops

model = md.load_model('transformer', seq_len=60, path=MODEL_PATH)
results = {}
for pid, chain, cls in PROTEINS:
    print(f'Evaluating {pid}...')
    try:
        path   = utils.fetch_pdb(pid)
        seq    = utils.pdb_sequence(path, chain=chain, max_residues=CROP_LEN)
        coords = utils.pdb_ca_coords(path, chain=chain, max_residues=CROP_LEN)
        N = min(len(seq), len(coords))
        seq = seq[:N]; coords = coords[:N]
        print(f'  L={N}  seq={seq[:20]}...')
        r = evaluate_model(model, seq, coords)
        results[pid] = dict(r, cls=cls, L=N)
        ldt  = r['local_lDDT']
        f1   = r['contact_f1']
        lr   = r['long_range_precision_L5']
        tm   = r['tm_proxy']
        rmsd = r['rmsd_aligned']
        print(f'  lDDT={ldt:.1f}  F1={f1:.3f}  LR={lr:.3f}  TM={tm:.3f}  RMSD={rmsd:.1f}')
    except Exception as e:
        import traceback
        print(f'  ERROR: {e}')
        traceback.print_exc()

out = 'results/eval_3proteins.json'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)
print(f'Saved {out}')
