#!/usr/bin/env python3
"""eval_v6.py — Evaluate v6 (ESM-2 input) model on all 7 held-out test proteins.

Usage:
    python src/eval_v6.py
    python src/eval_v6.py --model model_v6.pt --out results/eval_v6.json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md
from src.esm_utils import ESM_RICH_DIM
from src.evaluate import evaluate_model

PROTEINS = [
    ('1crn', 'A'),
    ('1vii', 'A'),
    ('1lyz', 'A'),
    ('1trz', 'A'),
    ('1aho', 'A'),
    ('2ptl', 'A'),
    ('1tig', 'A'),
]
CROP_LEN = 60


def main():
    parser = argparse.ArgumentParser(description='Evaluate v6 ESM-2 model')
    parser.add_argument('--model', default='model_v6.pt',
                        help='Path to trained v6 model checkpoint')
    parser.add_argument('--out',   default='results/eval_v6.json',
                        help='Output JSON path')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'ERROR: model file not found: {args.model}')
        print('Train v6 first:  python src/train_v6.py')
        sys.exit(1)

    print(f'Loading v6 model ({args.model}) ...')
    # Load with aa_dim=368 (ESM_RICH_DIM)
    import torch
    raw = torch.load(args.model, map_location='cpu', weights_only=False)
    state = raw['state_dict'] if isinstance(raw, dict) else raw
    aa_dim = raw.get('aa_dim', ESM_RICH_DIM) if isinstance(raw, dict) else ESM_RICH_DIM
    # Auto-detect v8/v9 model: coord_head present → coord_head=True
    # Also detect seq_len from pos_embed shape in checkpoint
    is_v8 = any('coord_head' in k for k in state.keys())
    if 'pos_embed' in state:
        seq_len = state['pos_embed'].shape[0]
    else:
        seq_len = 100 if is_v8 else CROP_LEN
    model = md.TransformerDistancePredictor(
        seq_len=seq_len, aa_dim=aa_dim, coord_head=is_v8)
    model.load_state_dict(state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} parameters  |  aa_dim={aa_dim}  seq_len={seq_len}  coord_head={is_v8}\n')

    results = {}
    for pid, chain in PROTEINS:
        print(f'Evaluating {pid} ...', end=' ', flush=True)
        try:
            path   = utils.fetch_pdb(pid)
            seq    = utils.pdb_sequence(path, chain=chain, max_residues=seq_len)
            coords = utils.pdb_ca_coords(path, chain=chain, max_residues=seq_len)
            N = min(len(seq), len(coords))
            seq_c, crd_c = seq[:N], coords[:N]

            # evaluate_model auto-detects aa_dim=368 and uses esm2_rich_encoding
            r = evaluate_model(model, seq_c, crd_c)
            results[pid] = r

            ldt = r['local_lDDT']
            f1  = r['contact_f1']
            lr  = r['long_range_precision_L5']
            tm  = r['tm_proxy']
            print(f'lDDT={ldt:.1f}  F1={f1:.3f}  LR={lr:.3f}  TM={tm:.3f}')
        except Exception:
            import traceback
            traceback.print_exc()

    if results:
        lrs  = [results[p]['long_range_precision_L5'] for p in results]
        f1s  = [results[p]['contact_f1']              for p in results]
        ldts = [results[p]['local_lDDT']              for p in results]
        tms  = [results[p]['tm_proxy']                for p in results]
        print()
        print(f'{"MEAN":>10}  lDDT={np.mean(ldts):.1f}  '
              f'F1={np.mean(f1s):.3f}  LR={np.mean(lrs):.3f}  TM={np.mean(tms):.3f}')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
