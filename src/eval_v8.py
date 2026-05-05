#!/usr/bin/env python3
"""eval_v8.py — Evaluate v8 model using the direct coordinate head output.

Key difference from eval_v6.py:
  - Uses CoordinateHead output directly (no MDS post-processing).
  - The coord head predicts 3D coordinates end-to-end, then we Kabsch-align
    to the native structure for scoring.
  - Evaluates at full protein length (up to COORD_LEN=100aa), so 1AHO, 2PTL,
    and 1TIG are evaluated without being cropped.

Usage:
    python src/eval_v8.py
    python src/eval_v8.py --model model_v8.pt --out results/eval_v8.json
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md
from src.esm_utils import ESM_RICH_DIM, esm2_rich_encoding_cached

PROTEINS = [
    ('1crn', 'A'),
    ('1vii', 'A'),
    ('1lyz', 'A'),
    ('1trz', 'A'),
    ('1aho', 'A'),
    ('2ptl', 'A'),
    ('1tig', 'A'),
]
COORD_LEN = 100   # max eval length (pos_embed supports up to this)


def eval_one_protein(model, seq, true_coords, crop_len):
    """Evaluate model on one protein using the coordinate head.

    Returns a dict with lDDT, Contact F1, LR P@L/5, TM-score.
    Also computes the distogram-based metrics for comparison.
    """
    import torch
    from src.esm_utils import esm2_rich_encoding

    L = min(len(seq), crop_len)
    seq_c, crd_c = seq[:L], true_coords[:L]

    # Encoding: use disk cache if available, else compute on the fly
    try:
        enc = esm2_rich_encoding_cached(seq, cache_path='data/esm2_cache.npz')[:L]
    except Exception:
        enc = esm2_rich_encoding(seq)[:L]

    X = torch.tensor(enc[None], dtype=torch.float32)  # (1, L, 368)

    model.eval()
    with torch.no_grad():
        if model.coord_head is not None:
            logits, _, _, coords = model.forward_with_coords(X)
            pred_coords_raw = coords[0].numpy()    # (L, 3) mean-centered
        else:
            logits, _, _ = model.forward_full(X)
            pred_coords_raw = None

        pred_dist = md.bin_to_dist(logits)[0].numpy()  # (L, L) from distogram

    pred_dist = 0.5 * (pred_dist + pred_dist.T)
    true_dist = utils.coords_to_distances(crd_c)

    # ── Coordinate-head based metrics ─────────────────────────────────────────
    if pred_coords_raw is not None:
        # Kabsch align predicted coords to native (handles reflections)
        rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords_raw, crd_c)

        # Distances from aligned predicted coords (for lDDT)
        aligned_dist = utils.coords_to_distances(aligned_pred)

        local_ldt  = float(utils.local_lddt(aligned_dist, true_dist).mean())
        tm         = float(utils.tm_score(aligned_pred, crd_c))
        cmap       = md.contact_map_score(pred_dist, true_dist)   # use distogram F1
    else:
        # Fallback: run gradient MDS on distogram distances
        pred_coords_mds = utils.gradient_mds(pred_dist, dim=3, n_iter=600)
        rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords_mds[:L], crd_c)
        local_ldt  = float(utils.local_lddt(pred_dist, true_dist).mean())
        tm         = float(utils.tm_score(aligned_pred, crd_c))
        cmap       = md.contact_map_score(pred_dist, true_dist)

    return {
        'rmsd_aligned':             float(rmsd_aligned),
        'local_lDDT':               local_ldt,
        'contact_f1':               float(cmap['f1']),
        'long_range_precision_L5':  float(cmap.get('long_range_precision_L5', 0.0)),
        'tm_proxy':                 float(tm),
        'length_evaluated':         L,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate v8 coord-head model')
    parser.add_argument('--model', default='model_v8.pt')
    parser.add_argument('--out',   default='results/eval_v8.json')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'ERROR: model file not found: {args.model}')
        print('Train v8 first:  python src/train_v8.py')
        sys.exit(1)

    print(f'Loading v8 model ({args.model}) ...')
    import torch
    raw   = torch.load(args.model, map_location='cpu', weights_only=False)
    aa_dim     = raw.get('aa_dim', ESM_RICH_DIM) if isinstance(raw, dict) else ESM_RICH_DIM
    has_coords = raw.get('coord_head', False)     if isinstance(raw, dict) else False
    seq_len    = raw.get('seq_len', COORD_LEN)    if isinstance(raw, dict) else COORD_LEN

    model = md.TransformerDistancePredictor(
        seq_len=seq_len, aa_dim=aa_dim, coord_head=has_coords)
    state = raw['state_dict'] if isinstance(raw, dict) else raw
    model.load_state_dict(state)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  {n_params:,} parameters  |  aa_dim={aa_dim}  coord_head={has_coords}  '
          f'seq_len={seq_len}\n')

    results = {}
    for pid, chain in PROTEINS:
        print(f'Evaluating {pid} ...', end=' ', flush=True)
        try:
            path   = utils.fetch_pdb(pid)
            seq    = utils.pdb_sequence(path, chain=chain, max_residues=COORD_LEN)
            coords = utils.pdb_ca_coords(path, chain=chain, max_residues=COORD_LEN)
            N = min(len(seq), len(coords), COORD_LEN)
            r = eval_one_protein(model, seq[:N], coords[:N], crop_len=COORD_LEN)
            results[pid] = r
            print(f'L={r["length_evaluated"]}  lDDT={r["local_lDDT"]:.1f}  '
                  f'F1={r["contact_f1"]:.3f}  LR={r["long_range_precision_L5"]:.3f}  '
                  f'TM={r["tm_proxy"]:.3f}')
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
