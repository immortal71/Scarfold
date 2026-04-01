"""ablation.py — Systematic ablation study: isolates the contribution of each component.

Tests all meaningful combinations of:
  - Model type:           MLP  |  Transformer
  - Input features:       one_hot(20)  |  rich_encoding(48)  |  pssm_encoding(50)
  - Loss:                 MSE only  |  MSE + Contact BCE
  - Coord reconstruction: classical MDS  |  gradient MDS

This allows us to answer the key question: *which component contributes how much?*

Usage:
    python src/ablation.py --samples 400 --length 40 --epochs 60 --n-test 30
    python src/ablation.py --samples 400 --length 40 --epochs 60 --quick   (fast 8-epoch run)

Outputs:
    - results/ablation_<timestamp>.json   full per-sample results
    - results/ablation_<timestamp>.csv    summary table (import into Excel/LaTeX)
    - Printed ASCII table for immediate reading
"""

import argparse
import csv
import json
import os
import sys
import datetime
import itertools
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import utils
from src import model as md
from src.pssm import encoding_with_pssm, pseudo_pssm, PSSM_FEAT_DIM


# ── feature builders ──────────────────────────────────────────────────────────

def _make_features(seqs, feature_set):
    """Return (N, L, D) feature array for the given feature_set string."""
    if feature_set == 'one_hot':
        return np.stack([utils.one_hot(s) for s in seqs])
    if feature_set == 'rich':
        return np.stack([utils.rich_encoding(s) for s in seqs])
    if feature_set == 'pssm':
        return np.stack([encoding_with_pssm(s) for s in seqs])
    raise ValueError(f'Unknown feature set: {feature_set}')


def _feat_dim(feature_set):
    if feature_set == 'one_hot':
        return 20
    if feature_set == 'rich':
        return utils.RICH_AA_DIM
    if feature_set == 'pssm':
        return PSSM_FEAT_DIM
    raise ValueError(feature_set)


# ── evaluation ────────────────────────────────────────────────────────────────

def _evaluate_one(model, seq, true_coords, feature_set, use_grad_mds):
    enc = _make_features([seq], feature_set)[0]
    true_dist = utils.coords_to_distances(true_coords)
    pred_dist = md.predict(model, enc)
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    L_pred = pred_dist.shape[0]
    L_true = true_coords.shape[0]
    N = min(L_pred, L_true)

    if use_grad_mds:
        pred_coords = utils.gradient_mds(pred_dist, dim=3, n_iter=200)
    else:
        pred_coords = utils.classical_mds(pred_dist, dim=3)

    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords[:N], true_coords[:N])
    cmap = md.contact_map_score(pred_dist, true_dist[:L_pred, :L_pred])
    plddt = float(md.pseudo_plddt(pred_dist, true_dist[:L_pred, :L_pred]).mean())
    local_lddt = float(utils.local_lddt(pred_dist, true_dist[:L_pred, :L_pred]).mean())
    tm = utils.tm_score(aligned_pred, true_coords[:N])

    return {
        'rmsd_aligned': float(rmsd_aligned),
        'contact_f1': float(cmap['f1']),
        'contact_precision': float(cmap['precision']),
        'contact_recall': float(cmap['recall']),
        'pLDDT': plddt,
        'local_lDDT': local_lddt,
        'tm_proxy': float(tm),
    }


# ── one condition ─────────────────────────────────────────────────────────────

def _run_condition(name, model_type, feature_set, use_contact_loss, use_grad_mds,
                   train_X, train_Y, test_seqs, test_coords, seq_len, epochs, lr, device):
    aa_dim = _feat_dim(feature_set)
    model = md.get_model(model_type, seq_len, aa_dim=aa_dim)
    contact_w = 0.5 if use_contact_loss else 0.0
    model = md.train_simple(model, train_X, train_Y, epochs=epochs, lr=lr,
                            device=device, contact_weight=contact_w)

    results = []
    for seq, coords in zip(test_seqs, test_coords):
        try:
            s = seq[:model.seq_len]
            c = coords[:model.seq_len]
            m = _evaluate_one(model, s, c, feature_set, use_grad_mds)
            results.append(m)
        except Exception as exc:
            print(f'    Warning [{name}]: {exc}')

    summary = {}
    if results:
        for k in results[0]:
            vals = [r[k] for r in results]
            summary[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}
    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Ablation study: isolate each component\'s contribution')
    p.add_argument('--samples', type=int, default=400)
    p.add_argument('--length',  type=int, default=40)
    p.add_argument('--n-test',  type=int, default=30)
    p.add_argument('--epochs',  type=int, default=60)
    p.add_argument('--lr',      type=float, default=5e-4)
    p.add_argument('--device',  type=str, default='cpu')
    p.add_argument('--quick',   action='store_true', help='Fast 8-epoch run to verify pipeline')
    p.add_argument('--result-dir', type=str, default='results')
    args = p.parse_args()

    if args.quick:
        args.epochs = 8
        args.samples = 100
        args.n_test = 10
        print('Quick mode: epochs=8, samples=100, n_test=10')

    os.makedirs(args.result_dir, exist_ok=True)

    # ── build shared dataset ─────────────────────────────────────────────────
    print(f'Generating dataset: {args.samples} train + {args.n_test} test, L={args.length} ...')
    all_seqs, all_dists = utils.make_synthetic_dataset(
        num=args.samples + args.n_test, L=args.length, seed=42
    )
    test_seqs  = all_seqs[args.samples:]
    test_coords = [utils.synthetic_native_coords(s, seed=999 + i)
                   for i, s in enumerate(test_seqs)]

    # Pre-compute features for all three feature sets (match train split)
    print('Pre-computing features ...')
    feats = {
        'one_hot': _make_features(all_seqs[:args.samples], 'one_hot'),
        'rich':    _make_features(all_seqs[:args.samples], 'rich'),
        'pssm':    _make_features(all_seqs[:args.samples], 'pssm'),
    }
    train_Y = all_dists[:args.samples]
    seq_len = args.length

    # ── define ablation conditions ───────────────────────────────────────────
    # Format: (display_name, model_type, feature_set, contact_loss, grad_mds)
    conditions = [
        ('MLP  | one-hot  | MSE      | classMDS', 'mlp',         'one_hot', False, False),
        ('MLP  | rich-48  | MSE      | classMDS', 'mlp',         'rich',    False, False),
        ('MLP  | pssm-50  | MSE      | classMDS', 'mlp',         'pssm',    False, False),
        ('MLP  | rich-48  | MSE+CBC  | classMDS', 'mlp',         'rich',    True,  False),
        ('MLP  | rich-48  | MSE+CBC  | gradMDS ', 'mlp',         'rich',    True,  True),
        ('TRF  | one-hot  | MSE      | classMDS', 'transformer', 'one_hot', False, False),
        ('TRF  | rich-48  | MSE      | classMDS', 'transformer', 'rich',    False, False),
        ('TRF  | pssm-50  | MSE      | classMDS', 'transformer', 'pssm',    False, False),
        ('TRF  | rich-48  | MSE+CBC  | classMDS', 'transformer', 'rich',    True,  False),
        ('TRF  | pssm-50  | MSE+CBC  | classMDS', 'transformer', 'pssm',    True,  False),
        ('TRF  | pssm-50  | MSE+CBC  | gradMDS ', 'transformer', 'pssm',    True,  True),  # BEST
    ]

    print(f'\nRunning {len(conditions)} ablation conditions x {args.epochs} epochs ...\n')
    all_results = {}

    for name, model_type, feature_set, use_cbc, use_grad_mds in conditions:
        train_X = feats[feature_set]
        print(f'[{name}] training ...')
        summary = _run_condition(
            name, model_type, feature_set, use_cbc, use_grad_mds,
            train_X, train_Y, test_seqs, test_coords,
            seq_len, args.epochs, args.lr, args.device,
        )
        all_results[name] = summary
        f1     = summary.get('contact_f1', {}).get('mean', float('nan'))
        rmsd   = summary.get('rmsd_aligned', {}).get('mean', float('nan'))
        plddt  = summary.get('pLDDT', {}).get('mean', float('nan'))
        print(f'    contact F1={f1:.4f}  RMSD={rmsd:.3f} Å  pLDDT={plddt:.1f}')

    # ── print summary table ──────────────────────────────────────────────────
    metrics_to_show = ['contact_f1', 'rmsd_aligned', 'pLDDT', 'local_lDDT', 'tm_proxy']
    col_w = 14
    header = f'{"Condition":<46}' + ''.join(f'{m:>{col_w}}' for m in metrics_to_show)
    print('\n' + '=' * (46 + col_w * len(metrics_to_show)))
    print('ABLATION RESULTS  (mean over test set)')
    print('=' * (46 + col_w * len(metrics_to_show)))
    print(header)
    print('-' * (46 + col_w * len(metrics_to_show)))

    rows_for_csv = []
    for name, _, _, _, _ in conditions:
        s = all_results.get(name, {})
        row = {'condition': name}
        line = f'{name:<46}'
        for m in metrics_to_show:
            val = s.get(m, {}).get('mean', float('nan'))
            std = s.get(m, {}).get('std', float('nan'))
            line += f'{val:>{col_w}.4f}'
            row[m + '_mean'] = val
            row[m + '_std']  = std
        print(line)
        rows_for_csv.append(row)

    print('=' * (46 + col_w * len(metrics_to_show)))

    # Best condition by contact F1
    best = max(all_results.items(),
               key=lambda kv: kv[1].get('contact_f1', {}).get('mean', -1))
    print(f'\nBest condition by contact F1: [{best[0]}]')
    print(f"  contact F1 = {best[1]['contact_f1']['mean']:.4f} ± {best[1]['contact_f1']['std']:.4f}")

    # ── save results ─────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    json_path = os.path.join(args.result_dir, f'ablation_{ts}.json')
    csv_path  = os.path.join(args.result_dir, f'ablation_{ts}.csv')

    with open(json_path, 'w') as f:
        json.dump({'config': vars(args), 'results': all_results}, f, indent=2)

    fieldnames = ['condition'] + [m + s for m in metrics_to_show for s in ('_mean', '_std')]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_for_csv)

    print(f'\nJSON: {json_path}')
    print(f'CSV:  {csv_path}  ← paste into Excel or LaTeX table generator')
    print('\nTip: use the CSV to fill in Table 2 of report/report.md')


if __name__ == '__main__':
    main()
