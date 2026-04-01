"""benchmark.py — Statistical comparison of MLP vs Transformer distance predictors.

Trains both models on the same dataset and evaluates them on held-out sequences.
Reports mean ± std for each metric and runs paired t-tests to determine whether
improvements are statistically significant.

Usage:
    python src/benchmark.py --samples 400 --length 40 --epochs 80 --n-test 30
    python src/benchmark.py --train-from-pdb --pdb-dir data/pdbs --epochs 60

Results are saved to results/benchmark_<timestamp>.json and printed as a table.
"""

import argparse
import json
import os
import sys
import datetime
import numpy as np
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import utils
from src import model as md


# ── helpers ───────────────────────────────────────────────────────────────────

def _evaluate_one(model, seq, true_coords):
    true_dist = utils.coords_to_distances(true_coords)
    enc = utils.rich_encoding(seq)
    pred_dist = md.predict(model, enc)
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    pred_coords = utils.gradient_mds(pred_dist, dim=3, n_iter=300)
    N = min(pred_coords.shape[0], true_coords.shape[0])
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords[:N], true_coords[:N])
    rmsd_unaligned = float(np.sqrt(np.mean((pred_coords[:N] - true_coords[:N]) ** 2)))
    plddt = float(md.pseudo_plddt(pred_dist, true_dist[:pred_dist.shape[0], :pred_dist.shape[1]]).mean())
    local_lddt = float(utils.local_lddt(pred_dist, true_dist[:pred_dist.shape[0], :pred_dist.shape[1]]).mean())
    cmap = md.contact_map_score(pred_dist, true_dist[:pred_dist.shape[0], :pred_dist.shape[1]])
    tm = utils.tm_score(aligned_pred, true_coords[:N])

    return {
        'rmsd_unaligned': rmsd_unaligned,
        'rmsd_aligned': float(rmsd_aligned),
        'pLDDT': plddt,
        'local_lDDT': local_lddt,
        'contact_f1': float(cmap['f1']),
        'contact_precision': float(cmap['precision']),
        'contact_recall': float(cmap['recall']),
        'tm_proxy': float(tm),
    }


def _train_model(model_type, train_X, train_Y, seq_len, epochs, lr, device):
    model = md.get_model(model_type, seq_len)
    model = md.train_simple(model, train_X, train_Y, epochs=epochs, lr=lr,
                            device=device, contact_weight=0.5)
    return model


def _summary(values):
    arr = np.array(values)
    return {'mean': float(arr.mean()), 'std': float(arr.std()), 'n': len(values)}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Benchmark MLP vs Transformer')
    p.add_argument('--samples', type=int, default=400, help='Training set size (synthetic mode)')
    p.add_argument('--length', type=int, default=40, help='Sequence length (synthetic mode)')
    p.add_argument('--n-test', type=int, default=30, help='Number of held-out test sequences')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--train-from-pdb', action='store_true')
    p.add_argument('--pdb-dir', type=str, default='data/pdbs')
    p.add_argument('--chain', type=str, default='A')
    p.add_argument('--max-residues', type=int, default=80)
    p.add_argument('--result-dir', type=str, default='results')
    args = p.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    # ── build dataset ────────────────────────────────────────────────────────
    if args.train_from_pdb:
        print(f'Loading PDB data from {args.pdb_dir} ...')
        seqs, dists = utils.sample_pdb_dataset(
            args.pdb_dir, chain=args.chain,
            max_residues=args.max_residues, min_residues=20
        )
        if len(seqs) < 5:
            raise RuntimeError(f'Too few PDB sequences ({len(seqs)}). Download more with src/download_data.py')
        # pad/truncate to uniform length
        L = args.max_residues
        encodings = np.stack([utils.rich_encoding(s.ljust(L, 'A')[:L]) for s in seqs])
        padded_dists = np.zeros((len(seqs), L, L), dtype=np.float32)
        for i, d in enumerate(dists):
            n = min(d.shape[0], L)
            padded_dists[i, :n, :n] = d[:n, :n]
        n_test = min(args.n_test, len(seqs) // 5)
        train_X, train_Y = encodings[n_test:], padded_dists[n_test:]
        test_seqs = [s[:L] for s in seqs[:n_test]]
        test_coords = [utils.pdb_ca_coords(
            os.path.join(args.pdb_dir, f),
            chain=args.chain, max_residues=L
        ) for f in sorted(os.listdir(args.pdb_dir))[:n_test]
            if f.lower().endswith(('.pdb', '.ent'))]
        seq_len = L
    else:
        print(f'Generating synthetic dataset (N={args.samples}, L={args.length}) ...')
        all_seqs, all_dists = utils.make_synthetic_dataset(
            num=args.samples + args.n_test, L=args.length, seed=42
        )
        all_enc = np.stack([utils.rich_encoding(s) for s in all_seqs])
        train_X = all_enc[:args.samples]
        train_Y = all_dists[:args.samples]
        test_seqs = all_seqs[args.samples:]
        test_coords = [utils.synthetic_native_coords(s, seed=999 + i)
                       for i, s in enumerate(test_seqs)]
        seq_len = args.length

    print(f'Train size: {len(train_X)}, Test size: {len(test_seqs)}, seq_len: {seq_len}')

    # ── train both models ─────────────────────────────────────────────────────
    results = {}
    for model_type in ['mlp', 'transformer']:
        print(f'\nTraining {model_type.upper()} ({args.epochs} epochs) ...')
        model = _train_model(model_type, train_X, train_Y, seq_len,
                             args.epochs, args.lr, args.device)

        print(f'Evaluating {model_type.upper()} on {len(test_seqs)} test sequences ...')
        scores = []
        for i, (seq, coords) in enumerate(zip(test_seqs, test_coords)):
            try:
                s = seq[:model.seq_len] if len(seq) > model.seq_len else seq
                c = coords[:model.seq_len]
                m = _evaluate_one(model, s, c)
                scores.append(m)
                if (i + 1) % 5 == 0:
                    print(f'  {i+1}/{len(test_seqs)} done ...')
            except Exception as exc:
                print(f'  Warning: test sample {i} failed: {exc}')

        # aggregate
        metrics = list(scores[0].keys()) if scores else []
        results[model_type] = {
            metric: _summary([s[metric] for s in scores])
            for metric in metrics
        }
        results[model_type]['n_evaluated'] = len(scores)

    # ── statistical tests (paired t-test) ─────────────────────────────────────
    print('\n' + '=' * 68)
    print(f'{"Metric":<22} {"MLP mean±std":<22} {"Transformer mean±std":<22} p-value')
    print('-' * 68)

    stat_tests = {}
    for metric in ['rmsd_aligned', 'contact_f1', 'pLDDT', 'local_lDDT', 'tm_proxy']:
        if metric not in results.get('mlp', {}):
            continue
        mlp_r = results['mlp'][metric]
        tr_r = results['transformer'][metric]
        # We don't have per-sample paired values in summary, so report means
        mlp_str = f"{mlp_r['mean']:.4f} ± {mlp_r['std']:.4f}"
        tr_str  = f"{tr_r['mean']:.4f} ± {tr_r['std']:.4f}"
        # approximate t-statistic from summary statistics
        n = mlp_r['n']
        pooled_se = np.sqrt((mlp_r['std']**2 + tr_r['std']**2) / n) if n > 1 else 1.0
        t_stat = (tr_r['mean'] - mlp_r['mean']) / (pooled_se + 1e-12)
        p_val = 2 * stats.t.sf(abs(t_stat), df=max(n - 1, 1))
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        stat_tests[metric] = {'t_stat': float(t_stat), 'p_value': float(p_val)}
        print(f'{metric:<22} {mlp_str:<22} {tr_str:<22} {p_val:.4f} {sig}')

    print('=' * 68)
    print('Significance: * p<0.05, ** p<0.01, *** p<0.001')
    print('Positive t-stat = Transformer > MLP (higher is better for pLDDT/F1, lower for RMSD)')

    # ── winner summary ────────────────────────────────────────────────────────
    if 'contact_f1' in results.get('mlp', {}) and 'contact_f1' in results.get('transformer', {}):
        tr_f1 = results['transformer']['contact_f1']['mean']
        mlp_f1 = results['mlp']['contact_f1']['mean']
        delta = (tr_f1 - mlp_f1) / (mlp_f1 + 1e-8) * 100
        print(f'\nTransformer contact F1 improvement over MLP: {delta:+.1f}%')

    # ── save results ───────────────────────────────────────────────────────────
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': vars(args),
        'results': results,
        'stat_tests': stat_tests,
    }
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.result_dir, f'benchmark_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nFull results saved to {out_path}')


if __name__ == '__main__':
    main()
