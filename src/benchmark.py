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


# ── naive baselines ───────────────────────────────────────────────────────────

def _random_pred_dist(L, rng=None):
    """Fully random symmetric distance matrix — absolute lower bound."""
    if rng is None:
        rng = np.random.default_rng(0)
    raw = rng.uniform(3.0, 35.0, (L, L)).astype(np.float32)
    m = 0.5 * (raw + raw.T)
    np.fill_diagonal(m, 0.0)
    return m


def _separation_pred_dist(L):
    """Sequence-separation heuristic: dist[i,j] = |i-j| × 3.8 Å.

    3.8 Å is the average Cα–Cα step between consecutive residues.
    This ignores all sequence content — so beating it means the model
    has learned something *structural* rather than just *local chain geometry*.
    """
    idx = np.arange(L, dtype=np.float32)
    sep = np.abs(idx[:, None] - idx[None, :])
    return sep * 3.8


def _mean_pred_dist(train_Y, L):
    """Predict the global mean inter-residue distance for every pair.

    Training-set mean is an unbiased estimate if the test set is drawn from
    the same distribution — this baseline has 0 sequence sensitivity.
    """
    nonzero = train_Y[train_Y > 0]
    mean_d = float(nonzero.mean()) if len(nonzero) > 0 else 10.0
    m = np.full((L, L), mean_d, dtype=np.float32)
    np.fill_diagonal(m, 0.0)
    return m


def _eval_naive(pred_dist, seq, true_coords):
    """Compute eval metrics for a pre-built naive distance matrix."""
    true_dist = utils.coords_to_distances(true_coords)
    pred_coords = utils.gradient_mds(pred_dist, dim=3, n_iter=200)
    N = min(pred_coords.shape[0], true_coords.shape[0])
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords[:N], true_coords[:N])
    plddt = float(md.pseudo_plddt(pred_dist[:N, :N], true_dist[:N, :N]).mean())
    local_ldt = float(utils.local_lddt(pred_dist[:N, :N], true_dist[:N, :N]).mean())
    cmap = md.contact_map_score(pred_dist[:N, :N], true_dist[:N, :N])
    tm = utils.tm_score(aligned_pred, true_coords[:N])
    return {
        'rmsd_unaligned': float(np.sqrt(np.mean((pred_coords[:N] - true_coords[:N]) ** 2))),
        'rmsd_aligned': float(rmsd_aligned),
        'pLDDT': plddt,
        'local_lDDT': local_ldt,
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

        # aggregate — keep raw per-sample results for paired t-test
        metrics = list(scores[0].keys()) if scores else []
        results[model_type] = {
            metric: _summary([s[metric] for s in scores])
            for metric in metrics
        }
        results[model_type]['_raw'] = scores     # per-sample dicts for paired t-test
        results[model_type]['n_evaluated'] = len(scores)

    # ── naive baseline evaluation ──────────────────────────────────────────────
    print('\nEvaluating naive baselines ...')
    rng_base = np.random.default_rng(0)
    for baseline_name, bl_fn in [
        ('random', lambda seq, true: _eval_naive(
            _random_pred_dist(len(seq), rng=rng_base), seq, true)),
        ('seq_separation', lambda seq, true: _eval_naive(
            _separation_pred_dist(len(seq)), seq, true)),
        ('mean_distance', lambda seq, true: _eval_naive(
            _mean_pred_dist(train_Y, len(seq)), seq, true)),
    ]:
        bl_scores = []
        for seq, true_coords in zip(test_seqs, test_coords):
            try:
                bl_scores.append(bl_fn(seq, true_coords))
            except Exception as exc:
                print(f'  Warning: baseline {baseline_name} sample failed: {exc}')
        if bl_scores:
            metrics = list(bl_scores[0].keys())
            results[baseline_name] = {
                metric: _summary([s[metric] for s in bl_scores])
                for metric in metrics
            }
            results[baseline_name]['_raw'] = bl_scores   # per-sample for paired t-test
            results[baseline_name]['n_evaluated'] = len(bl_scores)
        print(f'  {baseline_name}: {len(bl_scores)} samples evaluated')

    # ── statistical tests (paired t-test) ─────────────────────────────────────
    model_order = ['random', 'seq_separation', 'mean_distance', 'mlp', 'transformer']
    model_labels = {
        'random': 'Random',
        'seq_separation': 'Seq-sep',
        'mean_distance': 'Mean-dist',
        'mlp': 'MLP',
        'transformer': 'Transformer',
    }

    print('\n' + '═' * 92)
    print(f'{"Metric":<20}', end='')
    for m in model_order:
        if m in results:
            print(f' {model_labels[m]:>17}', end='')
    print()
    print('─' * 92)

    for metric in ['rmsd_aligned', 'contact_f1', 'pLDDT', 'local_lDDT', 'tm_proxy']:
        if metric not in results.get('mlp', {}):
            continue
        print(f'{metric:<20}', end='')
        for m in model_order:
            if m not in results or metric not in results[m]:
                continue
            r = results[m][metric]
            print(f' {r["mean"]:8.4f}±{r["std"]:6.4f}', end='')
        print()
    print('═' * 92)

    # t-tests: Transformer vs each other method
    stat_tests = {}
    print('\nPaired t-tests (Transformer vs other methods, on matched test samples):')
    print(f'{"vs":<20} {"Metric":<18} {"t-stat":>8} {"p-value":>10} {"sig":>5}')
    print('-' * 65)
    for other in ['random', 'seq_separation', 'mean_distance', 'mlp']:
        if other not in results or '_raw' not in results.get('transformer', {}):
            continue
        for metric in ['contact_f1', 'rmsd_aligned', 'pLDDT']:
            if metric not in results.get('transformer', {}) or metric not in results.get(other, {}):
                continue
            # True paired t-test: same test sequence evaluated by both models
            tr_raw = results['transformer']['_raw']
            ot_raw = results[other].get('_raw', [])
            n = min(len(tr_raw), len(ot_raw))
            if n < 2:
                continue
            tr_vals = [s[metric] for s in tr_raw[:n]]
            ot_vals = [s[metric] for s in ot_raw[:n]]
            t_stat, p_val = stats.ttest_rel(tr_vals, ot_vals)
            sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
            key = f'transformer_vs_{other}_{metric}'
            stat_tests[key] = {'t_stat': float(t_stat), 'p_value': float(p_val)}
            print(f'{model_labels[other]:<20} {metric:<18} {t_stat:>8.3f} {p_val:>10.4f} {sig:>5}')
    print('Significance: * p<0.05, ** p<0.01, *** p<0.001, ns=not significant')

    # ── winner summary ────────────────────────────────────────────────────────
    if 'contact_f1' in results.get('mlp', {}) and 'contact_f1' in results.get('transformer', {}):
        tr_f1 = results['transformer']['contact_f1']['mean']
        mlp_f1 = results['mlp']['contact_f1']['mean']
        delta = (tr_f1 - mlp_f1) / (mlp_f1 + 1e-8) * 100
        print(f'\nTransformer contact F1 improvement over MLP: {delta:+.1f}%')
    if 'contact_f1' in results.get('seq_separation', {}):
        tr_f1 = results['transformer']['contact_f1']['mean']
        sep_f1 = results['seq_separation']['contact_f1']['mean']
        delta = (tr_f1 - sep_f1) / (sep_f1 + 1e-8) * 100
        print(f'Transformer contact F1 improvement over sequence-separation baseline: {delta:+.1f}%')

    # ── save results ───────────────────────────────────────────────────────────
    output = {
        'timestamp': datetime.datetime.now().isoformat(),
        'config': vars(args),
        # Remove _raw from saved output (too large) — keep summary stats only
        'results': {k: {mk: mv for mk, mv in v.items() if mk != '_raw'}
                    for k, v in results.items()},
        'stat_tests': stat_tests,
    }
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.result_dir, f'benchmark_{ts}.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nFull results saved to {out_path}')


if __name__ == '__main__':
    main()
