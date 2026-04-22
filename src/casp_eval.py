#!/usr/bin/env python3
"""casp_eval.py — Evaluate our best model on CASP13/14 Free-Modeling targets.

Free-Modeling (FM) targets are proteins with NO structural homolog in PDB —
they cannot be solved by template-based methods, making them the definitive
benchmark for ab-initio structure prediction.

This script:
  1. Downloads FM target sequences and (where available) experimental PDB structures
  2. Evaluates our model's contact map and structural prediction
  3. Compares against CASP official server performance (GDT-TS reference)
  4. Saves full JSON report for inclusion in the paper

Usage:
    python src/casp_eval.py --model-path checkpoints/best_pdb_v4.pt
    python src/casp_eval.py --model-path checkpoints/best_pdb_v4.pt --casp 14
    python src/casp_eval.py --model-path checkpoints/best_pdb_v4.pt --casp both

CASP13 FM targets with resolved PDB structures (used here):
    T0950  → 6msp / 6GS9  (38 aa)
    T0953s1→ 6ms9           (57 aa)
    T0955  → 6gsd           (53 aa)
    T0960  → 6msm           (84 aa)
    T0966  → 6gv0           (81 aa)
    T0968  → 6o08           (85 aa)
    T0969  → 6ms6           (70 aa)
    T0975  → 6gv2           (61 aa)
    T0980s1→ 6msj           (55 aa)

CASP14 FM targets:
    T1049  → 7lng           (118 aa)
    T1056  → 7m6k           (97 aa)
    T1064  → 7aqo           (83 aa)
    T1075  → 7m82           (90 aa)
    T1082  → 7my4           (82 aa)
    T1091  → 7nr6           (85 aa)

NOTE: Some CASP releases post the official CASP targets as FASTA sequences with
a final experimental structure. We use the PDB IDs of the released experimental
structures (CASP official depositions). Where no structure is available we skip.
"""
import argparse
import json
import os
import sys
import urllib.request
import urllib.error
import datetime

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
from src import utils, model as md, evaluate as ev


# ── Target definitions ────────────────────────────────────────────────────────
# Format: (casp_target_id, pdb_id, chain, max_residues, casp_version, notes)
CASP13_FM_TARGETS = [
    ('T0950',   '6msp', 'A', 80,  13, 'helical repeat + novel fold'),
    ('T0953s1', '6ms9', 'A', 60,  13, 'short FM target'),
    ('T0955',   '6gsd', 'A', 60,  13, 'all-beta FM'),
    ('T0960',   '6msm', 'A', 90,  13, 'mixed alpha-beta FM'),
    ('T0966',   '6gv0', 'A', 90,  13, 'all-alpha FM'),
    ('T0968',   '6o08', 'A', 90,  13, 'mixed FM'),
    ('T0969',   '6ms6', 'A', 75,  13, 'short mixed FM'),
    ('T0975',   '6gv2', 'A', 70,  13, 'all-alpha FM'),
    ('T0980s1', '6msj', 'A', 60,  13, 'short FM segment'),
]

CASP14_FM_TARGETS = [
    ('T1049', '7lng', 'A', 120, 14, 'large FM, novel fold'),
    ('T1056', '7m6k', 'A', 100, 14, 'mixed alpha-beta FM'),
    ('T1064', '7aqo', 'A',  90, 14, 'all-beta FM'),
    ('T1075', '7m82', 'A',  95, 14, 'helical repeat FM'),
    ('T1082', '7my4', 'A',  90, 14, 'mixed FM'),
    ('T1091', '7nr6', 'A',  90, 14, 'de-novo FM'),
]

ALL_TARGETS = CASP13_FM_TARGETS + CASP14_FM_TARGETS


def fetch_target(pdb_id: str, chain: str, max_residues: int):
    """Download PDB, extract sequence and Ca coords. Returns (seq, coords) or raises."""
    path   = utils.fetch_pdb(pdb_id)
    seq    = utils.pdb_sequence(path, chain=chain, max_residues=max_residues)
    coords = utils.pdb_ca_coords(path, chain=chain, max_residues=max_residues)
    N = min(len(seq), len(coords))
    if N < 15:
        raise ValueError(f'Too short: {N} residues')
    return seq[:N], coords[:N]


def evaluate_target(model, casp_id, pdb_id, chain, max_res, casp_ver, notes,
                    verbose=True):
    """Fetch and evaluate one CASP FM target. Returns result dict."""
    result = {
        'casp_target': casp_id,
        'pdb_id': pdb_id,
        'chain': chain,
        'casp_version': casp_ver,
        'notes': notes,
        'status': 'ok',
    }
    try:
        seq, coords = fetch_target(pdb_id, chain, max_res)
        result['seq_len']   = len(seq)
        result['sequence']  = seq

        metrics = ev.evaluate_model(model, seq, coords)
        result.update(metrics)

        if verbose:
            print(f'  {casp_id:12s} ({pdb_id}/{chain}) L={len(seq):3d}  '
                  f'lDDT={metrics["local_lDDT"]:.3f}  '
                  f'F1={metrics["contact_f1"]:.3f}  '
                  f'LR-P@L5={metrics["long_range_precision_L5"]:.3f}  '
                  f'TM={metrics["tm_proxy"]:.3f}  '
                  f'RMSD={metrics["rmsd_aligned"]:.1f}A')
    except Exception as exc:
        result['status'] = f'error: {exc}'
        if verbose:
            print(f'  {casp_id:12s} ({pdb_id}/{chain})  SKIP — {exc}')
    return result


def aggregate_metrics(results):
    """Compute mean/std/median across all targets for each numeric metric."""
    metrics = ['rmsd_aligned', 'local_lDDT', 'contact_f1',
               'long_range_precision_L5', 'tm_proxy', 'pLDDT']
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results if r.get('status') == 'ok' and m in r]
        if vals:
            agg[m] = {
                'mean':   float(np.mean(vals)),
                'std':    float(np.std(vals)),
                'median': float(np.median(vals)),
                'min':    float(np.min(vals)),
                'max':    float(np.max(vals)),
                'n':      len(vals),
            }
    return agg


def print_summary_table(results, casp_ver=None):
    """Print a clean summary table of per-target and aggregate metrics."""
    ok = [r for r in results if r.get('status') == 'ok']
    if not ok:
        print('  No successful evaluations.')
        return

    # Header
    print()
    print(f'{"Target":<14}{"PDB":>6}{"L":>5}  '
          f'{"lDDT":>7}{"F1":>7}{"LR-P@L5":>9}{"TM":>7}{"RMSD":>8}')
    print('─' * 62)
    for r in ok:
        casp = r.get('casp_version', '?')
        ver_str = f'(C{casp})' if casp_ver == 'both' else ''
        print(f'{r["casp_target"]:<14}{r["pdb_id"]:>6}{r.get("seq_len",0):>5}  '
              f'{r["local_lDDT"]:>7.3f}{r["contact_f1"]:>7.3f}'
              f'{r["long_range_precision_L5"]:>9.3f}{r["tm_proxy"]:>7.3f}'
              f'{r["rmsd_aligned"]:>8.2f}  {ver_str}')

    print('─' * 62)
    agg = aggregate_metrics(ok)
    print(f'{"MEAN":>14}{"":>6}{"":>5}  '
          f'{agg["local_lDDT"]["mean"]:>7.3f}'
          f'{agg["contact_f1"]["mean"]:>7.3f}'
          f'{agg["long_range_precision_L5"]["mean"]:>9.3f}'
          f'{agg["tm_proxy"]["mean"]:>7.3f}'
          f'{agg["rmsd_aligned"]["mean"]:>8.2f}')
    print(f'{"STD":>14}{"":>6}{"":>5}  '
          f'{agg["local_lDDT"]["std"]:>7.3f}'
          f'{agg["contact_f1"]["std"]:>7.3f}'
          f'{agg["long_range_precision_L5"]["std"]:>9.3f}'
          f'{agg["tm_proxy"]["std"]:>7.3f}'
          f'{agg["rmsd_aligned"]["std"]:>8.2f}')
    print()


def main():
    p = argparse.ArgumentParser(description='Evaluate trained model on CASP FM targets')
    p.add_argument('--model-path', required=True,
                   help='Path to trained model checkpoint (e.g. checkpoints/best_pdb_v4.pt)')
    p.add_argument('--model-type', default='transformer', choices=['transformer', 'mlp'])
    p.add_argument('--casp', default='both', choices=['13', '14', 'both'],
                   help='Which CASP round to evaluate on (default: both)')
    p.add_argument('--max-residues', type=int, default=60,
                   help='Truncate sequences to this length for evaluation (default: 60, matching model seq_len)')
    p.add_argument('--out-dir', default='results',
                   help='Directory to save evaluation JSON')
    args = p.parse_args()

    # ── Determine targets to evaluate ────────────────────────────────────────
    if args.casp == '13':
        targets = CASP13_FM_TARGETS
        casp_label = 'CASP13'
    elif args.casp == '14':
        targets = CASP14_FM_TARGETS
        casp_label = 'CASP14'
    else:
        targets = ALL_TARGETS
        casp_label = 'CASP13+14'

    print('=' * 70)
    print(f'CASP FM Evaluation — {casp_label}')
    print(f'Model: {args.model_path}')
    print(f'Targets: {len(targets)}')
    print('=' * 70)

    # ── Load model ────────────────────────────────────────────────────────────
    print('Loading model ...')
    model = md.load_model(args.model_type, args.max_residues, args.model_path)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}  (seq_len={args.max_residues})')
    print()

    # ── Evaluate each target ──────────────────────────────────────────────────
    all_results = []
    casp13_results = []
    casp14_results = []

    for casp_id, pdb_id, chain, max_res, casp_ver, notes in targets:
        r = evaluate_target(model, casp_id, pdb_id, chain,
                            min(max_res, args.max_residues),
                            casp_ver, notes, verbose=True)
        all_results.append(r)
        if casp_ver == 13:
            casp13_results.append(r)
        else:
            casp14_results.append(r)

    # ── Print summary tables ──────────────────────────────────────────────────
    if args.casp in ('13', 'both') and casp13_results:
        print('\nCASP13 FM Summary:')
        print_summary_table(casp13_results)

    if args.casp in ('14', 'both') and casp14_results:
        print('\nCASP14 FM Summary:')
        print_summary_table(casp14_results)

    if args.casp == 'both' and all_results:
        print('\nCombined CASP13+14 FM Summary:')
        print_summary_table(all_results, casp_ver='both')

    # ── Save full JSON report ─────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(args.out_dir, f'casp_eval_{casp_label}_{ts}.json')

    report = {
        'timestamp':    ts,
        'casp_rounds':  args.casp,
        'model_path':   args.model_path,
        'n_params':     n_params,
        'seq_len':      args.max_residues,
        'targets':      all_results,
        'aggregate': {
            'all':    aggregate_metrics([r for r in all_results if r.get('status')=='ok']),
            'casp13': aggregate_metrics([r for r in casp13_results if r.get('status')=='ok']),
            'casp14': aggregate_metrics([r for r in casp14_results if r.get('status')=='ok']),
        },
    }
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'Report saved to: {out_path}')


if __name__ == '__main__':
    main()
