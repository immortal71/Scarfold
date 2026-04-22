import argparse
import os
import sys
import json
import datetime
import numpy as np

# Ensure imports work when script run directly from repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import utils
from src import model as md
try:
    from src import pssm as pssm_utils
    _HAS_PSSM = True
except ImportError:
    _HAS_PSSM = False

try:
    from src.esm_utils import esm2_rich_encoding, ESM_RICH_DIM as _ESM_RICH_DIM
    _HAS_ESM = True
except Exception:
    _ESM_RICH_DIM = 368
    _HAS_ESM = False


def evaluate_model(model, seq, true_coords, pssm_path=None):
    """Evaluate model on one sequence.  Optionally supply a PSI-BLAST PSSM file."""
    true_dist = utils.coords_to_distances(true_coords)

    # ── build encoding — honour the aa_dim the model was trained with ────────
    model_aa_dim = getattr(model, 'aa_dim', utils.RICH_AA_DIM)
    if pssm_path and _HAS_PSSM:
        pssm_matrix = pssm_utils.parse_psiblast_pssm(pssm_path)
        enc = pssm_utils.encoding_with_pssm(seq, pssm=pssm_matrix)
    elif model_aa_dim == _ESM_RICH_DIM and _HAS_ESM:
        enc = esm2_rich_encoding(seq)     # v6 model: ESM-2 (320) + rich (48) = 368-dim
    elif model_aa_dim == 20:
        enc = utils.one_hot(seq)          # legacy model trained on one-hot
    else:
        enc = utils.rich_encoding(seq)    # modern model (48-dim rich encoding)

    pred_dist = md.predict(model, enc)
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    # ── reconstruct coordinates with gradient MDS ─────────────────────────────
    pred_coords = utils.gradient_mds(pred_dist, dim=3, n_iter=600)
    N = min(pred_coords.shape[0], true_coords.shape[0])
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords[:N], true_coords[:N])
    rmsd_unaligned = float(np.sqrt(np.mean((pred_coords[:N] - true_coords[:N]) ** 2)))
    plddt = float(md.pseudo_plddt(
        pred_dist, true_dist[:pred_dist.shape[0], :pred_dist.shape[1]]).mean())
    local_ldt = float(utils.local_lddt(
        pred_dist, true_dist[:pred_dist.shape[0], :pred_dist.shape[1]]).mean())
    cmap = md.contact_map_score(
        pred_dist, true_dist[:pred_dist.shape[0], :pred_dist.shape[1]])
    tm = utils.tm_score(aligned_pred, true_coords[:N])

    return {
        'rmsd_unaligned': rmsd_unaligned,
        'rmsd_aligned': float(rmsd_aligned),
        'pLDDT': plddt,
        'local_lDDT': local_ldt,
        'contact_f1': float(cmap['f1']),
        'long_range_precision_L5': float(cmap.get('long_range_precision_L5', 0.0)),
        'tm_proxy': float(tm),
    }




def save_evaluation_results(results, output_dir='results', filename=None):
    os.makedirs(output_dir, exist_ok=True)
    if filename is None:
        filename = f"eval_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Evaluate a trained model on a PDB/FASTA/auto dataset')
    p.add_argument('--model', choices=['mlp', 'transformer'], default='transformer')
    p.add_argument('--load-model', required=True)
    p.add_argument('--pdb', default='')
    p.add_argument('--pdb-id', default='')
    p.add_argument('--fasta', default='')
    p.add_argument('--chain', default='A')
    p.add_argument('--max-residues', type=int, default=120)
    p.add_argument('--pssm', default='', help='Path to PSI-BLAST ASCII PSSM file (optional; upgrades encoding to 50-dim)')
    p.add_argument('--result-dir', default='results', help='Directory to save evaluation json')
    p.add_argument('--result-file', default=None, help='Optional filename for evaluation JSON output')
    args = p.parse_args()

    if args.pdb_id:
        args.pdb = utils.fetch_pdb(args.pdb_id)

    if args.pdb:
        seq = utils.pdb_sequence(args.pdb, chain=args.chain, max_residues=args.max_residues)
        coords = utils.pdb_ca_coords(args.pdb, chain=args.chain, max_residues=args.max_residues)
    elif args.fasta:
        seq = utils.fasta_sequence(args.fasta, max_residues=args.max_residues)
        coords = utils.synthetic_native_coords(seq, seed=99)
    else:
        raise ValueError('Require --pdb, --pdb-id, or --fasta to evaluate.')

    model = md.load_model(args.model, len(seq), args.load_model)
    if len(seq) != model.seq_len:
        print(f'Warning: input sequence length {len(seq)} != model length {model.seq_len}. Truncating for evaluation.')
        seq = seq[:model.seq_len]
        coords = coords[:model.seq_len]

    pssm_path = args.pssm if args.pssm else None
    if pssm_path and not _HAS_PSSM:
        print('Warning: --pssm specified but src/pssm.py not found; falling back to rich_encoding.')
        pssm_path = None
    metrics = evaluate_model(model, seq, coords, pssm_path=pssm_path)

    print('Evaluation metrics:')
    for k,v in metrics.items():
        print(f'  {k}: {v:.5f}')

    output_path = save_evaluation_results(metrics, output_dir=args.result_dir, filename=args.result_file)
    print(f'Saved evaluation metrics to {output_path}')
