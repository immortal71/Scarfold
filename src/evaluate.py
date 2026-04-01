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


def evaluate_model(model, seq, true_coords):
    true_dist = utils.coords_to_distances(true_coords)
    pred_dist = md.predict(model, utils.one_hot(seq))
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    pred_coords = utils.classical_mds(pred_dist, dim=3)
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords, true_coords[:pred_coords.shape[0]])
    rmsd_unaligned = np.sqrt(np.mean((pred_coords - true_coords[:pred_coords.shape[0]])**2))
    plddt = md.pseudo_plddt(pred_dist, true_dist).mean()
    local_lddt = utils.local_lddt(pred_dist, true_dist).mean()
    cmap = md.contact_map_score(pred_dist, true_dist)
    tm = utils.tm_score(aligned_pred, true_coords[:aligned_pred.shape[0]])

    return {
        'rmsd_unaligned': float(rmsd_unaligned),
        'rmsd_aligned': float(rmsd_aligned),
        'pLDDT': float(plddt),
        'local_lDDT': float(local_lddt),
        'contact_f1': cmap['f1'],
        'tm_proxy': tm,
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

    metrics = evaluate_model(model, seq, coords)

    print('Evaluation metrics:')
    for k,v in metrics.items():
        print(f'  {k}: {v:.5f}')

    output_path = save_evaluation_results(metrics, output_dir=args.result_dir, filename=args.result_file)
    print(f'Saved evaluation metrics to {output_path}')
