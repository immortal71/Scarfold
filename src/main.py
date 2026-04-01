import argparse
import numpy as np
import os
import sys

# Ensure imports work when script run directly from repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import utils
from src import model as md
from src import visualize
from src import train as tr
from src import evaluate as ev

def demo_run(num_samples=250, L=40, seed=2, model_type='mlp', load_model_path=None, save_model_path=None):
    print('Generating synthetic dataset...')
    seqs, dists = utils.make_synthetic_dataset(num=num_samples, L=L, seed=seed)

    # rich encoding: one-hot + BLOSUM62 + physicochemical (48 features per residue)
    onehots = np.stack([utils.rich_encoding(s) for s in seqs])

    # train-test split
    n_train = min(200, num_samples - 5)
    train_X, train_Y = onehots[:n_train], dists[:n_train]
    test_idx = min(220, num_samples - 1)
    test_seq = seqs[test_idx]
    test_onehot = onehots[test_idx]

    true_coords = utils.synthetic_native_coords(test_seq, seed=222)
    true_dist = utils.coords_to_distances(true_coords)

    print('Initializing model...')
    seq_len = train_X.shape[1]
    if load_model_path:
        print('Loading model weights from', load_model_path)
        model = md.load_model(model_type, seq_len, load_model_path)
    else:
        model = md.get_model(model_type, seq_len)
        print(f'Training model ({model_type})...')
        model = md.train_simple(model, train_X, train_Y, epochs=120 if model_type == 'mlp' else 240, lr=1e-3)
        if save_model_path:
            md.save_model(model, save_model_path)
            print('Saved model weights to', save_model_path)

    print('Predicting distances for held-out sequence...')
    pred_dist = md.predict(model, test_onehot)

    # enforce symmetry and positive distances
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    print('Computing predicted coordinates with gradient-based optimisation...')
    pred_coords = utils.gradient_mds(pred_dist, dim=3)

    print('Computing alignment and RMSD')
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords, true_coords[:pred_coords.shape[0]])
    rmsd_unaligned = np.sqrt(np.mean((pred_coords - true_coords[:pred_coords.shape[0]])**2))

    print(f'RMSD (unaligned): {rmsd_unaligned:.3f} Å')
    print(f'RMSD (Kabsch aligned): {rmsd_aligned:.3f} Å')

    print('Computing AlphaFold-like confidence scores (pLDDT proxy)')
    plddt_per_res = md.pseudo_plddt(pred_dist, true_dist)
    local_lddt_per_res = utils.local_lddt(pred_dist, true_dist)
    print(f'Mean pseudo pLDDT: {plddt_per_res.mean():.2f}')
    print(f'Mean local lDDT: {local_lddt_per_res.mean():.2f}')

    print('Contact map metrics')
    cmap_metrics = md.contact_map_score(pred_dist, true_dist, threshold=8.0)
    print(f"Contact map F1: {cmap_metrics['f1']:.3f}, Precision: {cmap_metrics['precision']:.3f}, Recall: {cmap_metrics['recall']:.3f}")

    print('Visualizing predicted structure vs native')
    visualize.plot_pred_and_native(aligned_pred, true_coords[:aligned_pred.shape[0]], save_html='out_pred_vs_native.html')

    print('Visualizing predicted structure with per-residue pseudo pLDDT colors')
    visualize.plot_structure(pred_coords, title='Predicted structure (plddt colored)',
                             save_html='out_pred_struct_colored.html', colors=plddt_per_res)

    print('Visualizing contact maps')
    visualize.plot_contact_map(pred_dist, true_dist, save_html='out_contact_map.html')
    visualize.plot_plddt(plddt_per_res, save_html='out_plddt_profile.html')
    visualize.plot_plddt(local_lddt_per_res, save_html='out_local_lddt_profile.html', title='local lDDT profile')
    tm = utils.tm_score(aligned_pred, true_coords[:aligned_pred.shape[0]])
    visualize.plot_tm_score(tm, save_html='out_tm_score.html')

    print('Done. metrics:')
    print('  RMSD aligned:', rmsd_aligned)
    print('  RMSD unaligned:', rmsd_unaligned)
    print('  mean pLDDT:', float(plddt_per_res.mean()))
    print('  contact F1:', float(cmap_metrics['f1']))
    print('  TM score proxy:', tm)


def demo_run_pdb(pdb_path, chain='A', num_samples=250, L=80, seed=2, model_type='transformer', load_model_path=None, save_model_path=None):
    print('Loading PDB CA coordinates from', pdb_path)
    true_coords = utils.pdb_ca_coords(pdb_path, chain=chain, max_residues=L)
    seq = utils.pdb_sequence(pdb_path, chain=chain, max_residues=L)
    if len(seq) < 8:
        raise ValueError('PDB sequence too short or chain not found')
    true_dist = utils.coords_to_distances(true_coords)

    print('Generating synthetic training dataset (N={} L={} seq)'.format(num_samples, len(seq)))
    seqs, dists = utils.make_synthetic_dataset(num=num_samples, L=len(seq), seed=seed)
    onehots = np.stack([utils.rich_encoding(s) for s in seqs])
    train_X, train_Y = onehots[:max(1, num_samples-30)], dists[:max(1, num_samples-30)]

    print('Training predictor on synthetic data...')
    if load_model_path:
        model = md.load_model(model_type, len(seq), load_model_path)
    else:
        model = md.get_model(model_type, len(seq))
        model = md.train_simple(model, train_X, train_Y, epochs=180 if model_type == 'mlp' else 300, lr=5e-4)
        if save_model_path:
            md.save_model(model, save_model_path)
            print('Saved model weights to', save_model_path)

    if model.seq_len != len(seq):
        print(f'Warning: model expects seq_len {model.seq_len}, but PDB sequence is {len(seq)}. Truncating for model.')
        seq = seq[:model.seq_len]
        true_coords = true_coords[:model.seq_len]
        true_dist = utils.coords_to_distances(true_coords)

    print('Predicting on PDB sequence...')
    test_onehot = utils.rich_encoding(seq)
    pred_dist = md.predict(model, test_onehot)
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    pred_coords = utils.gradient_mds(pred_dist, dim=3)
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords, true_coords[:pred_coords.shape[0]])
    rmsd_unaligned = np.sqrt(np.mean((pred_coords - true_coords[:pred_coords.shape[0]])**2))

    plddt_per_res = md.pseudo_plddt(pred_dist, true_dist)
    local_lddt_per_res = utils.local_lddt(pred_dist, true_dist)
    cmap_metrics = md.contact_map_score(pred_dist, true_dist, threshold=8.0)
    tm = utils.tm_score(aligned_pred, true_coords[:aligned_pred.shape[0]])

    print('RMSD (unaligned):', rmsd_unaligned)
    print('RMSD (aligned):', rmsd_aligned)
    print('Mean pseudo pLDDT:', float(plddt_per_res.mean()))
    print('Mean local lDDT:', float(local_lddt_per_res.mean()))
    print('Contact map F1:', cmap_metrics['f1'])
    print('TM proxy:', tm)

    visualize.plot_pred_and_native(aligned_pred, true_coords[:aligned_pred.shape[0]], save_html='out_pred_vs_native_pdb.html')
    visualize.plot_structure(pred_coords, title='Predicted structure (pLDDT)', save_html='out_pred_struct_colored_pdb.html', colors=plddt_per_res)
    visualize.plot_contact_map(pred_dist, true_dist, save_html='out_contact_map_pdb.html')
    visualize.plot_plddt(plddt_per_res, save_html='out_plddt_profile_pdb.html')
    visualize.plot_plddt(local_lddt_per_res, save_html='out_local_lddt_profile_pdb.html', title='local lDDT profile')
    visualize.plot_tm_score(tm, save_html='out_tm_score_pdb.html')


def demo_run_fasta(fasta_path, num_samples=250, L=40, seed=2, model_type='mlp', load_model_path=None, save_model_path=None):
    print('Loading FASTA sequence:', fasta_path)
    seq = utils.fasta_sequence(fasta_path, max_residues=L)
    if len(seq) < 8:
        raise ValueError('FASTA sequence too short')

    true_coords = utils.synthetic_native_coords(seq, seed=seed + 99)
    true_dist = utils.coords_to_distances(true_coords)

    print('Generating synthetic training data...')
    seqs, dists = utils.make_synthetic_dataset(num=num_samples, L=len(seq), seed=seed)
    onehots = np.stack([utils.rich_encoding(s) for s in seqs])
    train_X, train_Y = onehots[:max(1, num_samples-30)], dists[:max(1, num_samples-30)]

    print('Initializing model...')
    if load_model_path:
        model = md.load_model(model_type, len(seq), load_model_path)
    else:
        model = md.get_model(model_type, len(seq))
        model = md.train_simple(model, train_X, train_Y, epochs=120 if model_type == 'mlp' else 240, lr=1e-3)
        if save_model_path:
            md.save_model(model, save_model_path)
            print('Saved model weights to', save_model_path)

    if model.seq_len != len(seq):
        print(f'Warning: model expects seq_len {model.seq_len}, but FASTA sequence is {len(seq)}. Truncating for model.')
        seq = seq[:model.seq_len]
        true_coords = true_coords[:model.seq_len]
        true_dist = utils.coords_to_distances(true_coords)

    test_onehot = utils.rich_encoding(seq)
    pred_dist = md.predict(model, test_onehot)
    pred_dist = 0.5 * (pred_dist + pred_dist.T)

    pred_coords = utils.gradient_mds(pred_dist, dim=3)
    rmsd_aligned, aligned_pred = utils.rmsd_kabsch(pred_coords, true_coords[:pred_coords.shape[0]])
    rmsd_unaligned = np.sqrt(np.mean((pred_coords - true_coords[:pred_coords.shape[0]])**2))

    plddt_per_res = md.pseudo_plddt(pred_dist, true_dist)
    local_lddt_per_res = utils.local_lddt(pred_dist, true_dist)
    cmap_metrics = md.contact_map_score(pred_dist, true_dist, threshold=8.0)
    tm = utils.tm_score(aligned_pred, true_coords[:aligned_pred.shape[0]])

    print('RMSD (unaligned):', rmsd_unaligned)
    print('RMSD (aligned):', rmsd_aligned)
    print('Mean pseudo pLDDT:', float(plddt_per_res.mean()))
    print('Mean local lDDT:', float(local_lddt_per_res.mean()))
    print('Contact map F1:', cmap_metrics['f1'])
    print('TM proxy:', tm)

    visualize.plot_pred_and_native(aligned_pred, true_coords[:aligned_pred.shape[0]], save_html='out_pred_vs_native_fasta.html')
    visualize.plot_structure(pred_coords, title='Predicted structure (pLDDT)', save_html='out_pred_struct_colored_fasta.html', colors=plddt_per_res)
    visualize.plot_contact_map(pred_dist, true_dist, save_html='out_contact_map_fasta.html')
    visualize.plot_plddt(plddt_per_res, save_html='out_plddt_profile_fasta.html')
    visualize.plot_plddt(local_lddt_per_res, save_html='out_local_lddt_profile_fasta.html', title='local lDDT profile')
    visualize.plot_tm_score(tm, save_html='out_tm_score_fasta.html')


def demo_run_pdbid(pdb_id, chain='A', num_samples=250, L=80, seed=2, model_type='transformer'):
    print('Downloading PDB', pdb_id)
    pdb_path = utils.fetch_pdb(pdb_id)
    return demo_run_pdb(pdb_path, chain=chain, num_samples=num_samples, L=L, seed=seed, model_type=model_type)


def run_training(args):
    print('Running training workflow...')
    if args.train_from_pdb:
        if not os.path.exists(args.pdb_dir):
            raise FileNotFoundError(f'PDB directory not found: {args.pdb_dir}')
        seqs, dists = utils.sample_pdb_dataset(args.pdb_dir, chain=args.chain,
                                              max_residues=args.max_residues,
                                              min_residues=args.min_residues)
        if len(seqs) < 2:
            print(f'Found only {len(seqs)} PDB entries; falling back to synthetic dataset for stability.')
            seqs, dists = utils.make_synthetic_dataset(num=args.samples, L=args.length, seed=42)
            onehots = np.stack([utils.rich_encoding(s) for s in seqs])
            n_train = int(args.samples * 0.8)
            train_X, val_X = onehots[:n_train], onehots[n_train:]
            train_Y, val_Y = dists[:n_train], dists[n_train:]
            model = md.get_model(args.model, args.length, aa_dim=utils.RICH_AA_DIM)
        else:
            onehots = np.stack([utils.rich_encoding(s.ljust(args.max_residues, 'A')[:args.max_residues]) for s in seqs])
            n_train = max(1, int(len(seqs) * 0.8))
            train_X, val_X = onehots[:n_train], onehots[n_train:]
            train_Y, val_Y = dists[:n_train], dists[n_train:]
            model_len = min(args.max_residues, onehots.shape[1])
            print(f'Using {len(seqs)} PDB entries for training; seq_len={model_len}.')
            model = md.get_model(args.model, model_len, aa_dim=utils.RICH_AA_DIM)
    else:
        seqs, dists = utils.make_synthetic_dataset(num=args.samples, L=args.length, seed=42)
        onehots = np.stack([utils.rich_encoding(s) for s in seqs])
        n_train = int(args.samples * 0.8)
        train_X, val_X = onehots[:n_train], onehots[n_train:]
        train_Y, val_Y = dists[:n_train], dists[n_train:]
        model = md.get_model(args.model, args.length, aa_dim=utils.RICH_AA_DIM)

    model, history = tr.train_and_validate(model, train_X, train_Y, val_X, val_Y,
                                           epochs=args.epochs, lr=args.lr,
                                           device=args.device, checkpoint_dir=args.checkpoint_dir)
    if args.save_model:
        md.save_model(model, args.save_model)
        print('Saved final model to', args.save_model)
    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
            writer.writeheader()
            writer.writerows(history)
        print('Training log written to', args.csv)


def run_evaluation(args):
    print('Running evaluation workflow...')
    if args.pdb_id:
        args.pdb = utils.fetch_pdb(args.pdb_id)

    if args.pdb:
        seq = utils.pdb_sequence(args.pdb, chain=args.chain, max_residues=args.max_residues)
        coords = utils.pdb_ca_coords(args.pdb, chain=args.chain, max_residues=args.max_residues)
    elif args.fasta:
        seq = utils.fasta_sequence(args.fasta, max_residues=args.max_residues)
        coords = utils.synthetic_native_coords(seq, seed=99)
    else:
        raise ValueError('Require --pdb, --pdb-id, or --fasta for evaluation')

    model = md.load_model(args.model, len(seq), args.load_model)
    if len(seq) != model.seq_len:
        print(f'Warning: input sequence length {len(seq)} != model length {model.seq_len}. Truncating for evaluation.')
        seq = seq[:model.seq_len]
        coords = coords[:model.seq_len]

    metrics = ev.evaluate_model(model, seq, coords)
    print('Evaluation metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v:.6f}')


def main():
    p = argparse.ArgumentParser(description='Simplified AlphaFold-like protein folding visualization demo')
    p.add_argument('--demo', action='store_true', help='Run the synthetic demo')
    p.add_argument('--pdb', type=str, default='', help='Path to PDB file for real-structure demo')
    p.add_argument('--pdb-id', type=str, default='', help='RCSB PDB ID for download demo (e.g. 1a3n)')
    p.add_argument('--fasta', type=str, default='', help='FASTA file path for experiment on a given sequence')
    p.add_argument('--chain', type=str, default='A', help='PDB chain for real-structure demo')
    p.add_argument('--model', type=str, default='mlp', choices=['mlp','transformer'],
                   help='Model type: mlp or transformer')
    p.add_argument('--samples', type=int, default=250, help='Number of synthetic sequences')
    p.add_argument('--length', type=int, default=40, help='Sequence length for synthetic data')
    p.add_argument('--save-model', type=str, default='', help='Save trained model weights path')
    p.add_argument('--load-model', type=str, default='', help='Load pretrained model weights path')
    p.add_argument('--train', action='store_true', help='Run training workflow')
    p.add_argument('--evaluate', action='store_true', help='Run evaluation workflow')
    p.add_argument('--epochs', type=int, default=100, help='Training epochs for --train mode')
    p.add_argument('--lr', type=float, default=1e-3, help='Learning rate for --train mode')
    p.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory for training')
    p.add_argument('--csv', type=str, default='train_log.csv', help='CSV log path for training history')
    p.add_argument('--device', type=str, default='cpu', help='Torch device for training/eval')
    p.add_argument('--max-residues', type=int, default=120, help='Maximum residues to load from PDB/FASTA')
    p.add_argument('--train-from-pdb', action='store_true', help='Use local PDB dataset for training instead of synthetic')
    p.add_argument('--pdb-dir', type=str, default='data/pdbs', help='Directory of PDB files for train-from-pdb')
    p.add_argument('--min-residues', type=int, default=10, help='Minimum residues in PDB chain for training example')
    args = p.parse_args()
    if args.train:
        run_training(args)
    elif args.evaluate:
        run_evaluation(args)
    elif args.pdb:
        demo_run_pdb(args.pdb, chain=args.chain, num_samples=args.samples, L=args.length, model_type=args.model,
                     load_model_path=args.load_model, save_model_path=args.save_model)
    elif args.pdb_id:
        demo_run_pdbid(args.pdb_id, chain=args.chain, num_samples=args.samples, L=args.length, model_type=args.model)
    elif args.fasta:
        demo_run_fasta(args.fasta, num_samples=args.samples, L=args.length, model_type=args.model,
                       load_model_path=args.load_model, save_model_path=args.save_model)
    elif args.demo:
        demo_run(num_samples=args.samples, L=args.length, model_type=args.model,
                 load_model_path=args.load_model, save_model_path=args.save_model)
    else:
        print('Run with --demo or --train or --evaluate or --pdb <file> to execute a workflow. Example: python src/main.py --demo')


if __name__ == '__main__':
    main()
