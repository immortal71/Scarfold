import argparse
import os
import sys
import numpy as np

# Ensure imports work when script run directly from repository root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import utils
from src import model as md


def train_and_validate(model, train_X, train_Y, val_X, val_Y, epochs=100, lr=1e-3,
                       device='cpu', checkpoint_dir='checkpoints', verbose=True):
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val = float('inf')
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        model = md.train_simple(model, train_X, train_Y, epochs=1, lr=lr, device=device)

        model.eval()
        with np.errstate(all='ignore'):
            pred_train = np.stack([md.predict(model, x, device=device) for x in train_X])
            pred_val = np.stack([md.predict(model, x, device=device) for x in val_X])

        train_loss = np.mean((pred_train - train_Y) ** 2)
        val_loss = np.mean((pred_val - val_Y) ** 2)

        history.append({'epoch': ep, 'train_loss': float(train_loss), 'val_loss': float(val_loss)})

        if verbose:
            print(f"Epoch {ep:03d}: train MSE={train_loss:.6f}, val MSE={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            out_path = os.path.join(checkpoint_dir, f'best_model_epoch_{ep}.pt')
            md.save_model(model, out_path)
            if verbose:
                print(f"  => Saved best model to {out_path} (val_loss improved)")

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the protein folding predictor with checkpoints')
    parser.add_argument('--model', choices=['mlp', 'transformer'], default='mlp')
    parser.add_argument('--samples', type=int, default=400, help='Total synthetic sequences')
    parser.add_argument('--length', type=int, default=40, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-path', type=str, default='model_final.pt')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--csv', type=str, default='training_log.csv', help='CSV history output path')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train-from-pdb', action='store_true', help='Load training data from local PDB files instead of synthetic')
    parser.add_argument('--pdb-dir', type=str, default='data/pdbs', help='Directory containing PDB files for train-from-pdb')
    parser.add_argument('--chain', type=str, default='A', help='Chain ID for PDB train/eval')
    parser.add_argument('--max-residues', type=int, default=120, help='Max residues for PDB sequence')
    parser.add_argument('--min-residues', type=int, default=10, help='Minimum residues for valid PDB entry')
    args = parser.parse_args()

    if args.train_from_pdb:
        if not os.path.exists(args.pdb_dir):
            raise FileNotFoundError(f'PDB directory not found: {args.pdb_dir}')
        print(f'Loading PDB data from {args.pdb_dir} ...')
        seqs, dists = utils.sample_pdb_dataset(args.pdb_dir, chain=args.chain,
                                             max_residues=args.max_residues,
                                             min_residues=args.min_residues)
        onehots = np.stack([utils.rich_encoding(s.ljust(args.max_residues, 'A')[:args.max_residues]) for s in seqs])
        n_train = int(len(seqs) * 0.8)
        train_X, train_Y = onehots[:n_train], dists[:n_train]
        val_X, val_Y = onehots[n_train:], dists[n_train:]
        seq_len = min(args.max_residues, onehots.shape[1])
        print(f'Using {len(seqs)} PDB sequences; train/val split {n_train}/{len(seqs)-n_train}; seq_len={seq_len}')
        print('Initializing model...')
        model = md.get_model(args.model, seq_len)
    else:
        print('Generating synthetic dataset...')
        seqs, dists = utils.make_synthetic_dataset(num=args.samples, L=args.length, seed=42)
        onehots = np.stack([utils.rich_encoding(s) for s in seqs])
        n_train = int(args.samples * 0.8)
        train_X, train_Y = onehots[:n_train], dists[:n_train]
        val_X, val_Y = onehots[n_train:], dists[n_train:]
        print('Initializing model...')
        model = md.get_model(args.model, args.length)

    model, history = train_and_validate(model, train_X, train_Y, val_X, val_Y,
                                        epochs=args.epochs, lr=args.lr,
                                        device=args.device, checkpoint_dir=args.checkpoint_dir)

    print('Saving final model to', args.save_path)
    md.save_model(model, args.save_path)

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
            writer.writeheader()
            writer.writerows(history)
        print('Training history written to', args.csv)

    print('Done.')
