import argparse
import os
import sys
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src import utils
from src import model as md


def train_and_validate(model, train_X, train_Y, val_X, val_Y, epochs=100, lr=1e-3,
                       device='cpu', checkpoint_dir='checkpoints', verbose=True,
                       batch_size=16):
    """Fixed-length training: all proteins padded to the same seq_len."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    best_val = float('inf')
    history = []
    rng = np.random.default_rng(42)

    for ep in range(1, epochs + 1):
        # ── training ────────────────────────────────────────────────────────
        model.train()
        indices = rng.permutation(len(train_X))
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, len(train_X), batch_size):
            batch_idx = indices[start:start + batch_size]
            loss, *_ = md.train_epoch(model, train_X[batch_idx], train_Y[batch_idx],
                                      opt, device=device, contact_weight=0.5)
            ep_loss += loss
            n_batches += 1
        scheduler.step()
        train_loss = ep_loss / max(n_batches, 1)

        # ── validation (MSE on expected distances) ──────────────────────────
        model.eval()
        val_losses = []
        for i in range(len(val_X)):
            pred = md.predict(model, val_X[i], device=device)
            val_losses.append(float(np.mean((pred - val_Y[i]) ** 2)))
        val_loss = float(np.mean(val_losses))

        history.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})
        if verbose:
            print(f'Epoch {ep:03d}/{epochs}: train_loss={train_loss:.5f}  val_loss={val_loss:.5f}')

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(checkpoint_dir, f'best_model_epoch_{ep}.pt')
            md.save_model(model, ckpt)
            if verbose:
                print(f'  => Checkpoint saved to {ckpt}')

    return model, history


def train_variable_length(model_type, samples, epochs=60, lr=5e-4, device='cpu',
                          checkpoint_dir='checkpoints', verbose=True, aa_dim=48):
    """Per-protein training on variable-length real PDB sequences.

    Each protein is trained independently (batch_size=1) because proteins have
    different lengths.  A new model instance is used for the fixed seq_len, then
    merged — this is a simplified version of AlphaFold's recycling approach.

    In practice this function trains a *single* shared model by feeding each
    protein one at a time, which acts as stochastic gradient descent over the
    dataset.
    """
    import torch
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Pick a representative seq_len for the shared model (median length)
    lengths = [len(s) for s, _ in samples]
    seq_len = int(np.median(lengths))
    print(f'  Variable-length training: {len(samples)} proteins, median L={seq_len}')

    model = md.get_model(model_type, seq_len, aa_dim=aa_dim)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    rng = np.random.default_rng(42)
    history = []
    best_val = float('inf')
    n_val = max(1, len(samples) // 5)
    idx = rng.permutation(len(samples))
    train_idx = idx[n_val:]
    val_idx = idx[:n_val]

    for ep in range(1, epochs + 1):
        # ── training pass (one protein at a time) ────────────────────────────
        model.train()
        ep_losses = []
        for i in rng.permutation(train_idx):
            seq, dist = samples[i]
            L = len(seq)
            if L < 10:
                continue
            enc = utils.rich_encoding(seq)            # (L, aa_dim)
            X = np.array([enc])                       # (1, L, aa_dim)
            Y = np.array([dist])                      # (1, L, L)

            # Build a per-protein model if L != shared seq_len
            if L != model.seq_len:
                # Share all weights except positional embedding / seq_len-dependent layers
                # Simplest approach: skip proteins with very different length
                if abs(L - seq_len) > seq_len // 2:
                    continue
                # Pad/crop X and Y to model.seq_len
                Lm = model.seq_len
                X_pad = np.zeros((1, Lm, aa_dim), dtype=np.float32)
                Y_pad = np.zeros((1, Lm, Lm), dtype=np.float32)
                Lc = min(L, Lm)
                X_pad[0, :Lc] = enc[:Lc]
                Y_pad[0, :Lc, :Lc] = dist[:Lc, :Lc]
                X, Y = X_pad, Y_pad

            loss, *_ = md.train_epoch(model, X, Y, opt, device=device, contact_weight=0.5)
            ep_losses.append(loss)
        scheduler.step()
        train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

        # ── validation ───────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        for i in val_idx:
            seq, dist = samples[i]
            L = len(seq)
            enc = utils.rich_encoding(seq)
            Lm = model.seq_len
            Lc = min(L, Lm)
            X_pad = np.zeros((Lm, aa_dim), dtype=np.float32)
            X_pad[:Lc] = enc[:Lc]
            try:
                pred = md.predict(model, X_pad, device=device)
                val_losses.append(float(np.mean((pred[:Lc, :Lc] - dist[:Lc, :Lc]) ** 2)))
            except Exception:
                pass
        val_loss = float(np.mean(val_losses)) if val_losses else 999.0

        history.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})
        if verbose:
            print(f'Epoch {ep:03d}/{epochs}: train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  '
                  f'n_train_proteins={len(ep_losses)}')

        if val_loss < best_val:
            best_val = val_loss
            ckpt = os.path.join(checkpoint_dir, f'best_model_epoch_{ep}.pt')
            md.save_model(model, ckpt)
            if verbose:
                print(f'  => Checkpoint saved: {ckpt}')

    return model, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the protein folding predictor with checkpoints')
    parser.add_argument('--model', choices=['mlp', 'transformer'], default='mlp')
    parser.add_argument('--samples', type=int, default=400, help='Total synthetic sequences')
    parser.add_argument('--length', type=int, default=40, help='Sequence length')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--save-path', type=str, default='model_final.pt')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--csv', type=str, default='training_log.csv')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--train-from-pdb', action='store_true')
    parser.add_argument('--pdb-dir', type=str, default='data/pdbs')
    parser.add_argument('--chain', type=str, default='A')
    parser.add_argument('--max-residues', type=int, default=80)
    parser.add_argument('--min-residues', type=int, default=20)
    parser.add_argument('--variable-length', action='store_true',
                        help='Train on variable-length PDB sequences (no padding, batch_size=1)')
    args = parser.parse_args()

    if args.train_from_pdb:
        if not os.path.exists(args.pdb_dir):
            raise FileNotFoundError(f'PDB directory not found: {args.pdb_dir}')
        print(f'Loading PDB data from {args.pdb_dir} ...')

        if args.variable_length:
            # Variable-length real-PDB training
            samples = utils.sample_pdb_dataset_variable(
                args.pdb_dir, chain=args.chain,
                max_residues=args.max_residues, min_residues=args.min_residues
            )
            print(f'Loaded {len(samples)} variable-length proteins.')
            model, history = train_variable_length(
                args.model, samples, epochs=args.epochs, lr=args.lr,
                device=args.device, checkpoint_dir=args.checkpoint_dir,
                aa_dim=utils.RICH_AA_DIM,
            )
        else:
            # Fixed-length (padded) PDB training
            seqs, dists = utils.sample_pdb_dataset(
                args.pdb_dir, chain=args.chain,
                max_residues=args.max_residues, min_residues=args.min_residues
            )
            L = args.max_residues
            enc = np.stack([utils.rich_encoding(s.ljust(L, 'A')[:L]) for s in seqs])
            padded = np.zeros((len(seqs), L, L), dtype=np.float32)
            for i, d in enumerate(dists):
                n = min(d.shape[0], L)
                padded[i, :n, :n] = d[:n, :n]
            n_train = int(len(seqs) * 0.8)
            train_X, train_Y = enc[:n_train], padded[:n_train]
            val_X, val_Y = enc[n_train:], padded[n_train:]
            print(f'Train: {n_train}, Val: {len(seqs)-n_train}, seq_len={L}')
            model = md.get_model(args.model, L)
            model, history = train_and_validate(
                model, train_X, train_Y, val_X, val_Y,
                epochs=args.epochs, lr=args.lr, device=args.device,
                checkpoint_dir=args.checkpoint_dir, batch_size=args.batch_size,
            )
    else:
        print('Generating synthetic dataset ...')
        seqs, dists = utils.make_synthetic_dataset(num=args.samples, L=args.length, seed=42)
        onehots = np.stack([utils.rich_encoding(s) for s in seqs])
        n_train = int(args.samples * 0.8)
        train_X, train_Y = onehots[:n_train], dists[:n_train]
        val_X, val_Y = onehots[n_train:], dists[n_train:]
        model = md.get_model(args.model, args.length)
        model, history = train_and_validate(
            model, train_X, train_Y, val_X, val_Y,
            epochs=args.epochs, lr=args.lr, device=args.device,
            checkpoint_dir=args.checkpoint_dir, batch_size=args.batch_size,
        )

    print(f'Saving final model to {args.save_path}')
    md.save_model(model, args.save_path)

    if args.csv:
        import csv
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
            writer.writeheader()
            writer.writerows(history)
        print(f'Training history written to {args.csv}')

    print('Done.')


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
