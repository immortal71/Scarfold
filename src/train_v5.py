#!/usr/bin/env python3
"""train_v5.py — Fine-tune v4 TriMul model with long-range weighted contact loss.

Key improvement over v4:
    The BCE contact loss now upweights long-range pairs (|i-j| >= 12) by lr_weight
    (default 8.0), forcing the model to rank long-range contacts correctly — directly
    optimising the primary reported metric: long_range_precision_L5.

Strategy:
    Fine-tune from model_v4.pt (already converged) for ~30 epochs with a reduced
    learning rate (2e-4).  Fine-tuning preserves all hard-won local-structure
    knowledge while shifting attention to long-range contacts.

Usage:
    python src/train_v5.py
    python src/train_v5.py --epochs 50 --lr-weight 8.0 --out model_v5.pt
"""
import argparse, copy, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md

CROP_LEN  = 60
CROPS_PER = 4
AA_DIM    = utils.RICH_AA_DIM  # 48

# Test proteins — excluded from training
TEST_PIDS = {'1crn', '1vii', '1lyz', '1trz', '1aho', '2ptl', '1tig'}


def _extract_pid(filename):
    """Extract 4-letter PDB ID from filenames like pdb1abc.ent or 1abc.pdb."""
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    if name.startswith('pdb') and len(name) >= 7:
        return name[3:7]   # pdb1abc.ent → 1abc
    return name[:4]         # 1abc.pdb → 1abc


def load_all_pdbs(pdb_dir, min_res=20, max_res=200):
    """Load every PDB/ENT file in *pdb_dir*, return list of (seq, Cα-coords, pid)."""
    samples = []
    paths = (sorted(glob.glob(os.path.join(pdb_dir, '*.pdb'))) +
             sorted(glob.glob(os.path.join(pdb_dir, '*.ent'))))
    for path in paths:
        pid = _extract_pid(path)
        if pid in TEST_PIDS:
            continue  # never train on test proteins
        try:
            seq    = utils.pdb_sequence(path, chain='A', max_residues=max_res)
            coords = utils.pdb_ca_coords(path, chain='A', max_residues=max_res)
            N = min(len(seq), len(coords))
            if N >= min_res:
                samples.append((seq[:N], coords[:N], pid))
        except Exception as e:
            print(f'  SKIP {pid}: {e}')
    return samples


def one_epoch(model, train_samples, opt, rng, lr_weight, lr_sep, contact_weight):
    """One training epoch over all proteins with random cropping."""
    model.train()
    ep_loss = 0.0
    n_steps = 0
    order = rng.permutation(len(train_samples))

    for i in order:
        seq_r, coords_r, _ = train_samples[i]
        n_crops = CROPS_PER if len(seq_r) > CROP_LEN else 1

        for _ in range(n_crops):
            L = len(seq_r)
            if L > CROP_LEN:
                start  = int(rng.integers(0, L - CROP_LEN + 1))
                seq_c  = seq_r[start:start + CROP_LEN]
                crd_c  = coords_r[start:start + CROP_LEN]
            else:
                seq_c, crd_c = seq_r, coords_r

            enc     = utils.rich_encoding(seq_c)
            dist_np = utils.coords_to_distances(crd_c).astype(np.float32)
            X = torch.tensor(enc[None],     dtype=torch.float32)
            Y = torch.tensor(dist_np[None], dtype=torch.float32)

            opt.zero_grad()
            logits, ss_logits, _ = model.forward_full(X)

            # Distogram CE loss (backbone)
            loss = md.distogram_loss(logits, Y, backbone_weight=1.0)

            # LR-weighted contact BCE — core v5 improvement
            loss += contact_weight * md._contact_bce_loss(
                logits, Y, is_logits=True,
                lr_weight=lr_weight, lr_sep=lr_sep
            )

            # Secondary-structure auxiliary loss
            ss_lbl = torch.tensor(
                md.ss_labels_from_dists(dist_np)[None], dtype=torch.long)
            ss_loss = F.cross_entropy(ss_logits.reshape(-1, 3), ss_lbl.reshape(-1))
            if not torch.isnan(ss_loss):
                loss += 0.2 * ss_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss)
            n_steps  += 1

    return ep_loss / max(n_steps, 1)


def quick_val_mse(model, val_samples):
    """MSE on a small validation subset (random 10 proteins, cropped to CROP_LEN)."""
    model.eval()
    mses = []
    with torch.no_grad():
        for seq_r, coords_r, _ in val_samples:
            L = min(len(seq_r), CROP_LEN)
            enc  = utils.rich_encoding(seq_r[:L])
            dist = utils.coords_to_distances(coords_r[:L]).astype(np.float32)
            pred = md.predict(model, enc)
            mses.append(float(np.mean((pred[:L, :L] - dist[:L, :L]) ** 2)))
    return float(np.mean(mses)) if mses else 999.0


def main():
    parser = argparse.ArgumentParser(description='Fine-tune v4 with LR-weighted contact loss')
    parser.add_argument('--base-model',    default='model_v4.pt',
                        help='Starting checkpoint (v4 TriMul model)')
    parser.add_argument('--pdb-dir',       default='data/pdbs')
    parser.add_argument('--epochs',        type=int,   default=30)
    parser.add_argument('--lr',            type=float, default=2e-4,
                        help='Fine-tuning learning rate (smaller than training)')
    parser.add_argument('--lr-weight',     type=float, default=8.0,
                        help='Upweight for |i-j|>=lr-sep contacts in BCE loss')
    parser.add_argument('--lr-sep',        type=int,   default=12,
                        help='|i-j| threshold for long-range contacts')
    parser.add_argument('--contact-weight', type=float, default=0.5,
                        help='Overall weight on contact BCE vs distogram loss')
    parser.add_argument('--out',           default='model_v5.pt',
                        help='Output model path')
    parser.add_argument('--seed',          type=int,   default=42)
    args = parser.parse_args()

    print('=' * 65)
    print(f'  v5 fine-tuning: {args.epochs} epochs, lr={args.lr}, '
          f'lr_weight={args.lr_weight}, lr_sep={args.lr_sep}')
    print('=' * 65)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # ── Load training proteins ─────────────────────────────────────────────
    all_samples = load_all_pdbs(args.pdb_dir, min_res=20, max_res=200)
    if not all_samples:
        print(f'ERROR: no PDB files found in {args.pdb_dir}')
        sys.exit(1)
    print(f'Loaded {len(all_samples)} training proteins '
          f'(excluded {len(TEST_PIDS)} test proteins)')

    # 10% validation split (held out from gradient updates)
    n_val = max(1, len(all_samples) // 10)
    idx   = rng.permutation(len(all_samples))
    val_samples   = [all_samples[i] for i in idx[:n_val]]
    train_samples = [all_samples[i] for i in idx[n_val:]]
    print(f'Split: {len(train_samples)} train / {len(val_samples)} val\n')

    # ── Load v4 model ─────────────────────────────────────────────────────
    if not os.path.exists(args.base_model):
        print(f'ERROR: base model not found: {args.base_model}')
        sys.exit(1)
    model = md.load_model('transformer', CROP_LEN, args.base_model)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Loaded {args.base_model}  ({n_params:,} parameters)\n')

    # ── Optimiser: cosine decay from lr to lr*0.05 ─────────────────────────
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.05)

    best_val   = float('inf')
    best_state = None
    history    = []
    t_start    = time.time()

    # ── Training loop ─────────────────────────────────────────────────────
    for ep in range(1, args.epochs + 1):
        ep_t = time.time()
        train_loss = one_epoch(model, train_samples, opt, rng,
                               lr_weight=args.lr_weight,
                               lr_sep=args.lr_sep,
                               contact_weight=args.contact_weight)
        sched.step()
        val_mse = quick_val_mse(model, val_samples)
        elapsed = time.time() - ep_t

        history.append({'epoch': ep, 'train_loss': train_loss, 'val_mse': val_mse})

        star = ''
        if val_mse < best_val:
            best_val   = val_mse
            best_state = copy.deepcopy(model.state_dict())
            star = ' ★'
            # Save incrementally so a crash doesn't lose the best weights
            _tmp = args.out + '.best_so_far.pt'
            torch.save({'state_dict': best_state, 'aa_dim': AA_DIM,
                        'epoch': ep, 'val_mse': best_val}, _tmp)

        print(f'Ep {ep:3d}/{args.epochs}  train={train_loss:.4f}  '
              f'val_MSE={val_mse:.2f}  ({elapsed:.0f}s){star}')

    # ── Save best model ────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    md.save_model(model, args.out)
    print(f'\nSaved: {args.out}  (best val_MSE={best_val:.3f})')

    total_min = (time.time() - t_start) / 60
    print(f'Total training time: {total_min:.1f} min')

    # Save history
    hist_path = args.out.replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)
    print(f'History saved: {hist_path}')


if __name__ == '__main__':
    main()
