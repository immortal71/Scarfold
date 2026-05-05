#!/usr/bin/env python3
"""train_v10.py — v10: three targeted fixes over v9.

v9 post-mortem (TM=0.147, overfitting at epoch 31/80):
  1. TRAIN/INFERENCE DISTRIBUTION SHIFT — v9 trains at CROP_LEN=80 but infers at
     up to 100aa. pos_embed positions 81-100 are never seen during training → the
     model is blind to those positions at inference. Fix: CROP_LEN=100 = SEQ_LEN.
  2. WEAK REGULARISATION — weight_decay=1e-4 is not enough for 484 proteins of
     training data. Overfitting starts at epoch 31 despite cosine LR decay.
     Fix: weight_decay=0.05 + embedding dropout (Bernoulli input noise, p=0.1).
  3. PLAIN BCE ON CONTACTS — standard BCE treats all negatives equally. 99%+ of
     residue pairs are non-contacts, so the loss is dominated by easy negatives
     the model already predicts correctly. Fix: focal loss (gamma=2.0) downweights
     easy negatives, concentrating gradients on hard long-range pairs.

Additional: label smoothing (epsilon=0.1) on distogram CE prevents overconfident
one-hot targets from dominating the distogram loss.

Strategy:
  - Resume from model_v9.pt.best_so_far.pt (TM=0.147, epoch 31 checkpoint)
  - CROP_LEN=100 (= SEQ_LEN, eliminates train/inference distribution mismatch)
  - weight_decay=0.05 (50× stronger than v9)
  - Embedding dropout p=0.1 (stochastic input noise = implicit data augmentation)
  - Focal loss for contact BCE (gamma=2.0)
  - Label smoothing for distogram CE (epsilon=0.1)
  - LR=5e-5 (half of v9 — fine-tuning from a solid checkpoint)
  - 60 epochs with cosine annealing to 1%

Usage:
  python src/train_v10.py --base-model "model_v9.pt.best_so_far.pt" --epochs 60 --out model_v10.pt
"""
import argparse, copy, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md
from src.esm_utils import ESM_RICH_DIM, esm2_rich_encoding_cached, build_disk_cache

# ── Key hyperparameters ───────────────────────────────────────────────────────
CROP_LEN       = 100   # = SEQ_LEN: eliminates train/inference distribution shift
SEQ_LEN        = 100   # pos_embed size
CROPS_PER      = 1     # crops per protein per epoch
EMBED_DROPOUT  = 0.10  # Bernoulli dropout on ESM-2 input features
FOCAL_GAMMA    = 2.0   # focal loss exponent for contact BCE
LABEL_SMOOTH   = 0.10  # label smoothing epsilon for distogram CE

# Auto-select best available device; overridden by --device CLI arg
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TEST_PIDS = {'1crn', '1vii', '1lyz', '1trz', '1aho', '2ptl', '1tig'}

# ── ESM-2 full-protein cache ──────────────────────────────────────────────────
_ESM_CACHE: dict = {}

def _cached_esm2_rich(seq: str) -> np.ndarray:
    if seq not in _ESM_CACHE:
        _ESM_CACHE[seq] = esm2_rich_encoding_cached(seq, cache_path='data/esm2_cache.npz')
    return _ESM_CACHE[seq]


# ── Data loading ──────────────────────────────────────────────────────────────
def _extract_pid(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    if name.startswith('pdb') and len(name) >= 7:
        return name[3:7]
    return name[:4]


def load_all_pdbs(pdb_dir: str, min_res: int = 20, max_res: int = 200):
    samples = []
    paths = (sorted(glob.glob(os.path.join(pdb_dir, '*.pdb'))) +
             sorted(glob.glob(os.path.join(pdb_dir, '*.ent'))))
    for path in paths:
        pid = _extract_pid(path)
        if pid in TEST_PIDS:
            continue
        try:
            seq    = utils.pdb_sequence(path, chain='A', max_residues=max_res)
            coords = utils.pdb_ca_coords(path, chain='A', max_residues=max_res)
            N = min(len(seq), len(coords))
            if N >= min_res:
                samples.append((seq[:N], coords[:N], pid))
        except Exception as e:
            print(f'  SKIP {pid}: {e}')
    return samples


# ── Model construction ────────────────────────────────────────────────────────
def build_v10_model(base_model_path: str) -> md.TransformerDistancePredictor:
    """Load v9 checkpoint into a plain v10 model."""
    model = md.TransformerDistancePredictor(
        seq_len=SEQ_LEN,
        aa_dim=ESM_RICH_DIM,
        hidden=256,
        pair_dim=64,
        nhead=4,
        num_layers=4,
        n_bins=md.NUM_BINS + 1,
        dropout=0.1,
        num_recycles=3,
        coord_head=False,
    )
    model.aa_dim = ESM_RICH_DIM

    if not os.path.exists(base_model_path):
        print(f'  [train_v10] WARNING: {base_model_path} not found — training from scratch.')
        return model

    raw = torch.load(base_model_path, map_location='cpu', weights_only=False)
    src_state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw

    new_state = model.state_dict()
    n_copied, n_skipped = 0, 0

    for key, src_tensor in src_state.items():
        if key not in new_state:
            n_skipped += 1
            continue

        if key == 'pos_embed' and src_tensor.shape[0] != SEQ_LEN:
            old_len = src_tensor.shape[0]
            pe_t = src_tensor.float().T.unsqueeze(0)
            new_pe = F.interpolate(pe_t, size=SEQ_LEN, mode='linear', align_corners=True)
            new_state['pos_embed'] = new_pe.squeeze(0).T
            print(f'  [train_v10] Interpolated pos_embed {old_len} → {SEQ_LEN}')
            n_copied += 1
            continue

        if new_state[key].shape != src_tensor.shape:
            print(f'  [train_v10] Shape mismatch — skipping {key}')
            n_skipped += 1
            continue

        new_state[key] = src_tensor
        n_copied += 1

    model.load_state_dict(new_state)
    print(f'  [train_v10] Copied {n_copied} tensors, skipped {n_skipped}.')
    return model


# ── Loss functions ────────────────────────────────────────────────────────────
def distogram_loss_smooth(logits: torch.Tensor, dist_true: torch.Tensor,
                          epsilon: float = LABEL_SMOOTH) -> torch.Tensor:
    """Distogram CE with label smoothing.

    Smoothed target: y_smooth = (1 - eps) * y_hard + eps / n_bins
    This prevents overconfident one-hot targets and acts as mild regularisation.
    """
    B, L, _, n_bins = logits.shape
    # get_bin_edges() returns NUM_BINS+1 edges; use inner edges for bucketize
    edges = torch.tensor(md.get_bin_edges()[1:], dtype=torch.float32)
    bin_idx = torch.bucketize(dist_true, edges)
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    # Hard targets → soft targets with label smoothing
    y_hard  = F.one_hot(bin_idx, num_classes=n_bins).float()   # (B, L, L, n_bins)
    y_soft  = (1.0 - epsilon) * y_hard + epsilon / n_bins

    log_p = F.log_softmax(logits, dim=-1)
    loss  = -(y_soft * log_p).sum(dim=-1)   # (B, L, L)

    # Mask diagonal
    mask = ~torch.eye(L, dtype=torch.bool, device=logits.device).unsqueeze(0)
    return loss[mask].mean()


def focal_contact_loss(logits: torch.Tensor, dist_true: torch.Tensor,
                       contact_thresh: float = 8.0,
                       lr_weight: float = 16.0, lr_sep: int = 12,
                       gamma: float = FOCAL_GAMMA) -> torch.Tensor:
    """Contact BCE with focal weighting and long-range upweighting.

    Focal loss: FL = -(1-p)^gamma * log(p) for positives
                    -p^gamma * log(1-p)     for negatives
    With gamma=2, easy confident negatives get ~0.01× the loss of hard examples.
    This focuses gradients on hard long-range contacts instead of easy short-range
    non-contacts which dominate plain BCE.
    """
    B, L, _, n_bins = logits.shape
    dist_pred = md.bin_to_dist(logits)   # (B, L, L)

    contact_pred = torch.sigmoid(10.0 * (contact_thresh - dist_pred) / contact_thresh)
    contact_true = (dist_true < contact_thresh).float()

    # Focal weights
    p_t = torch.where(contact_true == 1.0, contact_pred, 1.0 - contact_pred)
    focal_w = (1.0 - p_t).pow(gamma)

    bce = F.binary_cross_entropy(contact_pred.clamp(1e-6, 1-1e-6),
                                 contact_true, reduction='none')

    # Long-range upweight mask
    idx = torch.arange(L, device=logits.device)
    sep = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    lr_mask = (sep >= lr_sep).float()
    weight  = 1.0 + (lr_weight - 1.0) * lr_mask

    # Off-diagonal mask
    off_diag = 1.0 - torch.eye(L, device=logits.device)

    loss = (focal_w * bce * weight * off_diag).sum() / (off_diag.sum() * B)
    return loss


# ── Training loop ─────────────────────────────────────────────────────────────
def one_epoch(
    model: md.TransformerDistancePredictor,
    train_samples,
    opt: torch.optim.Optimizer,
    rng: np.random.Generator,
    lr_weight: float,
    lr_sep: int,
    contact_weight: float,
    device: str = 'cpu',
    accum_steps: int = 4,
    embed_dropout: float = EMBED_DROPOUT,
    crops_per: int = CROPS_PER,
) -> float:
    """Train one epoch.

    accum_steps: gradient accumulation steps — simulates batch_size=accum_steps
    on GPU where each forward pass processes one protein (batch=1). Accumulating
    4 gradients before stepping is equivalent to batch_size=4, stabilising
    training without requiring variable-length padding.
    """
    model.train()
    ep_loss, n_steps = 0.0, 0
    order = rng.permutation(len(train_samples))
    opt.zero_grad()

    for step_i, i in enumerate(order):
        seq_r, coords_r, _ = train_samples[i]
        n_crops = crops_per if len(seq_r) > CROP_LEN else 1

        for _ in range(n_crops):
            L = len(seq_r)
            if L > CROP_LEN:
                start = int(rng.integers(0, L - CROP_LEN + 1))
                seq_c = seq_r[start : start + CROP_LEN]
                crd_c = coords_r[start : start + CROP_LEN]
            else:
                seq_c, crd_c = seq_r, coords_r
                start = 0

            full_enc = _cached_esm2_rich(seq_r)
            enc      = full_enc[start : start + len(seq_c)] if len(seq_r) > CROP_LEN else full_enc
            dist_np  = utils.coords_to_distances(crd_c).astype(np.float32)

            X = torch.tensor(enc[None], dtype=torch.float32).to(device)   # (1, L, 368)

            # Embedding dropout: randomly zero out entire feature vectors
            if embed_dropout > 0.0:
                mask = torch.bernoulli(
                    torch.ones(X.shape[0], X.shape[1], 1, device=device) * (1.0 - embed_dropout)
                )
                X = X * mask / (1.0 - embed_dropout)

            Y = torch.tensor(dist_np[None], dtype=torch.float32).to(device)

            logits, ss_logits, _ = model.forward_full(X)

            # 1. Distogram CE with label smoothing
            loss = distogram_loss_smooth(logits, Y, epsilon=LABEL_SMOOTH)

            # 2. Focal contact loss with long-range upweighting
            loss += contact_weight * focal_contact_loss(
                logits, Y,
                lr_weight=lr_weight, lr_sep=lr_sep,
                gamma=FOCAL_GAMMA,
            )

            # 3. Secondary-structure auxiliary loss
            ss_lbl = torch.tensor(
                md.ss_labels_from_dists(dist_np)[None], dtype=torch.long, device=device)
            ss_loss = F.cross_entropy(ss_logits.reshape(-1, 3), ss_lbl.reshape(-1))
            if not torch.isnan(ss_loss):
                loss += 0.2 * ss_loss

            # Gradient accumulation: scale loss, accumulate, step every accum_steps
            (loss / accum_steps).backward()
            ep_loss += float(loss)
            n_steps  += 1

            if (step_i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

    # Final partial accumulation batch
    if n_steps % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

            ep_loss += float(loss)
            n_steps  += 1

    return ep_loss / max(n_steps, 1)


def quick_val_mse(model: md.TransformerDistancePredictor, val_samples,
                  device: str = 'cpu') -> float:
    model.eval()
    mses = []
    with torch.no_grad():
        for seq_r, coords_r, _ in val_samples:
            L = min(len(seq_r), CROP_LEN)
            enc  = _cached_esm2_rich(seq_r)[:L]
            dist = utils.coords_to_distances(coords_r[:L]).astype(np.float32)
            X    = torch.tensor(enc[None], dtype=torch.float32).to(device)
            logits, _, _ = model.forward_full(X)
            pred = md.bin_to_dist(logits)[0].cpu().numpy()
            mses.append(float(np.mean((pred[:L, :L] - dist[:L, :L]) ** 2)))
    return float(np.mean(mses)) if mses else 999.0


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='v10: focal contact loss + label smooth + CROP=100 + embed dropout')
    parser.add_argument('--base-model',     default='model_v9.pt.best_so_far.pt')
    parser.add_argument('--pdb-dir',        default='data/pdbs')
    parser.add_argument('--epochs',         type=int,   default=60)
    parser.add_argument('--lr',             type=float, default=5e-5)
    parser.add_argument('--lr-weight',      type=float, default=16.0)
    parser.add_argument('--lr-sep',         type=int,   default=12)
    parser.add_argument('--contact-weight', type=float, default=0.5)
    parser.add_argument('--crops',          type=int,   default=CROPS_PER)
    parser.add_argument('--out',            default='model_v10.pt')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--device',         default=DEFAULT_DEVICE,
                        help='Training device: cpu / cuda / cuda:0 / cuda:1 (default: auto)')
    parser.add_argument('--accum-steps',    type=int,   default=4,
                        help='Gradient accumulation steps (default 4 = effective batch_size=4)')
    args = parser.parse_args()

    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('WARNING: CUDA not available, falling back to CPU.')
        device = 'cpu'

    print('=' * 72)
    print('  v10  CROP_LEN=100 + focal contact loss + label smooth + embed dropout')
    print(f'  base: {args.base_model}')
    print(f'  device: {device}  |  accum_steps: {args.accum_steps} (eff. batch={args.accum_steps})')
    print(f'  {args.epochs} epochs  |  lr={args.lr}  |  weight_decay=0.05')
    print(f'  focal_gamma={FOCAL_GAMMA}  label_smooth={LABEL_SMOOTH}  embed_dropout={EMBED_DROPOUT}')
    print(f'  lr_weight={args.lr_weight}×  lr_sep={args.lr_sep}  contact_weight={args.contact_weight}')
    print('=' * 72)

    if device.startswith('cuda'):
        print(f'  GPU: {torch.cuda.get_device_name(device)}')
        print(f'  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB\n')

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    cache_path = 'data/esm2_cache.npz'
    if not os.path.exists(cache_path):
        print(f'\nBuilding ESM-2 disk cache → {cache_path} ...')
        build_disk_cache(args.pdb_dir, cache_path)
        print('  Done.\n')
    else:
        print(f'\nESM-2 disk cache: {cache_path}\n')

    print(f'Loading PDB files from {args.pdb_dir} ...')
    all_samples = load_all_pdbs(args.pdb_dir, min_res=20, max_res=200)
    if not all_samples:
        print(f'ERROR: no PDB files found in {args.pdb_dir}.')
        sys.exit(1)
    print(f'  {len(all_samples)} proteins (excluded {len(TEST_PIDS)} test proteins)')

    n_val = max(1, len(all_samples) // 10)
    idx   = rng.permutation(len(all_samples))
    val_samples   = [all_samples[i] for i in idx[:n_val]]
    train_samples = [all_samples[i] for i in idx[n_val:]]
    print(f'  Split: {len(train_samples)} train / {len(val_samples)} val\n')

    print(f'Building v10 model from {args.base_model} ...')
    model = build_v10_model(args.base_model)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {n_params:,}\n')

    # Stronger weight decay (50× vs v9) to fight overfitting
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.01)

    best_val   = float('inf')
    best_state = None
    history    = []
    t_start    = time.time()

    for ep in range(1, args.epochs + 1):
        ep_t = time.time()
        train_loss = one_epoch(model, train_samples, opt, rng,
                               lr_weight=args.lr_weight,
                               lr_sep=args.lr_sep,
                               contact_weight=args.contact_weight,
                               device=device,
                               accum_steps=args.accum_steps,
                               embed_dropout=EMBED_DROPOUT,
                               crops_per=args.crops)
        sched.step()
        val_mse = quick_val_mse(model, val_samples, device=device)
        elapsed = time.time() - ep_t

        history.append({'epoch': ep, 'train_loss': train_loss, 'val_mse': val_mse})

        star = ''
        if val_mse < best_val:
            best_val   = val_mse
            best_state = copy.deepcopy(model.state_dict())
            star = ' ☆'
            tmp = args.out + '.best_so_far.pt'
            torch.save({'state_dict': best_state, 'aa_dim': ESM_RICH_DIM,
                        'coord_head': False, 'seq_len': SEQ_LEN,
                        'epoch': ep, 'val_mse': best_val}, tmp)

        print(f'  Ep {ep:3d}/{args.epochs}  '
              f'train={train_loss:.4f}  val_MSE={val_mse:.2f}  ({elapsed:.0f}s){star}')

    # Save best model
    if best_state:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(), 'aa_dim': ESM_RICH_DIM,
                'coord_head': False, 'seq_len': SEQ_LEN}, args.out)
    print(f'\nSaved: {args.out}  (best val_MSE={best_val:.3f})')
    print(f'Total training time: {(time.time()-t_start)/60:.1f} min')

    hist_path = args.out.replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)
    print(f'History: {hist_path}')

    tmp_best = args.out + '.best_so_far.pt'
    if os.path.exists(tmp_best):
        os.remove(tmp_best)

    print('\nTRAIN_V10_DONE:0')


if __name__ == '__main__':
    main()
