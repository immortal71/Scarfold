#!/usr/bin/env python3
"""train_v8.py — v8: coordinate head + larger crop (80aa) for Level 10.

Key improvements over v7:
  1. CoordinateHead: directly predicts 3D Cα coordinates from the pair + sequence
     representation. Trained with an all-pairs Huber distance loss, forcing the
     network to learn 3D-embeddable geometry end-to-end — no MDS at inference.
  2. CROP_LEN=80: covers all 7 test proteins fully (1AHO=64, 2PTL=62 now fit; 1TIG=88
     gets 80aa crop vs old 60aa). pos_embed interpolated from v7 (60→80).
  3. pos_embed extended to COORD_LEN=100 so eval can handle proteins up to 100aa
     at inference without re-training.

Training strategy:
  Phase A (--warmup-epochs, default 5):
    Only the coord_head is trained; all other layers frozen.
    Teaches the head to produce reasonable coordinates from v7's pair representation.
  Phase B (remaining epochs):
    Full fine-tune with combined loss:
      L_total = L_distogram_CE + 0.5*L_contact_BCE + 0.5*L_coord_geom + 0.1*L_backbone

Usage:
  python src/train_v8.py --base-model model_v7_500data.pt --epochs 60 --out model_v8.pt
  python src/train_v8.py --base-model model_v7_500data.pt --epochs 60 --out model_v8.pt --crops 1
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
CROP_LEN  = 80    # training crop length (covers 1AHO=64, 2PTL=62 fully)
COORD_LEN = 100   # pos_embed size → supports eval up to 100aa at inference
CROPS_PER = 1     # crops per protein per epoch (already have 484 proteins)

# Test proteins — never seen during training
TEST_PIDS = {'1crn', '1vii', '1lyz', '1trz', '1aho', '2ptl', '1tig'}

# ── ESM-2 full-protein cache ──────────────────────────────────────────────────
_ESM_CACHE: dict = {}

def _cached_esm2_rich(seq: str) -> np.ndarray:
    """Return (L, 368) ESM-2+rich encoding, checking disk cache first."""
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


# ── Coordinate losses ─────────────────────────────────────────────────────────
def coord_geom_loss(pred_coords: torch.Tensor, true_coords_np: np.ndarray) -> torch.Tensor:
    """All-pairs Huber distance loss between predicted and true Cα coordinates.

    This loss is rotation/translation invariant (uses distances, not raw coords),
    so it trains the coord head without needing Kabsch alignment during backprop.

    pred_coords: (1, L, 3) tensor  (output of CoordinateHead, already centered)
    true_coords_np: (L, 3) numpy array
    """
    tc = torch.tensor(true_coords_np, dtype=pred_coords.dtype, device=pred_coords.device)
    p  = pred_coords[0]   # (L, 3)

    # Predicted pairwise distances
    diff_p = p.unsqueeze(0) - p.unsqueeze(1)         # (L, L, 3)
    d_pred = (diff_p ** 2).sum(-1).add(1e-4).sqrt()  # (L, L)

    # True pairwise distances
    diff_t = tc.unsqueeze(0) - tc.unsqueeze(1)
    d_true = (diff_t ** 2).sum(-1).add(1e-4).sqrt()  # (L, L)

    return F.huber_loss(d_pred, d_true, delta=4.0)


def backbone_bond_loss(pred_coords: torch.Tensor) -> torch.Tensor:
    """Encourage consecutive Cα bonds to be ~3.8 Å (physical constraint)."""
    p = pred_coords[0]              # (L, 3)
    diffs = p[1:] - p[:-1]         # (L-1, 3)
    bond_lengths = (diffs ** 2).sum(-1).add(1e-4).sqrt()   # (L-1,)
    target = torch.full_like(bond_lengths, 3.8)
    return F.huber_loss(bond_lengths, target, delta=0.5)


# ── Model construction ────────────────────────────────────────────────────────
def build_v8_model(base_model_path: str) -> md.TransformerDistancePredictor:
    """Load v7 weights into a new v8 model with coord_head=True and COORD_LEN pos_embed.

    Differences from v7:
      - coord_head=True: new CoordinateHead added (freshly initialized)
      - seq_len=COORD_LEN (100): pos_embed interpolated from v7's 60→100
      - All other weights copied exactly from v7
    """
    model = md.TransformerDistancePredictor(
        seq_len=COORD_LEN,     # pos_embed supports up to COORD_LEN=100 residues
        aa_dim=ESM_RICH_DIM,
        hidden=256,
        pair_dim=64,
        nhead=4,
        num_layers=4,
        n_bins=md.NUM_BINS + 1,
        dropout=0.1,
        num_recycles=3,
        coord_head=True,
    )
    model.aa_dim = ESM_RICH_DIM

    if not os.path.exists(base_model_path):
        print(f'  [train_v8] WARNING: {base_model_path} not found — training from scratch.')
        return model

    raw = torch.load(base_model_path, map_location='cpu', weights_only=False)
    state_v7 = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw

    new_state = model.state_dict()
    n_copied, n_skipped = 0, 0

    for key, v7_tensor in state_v7.items():
        if key not in new_state:
            n_skipped += 1
            continue

        # Special case: interpolate pos_embed from old seq_len → COORD_LEN
        if key == 'pos_embed' and v7_tensor.shape[0] != COORD_LEN:
            old_len = v7_tensor.shape[0]
            pe_t = v7_tensor.float().T.unsqueeze(0)              # (1, hidden, old_len)
            new_pe_t = F.interpolate(pe_t, size=COORD_LEN, mode='linear', align_corners=True)
            new_state['pos_embed'] = new_pe_t.squeeze(0).T       # (COORD_LEN, hidden)
            print(f'  [train_v8] Interpolated pos_embed {old_len} -> {COORD_LEN}')
            n_copied += 1
            continue

        if new_state[key].shape != v7_tensor.shape:
            print(f'  [train_v8] Shape mismatch — skipping {key}')
            n_skipped += 1
            continue

        new_state[key] = v7_tensor
        n_copied += 1

    model.load_state_dict(new_state)
    print(f'  [train_v8] Copied {n_copied} tensors from {base_model_path}, '
          f'skipped {n_skipped} (shape mismatch / new keys).')
    return model


# ── Training loop ─────────────────────────────────────────────────────────────
def one_epoch(
    model: md.TransformerDistancePredictor,
    train_samples,
    opt: torch.optim.Optimizer,
    rng: np.random.Generator,
    lr_weight: float,
    lr_sep: int,
    contact_weight: float,
    coord_weight: float,
    crops_per: int = CROPS_PER,
) -> float:
    model.train()
    ep_loss, n_steps = 0.0, 0
    order = rng.permutation(len(train_samples))

    for i in order:
        seq_r, coords_r, _ = train_samples[i]
        n_crops = crops_per if len(seq_r) > CROP_LEN else 1

        for _ in range(n_crops):
            L = len(seq_r)
            if L > CROP_LEN:
                start  = int(rng.integers(0, L - CROP_LEN + 1))
                seq_c  = seq_r[start : start + CROP_LEN]
                crd_c  = coords_r[start : start + CROP_LEN]
            else:
                seq_c, crd_c = seq_r, coords_r
                start = 0

            # ESM-2 + rich encoding (368-dim) — sliced from cached full-protein embedding
            full_enc = _cached_esm2_rich(seq_r)
            enc      = full_enc[start : start + len(seq_c)] if len(seq_r) > CROP_LEN else full_enc
            dist_np  = utils.coords_to_distances(crd_c).astype(np.float32)

            X = torch.tensor(enc[None],     dtype=torch.float32)   # (1, L, 368)
            Y = torch.tensor(dist_np[None], dtype=torch.float32)   # (1, L, L)

            opt.zero_grad()

            if model.coord_head is not None:
                logits, ss_logits, _, coords = model.forward_with_coords(X)
            else:
                logits, ss_logits, _ = model.forward_full(X)
                coords = None

            # 1. Distogram cross-entropy (primary)
            loss = md.distogram_loss(logits, Y, backbone_weight=1.0)

            # 2. Long-range weighted contact BCE (inherited from v5/v7)
            loss += contact_weight * md._contact_bce_loss(
                logits, Y, is_logits=True,
                lr_weight=lr_weight, lr_sep=lr_sep,
            )

            # 3. Secondary-structure auxiliary loss
            ss_lbl = torch.tensor(
                md.ss_labels_from_dists(dist_np)[None], dtype=torch.long)
            ss_loss = F.cross_entropy(ss_logits.reshape(-1, 3), ss_lbl.reshape(-1))
            if not torch.isnan(ss_loss):
                loss += 0.2 * ss_loss

            # 4. Coordinate geometry loss (v8 addition)
            if coords is not None:
                cg_loss = coord_geom_loss(coords, crd_c)
                bb_loss = backbone_bond_loss(coords)
                if not (torch.isnan(cg_loss) or torch.isnan(bb_loss)):
                    loss += coord_weight * cg_loss + 0.1 * bb_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_loss += float(loss)
            n_steps  += 1

    return ep_loss / max(n_steps, 1)


def quick_val_mse(model: md.TransformerDistancePredictor, val_samples) -> float:
    model.eval()
    mses = []
    with torch.no_grad():
        for seq_r, coords_r, _ in val_samples:
            L = min(len(seq_r), CROP_LEN)
            enc  = _cached_esm2_rich(seq_r)[:L]
            dist = utils.coords_to_distances(coords_r[:L]).astype(np.float32)
            X    = torch.tensor(enc[None], dtype=torch.float32)
            logits, _, _ = model.forward_full(X)
            pred = md.bin_to_dist(logits)[0].cpu().numpy()
            mses.append(float(np.mean((pred[:L, :L] - dist[:L, :L]) ** 2)))
    return float(np.mean(mses)) if mses else 999.0


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='v8: coordinate head + CROP_LEN=80 for Level 10')
    parser.add_argument('--base-model',      default='model_v7_500data.pt',
                        help='v7 checkpoint to start from')
    parser.add_argument('--pdb-dir',         default='data/pdbs')
    parser.add_argument('--epochs',          type=int,   default=60,
                        help='Total training epochs (Phase A + Phase B)')
    parser.add_argument('--warmup-epochs',   type=int,   default=5,
                        help='Phase-A epochs: only coord_head trained')
    parser.add_argument('--lr',              type=float, default=5e-5,
                        help='Fine-tuning learning rate for Phase B')
    parser.add_argument('--warmup-lr',       type=float, default=1e-3,
                        help='Learning rate for Phase A (coord_head warm-up)')
    parser.add_argument('--lr-weight',       type=float, default=8.0)
    parser.add_argument('--lr-sep',          type=int,   default=12)
    parser.add_argument('--contact-weight',  type=float, default=0.5)
    parser.add_argument('--coord-weight',    type=float, default=0.5,
                        help='Weight of coordinate geometry loss')
    parser.add_argument('--crops',           type=int,   default=CROPS_PER)
    parser.add_argument('--out',             default='model_v8.pt')
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()

    print('=' * 68)
    print('  v8  Coordinate head + CROP_LEN=80 + pos_embed interpolation')
    print(f'  {args.epochs} epochs  |  {args.warmup_epochs} coord_head warm-up  |  lr={args.lr}')
    print(f'  coord_weight={args.coord_weight}  contact_weight={args.contact_weight}')
    print('=' * 68)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # Build / verify ESM-2 cache
    cache_path = 'data/esm2_cache.npz'
    if not os.path.exists(cache_path):
        print(f'\nESM-2 disk cache not found. Building once -> {cache_path} ...')
        build_disk_cache(args.pdb_dir, cache_path)
        print('  Cache built.\n')
    else:
        print(f'\nESM-2 disk cache found ({cache_path})\n')

    # Load training data
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

    # Build v8 model (v7 weights + coord_head + interpolated pos_embed)
    print(f'Building v8 model from {args.base_model} ...')
    model = build_v8_model(args.base_model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {n_params:,}  (coord_head adds ~{sum(p.numel() for p in model.coord_head.parameters()):,} params)\n')

    # ── Phase A: warm up coord_head only ─────────────────────────────────────
    if args.warmup_epochs > 0:
        print(f'── Phase A: coord_head warm-up ({args.warmup_epochs} epochs, lr={args.warmup_lr}) ──')
        # Freeze everything except coord_head
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith('coord_head'))

        opt_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.warmup_lr, weight_decay=1e-4)

        for ep in range(1, args.warmup_epochs + 1):
            t0 = time.time()
            loss = one_epoch(model, train_samples, opt_a, rng,
                             lr_weight=args.lr_weight,
                             lr_sep=args.lr_sep,
                             contact_weight=0.0,   # no contact loss in warmup
                             coord_weight=args.coord_weight,
                             crops_per=args.crops)
            val_mse = quick_val_mse(model, val_samples)
            print(f'  Warm-up ep {ep:2d}/{args.warmup_epochs}  '
                  f'train_loss={loss:.4f}  val_MSE={val_mse:.2f}  ({time.time()-t0:.0f}s)')

        # Unfreeze all
        for param in model.parameters():
            param.requires_grad_(True)

    # ── Phase B: full fine-tune ───────────────────────────────────────────────
    phase_b_epochs = args.epochs - args.warmup_epochs
    print(f'\n── Phase B: full fine-tune ({phase_b_epochs} epochs, lr={args.lr}) ──')
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(phase_b_epochs, 1), eta_min=args.lr * 0.05)

    best_val   = float('inf')
    best_state = None
    history    = []
    t_start    = time.time()

    for ep in range(1, phase_b_epochs + 1):
        ep_t = time.time()
        train_loss = one_epoch(model, train_samples, opt, rng,
                               lr_weight=args.lr_weight,
                               lr_sep=args.lr_sep,
                               contact_weight=args.contact_weight,
                               coord_weight=args.coord_weight,
                               crops_per=args.crops)
        sched.step()
        val_mse = quick_val_mse(model, val_samples)
        elapsed = time.time() - ep_t

        history.append({'epoch': args.warmup_epochs + ep,
                        'train_loss': train_loss, 'val_mse': val_mse})

        star = ''
        if val_mse < best_val:
            best_val   = val_mse
            best_state = copy.deepcopy(model.state_dict())
            star = ' ☆'
            tmp = args.out + '.best_so_far.pt'
            torch.save({'state_dict': best_state, 'aa_dim': ESM_RICH_DIM,
                        'coord_head': True, 'seq_len': COORD_LEN,
                        'epoch': args.warmup_epochs + ep, 'val_mse': best_val}, tmp)

        print(f'  Ep {ep:3d}/{phase_b_epochs}  '
              f'train={train_loss:.4f}  val_MSE={val_mse:.2f}  ({elapsed:.0f}s){star}')

    # Save best model
    if best_state:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(), 'aa_dim': ESM_RICH_DIM,
                'coord_head': True, 'seq_len': COORD_LEN}, args.out)
    print(f'\nSaved: {args.out}  (best val_MSE={best_val:.3f})')
    print(f'Total training time: {(time.time()-t_start)/60:.1f} min')

    hist_path = args.out.replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)
    print(f'History: {hist_path}')

    # Cleanup temp
    tmp = args.out + '.best_so_far.pt'
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == '__main__':
    main()
