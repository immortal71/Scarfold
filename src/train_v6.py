#!/usr/bin/env python3
"""train_v6.py â€” Fine-tune v5 Evoformer model with ESM-2 sequence embeddings.

Key improvement over v5:
    Replace hand-crafted 48-dim features (one-hot + BLOSUM62 + physicochemical)
    with 368-dim features = ESM-2 (320-dim) + rich encoding (48-dim).

    ESM-2 (esm2_t6_8M_UR50D, 8M params) is a protein language model trained on
    250M sequences.  Its per-residue embeddings provide implicit evolutionary
    co-variation signal without requiring a raw MSA pipeline â€” the single biggest
    representational gap between our model and AlphaFold-class methods.

Strategy:
    1. Load model_v5.pt (Evoformer + TriMul, aa_dim=48).
    2. Build a new TransformerDistancePredictor with aa_dim=368.
    3. Copy all layers whose weights are shape-compatible (everything except
       residue_proj, which changes from Linear(48â†’256) to Linear(368â†’256)).
    4. Phase A (warm-up, --warmup-epochs, default 5): only train residue_proj
       so the new projection learns to align ESM-2 features with the frozen
       Evoformer pair-track representation.
    5. Phase B (fine-tune, remaining epochs): unfreeze all parameters and train
       end-to-end with LR-weighted contact BCE loss (inherited from v5).

Usage:
    python src/train_v6.py                            # defaults: 60 epochs total
    python src/train_v6.py --epochs 80 --out model_v6.pt
    python src/train_v6.py --epochs 60 --warmup-epochs 10 --lr 1e-4

Requirements:
    pip install fair-esm
    (or the script will try torch.hub on first run)
"""
import argparse, copy, glob, json, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import utils, model as md
from src.esm_utils import esm2_rich_encoding, ESM_RICH_DIM

CROP_LEN  = 60
CROPS_PER = 4

# Test proteins â€” never seen during training
TEST_PIDS = {'1crn', '1vii', '1lyz', '1trz', '1aho', '2ptl', '1tig'}


# â”€â”€ Data loading â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_pid(filename: str) -> str:
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    if name.startswith('pdb') and len(name) >= 7:
        return name[3:7]
    return name[:4]


def load_all_pdbs(pdb_dir: str, min_res: int = 20, max_res: int = 200):
    """Load every PDB/ENT file; return list of (seq, Ca-coords, pid)."""
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


# â”€â”€ Model initialisation: transplant v5 weights into new aa_dim=368 model â”€â”€â”€â”€

def build_v6_model(base_model_path: str, new_aa_dim: int = ESM_RICH_DIM) -> md.TransformerDistancePredictor:
    """Create a 368-dim TransformerDistancePredictor pre-loaded with v5 weights.

    residue_proj is the only layer with a shape mismatch (48â†’256 vs 368â†’256).
    All other layers (Evoformer, pair track, TriMul, heads) are copied as-is.
    The new residue_proj is initialised with Kaiming-uniform and immediately
    enters Phase-A training (see main()).
    """
    # Build fresh model with new input dimension
    model = md.TransformerDistancePredictor(
        seq_len=CROP_LEN,
        aa_dim=new_aa_dim,
        hidden=256,
        pair_dim=64,
        nhead=4,
        num_layers=4,
        n_bins=md.NUM_BINS + 1,
        dropout=0.1,
        num_recycles=3,
    )
    model.aa_dim = new_aa_dim  # keep metadata up-to-date for save_model

    if not os.path.exists(base_model_path):
        print(f'  [train_v6] WARNING: {base_model_path} not found â€” training from scratch.')
        return model

    raw = torch.load(base_model_path, map_location='cpu', weights_only=False)
    state_v5 = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw

    new_state = model.state_dict()
    n_copied, n_skipped = 0, 0
    for key, v5_tensor in state_v5.items():
        if key not in new_state:
            n_skipped += 1
            continue
        if new_state[key].shape != v5_tensor.shape:
            # residue_proj.weight / residue_proj.bias â€” skip, keep Kaiming init
            print(f'  [train_v6] Shape mismatch â€” skipping {key} '
                  f'({list(v5_tensor.shape)} â†’ {list(new_state[key].shape)})')
            n_skipped += 1
            continue
        new_state[key] = v5_tensor
        n_copied += 1

    model.load_state_dict(new_state)
    print(f'  [train_v6] Copied {n_copied} tensors from {base_model_path}, '
          f'skipped {n_skipped} (shape mismatch or new keys).')
    return model


# â”€â”€ Training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def one_epoch(
    model: md.TransformerDistancePredictor,
    train_samples,
    opt: torch.optim.Optimizer,
    rng: np.random.Generator,
    lr_weight: float,
    lr_sep: int,
    contact_weight: float,
) -> float:
    """One training epoch: random crop each protein, encode with ESM-2+rich, forward+backward."""
    model.train()
    ep_loss, n_steps = 0.0, 0
    order = rng.permutation(len(train_samples))

    for i in order:
        seq_r, coords_r, _ = train_samples[i]
        n_crops = CROPS_PER if len(seq_r) > CROP_LEN else 1

        for _ in range(n_crops):
            L = len(seq_r)
            if L > CROP_LEN:
                start  = int(rng.integers(0, L - CROP_LEN + 1))
                seq_c  = seq_r[start : start + CROP_LEN]
                crd_c  = coords_r[start : start + CROP_LEN]
            else:
                seq_c, crd_c = seq_r, coords_r

            # ESM-2 + rich encoding (368-dim)
            enc     = esm2_rich_encoding(seq_c)
            dist_np = utils.coords_to_distances(crd_c).astype(np.float32)

            X = torch.tensor(enc[None],     dtype=torch.float32)   # (1, L, 368)
            Y = torch.tensor(dist_np[None], dtype=torch.float32)   # (1, L, L)

            opt.zero_grad()
            logits, ss_logits, _ = model.forward_full(X)

            # Distogram cross-entropy (primary objective)
            loss = md.distogram_loss(logits, Y, backbone_weight=1.0)

            # Long-range weighted contact BCE (v5 improvement, inherited)
            loss += contact_weight * md._contact_bce_loss(
                logits, Y, is_logits=True,
                lr_weight=lr_weight, lr_sep=lr_sep,
            )

            # Secondary-structure auxiliary loss (unsupervised, from CÎ± geometry)
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


def quick_val_mse(model: md.TransformerDistancePredictor, val_samples) -> float:
    model.eval()
    mses = []
    with torch.no_grad():
        for seq_r, coords_r, _ in val_samples:
            L = min(len(seq_r), CROP_LEN)
            enc  = esm2_rich_encoding(seq_r[:L])
            dist = utils.coords_to_distances(coords_r[:L]).astype(np.float32)
            X    = torch.tensor(enc[None], dtype=torch.float32)
            logits, _, _ = model.forward_full(X)
            pred = md.bin_to_dist(logits)[0].cpu().numpy()
            mses.append(float(np.mean((pred[:L, :L] - dist[:L, :L]) ** 2)))
    return float(np.mean(mses)) if mses else 999.0


# â”€â”€ CLI â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(
        description='v6: ESM-2 embeddings + Evoformer TriMul fine-tune')
    parser.add_argument('--base-model',      default='model_v5.pt',
                        help='Starting checkpoint (v5 model, aa_dim=48)')
    parser.add_argument('--pdb-dir',         default='data/pdbs')
    parser.add_argument('--epochs',          type=int,   default=60,
                        help='Total training epochs (Phase A + Phase B)')
    parser.add_argument('--warmup-epochs',   type=int,   default=5,
                        help='Phase-A epochs: only residue_proj trained (proj warm-up)')
    parser.add_argument('--lr',              type=float, default=1e-4,
                        help='Fine-tuning learning rate for Phase B')
    parser.add_argument('--warmup-lr',       type=float, default=5e-4,
                        help='Learning rate for Phase A (residue_proj warm-up)')
    parser.add_argument('--lr-weight',       type=float, default=8.0,
                        help='Long-range contact BCE upweight (|i-j|>=lr-sep)')
    parser.add_argument('--lr-sep',          type=int,   default=12)
    parser.add_argument('--contact-weight',  type=float, default=0.5,
                        help='Weight of contact BCE relative to distogram CE')
    parser.add_argument('--out',             default='model_v6.pt')
    parser.add_argument('--seed',            type=int,   default=42)
    args = parser.parse_args()

    print('=' * 68)
    print('  v6  ESM-2 (320-dim) + rich (48-dim) -> 368-dim Evoformer TriMul')
    print(f'  {args.epochs} epochs total  |  {args.warmup_epochs} warm-up  |  '
          f'lr={args.lr}  lr_weight={args.lr_weight}')
    print('=' * 68)

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # â”€â”€ Load training data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'\nLoading PDB files from {args.pdb_dir} ...')
    all_samples = load_all_pdbs(args.pdb_dir, min_res=20, max_res=200)
    if not all_samples:
        print(f'ERROR: no PDB files found in {args.pdb_dir}. '
              f'Run: python src/download_data.py --n 50 --out data/pdbs')
        sys.exit(1)
    print(f'  {len(all_samples)} training proteins '
          f'(excluded {len(TEST_PIDS)} test proteins)')

    # 10% validation split
    n_val = max(1, len(all_samples) // 10)
    idx   = rng.permutation(len(all_samples))
    val_samples   = [all_samples[i] for i in idx[:n_val]]
    train_samples = [all_samples[i] for i in idx[n_val:]]
    print(f'  Split: {len(train_samples)} train / {len(val_samples)} val\n')

    # â”€â”€ Build v6 model (ESM-2 input, v5 Evoformer weights) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f'Building v6 model from {args.base_model} ...')
    model = build_v6_model(args.base_model, new_aa_dim=ESM_RICH_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {n_params:,}')

    # â”€â”€ Phase A: warm up residue_proj only â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if args.warmup_epochs > 0:
        print(f'\nâ”€â”€ Phase A: residue_proj warm-up ({args.warmup_epochs} epochs, '
              f'lr={args.warmup_lr}) â”€â”€')
        # Freeze everything except residue_proj
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith('residue_proj'))

        opt_a = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.warmup_lr, weight_decay=1e-4)

        for ep in range(1, args.warmup_epochs + 1):
            t0 = time.time()
            loss = one_epoch(model, train_samples, opt_a, rng,
                             lr_weight=args.lr_weight,
                             lr_sep=args.lr_sep,
                             contact_weight=args.contact_weight)
            print(f'  Warm-up ep {ep:2d}/{args.warmup_epochs}  '
                  f'train_loss={loss:.4f}  ({time.time()-t0:.0f}s)')

        # Unfreeze all parameters for Phase B
        for param in model.parameters():
            param.requires_grad_(True)

    # â”€â”€ Phase B: full fine-tune â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    phase_b_epochs = args.epochs - args.warmup_epochs
    print(f'\nâ”€â”€ Phase B: full fine-tune ({phase_b_epochs} epochs, lr={args.lr}) â”€â”€')
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
                               contact_weight=args.contact_weight)
        sched.step()
        val_mse = quick_val_mse(model, val_samples)
        elapsed = time.time() - ep_t

        history.append({
            'epoch': args.warmup_epochs + ep,
            'train_loss': train_loss,
            'val_mse': val_mse,
        })

        star = ''
        if val_mse < best_val:
            best_val   = val_mse
            best_state = copy.deepcopy(model.state_dict())
            star = ' â˜…'
            _tmp = args.out + '.best_so_far.pt'
            torch.save({'state_dict': best_state, 'aa_dim': ESM_RICH_DIM,
                        'epoch': args.warmup_epochs + ep,
                        'val_mse': best_val}, _tmp)

        print(f'  Ep {ep:3d}/{phase_b_epochs}  '
              f'train={train_loss:.4f}  val_MSE={val_mse:.2f}  '
              f'({elapsed:.0f}s){star}')

    # â”€â”€ Save best model â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if best_state:
        model.load_state_dict(best_state)
    torch.save({'state_dict': model.state_dict(), 'aa_dim': ESM_RICH_DIM}, args.out)
    print(f'\nSaved: {args.out}  (best val_MSE={best_val:.3f})')
    print(f'Total training time: {(time.time()-t_start)/60:.1f} min')

    # Save training history
    hist_path = args.out.replace('.pt', '_history.json')
    with open(hist_path, 'w') as f:
        json.dump({'args': vars(args), 'history': history}, f, indent=2)
    print(f'History: {hist_path}')

    # â”€â”€ Cleanup temp file â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    tmp = args.out + '.best_so_far.pt'
    if os.path.exists(tmp):
        os.remove(tmp)


if __name__ == '__main__':
    main()

