#!/usr/bin/env python3
"""train_v9.py — v9: 80aa crop + 16× long-range contact loss (no coord head).

v8 post-mortem: coord geometry loss (~50% of gradient budget) hurt short proteins
(1CRN, 1VII, 1TRZ all regressed). CROP_LEN=80 was the real win (1TIG: 0.141,
2PTL: 0.130 with MDS eval). v9 keeps the 80aa crop and drops the coord head, then
cranks up long-range contact supervision (16× vs v8's 8×) to force better topology.

Strategy:
  - Start from model_v8.pt (already trained on 80aa crops, better LR representation)
  - No coord head: full gradient budget goes to distogram + LR contacts
  - lr_weight=16 for |i-j|>=12 (doubled vs v8, same logic that got v5 to LR=0.266)
  - contact_weight=0.5 (same as v8)
  - 80 epochs at lr=1e-4 with cosine annealing to 1%

Usage:
  python src/train_v9.py --base-model model_v8.pt --epochs 80 --out model_v9.pt
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
CROP_LEN  = 80    # training crop length
SEQ_LEN   = 100   # pos_embed size (same as v8, supports up to 100aa at inference)
CROPS_PER = 1     # crops per protein per epoch

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
def build_v9_model(base_model_path: str) -> md.TransformerDistancePredictor:
    """Load v8 (or v7) weights into a plain v9 model (no coord_head).

    coord_head keys from v8 are silently skipped.
    pos_embed is interpolated from any size → SEQ_LEN=100 if needed.
    """
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
        print(f'  [train_v9] WARNING: {base_model_path} not found — training from scratch.')
        return model

    raw = torch.load(base_model_path, map_location='cpu', weights_only=False)
    src_state = raw['state_dict'] if isinstance(raw, dict) and 'state_dict' in raw else raw

    new_state = model.state_dict()
    n_copied, n_skipped = 0, 0

    for key, src_tensor in src_state.items():
        if key not in new_state:
            # e.g. coord_head.* from v8 — skip silently
            n_skipped += 1
            continue

        if key == 'pos_embed' and src_tensor.shape[0] != SEQ_LEN:
            old_len = src_tensor.shape[0]
            pe_t = src_tensor.float().T.unsqueeze(0)          # (1, hidden, old_len)
            new_pe = F.interpolate(pe_t, size=SEQ_LEN, mode='linear', align_corners=True)
            new_state['pos_embed'] = new_pe.squeeze(0).T      # (SEQ_LEN, hidden)
            print(f'  [train_v9] Interpolated pos_embed {old_len} → {SEQ_LEN}')
            n_copied += 1
            continue

        if new_state[key].shape != src_tensor.shape:
            print(f'  [train_v9] Shape mismatch — skipping {key}')
            n_skipped += 1
            continue

        new_state[key] = src_tensor
        n_copied += 1

    model.load_state_dict(new_state)
    print(f'  [train_v9] Copied {n_copied} tensors from {base_model_path}, '
          f'skipped {n_skipped} (coord_head / shape mismatch).')
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
                start = int(rng.integers(0, L - CROP_LEN + 1))
                seq_c = seq_r[start : start + CROP_LEN]
                crd_c = coords_r[start : start + CROP_LEN]
            else:
                seq_c, crd_c = seq_r, coords_r
                start = 0

            full_enc = _cached_esm2_rich(seq_r)
            enc      = full_enc[start : start + len(seq_c)] if len(seq_r) > CROP_LEN else full_enc
            dist_np  = utils.coords_to_distances(crd_c).astype(np.float32)

            X = torch.tensor(enc[None],     dtype=torch.float32)   # (1, L, 368)
            Y = torch.tensor(dist_np[None], dtype=torch.float32)   # (1, L, L)

            opt.zero_grad()

            logits, ss_logits, _ = model.forward_full(X)

            # 1. Distogram cross-entropy
            loss = md.distogram_loss(logits, Y, backbone_weight=1.0)

            # 2. Long-range weighted contact BCE — the key lever for TM-score
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
        description='v9: 80aa crop + 16x LR contact loss (no coord head)')
    parser.add_argument('--base-model',     default='model_v8.pt')
    parser.add_argument('--pdb-dir',        default='data/pdbs')
    parser.add_argument('--epochs',         type=int,   default=80)
    parser.add_argument('--lr',             type=float, default=1e-4)
    parser.add_argument('--lr-weight',      type=float, default=16.0,
                        help='Upweight for |i-j|>=lr_sep contacts (default 16×)')
    parser.add_argument('--lr-sep',         type=int,   default=12)
    parser.add_argument('--contact-weight', type=float, default=0.5)
    parser.add_argument('--crops',          type=int,   default=CROPS_PER)
    parser.add_argument('--out',            default='model_v9.pt')
    parser.add_argument('--seed',           type=int,   default=42)
    args = parser.parse_args()

    print('=' * 68)
    print('  v9  CROP_LEN=80 + 16× LR contact loss  (no coord head)')
    print(f'  {args.epochs} epochs  |  lr={args.lr}  |  lr_weight={args.lr_weight}×')
    print(f'  contact_weight={args.contact_weight}  lr_sep={args.lr_sep}')
    print('=' * 68)

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

    print(f'Building v9 model from {args.base_model} ...')
    model = build_v9_model(args.base_model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Total parameters: {n_params:,}\n')

    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
                               crops_per=args.crops)
        sched.step()
        val_mse = quick_val_mse(model, val_samples)
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

    tmp = args.out + '.best_so_far.pt'
    if os.path.exists(tmp):
        os.remove(tmp)

    print('\nTRAIN_V9_DONE:0')


if __name__ == '__main__':
    main()
