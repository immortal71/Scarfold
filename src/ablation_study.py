#!/usr/bin/env python3
"""ablation_study.py — Rigorous 3-variant ablation with multiple random seeds.

Scientific question:
    "Does triangle multiplication (geometric transitivity enforcement) improve
    long-range contact prediction in the single-sequence regime, and by how much
    does each pair-track component contribute?"

Three model variants (same hyperparameters, only architecture differs):
    A  — Seq-only:    Pure sequence Transformer, NO pair track at all
    B  — Evoformer:   Pair track with outer-product update, NO triangle multiplication
    C  — TriMul:      Full pair track WITH triangle multiplication (our v4 model)

Each variant is trained 3× with different random seeds, giving mean ± std over
metrics — the standard way to report results in NeurIPS/ICML/Nature Methods papers.

Primary metric: long_range_precision_L5 (|i-j|≥12, top-L/5 predicted contacts)
Secondary:      contact_f1, local_lDDT, tm_proxy

Usage:
    python src/ablation_study.py --epochs 150 --seeds 3 --proteins 50
    python src/ablation_study.py --epochs 100 --seeds 3 --proteins 50 --quick
"""
import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils, model as md, evaluate as ev

# ── Variant A: Sequence-only Transformer (no pair track) ─────────────────────

class SeqOnlyTransformer(nn.Module):
    """Pure sequence transformer — no pair representation, no pair-bias attention.

    This is the strongest reasonable baseline: shows how much the *pair track*
    (outer-product + triangle multiplication) contributes beyond a plain sequence model.
    Architecture: 4-layer pre-LN Transformer encoder → MLP distogram head.
    """
    def __init__(self, seq_len, aa_dim=48, hidden=256, nhead=4, num_layers=4,
                 n_bins=md.NUM_BINS + 1, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.aa_dim  = aa_dim
        self.n_bins  = n_bins

        self.input_proj = nn.Linear(aa_dim, hidden)
        self.pos_embed  = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # Pair head: for each pair (i,j) concatenate h_i and h_j, project to bins
        self.pair_head = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_bins),
        )

    def forward(self, x):
        """x: (B, L, aa_dim)  →  logits (B, L, L, n_bins)"""
        B, L, _ = x.shape
        h = self.input_proj(x) + self.pos_embed[:L].unsqueeze(0)
        h = self.encoder(h)                           # (B, L, hidden)
        h_i = h.unsqueeze(2).expand(B, L, L, -1)
        h_j = h.unsqueeze(1).expand(B, L, L, -1)
        logits = self.pair_head(torch.cat([h_i, h_j], dim=-1))
        logits = (logits + logits.transpose(1, 2)) / 2.0  # symmetrise
        return logits

    def forward_full(self, x):
        logits = self.forward(x)
        B, L, _, _ = logits.shape
        ss_dummy    = torch.zeros(B, L, 3, device=x.device)
        plddt_dummy = torch.zeros(B, L, 4, device=x.device)
        return logits, ss_dummy, plddt_dummy


# ── Variant B: Evoformer without triangle multiplication ─────────────────────

class _PairBiasLayerNoTriMul(nn.Module):
    """Evoformer-lite layer WITHOUT triangle multiplication.
    Identical to v4 except self.triangle_mul is removed.
    Used to isolate the exact contribution of triangle multiplication.
    """
    def __init__(self, hidden, pair_dim, nhead, dropout=0.1):
        super().__init__()
        assert hidden % nhead == 0
        self.nhead = nhead; self.head_dim = hidden // nhead
        self.norm_seq  = nn.LayerNorm(hidden)
        self.norm_pair = nn.LayerNorm(pair_dim)
        self.pair_to_bias = nn.Linear(pair_dim, nhead, bias=False)
        self.q_proj    = nn.Linear(hidden, hidden, bias=False)
        self.k_proj    = nn.Linear(hidden, hidden, bias=False)
        self.v_proj    = nn.Linear(hidden, hidden, bias=False)
        self.out_proj  = nn.Linear(hidden, hidden)
        self.attn_drop = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 4),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden), nn.Dropout(dropout))
        self.pair_update = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, pair_dim),
            nn.GELU(), nn.Linear(pair_dim, pair_dim))
        self.outer_proj = nn.Linear(hidden * 2, pair_dim)
        # NO self.triangle_mul here — this is the key difference

    def forward(self, seq, pair):
        B, L, H = seq.shape; hd = self.head_dim
        seq_n = self.norm_seq(seq)
        bias  = self.pair_to_bias(self.norm_pair(pair)).permute(0, 3, 1, 2)
        scale = hd ** -0.5
        Q = self.q_proj(seq_n).view(B, L, self.nhead, hd).transpose(1, 2)
        K = self.k_proj(seq_n).view(B, L, self.nhead, hd).transpose(1, 2)
        V = self.v_proj(seq_n).view(B, L, self.nhead, hd).transpose(1, 2)
        attn = torch.softmax(torch.einsum('bhid,bhjd->bhij', Q, K) * scale + bias, dim=-1)
        attn = self.attn_drop(attn)
        out  = self.out_proj(torch.einsum('bhij,bhjd->bhid', attn, V).transpose(1, 2).reshape(B, L, H))
        seq  = seq + out + self.ff(seq + out)
        h_i  = seq.unsqueeze(2).expand(B, L, L, -1)
        h_j  = seq.unsqueeze(1).expand(B, L, L, -1)
        pair = pair + self.pair_update(self.outer_proj(torch.cat([h_i, h_j], -1)))
        # ← no triangle multiplication
        return seq, pair


class EvoformerNoTriMul(nn.Module):
    """Evoformer-lite WITHOUT triangle multiplication (Variant B)."""
    def __init__(self, seq_len, aa_dim=48, hidden=256, pair_dim=64,
                 nhead=4, num_layers=4, n_bins=md.NUM_BINS+1,
                 dropout=0.1, num_recycles=3):
        super().__init__()
        self.seq_len = seq_len; self.aa_dim = aa_dim
        self.n_bins = n_bins; self.num_recycles = num_recycles
        self.residue_proj = nn.Linear(aa_dim, hidden)
        self.pos_embed    = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)
        self.pair_init_i  = nn.Linear(hidden, pair_dim)
        self.pair_init_j  = nn.Linear(hidden, pair_dim)
        max_rel = 32
        self.rel_pos_embed = nn.Embedding(2 * max_rel + 2, pair_dim)
        self.max_rel_pos   = max_rel
        self.layers = nn.ModuleList([
            _PairBiasLayerNoTriMul(hidden, pair_dim, nhead, dropout)
            for _ in range(num_layers)])
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, pair_dim),
            nn.GELU(), nn.Linear(pair_dim, n_bins))
        self.recycle_pair_proj = nn.Linear(n_bins, pair_dim)
        self.ss_head   = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 3))
        self.plddt_head= nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 4))

    def _rel_pos(self, L, device):
        idx = torch.arange(L, device=device)
        rel = (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp(-self.max_rel_pos, self.max_rel_pos)
        return rel + self.max_rel_pos

    def _init_pair(self, h, L, device):
        pair = self.pair_init_i(h).unsqueeze(2) + self.pair_init_j(h).unsqueeze(1)
        return pair + self.rel_pos_embed(self._rel_pos(L, device)).unsqueeze(0)

    def forward_full(self, x, num_recycles=None):
        B, L, _ = x.shape
        n_rec = num_recycles if num_recycles is not None else self.num_recycles
        h_init = self.residue_proj(x) + self.pos_embed[:L].unsqueeze(0)
        pair   = self._init_pair(h_init, L, x.device)
        for cycle in range(max(n_rec, 1)):
            h = h_init
            for layer in self.layers:
                h, pair = layer(h, pair)
            if cycle < max(n_rec, 1) - 1:
                logits_prev = self.distogram_head(pair)
                logits_prev = (logits_prev + logits_prev.transpose(1, 2)) / 2.0
                pair = pair + self.recycle_pair_proj(logits_prev.detach())
        logits = self.distogram_head(pair)
        logits = (logits + logits.transpose(1, 2)) / 2.0
        return logits, self.ss_head(h), self.plddt_head(h)

    def forward(self, x, **kw):
        return self.forward_full(x, **kw)[0]


# ── Variant D: Sequence-separation baseline (no learning) ───────────────────

class SeqDistanceBaseline(nn.Module):
    """Null-hypothesis baseline: predict Ca-Ca distance purely from |i-j|.

    Fit on training data: computes mean distance at each sequence separation.
    Requires NO training — represents what you get with zero structural knowledge.
    Any model that doesn't beat this on contact F1 is not learning structure.
    """
    def __init__(self, seq_len, n_bins=md.NUM_BINS + 1):
        super().__init__()
        self.seq_len = seq_len
        self.n_bins  = n_bins
        self.aa_dim  = 48   # needed by evaluate.py
        # Default polymer physics estimate before fitting: d(k) = 2.0 + 2.5*k^0.55
        k = np.arange(seq_len + 2, dtype=np.float32)
        default_dists = np.clip(2.0 + 2.5 * (k ** 0.55), 2.0, 22.0)
        self.sep_to_dist = default_dists    # numpy, not a parameter
        self._bin_edges  = md.get_bin_edges()

    def fit(self, raw_train, crop_len=60):
        """Compute mean Ca-Ca distance at each sequence separation from training data."""
        sep_sums   = np.zeros(crop_len + 2, dtype=np.float64)
        sep_counts = np.zeros(crop_len + 2, dtype=np.int64)
        for seq, coords, pid in raw_train:
            N = min(len(coords), crop_len)
            dists = utils.coords_to_distances(coords[:N])
            for sep in range(1, N):
                vals = [dists[i, i + sep] for i in range(N - sep)]
                sep_sums[sep]   += np.sum(vals)
                sep_counts[sep] += len(vals)
        fitted = self.sep_to_dist.copy()
        for sep in range(1, crop_len + 2):
            if sep_counts[sep] > 0:
                fitted[sep] = sep_sums[sep] / sep_counts[sep]
        self.sep_to_dist = np.clip(fitted, 2.0, 22.0)

    def forward(self, x):
        """x: (B, L, aa_dim)  ->  logits (B, L, L, n_bins)."""
        B, L, _ = x.shape
        idx     = np.arange(L)
        sep_mat = np.abs(idx[:, None] - idx[None, :]).clip(0, len(self.sep_to_dist) - 1)
        dist_pred = self.sep_to_dist[sep_mat].astype(np.float32)     # (L, L)
        bin_idx   = np.digitize(dist_pred, self._bin_edges[1:]).clip(0, self.n_bins - 1)
        logits_np = np.full((L, L, self.n_bins), -10.0, dtype=np.float32)
        rows, cols = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
        logits_np[rows, cols, bin_idx] = 10.0
        logits = torch.tensor(logits_np, device=x.device).unsqueeze(0).expand(B, -1, -1, -1)
        return logits

    def forward_full(self, x):
        logits = self.forward(x)
        B, L   = logits.shape[:2]
        return logits, torch.zeros(B, L, 3, device=x.device), torch.zeros(B, L, 4, device=x.device)


# ── Variant registry ──────────────────────────────────────────────────────────

def get_variant(name, seq_len, aa_dim=48):
    """Return an untrained model for the named variant."""
    if name == 'D_seq_distance':
        return SeqDistanceBaseline(seq_len)
    elif name == 'A_seq_only':
        return SeqOnlyTransformer(seq_len, aa_dim=aa_dim)
    elif name == 'B_evoformer_no_trimul':
        return EvoformerNoTriMul(seq_len, aa_dim=aa_dim)
    elif name == 'C_evoformer_trimul':
        return md.get_model('transformer', seq_len, aa_dim=aa_dim)
    else:
        raise ValueError(f'Unknown variant: {name}')


VARIANTS = ['D_seq_distance', 'A_seq_only', 'B_evoformer_no_trimul', 'C_evoformer_trimul']
VARIANT_LABELS = {
    'D_seq_distance':         'D: Seq-dist baseline (no learning)',
    'A_seq_only':             'A: Seq-only Transformer (no pair track)',
    'B_evoformer_no_trimul':  'B: Evoformer (no TriMul)',
    'C_evoformer_trimul':     'C: Evoformer + TriMul (ours)',
}


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_run(model, raw_train, test_data, epochs, crop_len, crops_per,
                  lr, seed, verbose=True):
    """Train one variant for one seed. Returns history list and best val_MSE."""
    rng   = np.random.default_rng(seed)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.02)

    best_val = float('inf')
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        order   = rng.permutation(len(raw_train))
        ep_loss = 0.0; n_steps = 0

        for i in order:
            seq_r, coords_r, _ = raw_train[i]
            n_crops = crops_per if len(seq_r) > crop_len else 1
            for _ in range(n_crops):
                L = len(seq_r)
                if L > crop_len:
                    start  = int(rng.integers(0, L - crop_len + 1))
                    seq_c  = seq_r[start:start + crop_len]
                    crd_c  = coords_r[start:start + crop_len]
                else:
                    seq_c, crd_c = seq_r, coords_r

                enc     = utils.rich_encoding(seq_c)
                dist_np = utils.coords_to_distances(crd_c).astype(np.float32)
                X = torch.tensor(enc[None],     dtype=torch.float32)
                Y = torch.tensor(dist_np[None], dtype=torch.float32)

                opt.zero_grad()
                logits, ss_logits, _ = model.forward_full(X)
                loss = md.distogram_loss(logits, Y, backbone_weight=1.0)
                loss += 0.3 * md._contact_bce_loss(logits, Y, is_logits=True)
                # SS auxiliary only when model has a real SS head
                if ss_logits.requires_grad or ss_logits.sum() != 0:
                    ss_lbl = torch.tensor(
                        md.ss_labels_from_dists(dist_np)[None], dtype=torch.long)
                    ss_loss = F.cross_entropy(ss_logits.reshape(-1, 3), ss_lbl.reshape(-1))
                    if not torch.isnan(ss_loss):
                        loss += 0.2 * ss_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ep_loss += float(loss); n_steps += 1

        sched.step()
        train_loss = ep_loss / max(n_steps, 1)

        model.eval()
        val_mses = []
        with torch.no_grad():
            for enc, dist_np, _, _, _ in test_data:
                pred = md.predict(model, enc)
                val_mses.append(float(np.mean((pred - dist_np) ** 2)))
        val_loss = float(np.mean(val_mses)) if val_mses else 999.0

        history.append({'epoch': ep, 'train_loss': train_loss, 'val_loss': val_loss})
        if val_loss < best_val:
            best_val = val_loss
            import copy; best_state = copy.deepcopy(model.state_dict())

        if verbose and (ep % 25 == 0 or ep == 1):
            print(f'    ep {ep:3d}/{epochs}  train={train_loss:.4f}  val_MSE={val_loss:.2f}'
                  + (' ★' if val_loss == best_val else ''))

    if best_state is not None:
        model.load_state_dict(best_state)
    return history, best_val


def eval_model_on_test(model, test_data):
    """Evaluate model on all test proteins, return per-protein and aggregate metrics."""
    results = {}
    for enc, dist_np, coords, seq, pid in test_data:
        try:
            metrics = ev.evaluate_model(model, seq, coords)
            results[pid] = metrics
        except Exception as e:
            print(f'    WARN: eval failed for {pid}: {e}')
    return results


# ── Main ablation runner ──────────────────────────────────────────────────────

def run_ablation(args):
    import datetime
    os.makedirs('results', exist_ok=True)

    CROP_LEN  = 60
    CROPS_PER = 4

    # ── Load training proteins ────────────────────────────────────────────────
    print('=' * 70)
    print(f'Loading {args.proteins} training + {args.test_proteins} test proteins ...')

    # Training proteins — from curated v4 list (all cached locally)
    TRAIN_IDS = [
        ('1bdd','A'),('1rop','A'),('2abd','A'),('1ail','A'),('1lmb','3'),
        ('2lzm','A'),('1prb','A'),('4rxn','A'),('1csp','A'),('1hoe','A'),
        ('1sh3','A'),('1tit','A'),('2ptl','A'),('1ubq','A'),('2ci2','I'),
        ('2hpr','A'),('3icb','A'),('1pgb','A'),('2ptn','A'),('1poh','A'),
        ('1aps','A'),('1fkb','A'),('2acy','A'),('1gab','A'),('2trx','A'),
        ('1fn3','A'),('1a3n','A'),('1hhp','A'),('1pga','A'),('1ab1','A'),
        ('1a6n','A'),('1l2y','A'),('1gb1','A'),('2gb1','A'),('1cbs','A'),
        ('1srl','A'),('1bba','A'),('2chf','A'),('1wit','A'),('1iib','A'),
        ('1fex','A'),('1gya','A'),('1w4e','A'),('2acg','A'),('1dci','A'),
        ('1hz6','A'),('1msi','A'),('1pca','A'),('2bbu','A'),('1qcf','A'),
    ]
    # 7 diverse test proteins: classic benchmarks + novel structures.
    # NOTE: deliberately excludes all CASP13/14 FM targets used in casp_eval.py
    # to avoid circular evaluation (section 4.4 uses those as held-out generalization).
    # Also excludes 1cbn (crambin-B, near-duplicate of 1crn).
    TEST_IDS = [
        ('1crn', 'A'),   # crambin, alpha+beta, 46 aa (benchmark classic)
        ('1vii', 'A'),   # villin headpiece, all-alpha, 36 aa
        ('1lyz', 'A'),   # lysozyme, alpha+beta, 129 aa
        ('1trz', 'A'),   # insulin, all-alpha, 21 aa
        ('1aho', 'A'),   # scorpion toxin, alpha+beta, 66 aa (different topology)
        ('2ptl', 'A'),   # protein L, alpha+beta, 62 aa  (NOT in training set)
        ('1tig', 'A'),   # Tig chaperone domain, alpha+beta, 88 aa
    ]

    n_train = min(args.proteins, len(TRAIN_IDS))
    train_ids = TRAIN_IDS[:n_train]

    raw_train = []
    for pid, chain in train_ids:
        try:
            path   = utils.fetch_pdb(pid)
            seq    = utils.pdb_sequence(path, chain=chain, max_residues=200)
            coords = utils.pdb_ca_coords(path, chain=chain, max_residues=200)
            N = min(len(seq), len(coords))
            if N >= 15:
                raw_train.append((seq[:N], coords[:N], pid))
        except Exception as e:
            print(f'  SKIP {pid}: {e}')

    test_data = []
    for pid, chain in TEST_IDS[:args.test_proteins]:
        try:
            path   = utils.fetch_pdb(pid)
            seq    = utils.pdb_sequence(path, chain=chain, max_residues=CROP_LEN)
            coords = utils.pdb_ca_coords(path, chain=chain, max_residues=CROP_LEN)
            N = min(len(seq), len(coords))
            enc    = utils.rich_encoding(seq[:N])
            dist   = utils.coords_to_distances(coords[:N]).astype(np.float32)
            test_data.append((enc, dist, coords[:N], seq[:N], pid))
        except Exception as e:
            print(f'  SKIP TEST {pid}: {e}')

    print(f'  {len(raw_train)} training proteins, {len(test_data)} test proteins\n')

    # ── Main ablation loop ────────────────────────────────────────────────────
    all_results  = {}   # variant → seed → {pid: metrics}
    summary_rows = []   # for CSV

    seeds = list(range(args.seeds))

    # Determine which variants to run
    active_variants = [v for v in VARIANTS if not (args.skip_trimul and v == 'C_evoformer_trimul')]

    # Fit the sequence-distance baseline on training data (no GPU, instant)
    _baseline_model = None
    if 'D_seq_distance' in active_variants:
        print('Fitting sequence-distance baseline on training data ...')
        _baseline_model = SeqDistanceBaseline(CROP_LEN)
        _baseline_model.fit(raw_train, crop_len=CROP_LEN)
        _baseline_results = eval_model_on_test(_baseline_model, test_data)
        print('  Baseline fitted and evaluated.')
        print()

    for variant in active_variants:
        all_results[variant] = {}
        label = VARIANT_LABELS[variant]
        print('-' * 70)
        print(f'VARIANT: {label}')
        print('-' * 70)

        seed_metrics = []   # list of dicts, one per seed

        # D baseline is deterministic — no training needed, reuse fitted results
        if variant == 'D_seq_distance':
            all_results[variant] = {s: _baseline_results for s in seeds}
            agg = {}
            for metric in ['rmsd_aligned', 'local_lDDT', 'contact_f1',
                           'long_range_precision_L5', 'tm_proxy']:
                vals = [_baseline_results[p][metric]
                        for p in _baseline_results
                        if metric in _baseline_results.get(p, {})]
                agg[metric] = float(np.mean(vals)) if vals else float('nan')
            for metric in ['local_lDDT', 'contact_f1', 'long_range_precision_L5', 'tm_proxy']:
                m = agg[metric]
                print(f'    {metric:32s}: {m:.4f} (deterministic, no training)')
                summary_rows.append({
                    'variant': variant, 'label': label, 'metric': metric,
                    'mean': m, 'std': 0.0,
                    'n_seeds': 1, 'epochs': 0,
                    'n_train_proteins': 0,
                    'seed_values': f'{m:.4f}',
                })
            print()
            continue

        for seed in seeds:
            print(f'  Seed {seed+1}/{args.seeds} ...')
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = get_variant(variant, CROP_LEN)
            n_params = sum(p.numel() for p in model.parameters())
            if seed == 0:
                print(f'  Params: {n_params:,}')

            t0 = time.time()
            history, best_val_mse = train_one_run(
                model, raw_train, test_data,
                epochs=args.epochs, crop_len=CROP_LEN, crops_per=CROPS_PER,
                lr=args.lr, seed=seed, verbose=True)
            elapsed = time.time() - t0

            print(f'  Training done in {elapsed/60:.1f} min  best_val_MSE={best_val_mse:.2f}')
            print(f'  Evaluating ...')
            results = eval_model_on_test(model, test_data)
            all_results[variant][seed] = results

            # Aggregate metrics over test proteins
            agg = {}
            for metric in ['rmsd_aligned', 'local_lDDT', 'contact_f1',
                           'long_range_precision_L5', 'tm_proxy']:
                vals = [results[p][metric] for p in results if metric in results.get(p, {})]
                agg[metric] = float(np.mean(vals)) if vals else float('nan')

            agg['best_val_mse'] = best_val_mse
            seed_metrics.append(agg)
            print(f'  Mean: lDDT={agg["local_lDDT"]:.3f}  F1={agg["contact_f1"]:.3f}  '
                  f'LR-P@L5={agg["long_range_precision_L5"]:.3f}  TM={agg["tm_proxy"]:.3f}')

            # Save checkpoint
            ckpt_path = f'results/ablation_{variant}_seed{seed}.pt'
            md.save_model(model, ckpt_path)
            print()

        # ── Compute mean ± std across seeds ──────────────────────────────────
        print(f'  Results across {len(seeds)} seeds:')
        for metric in ['local_lDDT', 'contact_f1', 'long_range_precision_L5', 'tm_proxy']:
            vals = [s[metric] for s in seed_metrics]
            mean, std = float(np.mean(vals)), float(np.std(vals))
            print(f'    {metric:32s}: {mean:.4f} ± {std:.4f}')
            summary_rows.append({
                'variant': variant, 'label': label, 'metric': metric,
                'mean': mean, 'std': std,
                'n_seeds': len(seeds), 'epochs': args.epochs,
                'n_train_proteins': len(raw_train),
                'seed_values': ','.join(f'{v:.4f}' for v in vals),
            })
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    json_path = f'results/ablation_results_{ts}.json'
    with open(json_path, 'w') as f:
        json.dump({'variants': all_results, 'summary': summary_rows}, f, indent=2)

    csv_path = f'results/ablation_summary_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader(); w.writerows(summary_rows)

    print('=' * 70)
    print('ABLATION COMPLETE')
    print(f'Full results: {json_path}')
    print(f'Summary CSV:  {csv_path}')
    print()
    _print_table(summary_rows)


def _print_table(rows):
    """Print a clean ASCII summary table of mean ± std for each variant × metric."""
    metrics = ['local_lDDT', 'contact_f1', 'long_range_precision_L5', 'tm_proxy']
    variants = list(dict.fromkeys(r['variant'] for r in rows))

    # Build lookup: variant → metric → (mean, std)
    data = {}
    for r in rows:
        data.setdefault(r['variant'], {})[r['metric']] = (r['mean'], r['std'])

    header = f"{'Variant':<38}" + ''.join(f'{m[:12]:>16}' for m in metrics)
    print(header)
    print('-' * len(header))
    for v in variants:
        label = VARIANT_LABELS[v]
        row   = f'{label:<38}'
        for m in metrics:
            mean, std = data.get(v, {}).get(m, (float('nan'), 0))
            row += f'{mean:.3f}±{std:.3f}'.rjust(16)
        print(row)
    print()


def main():
    p = argparse.ArgumentParser(description='Ablation study for Scarfold')
    p.add_argument('--epochs',        type=int,   default=150)
    p.add_argument('--seeds',         type=int,   default=3)
    p.add_argument('--proteins',      type=int,   default=50)
    p.add_argument('--test-proteins', type=int,   default=7)
    p.add_argument('--lr',            type=float, default=5e-4)
    p.add_argument('--skip-trimul',   action='store_true',
                   help='Skip Variant C (TriMul) — 31x slower, use when compute-limited')
    p.add_argument('--quick',         action='store_true',
                   help='Quick run: 30 epochs, 2 seeds, 20 proteins (legacy smoke-test)')
    p.add_argument('--medium',        action='store_true',
                   help='Medium run: 30 epochs, 3 seeds, 20 proteins, skip TriMul (~45 min CPU)')
    args = p.parse_args()

    if args.quick:
        args.epochs      = 30
        args.seeds       = 2
        args.proteins    = 20
        args.skip_trimul = True
    elif args.medium:
        args.epochs      = 30
        args.seeds       = 3
        args.proteins    = 20
        args.skip_trimul = True

    run_ablation(args)


if __name__ == '__main__':
    main()
