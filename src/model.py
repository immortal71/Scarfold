import torch
import torch.nn as nn
import numpy as np

# Default feature dimension when using rich_encoding (one-hot + BLOSUM62 + physicochemical)
RICH_AA_DIM = 48

# ── Distogram binning (AlphaFold-style) ──────────────────────────────────────
# 64 bins covering 2–22 Å, plus one "too-far" bin  →  65 bins total.
# Using cross-entropy over bins gives sharper, better-calibrated contact signal
# than direct MSE regression.
DIST_MIN = 2.0
DIST_MAX = 22.0
NUM_BINS = 64

def get_bin_edges():
    return np.linspace(DIST_MIN, DIST_MAX, NUM_BINS + 1, dtype=np.float32)

def dist_to_bin(dist_mat, bin_edges=None):
    """Convert an (L,L) distance matrix to integer bin indices (0..NUM_BINS).
    Distances >= DIST_MAX fall into the last bin (index NUM_BINS).
    """
    if bin_edges is None:
        bin_edges = get_bin_edges()
    bins = np.digitize(dist_mat, bin_edges[1:]).astype(np.int64)  # 0..NUM_BINS
    bins = np.clip(bins, 0, NUM_BINS)
    return bins

def bin_to_dist(logits, bin_edges=None):
    """Convert (L,L,NUM_BINS+1) logits to expected distance (L,L) via softmax."""
    if bin_edges is None:
        bin_edges = get_bin_edges()
    centres_np = np.concatenate([
        0.5 * (bin_edges[:-1] + bin_edges[1:]),   # midpoints of 64 bins
        [DIST_MAX + 2.0],                           # "too far" bin centre
    ])
    centres = torch.tensor(centres_np, dtype=logits.dtype, device=logits.device)
    probs = torch.softmax(logits, dim=-1)           # (..., NUM_BINS+1)
    expected = (probs * centres).sum(-1)            # (...)
    return expected


class DistancePredictor(nn.Module):
    """Basic fully-connected distogram predictor (MLP baseline).

    Outputs (B, L, L, NUM_BINS+1) logits over distance bins instead of a
    single scalar — this is far better calibrated than direct regression.
    """
    def __init__(self, seq_len, aa_dim=RICH_AA_DIM, hidden=1024, n_bins=NUM_BINS + 1):
        super().__init__()
        in_dim = seq_len * aa_dim
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, seq_len * seq_len * n_bins),
        )
        self.seq_len = seq_len
        self.aa_dim = aa_dim

    def forward(self, x):
        B = x.shape[0]
        L = self.seq_len
        x = x.reshape(B, -1)
        out = self.net(x).reshape(B, L, L, self.n_bins)
        # symmetrise logits
        out = (out + out.transpose(1, 2)) / 2.0
        return out   # raw logits → caller uses bin_to_dist() or cross-entropy


class _PairBiasTransformerLayer(nn.Module):
    """Single Evoformer-lite layer: single-sequence track + pair bias.

    The pair bias lets the attention pattern be informed by the current pair
    representation — a key idea from AlphaFold2's Evoformer.

    Uses manual scaled dot-product attention so the (B, nhead, L, L) pair bias
    is added *per sample per head* without any batch averaging.
    """
    def __init__(self, hidden, pair_dim, nhead, dropout=0.1):
        super().__init__()
        assert hidden % nhead == 0, f'hidden ({hidden}) must be divisible by nhead ({nhead})'
        self.nhead = nhead
        self.head_dim = hidden // nhead
        self.norm_seq = nn.LayerNorm(hidden)
        self.norm_pair = nn.LayerNorm(pair_dim)
        # Project pair features to per-head bias (one scalar per head per pair)
        self.pair_to_bias = nn.Linear(pair_dim, nhead, bias=False)
        # Manual QKV projections (replaces nn.MultiheadAttention for correct batching)
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden)
        self.attn_dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 4, hidden),
            nn.Dropout(dropout),
        )
        # Pair update: outer-product mean (simplified triangle update)
        self.pair_update = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, pair_dim),
        )
        self.outer_proj = nn.Linear(hidden * 2, pair_dim)

    def forward(self, seq, pair):
        """
        seq:  (B, L, hidden)
        pair: (B, L, L, pair_dim)
        """
        B, L, H = seq.shape
        hd = self.head_dim

        # Pre-LN on sequence track
        seq_n = self.norm_seq(seq)

        # Pair bias: (B, L, L, nhead) → (B, nhead, L, L) — per sample, per head
        bias = self.pair_to_bias(self.norm_pair(pair)).permute(0, 3, 1, 2)

        # Scaled dot-product attention with correct per-batch pair bias
        scale = hd ** -0.5
        Q = self.q_proj(seq_n).view(B, L, self.nhead, hd).transpose(1, 2)  # (B, nhead, L, hd)
        K = self.k_proj(seq_n).view(B, L, self.nhead, hd).transpose(1, 2)
        V = self.v_proj(seq_n).view(B, L, self.nhead, hd).transpose(1, 2)

        attn_scores = torch.einsum('bhid,bhjd->bhij', Q, K) * scale   # (B, nhead, L, L)
        attn_scores = attn_scores + bias                               # add pair bias correctly
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.einsum('bhij,bhjd->bhid', attn_weights, V)        # (B, nhead, L, hd)
        out = out.transpose(1, 2).reshape(B, L, H)                    # (B, L, H)
        out = self.out_proj(out)

        seq = seq + out
        seq = seq + self.ff(seq)

        # Update pair representation with outer-product mean
        h_i = seq.unsqueeze(2).expand(B, L, L, -1)
        h_j = seq.unsqueeze(1).expand(B, L, L, -1)
        outer = self.outer_proj(torch.cat([h_i, h_j], dim=-1))
        pair = pair + self.pair_update(outer)
        return seq, pair


class TransformerDistancePredictor(nn.Module):
    """Evoformer-lite Transformer with recycling, secondary structure, and confidence heads.

    Implements key ideas from AlphaFold2:
    - **Pair-biased attention**: pair(i,j) features directly bias residue-track attention
    - **Pair track update**: pair representation updated at every Evoformer-lite layer
    - **Recycling** (AlphaFold2 Section 1.8): distogram logits are fed back into the
      pair representation for N_recycles passes (stop-gradient), allowing iterative
      refinement of the prediction.
    - **Secondary structure head**: 3-class (coil/helix/strand) auxiliary supervision
      derived from Cα geometry — provides additional training signal without labels.
    - **pLDDT confidence head**: 4-bin per-residue confidence prediction, trained to
      predict the model's own distance error. Matches AlphaFold2's plDDT head concept.

    API:
        forward(x)           → (B, L, L, n_bins) distogram logits  [backward-compat]
        forward_full(x)      → (logits, ss_logits, plddt_logits)
    """
    def __init__(self, seq_len, aa_dim=RICH_AA_DIM, hidden=256, pair_dim=64,
                 nhead=4, num_layers=4, n_bins=NUM_BINS + 1, dropout=0.1,
                 num_recycles=3):
        super().__init__()
        self.seq_len = seq_len
        self.aa_dim = aa_dim
        self.n_bins = n_bins
        self.num_recycles = num_recycles

        # Residue-level input projection + positional embedding
        self.residue_proj = nn.Linear(aa_dim, hidden)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)

        # Initial pair embedding: outer sum of projected residues
        self.pair_init_i = nn.Linear(hidden, pair_dim)
        self.pair_init_j = nn.Linear(hidden, pair_dim)
        # Relative-position embedding (binned sequence separation, clipped ±32)
        max_rel_pos = 32
        self.rel_pos_embed = nn.Embedding(2 * max_rel_pos + 2, pair_dim)
        self.max_rel_pos = max_rel_pos

        # Stack of Evoformer-lite layers
        self.layers = nn.ModuleList([
            _PairBiasTransformerLayer(hidden, pair_dim, nhead, dropout)
            for _ in range(num_layers)
        ])

        # Distogram head: pair_dim → n_bins logits per residue pair
        self.distogram_head = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, pair_dim),
            nn.GELU(),
            nn.Linear(pair_dim, n_bins),
        )

        # Recycling gate: project previous distogram output back to pair_dim
        self.recycle_pair_proj = nn.Linear(n_bins, pair_dim)

        # Aux head 1: per-residue secondary structure (3 classes: coil / helix / strand)
        self.ss_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # Aux head 2: per-residue pLDDT confidence (4 bins: >4 / 2-4 / 1-2 / <1 Å error)
        self.plddt_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def _rel_pos(self, L, device):
        """Clipped relative-position index matrix, shape (L, L)."""
        idx = torch.arange(L, device=device)
        rel = (idx.unsqueeze(0) - idx.unsqueeze(1)).clamp(-self.max_rel_pos, self.max_rel_pos)
        rel = rel + self.max_rel_pos   # shift to [0, 2*max_rel_pos]
        return rel

    def _init_pair_repr(self, h, L, device):
        """Build initial pair representation from residue embeddings + relative position."""
        pi = self.pair_init_i(h)
        pj = self.pair_init_j(h)
        pair = pi.unsqueeze(2) + pj.unsqueeze(1)                    # (B, L, L, pair_dim)
        rel_idx = self._rel_pos(L, device)
        pair = pair + self.rel_pos_embed(rel_idx).unsqueeze(0)       # broadcast B
        return pair

    def forward_full(self, x, num_recycles=None):
        """Full forward pass with recycling and all auxiliary outputs.

        Args:
            x:            (B, L, aa_dim) input sequence features
            num_recycles: override self.num_recycles if provided

        Returns:
            logits:       (B, L, L, n_bins)  distogram logits
            ss_logits:    (B, L, 3)           secondary structure (coil/helix/strand)
            plddt_logits: (B, L, 4)           per-residue pLDDT confidence bins
        """
        B, L, _ = x.shape
        n_rec = num_recycles if num_recycles is not None else self.num_recycles

        # Residue embedding — fixed across recycles (single-sequence Evoformer convention)
        h_init = self.residue_proj(x) + self.pos_embed[:L].unsqueeze(0)   # (B, L, hidden)

        # Initial pair representation
        pair = self._init_pair_repr(h_init, L, x.device)   # (B, L, L, pair_dim)

        for cycle in range(max(n_rec, 1)):
            # Reinitialise residue track at each recycle
            h = h_init
            for layer in self.layers:
                h, pair = layer(h, pair)

            if cycle < max(n_rec, 1) - 1:
                # Recycle: update pair with stop-gradient distogram from this cycle
                logits_prev = self.distogram_head(pair)
                logits_prev = (logits_prev + logits_prev.transpose(1, 2)) / 2.0
                pair = pair + self.recycle_pair_proj(logits_prev.detach())

        # Final outputs from last cycle
        logits = self.distogram_head(pair)
        logits = (logits + logits.transpose(1, 2)) / 2.0   # symmetrise  (B, L, L, n_bins)
        ss_logits = self.ss_head(h)                         # (B, L, 3)
        plddt_logits = self.plddt_head(h)                   # (B, L, 4)
        return logits, ss_logits, plddt_logits

    def forward(self, x, num_recycles=None):
        """Standard forward — returns distogram logits only (backward-compatible)."""
        logits, _, _ = self.forward_full(x, num_recycles=num_recycles)
        return logits   # (B, L, L, n_bins)


# ── Auxiliary target derivation ──────────────────────────────────────────────

def ss_labels_from_dists(dist_mat):
    """Derive 3-class secondary structure labels from a Cα distance matrix.

    Labels: 0 = coil, 1 = helix (α-helix), 2 = strand (β-extended)

    Rules (Chou & Fasman, 1974; standard Cα distance geometry):
      α-helix:  d(i, i+3) < 6.0 Å  AND  d(i, i+4) < 7.0 Å   (ideal: ~5.5, 6.2 Å)
      β-strand: d(i, i+2) > 5.5 Å  (extended chain; ideal β: ~6.7 Å for i+2)
      coil:     all other residues
    """
    L = dist_mat.shape[0]
    labels = np.zeros(L, dtype=np.int64)
    for i in range(L):
        d3 = float(dist_mat[i, i + 3]) if i + 3 < L else 99.0
        d4 = float(dist_mat[i, i + 4]) if i + 4 < L else 99.0
        d2 = float(dist_mat[i, i + 2]) if i + 2 < L else 99.0
        if d3 < 6.0 and d4 < 7.0:   # α-helix geometry
            labels[i] = 1
        elif d2 > 5.5:               # extended/β-strand geometry
            labels[i] = 2
        # else: labels[i] = 0 (coil)
    return labels


def plddt_bins_from_dist_error(pred_dists, true_dists):
    """Per-residue pLDDT bin labels (0-3) derived from mean absolute distance error.

    Bins match AlphaFold2's pLDDT colour-coding convention:
      3: mean error < 1 Å  (very high confidence — blue)
      2: mean error 1–2 Å  (confident — yellow)
      1: mean error 2–4 Å  (low confidence — orange)
      0: mean error > 4 Å  (very low confidence — red)
    """
    L = pred_dists.shape[0]
    mask = 1.0 - np.eye(L)
    mae = (np.abs(pred_dists - true_dists) * mask).sum(1) / (mask.sum(1) + 1e-8)  # (L,)
    bins = np.zeros(L, dtype=np.int64)
    bins[mae < 4.0] = 1
    bins[mae < 2.0] = 2
    bins[mae < 1.0] = 3
    return bins


def get_model(model_type, seq_len, aa_dim=RICH_AA_DIM, num_recycles=3):
    if model_type.lower() == 'transformer':
        return TransformerDistancePredictor(seq_len, aa_dim=aa_dim, num_recycles=num_recycles)
    return DistancePredictor(seq_len, aa_dim=aa_dim)


# ── Loss functions ────────────────────────────────────────────────────────────

def distogram_loss(logits, true_dists):
    """Cross-entropy over 64+1 distance bins.

    This is the same objective used by AlphaFold (distogram head).
    It gives much sharper gradients for contact prediction than MSE regression.

    Args:
        logits:     (B, L, L, NUM_BINS+1) — raw logits from the model
        true_dists: (B, L, L) — ground-truth distance matrices (Å)
    """
    bin_edges = torch.tensor(get_bin_edges(), device=logits.device)
    # convert continuous distances to bin indices
    # digitize: bins[i] = k if edges[k] <= d < edges[k+1]
    B, L, _, n_bins = logits.shape
    true_bins = torch.bucketize(true_dists, bin_edges[1:]).clamp(0, n_bins - 1).long()  # (B,L,L)
    # flatten spatial dims for cross-entropy
    logits_flat = logits.reshape(B * L * L, n_bins)
    target_flat = true_bins.reshape(B * L * L)
    # ignore self-distances (diagonal) — they're always 0
    diag_mask = torch.eye(L, device=logits.device, dtype=torch.bool)
    diag_mask = diag_mask.unsqueeze(0).expand(B, -1, -1).reshape(B * L * L)
    logits_flat = logits_flat[~diag_mask]
    target_flat = target_flat[~diag_mask]
    return nn.functional.cross_entropy(logits_flat, target_flat)


def _contact_bce_loss(logits_or_dists, true_dists, threshold=8.0, is_logits=True):
    """Binary contact BCE loss.

    If *is_logits* is True, convert distogram logits to expected distances first.
    """
    if is_logits:
        pred_dists = bin_to_dist(logits_or_dists)   # (B,L,L)
    else:
        pred_dists = logits_or_dists
    pred_contact = torch.sigmoid((threshold - pred_dists) / 2.0)
    true_contact = (true_dists < threshold).float()
    L = pred_dists.shape[-1]
    mask = 1.0 - torch.eye(L, device=pred_dists.device).unsqueeze(0)
    loss = nn.functional.binary_cross_entropy(
        pred_contact * mask, true_contact * mask, reduction='sum'
    ) / (mask.sum() + 1e-8)
    return loss


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, X, Y, optimizer, device='cpu', contact_weight=0.5,
                ss_weight=0.2, plddt_weight=0.1):
    """One step: distogram CE + contact BCE + (Transformer only) SS + pLDDT aux losses.

    For the MLP, only distogram CE and contact BCE are used.
    For the Transformer, two additional auxiliary heads are trained:
      - Secondary structure (coil/helix/strand) derived from Cα distance geometry
      - Per-residue pLDDT confidence bins derived from mean absolute distance error
    Both auxiliary losses provide extra gradient signal without requiring external labels.
    """
    model.train()
    model.to(device)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
    optimizer.zero_grad()

    is_transformer = hasattr(model, 'forward_full')

    if is_transformer:
        logits, ss_logits, plddt_logits = model.forward_full(X_t)
    else:
        logits = model(X_t)                   # (B, L, L, n_bins)

    loss_dg = distogram_loss(logits, Y_t)
    loss_cb = _contact_bce_loss(logits, Y_t, is_logits=True)
    loss = loss_dg + contact_weight * loss_cb

    if is_transformer:
        B = X.shape[0]
        # ── Secondary-structure auxiliary loss ──────────────────────────────
        if ss_weight > 0.0:
            ss_lbl = np.stack([ss_labels_from_dists(Y[b]) for b in range(B)])  # (B, L)
            ss_lbl_t = torch.tensor(ss_lbl, dtype=torch.long, device=device)
            loss_ss = nn.functional.cross_entropy(ss_logits.reshape(-1, 3), ss_lbl_t.reshape(-1))
            loss = loss + ss_weight * loss_ss
        # ── pLDDT confidence auxiliary loss ─────────────────────────────────
        if plddt_weight > 0.0:
            with torch.no_grad():
                pred_d_np = bin_to_dist(logits).cpu().numpy()        # (B, L, L)
            pbin = np.stack([plddt_bins_from_dist_error(pred_d_np[b], Y[b]) for b in range(B)])
            pbin_t = torch.tensor(pbin, dtype=torch.long, device=device)
            loss_pl = nn.functional.cross_entropy(plddt_logits.reshape(-1, 4), pbin_t.reshape(-1))
            loss = loss + plddt_weight * loss_pl

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item()), float(loss_dg.item()), float(loss_cb.item())


def train_simple(model, seqs_onehot, targets, epochs=30, lr=1e-3, device='cpu',
                 contact_weight=0.5, contact_threshold=8.0, batch_size=16,
                 verbose=False):
    """Train the model on a dataset, using mini-batches and cosine LR annealing.

    Upgraded from full-batch to mini-batch training — essential for real-PDB
    datasets where each protein may have a different length (use batch_size=1
    for variable-length sequences, or pad to a fixed length first).
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    N = len(seqs_onehot)
    rng = np.random.default_rng(0)
    for ep in range(epochs):
        indices = rng.permutation(N)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            batch_idx = indices[start:start + batch_size]
            X_b = seqs_onehot[batch_idx]
            Y_b = targets[batch_idx]
            loss, *_ = train_epoch(model, X_b, Y_b, opt, device=device,
                                   contact_weight=contact_weight)
            ep_loss += loss
            n_batches += 1
        scheduler.step()
        if verbose and (ep + 1) % 10 == 0:
            print(f'  epoch {ep+1}/{epochs}  loss={ep_loss/n_batches:.4f}')
    return model


def predict(model, seq_onehot, device='cpu'):
    """Run inference and return expected distance matrix (L, L) in Å."""
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq_onehot[None], dtype=torch.float32, device=device)
        logits = model(x)[0]   # (L, L, n_bins)
        dist = bin_to_dist(logits).cpu().numpy()
    return dist


def predict_with_confidence(model, seq_onehot, device='cpu'):
    """Run inference and return (distance_matrix, per-residue pLDDT score 0-100).

    The pLDDT confidence score comes from the model's own pLDDT head (Transformer)
    or is set to a uniform 70.0 for the MLP baseline.

    Bin midpoints: [0→25, 1→60, 2→80, 3→95] matching AlphaFold2 colour coding.

    Returns:
        dist:   np.array (L, L)  expected distance matrix in Å
        plddt:  np.array (L,)    per-residue pLDDT confidence (0-100)
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq_onehot[None], dtype=torch.float32, device=device)
        if hasattr(model, 'forward_full'):
            logits, _, plddt_logits = model.forward_full(x)
            probs = torch.softmax(plddt_logits[0], dim=-1).cpu().numpy()  # (L, 4)
            bin_midpoints = np.array([25.0, 60.0, 80.0, 95.0])
            plddt = (probs * bin_midpoints).sum(-1)                        # (L,)
        else:
            logits = model(x)
            plddt = np.full(seq_onehot.shape[0], 70.0, dtype=np.float32)
        dist = bin_to_dist(logits[0]).cpu().numpy()
    return dist, plddt


def contact_map_score(pred_dists, true_dists, threshold=8.0):
    from src.utils import contact_map_metrics
    return contact_map_metrics(pred_dists, true_dists, threshold=threshold)


def pseudo_plddt(pred_dists, true_dists):
    from src.utils import compute_plddt_from_distances
    return compute_plddt_from_distances(pred_dists, true_dists)


def save_model(model, path):
    """Save trained model weights and metadata."""
    torch.save({'state_dict': model.state_dict(), 'aa_dim': model.aa_dim}, path)



def _infer_seq_len_from_state_dict(state_dict, model_type):
    if model_type.lower() == 'transformer' and 'pos_embed' in state_dict:
        return state_dict['pos_embed'].shape[0]
    if 'net.0.weight' in state_dict:
        in_dim = state_dict['net.0.weight'].shape[1]
        # may be 20 (old one-hot) or 48 (rich encoding) — divide by stored aa_dim
        return None  # caller will try both
    raise ValueError('Unable to infer seq_len from state dict')


def load_model(model_type, seq_len, path, device='cpu'):
    """Load a saved model — handles old (plain state_dict) and new (dict+aa_dim) formats."""
    raw = torch.load(path, map_location=device, weights_only=False)

    if isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
        aa_dim = raw.get('aa_dim', RICH_AA_DIM)
    else:
        state = raw
        aa_dim = 20   # legacy one-hot only

    # Try to infer seq_len from saved weights
    inferred_len = None
    if model_type.lower() == 'transformer' and 'pos_embed' in state:
        inferred_len = state['pos_embed'].shape[0]
    elif 'net.0.weight' in state:
        inferred_len = state['net.0.weight'].shape[1] // aa_dim

    try:
        model = get_model(model_type, seq_len, aa_dim=aa_dim)
        # strict=False: tolerates new aux heads (ss_head, plddt_head, recycle_pair_proj)
        # that may be absent in older checkpoints; new heads stay randomly initialised.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f'  Note: {len(missing)} new parameters initialised randomly '
                  f'(new arch features: SS head, pLDDT head, recycling).')
    except RuntimeError:
        if inferred_len is None:
            raise
        print(f'  Note: resizing model seq_len {seq_len}→{inferred_len} to match saved weights.')
        model = get_model(model_type, inferred_len, aa_dim=aa_dim)
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model

