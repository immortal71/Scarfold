import torch
import torch.nn as nn
import numpy as np

# Default feature dimension when using rich_encoding (one-hot + BLOSUM62 + physicochemical)
RICH_AA_DIM = 48


class DistancePredictor(nn.Module):
    """Basic fully-connected distance predictor (baseline)."""
    def __init__(self, seq_len, aa_dim=RICH_AA_DIM, hidden=1024):
        super().__init__()
        in_dim = seq_len * aa_dim
        out_dim = seq_len * seq_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, out_dim),
        )
        self.seq_len = seq_len
        self.aa_dim = aa_dim

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        out = self.net(x)
        out = out.reshape(B, self.seq_len, self.seq_len)
        out = (out + out.transpose(1, 2)) / 2.0
        out = torch.relu(out) + 1e-3
        return out


class TransformerDistancePredictor(nn.Module):
    """Transformer-based distance predictor with richer pair features."""
    def __init__(self, seq_len, aa_dim=RICH_AA_DIM, hidden=256, nhead=4, num_layers=3):
        super().__init__()
        self.seq_len = seq_len
        self.aa_dim = aa_dim
        self.residue_proj = nn.Linear(aa_dim, hidden)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
            activation='gelu', dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Pair MLP with residual — captures interactions between every residue pair
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        B, L, _ = x.shape
        h = self.residue_proj(x) + self.pos_embed[:L].unsqueeze(0)
        h = self.transformer(h)

        h_i = h.unsqueeze(2).expand(B, L, L, -1)
        h_j = h.unsqueeze(1).expand(B, L, L, -1)
        pair = torch.cat([h_i, h_j], dim=-1)

        dist = self.pair_mlp(pair).squeeze(-1)
        dist = (dist + dist.transpose(1, 2)) / 2.0
        dist = torch.relu(dist) + 1e-3
        return dist


def get_model(model_type, seq_len, aa_dim=RICH_AA_DIM):
    if model_type.lower() == 'transformer':
        return TransformerDistancePredictor(seq_len, aa_dim=aa_dim)
    return DistancePredictor(seq_len, aa_dim=aa_dim)


def _contact_bce_loss(pred_dists, true_dists, threshold=8.0):
    """Binary cross-entropy loss on contact predictions (< threshold Å).

    Adds an explicit signal for short-range contacts on top of MSE, which
    greatly improves contact-map F1.
    """
    pred_contact = torch.sigmoid((threshold - pred_dists) / 2.0)
    true_contact = (true_dists < threshold).float()
    # ignore diagonal
    L = pred_dists.shape[-1]
    mask = 1.0 - torch.eye(L, device=pred_dists.device).unsqueeze(0)
    loss = nn.functional.binary_cross_entropy(
        pred_contact * mask, true_contact * mask, reduction='sum'
    ) / (mask.sum() + 1e-8)
    return loss


def train_simple(model, seqs_onehot, targets, epochs=30, lr=1e-3, device='cpu',
                 contact_weight=0.5, contact_threshold=8.0):
    """Train with combined MSE + contact BCE loss and cosine LR annealing."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    X = torch.tensor(seqs_onehot, dtype=torch.float32).to(device)
    Y = torch.tensor(targets, dtype=torch.float32).to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        mse = nn.functional.mse_loss(pred, Y)
        cbc = _contact_bce_loss(pred, Y, threshold=contact_threshold)
        loss = mse + contact_weight * cbc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()
    return model


def predict(model, seq_onehot, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = torch.tensor(seq_onehot[None], dtype=torch.float32).to(device)
        out = model(x)[0].cpu().numpy()
    return out


def contact_map_score(pred_dists, true_dists, threshold=8.0):
    """Pseudo-AlphaFold-style contact map score (F1) for predicted inter-residue distances."""
    from src.utils import contact_map_metrics
    metrics = contact_map_metrics(pred_dists, true_dists, threshold=threshold)
    return metrics


def pseudo_plddt(pred_dists, true_dists):
    """Simplified confidence scoring function from distance agreement."""
    from src.utils import compute_plddt_from_distances
    return compute_plddt_from_distances(pred_dists, true_dists)


def save_model(model, path):
    """Save trained model weights and its aa_dim so it can be reloaded correctly."""
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
    """Load model — supports both old (state_dict only) and new (dict with aa_dim) formats."""
    raw = torch.load(path, map_location=device)

    # New save format: dict with 'state_dict' and 'aa_dim'
    if isinstance(raw, dict) and 'state_dict' in raw:
        state = raw['state_dict']
        aa_dim = raw.get('aa_dim', RICH_AA_DIM)
    else:
        # Legacy format: plain state_dict
        state = raw
        aa_dim = 20  # old models were trained with plain one-hot

    # Infer seq_len from weights when possible
    inferred_len = None
    if model_type.lower() == 'transformer' and 'pos_embed' in state:
        inferred_len = state['pos_embed'].shape[0]
    elif 'net.0.weight' in state:
        in_dim = state['net.0.weight'].shape[1]
        inferred_len = in_dim // aa_dim

    try:
        model = get_model(model_type, seq_len, aa_dim=aa_dim)
        model.load_state_dict(state)
    except RuntimeError as error:
        if inferred_len is None:
            raise RuntimeError(f'Failed to load model weights for {model_type} seq_len {seq_len}: {error}')
        print(f'Warning: model weights mismatch for seq_len={seq_len}. '
              f'Using saved seq_len={inferred_len} from weights.')
        model = get_model(model_type, inferred_len, aa_dim=aa_dim)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model

