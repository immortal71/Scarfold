import torch
import torch.nn as nn
import numpy as np


class DistancePredictor(nn.Module):
    """Basic fully-connected distance predictor (baseline)."""
    def __init__(self, seq_len, aa_dim=20, hidden=1024):
        super().__init__()
        in_dim = seq_len * aa_dim
        out_dim = seq_len * seq_len
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, out_dim),
        )
        self.seq_len = seq_len

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        out = self.net(x)
        out = out.reshape(B, self.seq_len, self.seq_len)
        out = (out + out.transpose(1, 2)) / 2.0
        out = torch.relu(out) + 1e-3
        return out


class TransformerDistancePredictor(nn.Module):
    """Small transformer-based distance predictor for better sequence context."""
    def __init__(self, seq_len, aa_dim=20, hidden=256, nhead=4, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.residue_proj = nn.Linear(aa_dim, hidden)
        self.pos_embed = nn.Parameter(torch.randn(seq_len, hidden) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
                                                   activation='gelu', dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pair_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        # x: (B, L, aa_dim)
        B, L, _ = x.shape
        h = self.residue_proj(x) + self.pos_embed.unsqueeze(0)
        h = self.transformer(h)

        h_i = h.unsqueeze(2).expand(B, L, L, -1)
        h_j = h.unsqueeze(1).expand(B, L, L, -1)
        pair = torch.cat([h_i, h_j], dim=-1)

        dist = self.pair_mlp(pair).squeeze(-1)
        dist = (dist + dist.transpose(1, 2)) / 2.0
        dist = torch.relu(dist) + 1e-3
        return dist


def get_model(model_type, seq_len):
    if model_type.lower() == 'transformer':
        return TransformerDistancePredictor(seq_len)
    return DistancePredictor(seq_len)


def train_simple(model, seqs_onehot, targets, epochs=30, lr=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    X = torch.tensor(seqs_onehot, dtype=torch.float32).to(device)
    Y = torch.tensor(targets, dtype=torch.float32).to(device)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
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
    """Save trained model weights."""
    torch.save(model.state_dict(), path)


def _infer_seq_len_from_state_dict(state_dict, model_type):
    if model_type.lower() == 'transformer' and 'pos_embed' in state_dict:
        return state_dict['pos_embed'].shape[0]
    if 'net.0.weight' in state_dict:
        in_dim = state_dict['net.0.weight'].shape[1]
        return in_dim // 20
    raise ValueError('Unable to infer seq_len from state dict')


def load_model(model_type, seq_len, path, device='cpu'):
    """Load model parameters for given model type and sequence length.

    If the model weights were trained at a different sequence length, loads the saved
    length and returns that model. The caller may need to truncate input sequences
    to model.seq_len.
    """
    state = torch.load(path, map_location=device)
    inferred_len = None
    try:
        inferred_len = _infer_seq_len_from_state_dict(state, model_type)
    except Exception:
        pass

    try:
        model = get_model(model_type, seq_len)
        model.load_state_dict(state)
    except RuntimeError as error:
        if inferred_len is None:
            raise RuntimeError(f'Failed to load model weights for {model_type} seq_len {seq_len}: {error}')
        print(f'Warning: model weights mismatch for seq_len={seq_len}. ' 
              f'Using saved seq_len={inferred_len} from weights.')
        model = get_model(model_type, inferred_len)
        model.load_state_dict(state)
        seq_len = inferred_len

    model.to(device)
    model.eval()
    return model
