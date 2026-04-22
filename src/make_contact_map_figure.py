#!/usr/bin/env python3
"""make_contact_map_figure.py — Generate publication-quality contact map figure.

Creates a 3-panel PNG figure for report/figures/contact_map_1crn.png comparing:
  (A) True binary contact map for 1CRN
  (B) v4 model (ours) predicted contact probability with true contact overlay
  (C) Seq-distance baseline (D) predicted contact probability with true contact overlay

The visual contrast between panels B and C motivates the paper's main claim:
  – D (C baseline) is bright only near the diagonal (sequential contacts)
  – v4 (B) detects some non-sequential long-range contacts
  – True contacts (red dots) show where the model is right/wrong

Usage:
    python src/make_contact_map_figure.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server/CI use
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)

from src import utils, model as md

THRESHOLD = 8.0   # Å, standard contact definition
MIN_SEP   = 6     # only show contacts with |i-j| >= MIN_SEP in overlay


def seq_distance_baseline(L):
    """Predict Cα–Cα distances from sequence separation alone.

    Uses the same polynomial calibrated in ablation_study.py:
      d(|i-j|) = 2.0 + 2.5 × |i-j|^0.55
    This is the zero-learning baseline (Variant D).
    """
    separations = np.arange(L, dtype=np.float32)
    d_k = 2.0 + 2.5 * np.maximum(separations, 1) ** 0.55
    d_k[0] = 0.0   # self-distance is 0
    dist_mat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        for j in range(L):
            dist_mat[i, j] = d_k[abs(i - j)]
    return dist_mat


def dist_to_contact_prob(dist_matrix, steepness=1.5):
    """Soft contact probability via logistic function centred at threshold.

    Lower distance → higher probability. Steepness controls sharpness of transition.
    """
    return 1.0 / (1.0 + np.exp(steepness * (dist_matrix - THRESHOLD)))


def make_figure():
    os.chdir(REPO_ROOT)

    # ── Load 1CRN ─────────────────────────────────────────────────────────────
    print('Fetching 1CRN (crambin) from PDB...')
    pdb_path = utils.fetch_pdb('1crn')
    seq      = utils.pdb_sequence(pdb_path, chain='A', max_residues=60)
    coords   = utils.pdb_ca_coords(pdb_path, chain='A', max_residues=60)
    L        = len(seq)
    print(f'  Length: {L}, Sequence: {seq}')

    # ── True contact map ──────────────────────────────────────────────────────
    true_dist    = utils.coords_to_distances(coords)
    true_contact = (true_dist < THRESHOLD)
    np.fill_diagonal(true_contact, False)

    # ── v4 model predictions ──────────────────────────────────────────────────
    print('Loading v4 model (checkpoints/best_pdb_v4.pt)...')
    v4_model  = md.load_model('transformer', seq_len=L,
                              path='checkpoints/best_pdb_v4.pt')
    enc       = utils.rich_encoding(seq)
    pred_dist = md.predict(v4_model, enc)
    pred_dist = 0.5 * (pred_dist + pred_dist.T)   # symmetrize
    pred_prob = dist_to_contact_prob(pred_dist)

    # ── D seq-distance baseline ───────────────────────────────────────────────
    baseline_dist = seq_distance_baseline(L)
    baseline_prob = dist_to_contact_prob(baseline_dist)

    # ── Compute metrics ───────────────────────────────────────────────────────
    m_v4 = md.contact_map_score(pred_dist[:L, :L], true_dist[:L, :L])
    m_d  = md.contact_map_score(baseline_dist[:L, :L], true_dist[:L, :L])
    print(f'\nMetrics on 1CRN (L={L}):')
    print(f'  v4 model : F1={m_v4["f1"]:.3f}  LR P@L/5={m_v4["long_range_precision_L5"]:.3f}')
    print(f'  D baseln : F1={m_d["f1"]:.3f}  LR P@L/5={m_d["long_range_precision_L5"]:.3f}')

    # ── True contact positions to overlay (|i-j| >= MIN_SEP) ─────────────────
    i_tc, j_tc = np.where(np.triu(true_contact, k=MIN_SEP))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))
    fig.patch.set_facecolor('white')

    cmap_prob   = 'Blues'
    cmap_true   = 'Greys'
    dot_color   = '#d62728'   # red for true contact overlay dots
    dot_size    = 12
    dot_alpha   = 0.85

    # ── Panel A: True contact map ─────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(true_contact.astype(float), cmap=cmap_true, vmin=0, vmax=1,
              origin='upper', aspect='equal')
    ax.set_title('(A) True contact map\n(Cα–Cα < 8 Å, |i−j| ≥ 1)',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Residue j', fontsize=10)
    ax.set_ylabel('Residue i', fontsize=10)
    # Annotate long-range region boundary
    ax.axhline(y=12, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.axvline(x=12, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.text(14, 1, '|i−j|≥12\n(long-range)', fontsize=7, color='darkorange',
            va='top')

    # ── Panel B: v4 model predicted probability with true contact overlay ─────
    ax = axes[1]
    im = ax.imshow(pred_prob, cmap=cmap_prob, vmin=0, vmax=1,
                   origin='upper', aspect='equal')
    ax.scatter(j_tc, i_tc, c=dot_color, s=dot_size, alpha=dot_alpha,
               linewidths=0, zorder=5, label='True contact')
    ax.set_title(f'(B) v4 model (ours)\nF1={m_v4["f1"]:.3f}  LR P@L/5={m_v4["long_range_precision_L5"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Residue j', fontsize=10)
    ax.set_ylabel('Residue i', fontsize=10)
    ax.axhline(y=12, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.axvline(x=12, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='P(contact)')
    ax.legend(handles=[mpatches.Patch(color=dot_color, label='True contact (|i−j|≥6)')],
              fontsize=8, loc='upper right', markerscale=0.8)

    # ── Panel C: D baseline probability with true contact overlay ─────────────
    ax = axes[2]
    im2 = ax.imshow(baseline_prob, cmap=cmap_prob, vmin=0, vmax=1,
                    origin='upper', aspect='equal')
    ax.scatter(j_tc, i_tc, c=dot_color, s=dot_size, alpha=dot_alpha,
               linewidths=0, zorder=5, label='True contact')
    ax.set_title(f'(C) Seq-distance baseline (D)\nF1={m_d["f1"]:.3f}  LR P@L/5={m_d["long_range_precision_L5"]:.3f}',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Residue j', fontsize=10)
    ax.set_ylabel('Residue i', fontsize=10)
    ax.axhline(y=12, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
    ax.axvline(x=12, color='orange', linewidth=1.2, linestyle='--', alpha=0.8)
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='P(contact)')
    ax.legend(handles=[mpatches.Patch(color=dot_color, label='True contact (|i−j|≥6)')],
              fontsize=8, loc='upper right', markerscale=0.8)

    # ── Global title ──────────────────────────────────────────────────────────
    fig.suptitle('1CRN Crambin (L=46): Contact Map Comparison',
                 fontsize=13, fontweight='bold', y=1.02)

    # ── Caption note ─────────────────────────────────────────────────────────
    caption = (
        'Blue intensity = predicted P(contact | model). Red dots = true contacts with |i−j| ≥ 6. '
        'Dashed orange lines mark the long-range boundary (|i−j| = 12). '
        'Panel C shows the zero-learning baseline (D) is bright only near the diagonal '
        '(sequential contacts); panel B shows v4 detects some long-range contacts missed by D.'
    )
    fig.text(0.5, -0.04, caption, ha='center', va='top', fontsize=8.5,
             color='#444444', wrap=True)

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = 'report/figures'
    out_path = os.path.join(out_dir, 'contact_map_1crn.png')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nSaved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    make_figure()
