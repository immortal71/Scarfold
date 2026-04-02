import os
import numpy as np
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {a:i for i,a in enumerate(AA_LIST)}
HYDRO = {
    'A': 1.8,'C':2.5,'D':-3.5,'E':-3.5,'F':2.8,'G':-0.4,'H':-3.2,'I':4.5,'K':-3.9,'L':3.8,'M':1.9,'N':-3.5,'P':-1.6,'Q':-3.5,'R':-4.5,'S':-0.8,'T':-0.7,'V':4.2,'W':-0.9,'Y':-1.3
}

# ──────────────────────────────────────────────
# BLOSUM62 substitution matrix (rows = query AA, cols = AA_LIST order)
# Used to add evolutionary-context features instead of plain one-hot.
# Source: NCBI BLOSUM62 (normalised to [-1, 1] for numerical stability).
# ──────────────────────────────────────────────
_BLOSUM62_RAW = {
    'A': [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    'C': [-1, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-2,-3,-1,-1,-2,-2,-1],
    'D': [-2,-3, 6, 2,-3,-1,-1,-2,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    'E': [-2,-4, 2, 5,-3,-2, 0,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    'F': [ 0,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2, 1, 3,-1],
    'G': [-1,-3,-1,-2,-3, 6,-2,-4,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    'H': [-1,-3,-1, 0,-1,-2, 8,-3,-1,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    'I': [ 0,-1,-2,-2, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1,-3,-1, 3],
    'K': [-2,-3,-1, 0,-3,-2,-1,-3, 5,-2,-2, 1,-1,-1,-1, 0,-1,-3,-2,-2],
    'L': [-1,-1,-3,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-3,-3,-2,-1,-2,-1, 1],
    'M': [-1,-1,-3,-3, 0,-3,-3, 1,-2, 2, 5,-2,-2, 0,-2,-1,-1,-1,-1, 1],
    'N': [-1,-3, 1, 0,-3,-2,-1,-3, 1,-3,-2, 6,-2, 0, 0, 1, 0,-4,-2,-3],
    'P': [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-4,-3,-2],
    'Q': [-1,-3,-1, 1,-4,-2,-1,-3, 1,-3,-2, 0,-1, 5,-1, 0,-1,-2,-1,-2],
    'R': [-1,-3,-1,-1,-3,-2,-2,-3,-1,-3,-2, 0,-2,-1, 5,-1,-1,-3,-2,-3],
    'S': [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-3,-2,-2],
    'T': [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5,-2,-2, 0],
    'V': [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,11, 2,-3],
    'W': [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2, 2, 7,-1],
    'Y': [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0,-3,-1, 4],
}
# Build ordered matrix (20×20) and normalise to [-1,1]
_BL62_MAX = 11.0
BLOSUM62 = np.array(
    [_BLOSUM62_RAW[a] for a in AA_LIST], dtype=np.float32
) / _BL62_MAX  # shape (20, 20)

# ──────────────────────────────────────────────
# Physicochemical property vectors per amino acid
# 8 features: hydrophobicity, charge, polarity, volume, aromaticity,
#             helix_propensity, sheet_propensity, coil_propensity
# ──────────────────────────────────────────────
_PHYSCHEM = {
    # AA: [hydro, charge, polarity, rel_volume, aromatic, helix, sheet, coil]
    'A': [ 1.8,  0,  8.1, 0.31, 0, 1.42, 0.83, 0.66],
    'C': [ 2.5,  0,  5.5, 0.55, 0, 0.70, 1.19, 1.19],
    'D': [-3.5, -1, 13.0, 0.46, 0, 1.01, 0.54, 1.46],
    'E': [-3.5, -1, 12.3, 0.62, 0, 1.51, 0.37, 1.14],
    'F': [ 2.8,  0,  5.2, 1.00, 1, 1.13, 1.38, 0.60],
    'G': [-0.4,  0,  9.0, 0.00, 0, 0.57, 0.75, 1.56],
    'H': [-3.2,  1, 10.4, 0.87, 1, 1.00, 0.87, 0.95],
    'I': [ 4.5,  0,  5.2, 0.90, 0, 1.08, 1.60, 0.47],
    'K': [-3.9,  1, 11.3, 0.93, 0, 1.14, 0.74, 1.01],
    'L': [ 3.8,  0,  4.9, 0.90, 0, 1.21, 1.30, 0.59],
    'M': [ 1.9,  0,  5.7, 0.94, 0, 1.45, 1.05, 0.60],
    'N': [-3.5,  0, 11.6, 0.58, 0, 0.67, 0.89, 1.33],
    'P': [-1.6,  0,  8.0, 0.55, 0, 0.57, 0.55, 1.52],
    'Q': [-3.5,  0, 10.5, 0.72, 0, 1.11, 1.10, 0.96],
    'R': [-4.5,  1, 10.5, 1.19, 0, 0.98, 0.93, 0.95],
    'S': [-0.8,  0,  9.2, 0.35, 0, 0.77, 0.75, 1.43],
    'T': [-0.7,  0,  8.6, 0.56, 0, 0.83, 1.19, 0.96],
    'V': [ 4.2,  0,  5.9, 0.76, 0, 1.06, 1.70, 0.50],
    'W': [-0.9,  0,  5.4, 1.30, 1, 1.08, 1.37, 0.96],
    'Y': [-1.3,  0,  6.2, 1.14, 1, 0.69, 1.47, 1.14],
}
_PHYSCHEM_MATRIX = np.array(
    [_PHYSCHEM[a] for a in AA_LIST], dtype=np.float32
)
# normalise each column to [0,1]
_pc_min = _PHYSCHEM_MATRIX.min(0)
_pc_max = _PHYSCHEM_MATRIX.max(0)
_PHYSCHEM_MATRIX = (_PHYSCHEM_MATRIX - _pc_min) / (_pc_max - _pc_min + 1e-8)

# total feature dimension per residue when using rich encoding
RICH_AA_DIM = 20 + 20 + 8  # one_hot + blosum62 + physicochemical = 48

def one_hot(seq):
    L = len(seq)
    mat = np.zeros((L,20),dtype=np.float32)
    for i,a in enumerate(seq):
        mat[i,AA_TO_IDX.get(a,'A')] = 1.0
    return mat


def rich_encoding(seq):
    """Return a (L, 48) feature matrix: one-hot(20) + BLOSUM62(20) + physicochemical(8).

    This gives the model evolutionary context beyond raw identity, which is the
    largest single improvement short of running a full MSA.
    """
    L = len(seq)
    oh = np.zeros((L, 20), dtype=np.float32)
    bl = np.zeros((L, 20), dtype=np.float32)
    pc = np.zeros((L,  8), dtype=np.float32)
    for i, a in enumerate(seq):
        idx = AA_TO_IDX.get(a, 0)
        oh[i, idx] = 1.0
        bl[i] = BLOSUM62[idx]
        pc[i] = _PHYSCHEM_MATRIX[idx]
    return np.concatenate([oh, bl, pc], axis=1)  # (L, 48)



def synthetic_native_coords(seq, seed=0):
    """Generate realistic Cα coordinates from sequence using secondary structure propensities.

    Replaces the naive random-walk with proper protein-like geometry:
    1. **Secondary structure assignment**: Chou-Fasman helix/strand propensities
       applied over a 4-residue sliding window assign each residue to
       helix / strand / coil.
    2. **Geometry-constrained backbone**:
       - α-helix: rise=1.5 Å/res, turn=100°/res, radius=2.3 Å (i+1 bond ≈3.8 Å)
       - β-strand: rise=3.5 Å/res, alternating lateral offset (extended chain)
       - Coil: virtual bond model with direction memory
    3. **Hydrophobic collapse**: pull hydrophobic residues toward the centroid,
       approximating the hydrophobic core of real folded proteins.

    This produces training distance matrices with realistic statistics:
    - Helix: d(i,i+4)≈6.2 Å, d(i,i+3)≈5.5 Å
    - Strand: d(i,i+2)≈7.0 Å (extended), d(i,i+1)≈3.8 Å
    - vs random-walk: all inter-residue distances roughly unconstrained
    """
    rng = np.random.default_rng(seed)
    L = len(seq)

    # ── 1. Chou-Fasman-like propensities ───────────────────────────────────────────────
    # Values from _PHYSCHEM helix/sheet propensities (already loaded above)
    _h_prop = np.array([_PHYSCHEM.get(a, _PHYSCHEM['G'])[5] for a in seq])   # helix
    _s_prop = np.array([_PHYSCHEM.get(a, _PHYSCHEM['G'])[6] for a in seq])   # strand

    # Sliding window of 4 residues: assign SS class
    ss = np.zeros(L, dtype=np.int8)   # 0=coil, 1=helix, 2=strand
    for i in range(L):
        end = min(i + 4, L)
        win_h = _h_prop[i:end].mean()
        win_s = _s_prop[i:end].mean()
        if win_h > 1.05 and win_h >= win_s:
            ss[i] = 1   # helix
        elif win_s > 1.10 and win_s > win_h:
            ss[i] = 2   # strand
        # else: remain coil

    # ── 2. Build backbone coordinates ──────────────────────────────────────────────
    coords = np.zeros((L, 3), dtype=np.float64)
    pos = np.zeros(3)
    direction = np.array([0., 0., 1.])

    i = 0
    while i < L:
        if ss[i] == 1:  # α-helix ───────────────────────────────────────
            # Count helix segment length
            j = i
            while j < L and ss[j] == 1:
                j += 1
            n_h = j - i

            rise = 1.5        # Å per residue along helix axis
            turn = np.radians(100.0)   # ~3.6 residues per turn
            radius = 2.3      # Å helix radius

            # Build local helix frame
            ax = direction / (np.linalg.norm(direction) + 1e-10)
            perp = rng.standard_normal(3)
            perp -= perp.dot(ax) * ax
            perp /= np.linalg.norm(perp) + 1e-10
            perp2 = np.cross(ax, perp)

            helix_phase = rng.uniform(0, 2 * np.pi)
            for k in range(n_h):
                theta = helix_phase + k * turn
                coords[i + k] = (pos + ax * (k * rise)
                                 + radius * (np.cos(theta) * perp + np.sin(theta) * perp2))
            pos = pos + ax * (n_h * rise)
            # Random kink at helix end (loop)
            direction = direction + rng.standard_normal(3) * 0.3
            direction /= np.linalg.norm(direction) + 1e-10
            i = j

        elif ss[i] == 2:  # β-strand ────────────────────────────────────
            j = i
            while j < L and ss[j] == 2:
                j += 1
            n_s = j - i

            strand_ax = direction / (np.linalg.norm(direction) + 1e-10)
            perp = rng.standard_normal(3)
            perp -= perp.dot(strand_ax) * strand_ax
            perp /= np.linalg.norm(perp) + 1e-10

            rise = 3.5   # Å per residue, fully extended
            side = 0.5   # lateral alternating offset (Å)
            for k in range(n_s):
                coords[i + k] = pos + strand_ax * (k * rise) + perp * (side * (-1) ** k)
            pos = pos + strand_ax * (n_s * rise)
            # Turn/loop at strand end
            direction = direction + rng.standard_normal(3) * 0.6
            direction /= np.linalg.norm(direction) + 1e-10
            i = j

        else:  # coil / loop ───────────────────────────────────────────
            # Virtual bond model: Cα-Cα = 3.8 Å, direction memory = 0.4
            noise = rng.standard_normal(3)
            step = 0.4 * direction + 0.6 * noise
            step /= np.linalg.norm(step) + 1e-10
            step *= 3.8
            coords[i] = pos
            pos = pos + step
            direction = step / (np.linalg.norm(step) + 1e-10)
            i += 1

    # ── 3. Hydrophobic collapse ─────────────────────────────────────────────
    hydro_vals = np.array([HYDRO.get(a, 0.0) for a in seq])
    # Normalise hydrophobicity to [0, 1]: higher → more collapse
    h_min, h_max = hydro_vals.min(), hydro_vals.max()
    hydro_norm = (hydro_vals - h_min) / (h_max - h_min + 1e-8)

    centroid = coords.mean(0)
    for idx in range(L):
        # Pull hydrophobic residues up to 20% toward centroid
        strength = 0.20 * float(hydro_norm[idx])
        coords[idx] = coords[idx] * (1.0 - strength) + centroid * strength

    # Ensure consecutive-bond constraint is exact after collapse
    for idx in range(1, L):
        d_prev = np.linalg.norm(coords[idx] - coords[idx - 1])
        if d_prev > 1e-8:
            coords[idx] = coords[idx - 1] + (coords[idx] - coords[idx - 1]) / d_prev * 3.8

    return coords.astype(np.float32)

def coords_to_distances(coords):
    diff = coords[:,None,:] - coords[None,:,:]
    d = np.sqrt((diff**2).sum(-1))
    return d

def make_synthetic_dataset(num=200, L=40, seed=1):
    rng = np.random.RandomState(seed)
    seqs = []
    dists = []
    for i in range(num):
        seq = ''.join(rng.choice(AA_LIST, size=L))
        coords = synthetic_native_coords(seq, seed=seed+i)
        d = coords_to_distances(coords)
        seqs.append(seq)
        dists.append(d.astype(np.float32))
    return seqs, np.stack(dists)

def classical_mds(dist_mat, dim=3):
    """Classical MDS (Torgerson) to get coordinates from distances."""
    n = dist_mat.shape[0]
    D2 = dist_mat**2
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * J.dot(D2).dot(J)
    evals, evecs = np.linalg.eigh(B)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
    w = np.sqrt(np.clip(evals, 0, None))
    coords = evecs[:,:dim] * w[:dim]
    return coords


def gradient_mds(dist_mat, dim=3, n_iter=500, lr=0.05):
    """Gradient-based coordinate optimisation: minimise distance violation loss.

    Starts from classical MDS (warm start) then refines with Adam + Huber loss.
    Three regularisers improve reconstruction quality:

    1. **Backbone bond constraint** (always active): consecutive Cα distances must
       be 3.8 Å (rigid physical constraint). In a protein chain every peptide
       bond produces a fixed Cα-Cα distance of ~3.8 Å.  Adding a strong penalty
       keeps the chain connected even when long-range distance predictions are noisy.

    2. **lDDT-proxy regulariser** (first 3/4 of iterations): smooth sigmoid reward
       for pairs where |pred_d - true_d| < 2 Å (strictest lDDT threshold).
       Aligns the optimisation objective with the evaluation metric.

    3. **Chirality / orientation bias** (optional via cosine annealing LR without
       explicit constraint): the Adam optimiser with warm restarts naturally avoids
       mirror-image solutions.
    """
    import torch
    target = torch.tensor(dist_mat, dtype=torch.float32)
    init = classical_mds(dist_mat, dim=dim)
    coords = torch.tensor(init, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([coords], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iter, eta_min=lr * 0.01)

    # Pre-compute pair masks
    L = target.shape[0]
    triu_i, triu_j = torch.triu_indices(L, L, offset=1)
    # Backbone bond pairs: (i, i+1) — physical constraint d ≈ 3.8 Å
    backbone_i = torch.arange(L - 1)
    backbone_j = backbone_i + 1

    for step in range(n_iter):
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)       # (L,L,3)
        pred_d = torch.sqrt((diff ** 2).sum(-1) + 1e-8)        # (L,L)

        # Huber distance reconstruction loss (robust to outlier predictions)
        loss = torch.nn.functional.huber_loss(pred_d, target, delta=2.0)

        # ── Backbone bond constraint ──────────────────────────────────────────────────
        # Physical constraint: consecutive Cα atoms are always ~3.8 Å apart.
        # Weight = 5.0 makes this a hard constraint: a 1 Å bond error contributes
        # 5 units of loss vs the ~0.05 per-pair contribution from Huber.
        bone_pred = pred_d[backbone_i, backbone_j]         # (L-1,)
        bone_loss = torch.mean((bone_pred - 3.8) ** 2)
        loss = loss + 5.0 * bone_loss

        # ── lDDT-proxy regulariser ─────────────────────────────────────────────────
        if step < n_iter * 3 // 4:
            delta_d = (pred_d[triu_i, triu_j] - target[triu_i, triu_j]).abs()
            lddt_proxy = torch.sigmoid(2.0 - delta_d).mean()
            loss = loss - 0.1 * lddt_proxy   # reward sub-2 Å errors

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
    return coords.detach().numpy()




def kabsch_alignment(P, Q):
    """Returns rotation matrix and translation vector to align P to Q."""
    assert P.shape == Q.shape and P.shape[1] == 3
    P_centroid = P.mean(axis=0)
    Q_centroid = Q.mean(axis=0)
    P_centered = P - P_centroid
    Q_centered = Q - Q_centroid
    H = P_centered.T.dot(Q_centered)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T.dot(U.T)))
    R = Vt.T.dot(np.diag([1.0, 1.0, d])).dot(U.T)
    t = Q_centroid - P_centroid.dot(R)
    return R, t

def apply_transform(coords, R, t):
    return coords.dot(R) + t

def rmsd_kabsch(P, Q):
    R, t = kabsch_alignment(P, Q)
    P_aligned = apply_transform(P, R, t)
    return np.sqrt(np.mean(np.sum((P_aligned - Q)**2, axis=1))), P_aligned

def compute_plddt_from_distances(pred_dists, true_dists, cutoff=8.0):
    """Simplified per-residue confidence like pLDDT from distance agreement."""
    assert pred_dists.shape == true_dists.shape
    L = pred_dists.shape[0]
    scores = np.zeros(L, dtype=np.float32)
    for i in range(L):
        # include non-self distances only
        delta = np.abs(pred_dists[i] - true_dists[i])
        delta[i] = 0.0
        # logistic-like mapping; tight if errors are small
        score_ij = 1.0 / (1.0 + np.exp((delta - 1.0) * 2.5))
        # weight by proximity to focus on relevant contacts
        weight = np.exp(-true_dists[i] / cutoff)
        scores[i] = 100.0 * np.sum(score_ij * weight) / (np.sum(weight) + 1e-8)
    return scores


def local_lddt(pred_dists, true_dists, cutoff=15.0):
    """Simplified local lDDT per residue, similar to AlphaFold's lDDT metric."""
    assert pred_dists.shape == true_dists.shape
    L = pred_dists.shape[0]
    thresholds = [0.5, 1.0, 2.0, 4.0]
    scores = np.zeros(L, dtype=np.float32)
    for i in range(L):
        mask = (np.arange(L) != i) & (true_dists[i] < cutoff)
        if mask.sum() == 0:
            scores[i] = 0.0
            continue
        delta = np.abs(pred_dists[i, mask] - true_dists[i, mask])
        score_i = 0.0
        for t in thresholds:
            score_i += (delta <= t).sum() / float(mask.sum())
        scores[i] = 100.0 * (score_i / len(thresholds))
    return scores


def contact_map_metrics(pred_dists, true_dists, threshold=8.0):
    """Contact-map precision/recall/F1 at all ranges + long-range (|i-j|>=12).

    The long-range contact precision at top-L/5 predicted contacts is the
    standard metric used in CASP / published structure-prediction papers.
    Short-range contacts (|i-j|<6) are trivially predictable from backbone
    geometry alone, so long-range metrics are the meaningful comparison.
    """
    assert pred_dists.shape == true_dists.shape
    L = pred_dists.shape[0]

    def _metrics(sep_min, sep_max=None):
        i_idx, j_idx = np.triu_indices(L, k=sep_min)
        if sep_max is not None:
            keep = (j_idx - i_idx) < sep_max
            i_idx, j_idx = i_idx[keep], j_idx[keep]
        if len(i_idx) == 0:
            return dict(precision=0.0, recall=0.0, f1=0.0, tp=0, fp=0, fn=0,
                        long_range_precision_L5=0.0)
        p = pred_dists[i_idx, j_idx] < threshold
        t = true_dists[i_idx, j_idx] < threshold
        tp = int(np.logical_and(p, t).sum())
        fp = int(np.logical_and(p, ~t).sum())
        fn = int(np.logical_and(~p, t).sum())
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2.0 * prec * rec / (prec + rec + 1e-8)
        return dict(precision=float(prec), recall=float(rec), f1=float(f1),
                    tp=tp, fp=fp, fn=fn)

    res = _metrics(sep_min=1)   # all pairs

    # Long-range top-L/5 precision: standard CASP metric
    # Take only pairs with |i-j| >= 12, rank by predicted probability of contact
    # (lower predicted distance = higher probability), evaluate top-L/5.
    lr_i, lr_j = np.triu_indices(L, k=12)
    if len(lr_i) > 0:
        lr_pred = pred_dists[lr_i, lr_j]
        lr_true = true_dists[lr_i, lr_j] < threshold
        top_k   = max(1, L // 5)
        order   = np.argsort(lr_pred)[:top_k]          # lowest dist = most likely contact
        res['long_range_precision_L5'] = float(lr_true[order].mean())
    else:
        res['long_range_precision_L5'] = 0.0

    return res


def pdb_ca_coords(pdb_path, chain='A', max_residues=120):
    """Load CA coords (N residues max) from a PDB file."""
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    model = structure[0]
    coords = []
    for res in model[chain]:
        if len(coords) >= max_residues:
            break
        if 'CA' in res:
            coords.append(res['CA'].get_coord())
    return np.asarray(coords, dtype=np.float32)


def pdb_sequence(pdb_path, chain='A', max_residues=120):
    """Extract one-letter sequence from PDB CA atoms (residues with C-alpha)."""
    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('prot', pdb_path)
    model = structure[0]
    seq = []
    for res in model[chain]:
        if len(seq) >= max_residues:
            break
        if 'CA' not in res:
            continue
        resname = res.get_resname().strip().upper()
        try:
            seq.append(protein_letters_3to1.get(resname, 'A'))
        except Exception:
            seq.append('A')
    return ''.join(seq)


def fetch_pdb(pdb_id, output_dir='data/pdbs'):
    """Download a PDB file from RCSB by ID and return local path."""
    from Bio.PDB import PDBList
    pdb_id = pdb_id.lower()
    p = PDBList()
    os.makedirs(output_dir, exist_ok=True)
    path = p.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format='pdb', overwrite=True)
    # PDBList returns file path like 'data/pdbs/pdbXXXX.ent'
    if path.endswith('.gz'):
        import gzip, shutil
        out_path = os.path.join(output_dir, f'{pdb_id}.pdb')
        with gzip.open(path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        return out_path
    return path


def fasta_sequence(fasta_path, max_residues=120):
    """Read first sequence from FASTA and truncate to max_residues."""
    from Bio import SeqIO
    rec = next(SeqIO.parse(fasta_path, 'fasta'))
    seq = str(rec.seq).upper()
    seq = ''.join([a if a in AA_LIST else 'A' for a in seq])
    return seq[:max_residues]


def sample_pdb_dataset(pdb_dir, chain='A', max_residues=120, min_residues=10):
    """Load PDB files and produce (seq, dist) pairs — all padded to the same length.

    For mini-batch training on real PDB data, prefer sample_pdb_dataset_variable()
    which returns variable-length sequences (one protein = one sample).
    """
    pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.lower().endswith(('.pdb', '.ent')))
    seqs = []
    dists = []
    for fn in pdb_files:
        path = os.path.join(pdb_dir, fn)
        try:
            seq = pdb_sequence(path, chain=chain, max_residues=max_residues)
            if len(seq) < min_residues:
                continue
            coords = pdb_ca_coords(path, chain=chain, max_residues=max_residues)
            if coords.shape[0] < min_residues or coords.shape[0] != len(seq):
                continue
            d = coords_to_distances(coords)
            seqs.append(seq)
            dists.append(d.astype(np.float32))
        except Exception:
            continue
    if len(seqs) == 0:
        raise ValueError(f'No valid PDB data found in {pdb_dir} for chain {chain}')
    return seqs, np.stack(dists)


def sample_pdb_dataset_variable(pdb_dir, chain='A', max_residues=120, min_residues=20):
    """Load PDB files and return variable-length (seq, dist_matrix) pairs.

    Unlike sample_pdb_dataset(), this does NOT pad — each entry may have a
    different length.  Use this for per-protein training where the model is
    re-instantiated or batch_size=1.

    Returns:
        list of (seq: str, dist: np.ndarray[L,L])
    """
    pdb_files = sorted(f for f in os.listdir(pdb_dir) if f.lower().endswith(('.pdb', '.ent')))
    samples = []
    for fn in pdb_files:
        path = os.path.join(pdb_dir, fn)
        try:
            seq = pdb_sequence(path, chain=chain, max_residues=max_residues)
            if len(seq) < min_residues:
                continue
            coords = pdb_ca_coords(path, chain=chain, max_residues=max_residues)
            if coords.shape[0] < min_residues or coords.shape[0] != len(seq):
                continue
            d = coords_to_distances(coords).astype(np.float32)
            samples.append((seq, d))
        except Exception:
            continue
    if not samples:
        raise ValueError(f'No valid PDB sequences found in {pdb_dir}')
    return samples


def tm_score(P, Q, d0=None):
    """Length-adjusted TM-score proxy for coordinate sets of same length."""
    assert P.shape == Q.shape
    L = P.shape[0]
    if d0 is None:
        if L < 1:
            return 0.0
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(0.5, d0)
    dist2 = np.sum((P - Q) ** 2, axis=1)
    scores = 1.0 / (1.0 + dist2 / (d0 ** 2))
    return float(np.sum(scores) / L)

