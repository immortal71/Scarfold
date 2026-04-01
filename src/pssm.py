"""pssm.py — Position-Specific Scoring Matrix (PSSM) utilities.

This module provides three levels of evolutionary information, in increasing
quality order:

  1. pseudo_pssm(seq)        — BLOSUM62-derived approximation. No external tools.
                               Encodes which substitutions are tolerated at each
                               position based on the residue's BLOSUM row.

  2. parse_psiblast_pssm()   — Parses the ASCII PSSM file output by PSI-BLAST
                               (-out_ascii_pssm flag). Requires running PSI-BLAST
                               against UniRef50 or similar.

  3. encoding_with_pssm()    — Builds the full 68-dim feature vector:
                               one-hot (20) + PSSM (20) + physicochemical (8)
                               + relative position (1) + local_complexity (1)
                               = 50 core + 18 context = 68 dim

Why PSSM matters:
    Consider residue i that is always Leu or Val in homologs — both are
    hydrophobic. The PSSM will show high scores for I, V, L, M at position i.
    The network sees that position i MUST be hydrophobic, likely buried in the
    core, strongly constraining which residues can be adjacent. This is the
    core insight of AlphaFold's evolutionary coupling approach.

Usage:
    # Level 1 — no tools needed, immediate improvement
    from src.pssm import pseudo_pssm, encoding_with_pssm
    feat = encoding_with_pssm(seq)           # (L, 50)

    # Level 2 — after running PSI-BLAST
    # psiblast -query seq.fasta -db uniref50 -num_iterations 3 \\
    #          -out_ascii_pssm output.pssm -num_threads 4
    from src.pssm import parse_psiblast_pssm, encoding_with_pssm
    pssm = parse_psiblast_pssm('output.pssm')  # (L, 20)
    feat = encoding_with_pssm(seq, pssm=pssm)  # (L, 50)
"""

import os
import numpy as np

# ── same ordering as utils.py ─────────────────────────────────────────────────
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
AA_TO_IDX = {a: i for i, a in enumerate(AA_LIST)}

# BLOSUM62 (20×20, rows = query AA in AA_LIST order, normalised to [-1,1])
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
_BL62_MAX = 11.0
BLOSUM62 = np.array([_BLOSUM62_RAW[a] for a in AA_LIST], dtype=np.float32) / _BL62_MAX

# Physicochemical (same 8 features as utils.py)
_PHYSCHEM = {
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
_PC_MAT = np.array([_PHYSCHEM[a] for a in AA_LIST], dtype=np.float32)
_pc_min = _PC_MAT.min(0); _pc_max = _PC_MAT.max(0)
_PC_MAT = (_PC_MAT - _pc_min) / (_pc_max - _pc_min + 1e-8)

# Total feature dimension: one_hot(20) + pssm(20) + physicochemical(8) + rel_pos(1) + lc(1) = 50
PSSM_FEAT_DIM = 50


# ── Level 1: Pseudo-PSSM from BLOSUM62 ───────────────────────────────────────

def pseudo_pssm(seq):
    """Return a (L, 20) pseudo-PSSM derived from BLOSUM62 rows.

    This approximates evolutionary information by using the substitution
    profile of each observed residue.  It is better than nothing and requires
    no external tools.  The values are normalised to [0, 1].
    """
    L = len(seq)
    mat = np.zeros((L, 20), dtype=np.float32)
    for i, a in enumerate(seq):
        idx = AA_TO_IDX.get(a, 0)
        row = BLOSUM62[idx].copy()
        # shift to [0,1]: BLOSUM62 is in [-1,1] after our normalisation
        mat[i] = (row + 1.0) / 2.0
    return mat


# ── Level 2: Parse real PSI-BLAST PSSM ───────────────────────────────────────

def parse_psiblast_pssm(pssm_path, aa_order=None):
    """Parse the ASCII PSSM file produced by PSI-BLAST.

    Run PSI-BLAST with:
        psiblast -query seq.fasta -db uniref50 \\
                 -num_iterations 3 -out_ascii_pssm output.pssm

    Returns:
        pssm : np.ndarray of shape (L, 20), float32, values in [-1, 1]
               Columns are in AA_LIST order ('ACDEFGHIKLMNPQRSTVWY').
    """
    if aa_order is None:
        # PSI-BLAST uses alphabetical AA order: ACDEFGHIKLMNPQRSTVWY
        aa_order = list('ACDEFGHIKLMNPQRSTVWY')

    rows = []
    with open(pssm_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 22:
                continue
            try:
                int(parts[0])  # first column is residue number
            except ValueError:
                continue
            # columns 2–21 are raw PSSM scores; columns 22–41 are frequencies
            scores = [int(x) for x in parts[2:22]]
            rows.append(scores)

    if not rows:
        raise ValueError(f'No PSSM data found in {pssm_path}. '
                         'Ensure the file is a PSI-BLAST ASCII PSSM (-out_ascii_pssm).')

    pssm_raw = np.array(rows, dtype=np.float32)  # (L, 20) in PSI-BLAST AA order

    # Reorder columns to match AA_LIST if needed
    if aa_order != AA_LIST:
        col_map = [aa_order.index(a) for a in AA_LIST]
        pssm_raw = pssm_raw[:, col_map]

    # Normalise raw PSSM scores to [-1, 1] (typical range is about -8 to +14)
    pssm_norm = np.clip(pssm_raw / 10.0, -1.0, 1.0)
    return pssm_norm.astype(np.float32)


# ── local complexity (Shannon entropy of window) ──────────────────────────────

def _local_complexity(seq, window=5):
    """Per-residue local sequence complexity as normalised Shannon entropy.

    Low-complexity regions (e.g. poly-Q runs) tend to be disordered; high
    complexity regions are more likely to be structured.
    """
    L = len(seq)
    scores = np.zeros(L, dtype=np.float32)
    for i in range(L):
        lo = max(0, i - window // 2)
        hi = min(L, lo + window)
        window_seq = seq[lo:hi]
        counts = np.zeros(20)
        for a in window_seq:
            counts[AA_TO_IDX.get(a, 0)] += 1
        probs = counts / counts.sum()
        # Shannon entropy normalised to [0,1] by log(20)
        nonzero = probs[probs > 0]
        h = -np.sum(nonzero * np.log(nonzero)) / np.log(20)
        scores[i] = float(h)
    return scores


# ── Level 3: Full 50-dim feature vector ───────────────────────────────────────

def encoding_with_pssm(seq, pssm=None):
    """Build the full 50-dim per-residue feature vector.

    Breakdown:
        [0:20]  one-hot identity
        [20:40] PSSM (real PSI-BLAST if pssm given, else pseudo-PSSM from BLOSUM62)
        [40:48] physicochemical properties (8 features, normalised)
        [48]    relative sequence position (0 → 1 from N- to C-terminus)
        [49]    local sequence complexity (Shannon entropy, window=5)

    Args:
        seq  : str, amino-acid sequence (standard 1-letter codes)
        pssm : (L, 20) float32 array from parse_psiblast_pssm(), or None.
               If None, pseudo_pssm(seq) is used.

    Returns:
        features : (L, 50) float32 array
    """
    L = len(seq)

    # one-hot
    oh = np.zeros((L, 20), dtype=np.float32)
    for i, a in enumerate(seq):
        oh[i, AA_TO_IDX.get(a, 0)] = 1.0

    # PSSM block
    if pssm is not None:
        assert pssm.shape == (L, 20), \
            f'PSSM shape {pssm.shape} does not match sequence length {L}'
        pssm_block = pssm.astype(np.float32)
    else:
        pssm_block = pseudo_pssm(seq)

    # physicochemical
    pc = np.zeros((L, 8), dtype=np.float32)
    for i, a in enumerate(seq):
        pc[i] = _PC_MAT[AA_TO_IDX.get(a, 0)]

    # relative position [0,1]
    rel_pos = np.linspace(0.0, 1.0, L, dtype=np.float32).reshape(L, 1)

    # local complexity
    lc = _local_complexity(seq).reshape(L, 1)

    return np.concatenate([oh, pssm_block, pc, rel_pos, lc], axis=1)  # (L, 50)


# ── Utility: write a FASTA file for PSI-BLAST input ──────────────────────────

def write_fasta(seq, path, seq_id='query'):
    """Write a single sequence to a FASTA file for PSI-BLAST input."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write(f'>{seq_id}\n')
        for i in range(0, len(seq), 60):
            f.write(seq[i:i+60] + '\n')


# ── PSI-BLAST runner (if available) ──────────────────────────────────────────

def run_psiblast(seq, db_path, output_pssm_path, num_iterations=3, num_threads=4):
    """Run PSI-BLAST and return the parsed PSSM.

    Requires BLAST+ to be installed and 'psiblast' in PATH.
    Requires a local copy of UniRef50 or nr database indexed with makeblastdb.

    Args:
        seq              : amino-acid sequence string
        db_path          : path to BLAST database (e.g. 'data/uniref50/uniref50')
        output_pssm_path : path to write the .pssm file
        num_iterations   : PSI-BLAST iterations (3 is standard)
        num_threads      : number of CPU threads

    Returns:
        pssm : (L, 20) float32 PSSM array, or None if PSI-BLAST not available
    """
    import subprocess
    import tempfile

    fasta_path = output_pssm_path + '.query.fasta'
    write_fasta(seq, fasta_path)

    cmd = [
        'psiblast',
        '-query', fasta_path,
        '-db', db_path,
        '-num_iterations', str(num_iterations),
        '-out_ascii_pssm', output_pssm_path,
        '-num_threads', str(num_threads),
        '-evalue', '0.001',
        '-inclusion_ethresh', '0.001',
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode != 0:
            print(f'PSI-BLAST failed: {result.stderr.decode()[:200]}')
            return None
        if os.path.exists(output_pssm_path):
            return parse_psiblast_pssm(output_pssm_path)
        return None
    except FileNotFoundError:
        print('PSI-BLAST not found. Install BLAST+ from https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/')
        return None
    except subprocess.TimeoutExpired:
        print('PSI-BLAST timed out after 300s.')
        return None
    finally:
        if os.path.exists(fasta_path):
            os.remove(fasta_path)
