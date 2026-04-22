"""esm_utils.py — ESM-2 per-residue embeddings as drop-in feature extractor.

Uses Meta's ESM-2 protein language model (esm2_t6_8M_UR50D, 8M params, 320-dim)
to produce per-residue embeddings from a single amino-acid sequence with no MSA
required.  The embeddings encode evolutionary signal distilled from 250M sequences,
providing the model with implicit co-variation information that is the the largest
single gap between our hand-crafted features and AlphaFold-style representations.

Combined with the existing 48-dim rich encoding (one-hot + BLOSUM62 + physicochemical),
the total input feature dimension becomes 368—dim (ESM_RICH_DIM).

Install the ESM library:
    pip install fair-esm

Or the model will attempt a torch.hub download automatically.

Typical usage (drop-in replacement for utils.rich_encoding):
    from src.esm_utils import esm2_rich_encoding, ESM_RICH_DIM
    features = esm2_rich_encoding(sequence)   # (L, 368) float32 numpy array
"""

import numpy as np
import torch

# Output dimension of esm2_t6_8M_UR50D (6-layer, 320-dim)
ESM_DIM = 320

# Combined ESM-2 + rich-encoding dimension
ESM_RICH_DIM = ESM_DIM + 48   # 368

# Module-level singletons — lazy-loaded on first call
_ESM_MODEL = None
_ESM_ALPHABET = None
_ESM_BATCH_CONVERTER = None


def _load_esm() -> None:
    """Load ESM-2 weights once and cache them.  Tries `fair-esm` pip package
    first; falls back to torch.hub (requires internet on first run)."""
    global _ESM_MODEL, _ESM_ALPHABET, _ESM_BATCH_CONVERTER
    if _ESM_MODEL is not None:
        return

    try:
        import esm as esm_lib  # pip install fair-esm
        _ESM_MODEL, _ESM_ALPHABET = esm_lib.pretrained.esm2_t6_8M_UR50D()
    except (ImportError, AttributeError):
        # Fallback: download via PyTorch Hub (cached after first run)
        print("  [esm_utils] 'fair-esm' not installed — loading via torch.hub ...")
        _ESM_MODEL, _ESM_ALPHABET = torch.hub.load(
            "facebookresearch/esm:main",
            "esm2_t6_8M_UR50D",
            verbose=False,
        )

    _ESM_MODEL.eval()
    for param in _ESM_MODEL.parameters():
        param.requires_grad_(False)   # freeze — feature extractor only

    _ESM_BATCH_CONVERTER = _ESM_ALPHABET.get_batch_converter()
    print(f"  [esm_utils] ESM-2 (esm2_t6_8M_UR50D, {ESM_DIM}-dim) loaded.")


def esm2_encoding(seq: str) -> np.ndarray:
    """Return (L, 320) ESM-2 per-residue embeddings for a single sequence.

    The ESM-2 model is loaded lazily on the first call and cached for all
    subsequent calls in the same process.

    Args:
        seq: Amino-acid sequence string (single-letter codes, A-Z).

    Returns:
        float32 numpy array of shape (L, 320).
    """
    _load_esm()

    data = [("protein", seq)]
    _, _, tokens = _ESM_BATCH_CONVERTER(data)

    with torch.no_grad():
        results = _ESM_MODEL(tokens, repr_layers=[6], return_contacts=False)

    # Layer-6 representations: shape (1, L+2, 320) — strip <cls> and <eos> tokens
    embs = results["representations"][6][0, 1 : len(seq) + 1].cpu().numpy()
    return embs.astype(np.float32)   # (L, 320)


def esm2_rich_encoding(seq: str) -> np.ndarray:
    """Return (L, 368) combined features: ESM-2 (320) + rich encoding (48).

    Concatenating ESM-2 embeddings with hand-crafted physicochemical features
    gives the model:
    - Implicit evolutionary co-variation (from the LM's 250M-sequence training)
    - Explicit biochemical properties (hydrophobicity, charge, etc.)

    This is complementary: the LM captures what evolution considers equivalent,
    the physicochemical features capture why two residues might be structurally
    similar without their being evolutionarily related.

    Args:
        seq: Amino-acid sequence string.

    Returns:
        float32 numpy array of shape (L, 368).
    """
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.utils import rich_encoding

    esm_feats  = esm2_encoding(seq)    # (L, 320)
    rich_feats = rich_encoding(seq)    # (L,  48)
    return np.concatenate([esm_feats, rich_feats], axis=1)   # (L, 368)


def is_esm_available() -> bool:
    """Return True if the ESM library can be imported without error."""
    try:
        import esm  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        # Check torch.hub cache without downloading
        torch.hub.list("facebookresearch/esm:main")
        return True
    except Exception:
        return False
