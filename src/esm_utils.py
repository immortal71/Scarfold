"""esm_utils.py — ESM-2 per-residue embeddings as drop-in feature extractor.

Uses Meta's ESM-2 protein language model to produce per-residue embeddings from
a single amino-acid sequence with no MSA required.  The embeddings encode
evolutionary signal distilled from 250M sequences, providing the model with
implicit co-variation information.

Two model tiers are supported:
    esm2_t6_8M_UR50D   — 8 M params,  6 layers,  320-dim  (fast, default)
    esm2_t12_35M_UR50D — 35 M params, 12 layers, 480-dim  (accurate, ESM_MODEL=35M)

The larger 35M model substantially improves long-range contact and TM-score
because more layers = richer residue-residue context captured by the LM.

Set via environment variable or function argument:
    import os; os.environ['ESM_MODEL'] = '35M'  # before first call
    esm2_rich_encoding(seq, model_size='35M')

Combined with the existing 48-dim rich encoding:
    ESM_RICH_DIM_8M  = 320 + 48 = 368
    ESM_RICH_DIM_35M = 480 + 48 = 528

Install the ESM library:
    pip install fair-esm
"""

import os
import numpy as np
import torch

# Model configs: (hub_name, n_layers, embed_dim)
_ESM_CONFIGS = {
    '8M':  ('esm2_t6_8M_UR50D',   6,  320),
    '35M': ('esm2_t12_35M_UR50D', 12, 480),
}

# Default model size (override with ESM_MODEL env var)
_DEFAULT_SIZE = os.environ.get('ESM_MODEL', '8M')

# Output dimension shortcuts
ESM_DIM      = 320            # default 8M model
ESM_DIM_35M  = 480            # 35M model
ESM_RICH_DIM = ESM_DIM + 48   # 368  (default)

# Module-level singletons — lazy-loaded on first call, keyed by model size
_ESM_MODELS    = {}
_ESM_ALPHABETS = {}
_ESM_BATCH_CONVERTERS = {}


def _load_esm(model_size: str = _DEFAULT_SIZE) -> None:
    """Load ESM-2 weights once and cache them by model_size ('8M' or '35M')."""
    if model_size in _ESM_MODELS:
        return

    hub_name, n_layers, embed_dim = _ESM_CONFIGS[model_size]

    try:
        import esm as esm_lib  # pip install fair-esm
        load_fn = getattr(esm_lib.pretrained, hub_name)
        _model, _alphabet = load_fn()
    except (ImportError, AttributeError):
        print(f"  [esm_utils] 'fair-esm' not installed — loading {hub_name} via torch.hub ...")
        _model, _alphabet = torch.hub.load(
            "facebookresearch/esm:main",
            hub_name,
            verbose=False,
        )

    _model.eval()
    for param in _model.parameters():
        param.requires_grad_(False)

    _ESM_MODELS[model_size]            = _model
    _ESM_ALPHABETS[model_size]         = _alphabet
    _ESM_BATCH_CONVERTERS[model_size]  = _alphabet.get_batch_converter()
    print(f"  [esm_utils] ESM-2 ({hub_name}, {embed_dim}-dim) loaded.")


def esm2_encoding(seq: str, model_size: str = _DEFAULT_SIZE) -> np.ndarray:
    """Return (L, D) ESM-2 per-residue embeddings.

    D = 320 for model_size='8M', 480 for model_size='35M'.
    """
    _load_esm(model_size)
    _, n_layers, _ = _ESM_CONFIGS[model_size]

    batch_converter = _ESM_BATCH_CONVERTERS[model_size]
    model           = _ESM_MODELS[model_size]

    data = [("protein", seq)]
    _, _, tokens = batch_converter(data)

    with torch.no_grad():
        results = model(tokens, repr_layers=[n_layers], return_contacts=False)

    embs = results["representations"][n_layers][0, 1: len(seq) + 1].cpu().numpy()
    return embs.astype(np.float32)


def esm2_rich_encoding(seq: str, model_size: str = _DEFAULT_SIZE) -> np.ndarray:
    """Return (L, D+48) combined features: ESM-2 + rich encoding.

    D+48 = 368 for model_size='8M', 528 for model_size='35M'.
    """
    import sys, os as _os
    sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..')))
    from src.utils import rich_encoding

    esm_feats  = esm2_encoding(seq, model_size=model_size)
    rich_feats = rich_encoding(seq)
    return np.concatenate([esm_feats, rich_feats], axis=1)


def get_esm_rich_dim(model_size: str = _DEFAULT_SIZE) -> int:
    """Return the combined feature dimension for a given model_size."""
    _, _, embed_dim = _ESM_CONFIGS[model_size]
    return embed_dim + 48


def is_esm_available() -> bool:
    """Return True if the ESM library can be imported without error."""
    try:
        import esm  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        torch.hub.list("facebookresearch/esm:main")
        return True
    except Exception:
        return False

