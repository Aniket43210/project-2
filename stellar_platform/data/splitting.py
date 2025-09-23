"""Data splitting utilities with leakage-safe strategies.

Provides:
- object_id_split: deterministic split by hashing object identifiers
- sky_region_split: split using HEALPix sky regions (if healpy available)
"""
from __future__ import annotations

import hashlib
from typing import Iterable, List, Tuple, Dict, Any, Sequence
import numpy as np

try:  # optional dependency
    import healpy as hp  # type: ignore
except Exception:  # pragma: no cover
    hp = None  # type: ignore


def _hash_to_float(value: str) -> float:
    h = hashlib.sha1(value.encode()).hexdigest()
    # Take first 8 hex digits -> int -> scale to [0,1)
    return int(h[:8], 16) / 16**8


def object_id_split(
    object_ids: Sequence[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Deterministic split by hashing object IDs.

    Ensures same object always goes to same split independent of order.
    """
    assert 0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1
    rng = np.random.default_rng(seed)
    # Shuffle copy to avoid order bias then assign by hash threshold
    ids = list(object_ids)
    # Hash each id to value
    hashed = np.array([_hash_to_float(str(i)) for i in ids])
    train_cut = train_frac
    val_cut = train_frac + val_frac
    train = [i for i, h in zip(ids, hashed) if h < train_cut]
    val = [i for i, h in zip(ids, hashed) if train_cut <= h < val_cut]
    test = [i for i, h in zip(ids, hashed) if h >= val_cut]
    return {'train': train, 'val': val, 'test': test}


def sky_region_split(
    ras: Sequence[float],
    decs: Sequence[float],
    object_ids: Sequence[str],
    nside: int = 32,
    holdout_frac: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split by holding out random HEALPix cells.

    If healpy is not installed, falls back to object_id_split.
    """
    if hp is None:  # pragma: no cover
        return object_id_split(object_ids, 1 - holdout_frac - 0.1, 0.1, seed)
    theta = np.radians(90 - np.array(decs))
    phi = np.radians(np.array(ras))
    pix = hp.ang2pix(nside, theta, phi, nest=False)
    unique_pix = np.unique(pix)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_pix)
    holdout_n = int(len(unique_pix) * holdout_frac)
    holdout_cells = set(unique_pix[:holdout_n])
    holdout_ids = [oid for oid, p in zip(object_ids, pix) if p in holdout_cells]
    remain_ids = [oid for oid, p in zip(object_ids, pix) if p not in holdout_cells]
    # Further split remain_ids deterministically
    remain_split = object_id_split(remain_ids, 0.8, 0.1, seed)
    return {
        'train': remain_split['train'],
        'val': remain_split['val'],
        'sky_holdout': holdout_ids,
        'test': remain_split['test'],
    }


__all__ = ['object_id_split', 'sky_region_split']
