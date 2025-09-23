"""Global deterministic seeding utilities.

Ensures reproducibility across numpy, Python stdlib random, and TensorFlow (if installed).
Extend later for torch / jax when added.
"""
from __future__ import annotations
import os
import random
import numpy as np

try:  # optional tf
    import tensorflow as tf  # type: ignore
except Exception:  # pragma: no cover
    tf = None  # type: ignore

_DEFAULT_ENV_VAR = "STELLAR_GLOBAL_SEED"


def set_global_seed(seed: int | None = None) -> int:
    """Set global deterministic seeds.

    If seed is None, will look for STELLAR_GLOBAL_SEED environment variable.
    If still None, generates a new seed from os.urandom randomness.

    Returns the seed actually used so callers can log it.
    """
    if seed is None:
        env_val = os.getenv(_DEFAULT_ENV_VAR)
        if env_val is not None:
            try:
                seed = int(env_val)
            except ValueError:  # pragma: no cover
                seed = None
    if seed is None:
        seed = int.from_bytes(os.urandom(4), 'big')

    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:  # pragma: no cover - environment dependent
        try:
            tf.random.set_seed(seed)
        except Exception:
            pass
    return seed

__all__ = ["set_global_seed"]
