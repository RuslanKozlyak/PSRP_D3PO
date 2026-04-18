from __future__ import annotations

import numpy as np


def ensure_at_least_one_valid(mask: np.ndarray, depot_index: int = 0) -> np.ndarray:
    """Guarantee that every row has at least one valid action by enabling depot."""

    fixed = np.asarray(mask, dtype=bool).copy()
    if fixed.ndim == 1:
        if not fixed.any():
            fixed[depot_index] = True
        return fixed

    empty_rows = ~fixed.any(axis=-1)
    if np.any(empty_rows):
        fixed[empty_rows, depot_index] = True
    return fixed


def masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    weights = np.asarray(mask, dtype=np.float32)
    total = np.sum(values * np.expand_dims(weights, axis=-1), axis=axis)
    denom = np.maximum(np.sum(weights, axis=axis, keepdims=True), 1.0)
    return total / denom


def normalize_entropy(entropy: np.ndarray, valid_counts: np.ndarray) -> np.ndarray:
    denom = np.maximum(np.log(np.maximum(valid_counts, 2)), 1.0)
    return entropy / denom
