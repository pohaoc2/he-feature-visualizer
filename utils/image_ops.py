"""Shared image-processing utilities for visualisation tools."""

from __future__ import annotations

import numpy as np


def colorize_label_mask(mask: np.ndarray, seed: int = 42) -> np.ndarray:
    """Map integer label IDs to deterministic random RGB colors (0 -> black).

    Uses sparse unique-inverse indexing, so memory usage scales with the number
    of distinct labels rather than the maximum label ID.
    """
    label_ids, inverse = np.unique(mask, return_inverse=True)
    colors = np.zeros((label_ids.shape[0], 3), dtype=np.uint8)
    non_bg = label_ids != 0
    if np.any(non_bg):
        rng = np.random.default_rng(seed)
        colors[non_bg] = rng.integers(
            30, 256, size=(int(non_bg.sum()), 3), dtype=np.uint8
        )
    return colors[inverse].reshape(mask.shape + (3,))
