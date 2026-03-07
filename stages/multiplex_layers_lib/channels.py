"""Channel and patch I/O helpers for multiplex layers."""

import numpy as np


def load_multiplex_patch(npy_path: str) -> np.ndarray:
    """Load and return (C, H, W) uint16 array from .npy file."""
    return np.load(npy_path)


def get_channel_index(channel_names: list[str], target: str) -> int:
    """Return 0-based index of target in channel_names (case-insensitive)."""
    target_lower = target.lower()
    for idx, name in enumerate(channel_names):
        if name.lower() == target_lower:
            return idx
    raise ValueError(
        f"Channel '{target}' not found in channel list {channel_names}. "
        f"Available (case-insensitive): {channel_names}"
    )


def get_first_matching_channel_index(
    channel_names: list[str], candidates: list[str]
) -> int | None:
    """Return index of the first candidate channel found, else None."""
    for name in candidates:
        try:
            return get_channel_index(channel_names, name)
        except ValueError:
            continue
    return None


def extract_channel(patch: np.ndarray, idx: int) -> np.ndarray:
    """Return patch[idx] as (H, W) uint16 array."""
    return patch[idx]
