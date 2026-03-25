"""Shared I/O helpers for CellViT JSON patch files and RGBA compositing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_patch_json(path: Path) -> list[dict]:
    """Load a CellViT JSON file and return the list of cell dicts."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        cells = data.get("cells", [])
        if isinstance(cells, list):
            return cells
    if isinstance(data, list):
        return data
    return []


def composite_rgba_on_rgb(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite an RGBA overlay onto an RGB base image."""
    base = base_rgb.astype(np.float32)
    if overlay_rgba.ndim != 3 or overlay_rgba.shape[-1] != 4:
        return base_rgb
    ov = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = (overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0).clip(0.0, 1.0)
    return (alpha * ov + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
