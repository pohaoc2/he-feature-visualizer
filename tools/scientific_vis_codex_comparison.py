#!/usr/bin/env python3
"""CellViT vs CODEX cell type comparison figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path("/tmp/matplotlib")
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.cell_assignment_reports import load_cell_assignments
from utils.normalize import percentile_norm

CELL_TYPES: tuple[str, str, str] = ("cancer", "immune", "healthy")

TYPE_COLORS = {
    "cancer":  "#DC3232",
    "immune":  "#3264DC",
    "healthy": "#32B432",
}

MARKER_COLORS = {
    "cancer":  "#DC3232",
    "immune":  "#3264DC",
    "healthy": "#32B432",
}

CROP_HALF = 20   # 40×40 px crop


# ── Data helpers ──────────────────────────────────────────────────────────────

def _filter_assignable(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows where cell_type is not one of the three canonical classes."""
    return df[df["cell_type"].isin(CELL_TYPES)].copy()


def _compute_codex_margin(df: pd.DataFrame) -> pd.Series:
    """Return per-row CODEX margin: p_model_winner - max(p_model_others).

    Caller must ensure df contains only rows where cell_type ∈ {cancer, immune, healthy}.
    """
    margins = []
    for _, row in df.iterrows():
        ct = str(row["cell_type"])
        winner = float(row.get(f"p_model_{ct}", 0.0))
        others = [
            float(row.get(f"p_model_{other}", 0.0))
            for other in CELL_TYPES
            if other != ct
        ]
        margins.append(winner - max(others))
    return pd.Series(margins, index=df.index, dtype=float)
