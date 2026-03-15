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


def _select_examples(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """Select one cell per bucket (agree, medium, disagree) with de-duplication.

    df must already have a 'codex_margin' column.
    Returns dict with keys 'agree', 'medium', 'disagree' (value is None if bucket empty).
    """
    used: set[int] = set()
    results: dict[str, pd.Series | None] = {}

    # agree: is_mismatch=False, highest codex_margin
    non_mismatch = df[~df["is_mismatch"]].sort_values("codex_margin", ascending=False)
    results["agree"] = None
    for idx, row in non_mismatch.iterrows():
        if idx not in used:
            used.add(idx)
            results["agree"] = row
            break

    # medium: is_mismatch=False, lowest codex_margin (least confident agreement), not already used
    results["medium"] = None
    for idx, row in non_mismatch.sort_values("codex_margin", ascending=True).iterrows():
        if idx not in used:
            used.add(idx)
            results["medium"] = row
            break

    # disagree: is_mismatch=True, highest codex_margin
    results["disagree"] = None
    for idx, row in df[df["is_mismatch"]].sort_values("codex_margin", ascending=False).iterrows():
        if idx not in used:
            used.add(idx)
            results["disagree"] = row
            break

    return results
