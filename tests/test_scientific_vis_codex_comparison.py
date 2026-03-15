# tests/test_scientific_vis_codex_comparison.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import tools.scientific_vis_codex_comparison as comp

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _base_df() -> pd.DataFrame:
    """Minimal synthetic DataFrame with all required columns."""
    return pd.DataFrame([
        {
            "cell_type": "cancer",
            "cellvit_mapped_type": "cancer",
            "type_astir": "cancer",
            "is_mismatch": False,
            "p_model_cancer": 0.80,
            "p_model_immune": 0.10,
            "p_model_healthy": 0.10,
            "patch_id": "p0",
            "centroid_x_local": 20.0,
            "centroid_y_local": 20.0,
        },
        {
            "cell_type": "immune",
            "cellvit_mapped_type": "immune",
            "type_astir": "immune",
            "is_mismatch": False,
            "p_model_cancer": 0.05,
            "p_model_immune": 0.85,
            "p_model_healthy": 0.10,
            "patch_id": "p0",
            "centroid_x_local": 40.0,
            "centroid_y_local": 40.0,
        },
        {
            "cell_type": "healthy",
            "cellvit_mapped_type": "cancer",   # mismatch
            "type_astir": "healthy",
            "is_mismatch": True,
            "p_model_cancer": 0.05,
            "p_model_immune": 0.10,
            "p_model_healthy": 0.85,
            "patch_id": "p0",
            "centroid_x_local": 60.0,
            "centroid_y_local": 60.0,
        },
        {
            "cell_type": "other",   # must be excluded
            "cellvit_mapped_type": "other",
            "type_astir": "other",
            "is_mismatch": False,
            "p_model_cancer": 0.30,
            "p_model_immune": 0.40,
            "p_model_healthy": 0.30,
            "patch_id": "p0",
            "centroid_x_local": 80.0,
            "centroid_y_local": 80.0,
        },
    ])


def test_codex_margin_computed_correctly() -> None:
    df = _base_df()
    # Only pass the first 3 rows (cell_type in {cancer, immune, healthy})
    filtered = comp._filter_assignable(df)
    result = comp._compute_codex_margin(filtered)
    # cancer row: 0.80 - max(0.10, 0.10) = 0.70
    assert abs(result.iloc[0] - 0.70) < 1e-6
    # immune row: 0.85 - max(0.05, 0.10) = 0.75
    assert abs(result.iloc[1] - 0.75) < 1e-6
    # healthy row: 0.85 - max(0.05, 0.10) = 0.75
    assert abs(result.iloc[2] - 0.75) < 1e-6


def test_codex_margin_excludes_other_cell_type() -> None:
    """Cells with cell_type == 'other' are excluded before margin computation."""
    df = _base_df()
    filtered = comp._filter_assignable(df)
    assert "other" not in filtered["cell_type"].values
    assert len(filtered) == 3
