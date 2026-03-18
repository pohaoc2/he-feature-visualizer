# tests/test_scientific_vis_codex_comparison.py
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
            "type_codex": "cancer",
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
            "type_codex": "immune",
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
            "type_codex": "healthy",
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
            "type_codex": "other",
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


def _selection_df() -> pd.DataFrame:
    """DataFrame for one class with known margins and mismatch flags."""
    return pd.DataFrame([
        # agree candidates (is_mismatch=False): margin 0.60 and 0.20
        {"cell_type": "cancer", "cellvit_mapped_type": "cancer", "type_codex": "cancer",
         "is_mismatch": False, "codex_margin": 0.60,
         "patch_id": "p0", "centroid_x_local": 10.0, "centroid_y_local": 10.0},
        {"cell_type": "cancer", "cellvit_mapped_type": "cancer", "type_codex": "cancer",
         "is_mismatch": False, "codex_margin": 0.20,
         "patch_id": "p0", "centroid_x_local": 20.0, "centroid_y_local": 20.0},
        # disagree candidates (is_mismatch=True): margin 0.70 and 0.40
        {"cell_type": "cancer", "cellvit_mapped_type": "immune", "type_codex": "cancer",
         "is_mismatch": True, "codex_margin": 0.70,
         "patch_id": "p0", "centroid_x_local": 30.0, "centroid_y_local": 30.0},
        {"cell_type": "cancer", "cellvit_mapped_type": "immune", "type_codex": "cancer",
         "is_mismatch": True, "codex_margin": 0.40,
         "patch_id": "p0", "centroid_x_local": 40.0, "centroid_y_local": 40.0},
    ])


def test_example_selection_agree_medium_disagree() -> None:
    df = _selection_df()
    examples = comp._select_examples(df)

    # agree: highest-margin non-mismatch cell (margin 0.60)
    agree = examples.get("agree")
    assert agree is not None
    assert abs(float(agree["codex_margin"]) - 0.60) < 1e-6

    # medium: lowest-margin non-mismatch, not same as agree (margin 0.20)
    medium = examples.get("medium")
    assert medium is not None
    assert abs(float(medium["codex_margin"]) - 0.20) < 1e-6
    # must not be the same row as agree
    assert medium["centroid_x_local"] != agree["centroid_x_local"]

    # disagree: highest-margin mismatch cell (margin 0.70, not 0.40)
    disagree = examples.get("disagree")
    assert disagree is not None
    assert abs(float(disagree["codex_margin"]) - 0.70) < 1e-6


def test_placeholder_when_disagree_bucket_empty() -> None:
    """When all rows are non-mismatch, disagree bucket returns None."""
    df = _selection_df().copy()
    df["is_mismatch"] = False   # no mismatches
    examples = comp._select_examples(df)
    assert examples["disagree"] is None
    # agree and medium still selected
    assert examples["agree"] is not None
    assert examples["medium"] is not None


def test_missing_canonical_marker_renders_placeholder() -> None:
    """If SMA column absent from norm_vals, bar renders as 'n/a' without error."""
    markers = {
        "cancer_marker": "Pan-CK",
        "immune_marker": "CD45",
        "healthy_marker": "SMA",
    }
    # norm_vals only has Pan-CK and CD45; SMA missing
    norm_vals: dict[str, np.ndarray] = {
        "Pan-CK": np.array([0.8]),
        "CD45":   np.array([0.2]),
    }
    row = pd.Series({
        "cell_type": "cancer",
        "cellvit_mapped_type": "cancer",
        "type_codex": "cancer",
        "codex_margin": 0.60,
    })
    fig, ax = plt.subplots(figsize=(3, 2))
    # should not raise
    comp._plot_marker_bar(ax, row, norm_vals, markers, row_index=0)
    plt.close(fig)


def _make_processed_codex(tmp_path: Path) -> tuple[Path, Path]:
    """Create minimal processed dir + CSV for codex comparison smoke test."""
    processed = tmp_path / "proc"
    (processed / "he").mkdir(parents=True)

    patch_id = "0_0"
    # H&E patch
    Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8), "RGB").save(
        processed / "he" / f"{patch_id}.png"
    )

    rows = []
    for i, (ct, cvit, codex, mismatch) in enumerate([
        ("cancer",  "cancer",  "cancer",  False),
        ("cancer",  "immune",  "cancer",  True),
        ("immune",  "immune",  "immune",  False),
        ("immune",  "cancer",  "immune",  True),
        ("healthy", "healthy", "healthy", False),
        ("healthy", "cancer",  "healthy", True),
    ]):
        rows.append({
            "patch_id": patch_id,
            "cell_type": ct,
            "cellvit_mapped_type": cvit,
            "type_codex": codex,
            "is_mismatch": mismatch,
            "type_cellvit": 1,
            "cell_type_confidence": "high",
            "centroid_x_local": float(10 + i * 8),
            "centroid_y_local": float(10 + i * 8),
            "p_model_cancer":  0.80 if ct == "cancer" else 0.05,
            "p_model_immune":  0.80 if ct == "immune" else 0.05,
            "p_model_healthy": 0.80 if ct == "healthy" else 0.10,
            "p_final_cancer":  0.80 if ct == "cancer" else 0.05,
            "p_final_immune":  0.80 if ct == "immune" else 0.05,
            "p_final_healthy": 0.80 if ct == "healthy" else 0.10,
            "Pan-CK": float(3000 if ct == "cancer" else 100),
            "CD45":   float(2000 if ct == "immune" else 50),
            "SMA":    float(1500 if ct == "healthy" else 30),
        })

    csv_path = processed / "cell_assignments.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return processed, csv_path


def test_codex_comparison_smoke(tmp_path: Path) -> None:
    """Full CLI run: exit code 0 and output PNG exists."""
    processed, csv_path = _make_processed_codex(tmp_path)
    out_prefix = processed / "codex_comparison"
    cmd = [
        sys.executable, "-m", "tools.scientific_vis_codex_comparison",
        "--processed", str(processed),
        "--assignments-csv", str(csv_path),
        "--out-prefix", str(out_prefix),
        "--formats", "png",
        "--dpi", "72",
        "--cancer-marker", "Pan-CK",
        "--immune-marker", "CD45",
        "--healthy-marker", "SMA",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()
