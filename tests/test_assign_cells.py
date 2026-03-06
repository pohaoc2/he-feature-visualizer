"""
Tests for assign_cells.py — Stage 3 of the histopathology pipeline.

Stage 3 contract:
  - Build a KD-tree index from CRC02.csv global pixel coordinates.
  - Match each CellViT-segmented cell to its nearest CSV row via KD-tree.
  - Assign a cell type ('tumor', 'immune', 'stromal', 'vasculature', 'other')
    and a cell state ('proliferating', 'emt', 'other') from marker intensities.
  - Rasterize cell contours as filled RGBA PNG images (one per patch).

All tests use synthetic data — no real TIFF or CSV files are required.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.spatial

# ---------------------------------------------------------------------------
# Color map constants (must match assign_cells.py)
# ---------------------------------------------------------------------------

CELL_TYPE_COLORS = {
    "tumor": (220, 50, 50, 200),
    "immune": (50, 100, 220, 200),
    "stromal": (50, 180, 50, 200),
    "vasculature": (255, 140, 0, 200),
    "other": (150, 150, 150, 150),
}

CELL_STATE_COLORS = {
    "proliferating": (0, 255, 0, 200),
    "emt": (255, 165, 0, 200),
    "other": (0, 0, 0, 0),
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _assign_cells_cmd() -> list[str]:
    """Return subprocess args for running stages.assign_cells as a module."""
    return [sys.executable, "-m", "stages.assign_cells"]


def _make_cell(centroid, contour, bbox=None, type_cellvit=1, type_prob=0.9):
    """Return a minimal CellViT cell dict."""
    if bbox is None:
        x, y = centroid
        bbox = [[y - 5, x - 5], [y + 5, x + 5]]
    return {
        "centroid": list(centroid),
        "contour": [list(pt) for pt in contour],
        "bbox": bbox,
        "type_cellvit": type_cellvit,
        "type_prob": type_prob,
    }


def _small_rect_contour(cx, cy, half=8):
    """Return a small rectangular contour centred at (cx, cy)."""
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


# ---------------------------------------------------------------------------
# Unit tests — build_csv_index
# ---------------------------------------------------------------------------


def test_build_csv_index_returns_kdtree():
    """
    Contract: build_csv_index returns a scipy.spatial.KDTree whose leaf
    coordinates come from the specified x/y columns of the DataFrame.

    - Result must be an instance of scipy.spatial.KDTree.
    - A nearest-neighbour query for (100, 100) must return distance ~0,
      because that exact point exists in the tree.
    """
    from stages.assign_cells import build_csv_index  # noqa: WPS433

    df = pd.DataFrame({"Xt": [100, 200, 300], "Yt": [100, 200, 300]})
    tree = build_csv_index(df, x_col="Xt", y_col="Yt")

    assert isinstance(tree, scipy.spatial.KDTree), f"Expected KDTree, got {type(tree)}"

    dist, _ = tree.query([100, 100])
    assert (
        dist < 1e-6
    ), f"Query for exact point (100,100) should return distance ~0, got {dist}"


# ---------------------------------------------------------------------------
# Unit tests — assign_type
# ---------------------------------------------------------------------------


def test_assign_type_priority_order():
    """
    Contract: assign_type follows a strict priority order:
      vasculature > tumor > immune > stromal > other

    When multiple markers exceed their threshold the highest-priority type wins.
    When only one marker is elevated the corresponding type is returned.
    """
    from stages.assign_cells import assign_type  # noqa: WPS433

    thresholds = {
        "CD31": 500.0,
        "Keratin": 500.0,
        "CD45": 500.0,
        "aSMA": 500.0,
    }

    # All markers above threshold → tumor wins (vasculature removed as cell type)
    all_high = pd.Series(
        {"CD31": 1000.0, "Keratin": 1000.0, "CD45": 1000.0, "aSMA": 1000.0}
    )
    assert (
        assign_type(all_high, thresholds) == "tumor"
    ), "When all markers are high, tumor (Keratin) wins"

    # Only Keratin → tumor
    tumor_row = pd.Series(
        {"CD31": 0.0, "Keratin": 1000.0, "CD45": 1000.0, "aSMA": 1000.0}
    )
    assert (
        assign_type(tumor_row, thresholds) == "tumor"
    ), "Keratin should win over CD45/aSMA"

    # Only CD45 → immune
    immune_row = pd.Series(
        {"CD31": 0.0, "Keratin": 0.0, "CD45": 1000.0, "aSMA": 1000.0}
    )
    assert (
        assign_type(immune_row, thresholds) == "immune"
    ), "Without CD31/Keratin, CD45 should win over aSMA"

    # Only aSMA → stromal
    stromal_row = pd.Series({"CD31": 0.0, "Keratin": 0.0, "CD45": 0.0, "aSMA": 1000.0})
    assert assign_type(stromal_row, thresholds) == "stromal"

    # No markers above threshold → other
    other_row = pd.Series({"CD31": 0.0, "Keratin": 0.0, "CD45": 0.0, "aSMA": 0.0})
    assert assign_type(other_row, thresholds) == "other"

    # Each type in isolation (CD31 now maps to stromal, no vasculature cell type)
    for marker, expected_type in [
        ("CD31", "stromal"),
        ("Keratin", "tumor"),
        ("CD45", "immune"),
        ("aSMA", "stromal"),
    ]:
        single = pd.Series({m: (1000.0 if m == marker else 0.0) for m in thresholds})
        assert (
            assign_type(single, thresholds) == expected_type
        ), f"Only {marker} high should yield '{expected_type}'"


def test_assign_type_missing_markers():
    """
    Contract: assign_type treats any marker absent from the row as 0.

    A row with no marker columns at all must return 'other' without raising
    an exception.
    """
    from stages.assign_cells import assign_type  # noqa: WPS433

    empty_row = pd.Series(dtype=float)
    thresholds = {"CD31": 500.0, "Keratin": 500.0, "CD45": 500.0, "aSMA": 500.0}

    result = assign_type(empty_row, thresholds)
    assert result == "other", f"Empty row should yield 'other', got '{result}'"


# ---------------------------------------------------------------------------
# Unit tests — assign_state
# ---------------------------------------------------------------------------


def test_assign_state_proliferating():
    """
    Contract: assign_state returns 'proliferating' when Ki67 OR PCNA exceeds
    its threshold, and this check takes priority over EMT.

    - Ki67 high alone → 'proliferating'
    - PCNA high alone → 'proliferating'
    - Both Ki67 and PCNA high (with EMT conditions also met) → 'proliferating'
    """
    from stages.assign_cells import assign_state  # noqa: WPS433

    thresholds = {
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,  # 5th-percentile threshold (low)
    }

    ki67_row = pd.Series(
        {"Ki67": 1000.0, "PCNA": 0.0, "Vimentin": 0.0, "Ecadherin": 500.0}
    )
    assert (
        assign_state(ki67_row, thresholds) == "proliferating"
    ), "Ki67 above threshold → 'proliferating'"

    pcna_row = pd.Series(
        {"Ki67": 0.0, "PCNA": 1000.0, "Vimentin": 0.0, "Ecadherin": 500.0}
    )
    assert (
        assign_state(pcna_row, thresholds) == "proliferating"
    ), "PCNA above threshold → 'proliferating'"

    # Both proliferating and EMT conditions met → proliferating wins
    both_row = pd.Series(
        {"Ki67": 1000.0, "PCNA": 1000.0, "Vimentin": 1000.0, "Ecadherin": 0.0}
    )
    assert (
        assign_state(both_row, thresholds) == "proliferating"
    ), "proliferating check takes priority over EMT"


def test_assign_state_emt():
    """
    Contract: assign_state returns 'emt' when Vimentin is high AND Ecadherin
    is LOW (below the 5th-percentile threshold), provided no proliferation
    marker is elevated.

    - High Vimentin + Low Ecadherin → 'emt'
    - High Vimentin + High Ecadherin → 'other' (Ecadherin not sufficiently low)
    - Low Vimentin + Low Ecadherin → 'other' (Vimentin not high enough)
    """
    from stages.assign_cells import assign_state  # noqa: WPS433

    thresholds = {
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
    }

    emt_row = pd.Series(
        {"Ki67": 0.0, "PCNA": 0.0, "Vimentin": 1000.0, "Ecadherin": 50.0}
    )
    assert (
        assign_state(emt_row, thresholds) == "emt"
    ), "High Vimentin + low Ecadherin (< threshold) → 'emt'"

    high_ecad_row = pd.Series(
        {"Ki67": 0.0, "PCNA": 0.0, "Vimentin": 1000.0, "Ecadherin": 500.0}
    )
    assert (
        assign_state(high_ecad_row, thresholds) == "other"
    ), "High Vimentin but Ecadherin NOT low → 'other'"

    low_vim_row = pd.Series(
        {"Ki67": 0.0, "PCNA": 0.0, "Vimentin": 0.0, "Ecadherin": 50.0}
    )
    assert (
        assign_state(low_vim_row, thresholds) == "other"
    ), "Low Vimentin even with low Ecadherin → 'other'"


# ---------------------------------------------------------------------------
# Unit tests — match_cells
# ---------------------------------------------------------------------------


def test_match_cells_finds_nearby_cell():
    """
    Contract: match_cells assigns 'tumor' when the nearest CSV cell has
    Keratin above threshold and falls within max_dist.

    Setup:
    - CSV has one cell at global coords (128, 128) with high Keratin.
    - Patch origin is (x0=0, y0=0), so local centroid (128, 128) maps to
      global (128, 128) — distance to CSV cell is 0, well within max_dist=15.

    Verifies: resulting cell dict contains cell_type == 'tumor'.
    """
    from stages.assign_cells import build_csv_index, match_cells  # noqa: WPS433

    thresholds = {
        "CD31": 500.0,
        "Keratin": 500.0,
        "CD45": 500.0,
        "aSMA": 500.0,
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
    }

    df = pd.DataFrame(
        {
            "Xt": [128.0],
            "Yt": [128.0],
            "CD31": [0.0],
            "Keratin": [5000.0],
            "CD45": [0.0],
            "aSMA": [0.0],
            "Ki67": [0.0],
            "PCNA": [0.0],
            "Vimentin": [0.0],
            "Ecadherin": [500.0],
        }
    )

    tree = build_csv_index(df, x_col="Xt", y_col="Yt")

    cell = _make_cell(
        centroid=[128, 128],
        contour=_small_rect_contour(128, 128),
    )

    result = match_cells(
        cells=[cell],
        kdtree=tree,
        df=df,
        thresholds=thresholds,
        x0=0,
        y0=0,
        max_dist=15.0,
    )

    assert len(result) == 1
    assert (
        result[0]["cell_type"] == "tumor"
    ), f"Expected cell_type='tumor', got '{result[0]['cell_type']}'"


def test_match_cells_unmatched_when_far():
    """
    Contract: match_cells assigns cell_type='other' when the nearest CSV cell
    is farther than max_dist.

    Setup:
    - CSV has one cell at global (500, 500).
    - Patch origin is (x0=0, y0=0).
    - CellViT cell centroid [10, 10] maps to global (10, 10).
    - Distance ≈ 693 pixels >> max_dist=15.

    Verifies: cell_type == 'other', cell_state == 'other'.
    """
    from stages.assign_cells import build_csv_index, match_cells  # noqa: WPS433

    thresholds = {
        "CD31": 500.0,
        "Keratin": 500.0,
        "CD45": 500.0,
        "aSMA": 500.0,
        "Ki67": 500.0,
        "PCNA": 500.0,
        "Vimentin": 500.0,
        "Ecadherin": 200.0,
    }

    df = pd.DataFrame(
        {
            "Xt": [500.0],
            "Yt": [500.0],
            "Keratin": [5000.0],
            "CD31": [0.0],
            "CD45": [0.0],
            "aSMA": [0.0],
            "Ki67": [0.0],
            "PCNA": [0.0],
            "Vimentin": [0.0],
            "Ecadherin": [500.0],
        }
    )

    tree = build_csv_index(df, x_col="Xt", y_col="Yt")

    cell = _make_cell(
        centroid=[10, 10],
        contour=_small_rect_contour(10, 10),
    )

    result = match_cells(
        cells=[cell],
        kdtree=tree,
        df=df,
        thresholds=thresholds,
        x0=0,
        y0=0,
        max_dist=15.0,
    )

    assert len(result) == 1
    assert (
        result[0]["cell_type"] == "other"
    ), f"No CSV match within max_dist → expected 'other', got '{result[0]['cell_type']}'"
    assert (
        result[0]["cell_state"] == "other"
    ), f"No CSV match within max_dist → expected state 'other', got '{result[0]['cell_state']}'"


# ---------------------------------------------------------------------------
# Unit tests — rasterize_cells
# ---------------------------------------------------------------------------


def test_rasterize_cells_produces_rgba_array():
    """
    Contract: rasterize_cells returns a (patch_size, patch_size, 4) uint8 RGBA
    array. Pixels inside each cell's filled contour reflect its color_map entry
    and pixels outside all cells are transparent (alpha == 0).

    Setup:
    - Cell A: cell_type='tumor', contour is a 30×30 square centred at (60, 60).
    - Cell B: cell_type='immune', contour is a 30×30 square centred at (180, 180).
    - Background pixel (0, 0) must be fully transparent.

    Verifies:
    - output.shape == (256, 256, 4), dtype uint8
    - Center of tumor contour: R > 150, B < 100
    - Center of immune contour: B > 150, R < 100
    - Pixel (0, 0) has alpha == 0
    """
    from stages.assign_cells import rasterize_cells  # noqa: WPS433

    tumor_cell = _make_cell(
        centroid=[60, 60],
        contour=_small_rect_contour(60, 60, half=15),
    )
    tumor_cell["cell_type"] = "tumor"
    tumor_cell["cell_state"] = "other"

    immune_cell = _make_cell(
        centroid=[180, 180],
        contour=_small_rect_contour(180, 180, half=15),
    )
    immune_cell["cell_type"] = "immune"
    immune_cell["cell_state"] = "other"

    output = rasterize_cells(
        cells=[tumor_cell, immune_cell],
        patch_size=256,
        color_key="cell_type",
        color_map=CELL_TYPE_COLORS,
    )

    assert output.shape == (
        256,
        256,
        4,
    ), f"Expected shape (256, 256, 4), got {output.shape}"
    assert output.dtype == np.uint8, f"Expected dtype uint8, got {output.dtype}"

    # Tumor region: red-ish (R > 150, B < 100)
    tumor_pixel = output[60, 60]
    assert (
        tumor_pixel[0] > 150
    ), f"Tumor center R channel should be > 150, got {tumor_pixel[0]}"
    assert (
        tumor_pixel[2] < 100
    ), f"Tumor center B channel should be < 100, got {tumor_pixel[2]}"

    # Immune region: blue-ish (B > 150, R < 100)
    immune_pixel = output[180, 180]
    assert (
        immune_pixel[2] > 150
    ), f"Immune center B channel should be > 150, got {immune_pixel[2]}"
    assert (
        immune_pixel[0] < 100
    ), f"Immune center R channel should be < 100, got {immune_pixel[0]}"

    # Background: transparent
    assert (
        output[0, 0, 3] == 0
    ), f"Background pixel (0,0) should be transparent (alpha=0), got {output[0, 0, 3]}"


def test_rasterize_cells_state_other_is_transparent():
    """
    Contract: rasterize_cells produces a fully transparent output when every
    cell has cell_state='other' and color_key='cell_state'.

    CELL_STATE_COLORS['other'] == (0, 0, 0, 0), so drawing it is equivalent
    to drawing nothing — the entire alpha channel must remain 0.
    """
    from stages.assign_cells import rasterize_cells  # noqa: WPS433

    cell = _make_cell(
        centroid=[128, 128],
        contour=_small_rect_contour(128, 128, half=20),
    )
    cell["cell_type"] = "tumor"
    cell["cell_state"] = "other"

    output = rasterize_cells(
        cells=[cell],
        patch_size=256,
        color_key="cell_state",
        color_map=CELL_STATE_COLORS,
    )

    assert output.shape == (
        256,
        256,
        4,
    ), f"Expected shape (256, 256, 4), got {output.shape}"
    assert (
        int(output[..., 3].max()) == 0
    ), "All pixels should be transparent (alpha=0) when state='other'"


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_cli_creates_output_dirs(tmp_path):
    """
    Contract: running assign_cells.py CLI produces cell_types/{i}_{j}.png and
    cell_states/{i}_{j}.png under --out, both 256×256 RGBA.

    Setup:
    - One CellViT JSON for patch 0_0 containing a single cell.
    - A features CSV with one matching row at the cell's global position.
    - A minimal index.json describing the single patch at (x0=0, y0=0).
    """
    from PIL import Image  # noqa: WPS433

    # Build directory structure
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    out_dir = tmp_path / "out"

    # CellViT JSON: one cell with a small rectangular contour
    cell_data = {
        "patch": "0_0",
        "cells": [
            {
                "centroid": [128, 128],
                "contour": _small_rect_contour(128, 128, half=10),
                "bbox": [[118, 118], [138, 138]],
                "type_cellvit": 1,
                "type_prob": 0.9,
            }
        ],
    }
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    # Features CSV: one cell at global (128, 128) with high Keratin
    features_path = tmp_path / "CRC02.csv"
    features_path.write_text(
        "Xt,Yt,CD31,Keratin,CD45,aSMA,Ki67,PCNA,Vimentin,Ecadherin\n"
        "128,128,0,5000,0,0,0,0,0,500\n"
    )

    # index.json: single patch 0_0 at origin
    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "stride": 256,
        "patch_size": 256,
        "tissue_min": 0.0,
        "img_w": 256,
        "img_h": 256,
        "channels": [],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    cmd = [
        *_assign_cells_cmd(),
        "--cellvit-dir",
        str(cellvit_dir),
        "--features-csv",
        str(features_path),
        "--index",
        str(index_path),
        "--out",
        str(out_dir),
        "--max-dist",
        "15.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.assign_cells exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    type_png = out_dir / "cell_types" / "0_0.png"
    state_png = out_dir / "cell_states" / "0_0.png"

    assert type_png.exists(), "cell_types/0_0.png must be created"
    assert state_png.exists(), "cell_states/0_0.png must be created"

    for png_path in (type_png, state_png):
        img = Image.open(str(png_path))
        assert img.size == (
            256,
            256,
        ), f"Expected 256×256 image, got {img.size} for {png_path.name}"
        assert (
            img.mode == "RGBA"
        ), f"Expected RGBA mode, got {img.mode} for {png_path.name}"


def test_cli_skips_patch_with_no_cellvit_json(tmp_path):
    """
    Contract: when a patch listed in index.json has no corresponding CellViT
    JSON in --cellvit-dir, the CLI silently skips it (no crash) and does NOT
    write an output PNG for that patch.

    Setup:
    - index.json lists two patches: 0_0 and 0_1.
    - Only 0_0.json exists in --cellvit-dir; 0_1.json is absent.

    Verifies:
    - CLI exits with code 0.
    - cell_types/0_0.png exists.
    - cell_types/0_1.png does NOT exist.
    """
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    out_dir = tmp_path / "out"

    # Only provide CellViT JSON for patch 0_0
    cell_data = {
        "patch": "0_0",
        "cells": [
            {
                "centroid": [64, 64],
                "contour": _small_rect_contour(64, 64, half=8),
                "bbox": [[56, 56], [72, 72]],
                "type_cellvit": 1,
                "type_prob": 0.9,
            }
        ],
    }
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))
    # 0_1.json intentionally absent

    features_path = tmp_path / "CRC02.csv"
    features_path.write_text(
        "Xt,Yt,CD31,Keratin,CD45,aSMA,Ki67,PCNA,Vimentin,Ecadherin\n"
        "64,64,0,5000,0,0,0,0,0,500\n"
    )

    # index.json lists both patches (0_0 and 0_1)
    index_data = {
        "patches": [
            {"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256},
            {"i": 0, "j": 1, "x0": 256, "y0": 0, "x1": 512, "y1": 256},
        ],
        "stride": 256,
        "patch_size": 256,
        "tissue_min": 0.0,
        "img_w": 512,
        "img_h": 256,
        "channels": [],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    cmd = [
        *_assign_cells_cmd(),
        "--cellvit-dir",
        str(cellvit_dir),
        "--features-csv",
        str(features_path),
        "--index",
        str(index_path),
        "--out",
        str(out_dir),
        "--max-dist",
        "15.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, (
        f"stages.assign_cells exited with code {result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )

    assert (
        out_dir / "cell_types" / "0_0.png"
    ).exists(), "cell_types/0_0.png must be created for the patch with a CellViT JSON"
    assert not (
        out_dir / "cell_types" / "0_1.png"
    ).exists(), "cell_types/0_1.png must NOT be created when its CellViT JSON is absent"
