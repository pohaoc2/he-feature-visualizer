"""Tests for compute_shape_features.py core functions."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from compute_shape_features import compute_circularity, load_mask_regions, match_cells_to_regions


def _make_mask_png(shape: tuple[int, int] = (64, 64), circles=None) -> Path:
    """Write a temporary binary mask PNG and return its path."""
    image = np.zeros(shape, dtype=np.uint8)
    for row, col, radius in circles or []:
        rr, cc = np.ogrid[: shape[0], : shape[1]]
        image[(rr - row) ** 2 + (cc - col) ** 2 <= radius**2] = 255
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(image).save(tmp.name)
    return Path(tmp.name)


def test_load_mask_regions_single_circle() -> None:
    path = _make_mask_png(circles=[(32, 32, 10)])
    regions = load_mask_regions(path)
    assert len(regions) == 1
    assert 290 < regions[0].area < 340


def test_load_mask_regions_two_circles() -> None:
    path = _make_mask_png(shape=(128, 128), circles=[(30, 30, 8), (90, 90, 8)])
    regions = load_mask_regions(path)
    assert len(regions) == 2


def test_load_mask_regions_empty_mask() -> None:
    path = _make_mask_png()
    regions = load_mask_regions(path)
    assert regions == []


def test_compute_circularity_circle() -> None:
    path = _make_mask_png(shape=(128, 128), circles=[(64, 64, 30)])
    region = load_mask_regions(path)[0]
    circularity = compute_circularity(region)
    assert 0.75 < circularity <= 1.0


def test_compute_circularity_zero_perimeter() -> None:
    class _Region:
        area = 1
        perimeter = 0.0

    assert compute_circularity(_Region()) == 0.0


def test_match_cells_to_regions_hit() -> None:
    path = _make_mask_png(circles=[(32, 32, 10)])
    image = np.asarray(Image.open(path))
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise AssertionError("OpenCV should be available for this test suite") from exc
    _, labeled = cv2.connectedComponents((image > 128).astype(np.uint8))
    regions = load_mask_regions(path)
    regions_by_label = {region.label: region for region in regions}

    cells = pd.DataFrame(
        [
            {
                "CellID": 1,
                "PatchID": "0_0",
                "centroid_x_local": 32.0,
                "centroid_y_local": 32.0,
            }
        ]
    )

    result = match_cells_to_regions(cells, labeled, regions_by_label)
    assert not pd.isna(result.loc[0, "circularity"])
    assert result.loc[0, "area_px"] > 0
    assert result.loc[0, "circularity"] == pytest.approx(
        compute_circularity(regions[0]), abs=1e-9
    )


def test_match_cells_to_regions_miss() -> None:
    path = _make_mask_png(circles=[(10, 10, 5)])
    image = np.asarray(Image.open(path))
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise AssertionError("OpenCV should be available for this test suite") from exc
    _, labeled = cv2.connectedComponents((image > 128).astype(np.uint8))
    regions_by_label = {region.label: region for region in load_mask_regions(path)}

    cells = pd.DataFrame(
        [
            {
                "CellID": 2,
                "PatchID": "0_0",
                "centroid_x_local": 55.0,
                "centroid_y_local": 55.0,
            }
        ]
    )

    result = match_cells_to_regions(cells, labeled, regions_by_label)
    assert pd.isna(result.loc[0, "circularity"])
