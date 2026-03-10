"""Unit tests for tools/viz_registration_debug.py helper functions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tools.viz_registration_debug as v


def test_extract_cells_from_common_dict_layout() -> None:
    """Should return dict entries from known list keys."""
    data = {
        "cells": [{"id": 1}, {"id": 2}, "bad", 123],
        "ignored": [{"id": 9}],
    }
    got = v._extract_cells(data)  # pylint: disable=protected-access
    assert got == [{"id": 1}, {"id": 2}]


def test_extract_cells_from_instances_dict_layout() -> None:
    """Should handle dict-of-dicts instance format."""
    data = {"0": {"id": 1}, "1": {"id": 2}}
    got = v._extract_cells(data)  # pylint: disable=protected-access
    assert got == [{"id": 1}, {"id": 2}]


def test_extract_contour_closes_open_polygon() -> None:
    """Open contour should be closed by appending first vertex."""
    contour = [[1, 1], [3, 1], [3, 3], [1, 3]]
    got = v._extract_contour({"contour": contour})  # pylint: disable=protected-access
    assert got is not None
    assert got.dtype == np.float32
    assert got.shape == (5, 2)
    assert np.allclose(got[0], got[-1])


def test_extract_contour_rejects_too_short() -> None:
    """Contours with fewer than 3 points should be ignored."""
    got = v._extract_contour(
        {"contour": [[1, 1], [2, 2]]}
    )  # pylint: disable=protected-access
    assert got is None


def test_load_he_contours_offsets_local_to_global(tmp_path: Path) -> None:
    """Local patch contours should be shifted by patch (x0, y0)."""
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir(parents=True, exist_ok=True)

    (cellvit_dir / "10_20.json").write_text(
        json.dumps(
            {
                "cells": [
                    {"contour": [[1, 1], [4, 1], [4, 5], [1, 5]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    contours = v.load_he_contours(cellvit_dir, [{"x0": 10, "y0": 20}])
    assert len(contours) == 1
    pts = contours[0]
    # first point shifted from (1,1) to (11,21)
    assert np.allclose(pts[0], [11, 21])


def test_filter_contours_bbox_keeps_intersections() -> None:
    """Contour bbox intersection test should include overlapping contours."""
    c1 = np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], dtype=np.float32)
    c2 = np.array([[50, 50], [52, 50], [52, 52], [50, 52], [50, 50]], dtype=np.float32)
    got = v._filter_contours_bbox(
        [c1, c2], 1, 10, 1, 10
    )  # pylint: disable=protected-access
    assert len(got) == 1
    assert np.array_equal(got[0], c1)


def test_transform_contours_applies_affine_translation() -> None:
    """Affine translation should be applied to every contour point."""
    m = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, -3.0]], dtype=np.float64)
    c = np.array([[1, 2], [3, 4], [1, 2]], dtype=np.float32)
    out = v._transform_contours(m, [c])  # pylint: disable=protected-access
    assert len(out) == 1
    assert np.allclose(
        out[0], np.array([[11, -1], [13, 1], [11, -1]], dtype=np.float64)
    )


def test_add_contours_returns_count() -> None:
    """Drawing helper should return number of drawn contours."""
    fig, ax = plt.subplots()
    contours = [
        np.array([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]], dtype=np.float32),
        np.array([[5, 5], [7, 5], [7, 7], [5, 7], [5, 5]], dtype=np.float32),
    ]
    n = v._add_contours(  # pylint: disable=protected-access
        ax=ax, contours=contours, color="cyan", linewidth=1.0, alpha=0.5
    )
    plt.close(fig)
    assert n == 2
