"""Tests for cell_vis.py data loading, plotting, and summary generation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from cell_vis import (
    load_data,
    plot_violin_area,
    plot_violin_circularity,
    plot_violin_markers,
    write_summary_md,
    zscore_markers,
)


def _make_csv_pair(tmp_dir: Path) -> None:
    """Write a minimal assignments/shapes pair for testing.

    Uses 6 cells (2 per type) so 99th-percentile clipping never empties any group.
    """
    assignments = pd.DataFrame(
        {
            "CellID": [1, 2, 3, 4, 5, 6],
            "PatchID": ["0_0"] * 6,
            "cell_type": ["cancer", "cancer", "immune", "immune", "healthy", "healthy"],
            "cell_state": ["proliferative", "quiescent", "quiescent", "quiescent", "quiescent", "quiescent"],
            "Area_cellvit_px": [500, 480, 200, 210, 300, 310],
            "Pan-CK": [10.0, 9.0, 1.0, 1.5, 2.0, 2.5],
            "CD45": [1.0, 1.2, 9.0, 8.5, 2.0, 2.1],
            "Ki67": [5.0, 4.5, 2.0, 1.8, 1.0, 1.1],
        }
    )
    shapes = pd.DataFrame(
        {
            "CellID": [1, 2, 3, 4, 5, 6],
            "PatchID": ["0_0"] * 6,
            "area_px": [480, 470, 190, 195, 290, 285],
            "perimeter_px": [80.0, 79.0, 50.0, 51.0, 65.0, 64.0],
            "circularity": [0.94, 0.93, 0.96, 0.95, 0.92, 0.91],
        }
    )
    assignments.to_csv(tmp_dir / "cell_assignments.csv", index=False)
    shapes.to_csv(tmp_dir / "cell_shape_features.csv", index=False)


def test_load_data_merges_shape_features() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
    assert "circularity" in df.columns
    assert len(df) == 6


def test_load_data_missing_shapes_raises() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        pd.DataFrame({"CellID": [1]}).to_csv(tmp_dir / "cell_assignments.csv", index=False)
        with pytest.raises(FileNotFoundError, match="compute_shape_features"):
            load_data(tmp_dir)


def test_zscore_markers_normalizes() -> None:
    df = pd.DataFrame({"Pan-CK": [1.0, 2.0, 3.0], "CD45": [10.0, 10.0, 10.0]})
    result = zscore_markers(df.copy(), ["Pan-CK", "CD45"])
    assert abs(float(result["Pan-CK"].mean())) < 1e-10
    assert abs(float(result["Pan-CK"].std()) - 1.0) < 1e-6
    assert (result["CD45"] == 0.0).all()


def test_zscore_markers_missing_column_warns(capsys) -> None:
    df = pd.DataFrame({"Pan-CK": [1.0, 2.0]})
    zscore_markers(df.copy(), ["Pan-CK", "MISSING_MARKER"])
    captured = capsys.readouterr()
    assert "MISSING_MARKER" in captured.out


def test_plot_violin_area_saves_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        out = tmp_dir / "violin_area.png"
        plot_violin_area(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_violin_circularity_saves_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        out = tmp_dir / "violin_circularity.png"
        plot_violin_circularity(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_plot_violin_markers_saves_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        out = tmp_dir / "violin_markers.png"
        plot_violin_markers(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_write_summary_md_contains_tables() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        _make_csv_pair(tmp_dir)
        df = load_data(tmp_dir)
        out = tmp_dir / "cell_summary.md"
        write_summary_md(df, out)
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "cancer" in text
        assert "immune" in text
        assert "healthy" in text
        assert "proliferative" in text
        assert "nonproliferative" in text
        assert "cell_type" in text
        assert "%" in text
