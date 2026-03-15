"""Tests for Astir-first assign_cells Stage 3 behavior."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.spatial

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_cells_cmd() -> list[str]:
    return [sys.executable, "-m", "stages.assign_cells"]


def _small_rect_contour(cx: int, cy: int, half: int = 8) -> list[list[int]]:
    return [
        [cx - half, cy - half],
        [cx + half, cy - half],
        [cx + half, cy + half],
        [cx - half, cy + half],
    ]


def _make_features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Xt": [64.0, 128.0, 192.0],
            "Yt": [64.0, 128.0, 192.0],
            "Pan-CK": [5000.0, 20.0, 5.0],
            "E-cadherin": [2500.0, 15.0, 40.0],
            "CD45": [5.0, 4500.0, 10.0],
            "CD3e": [5.0, 1800.0, 5.0],
            "CD4": [5.0, 900.0, 5.0],
            "CD8a": [5.0, 950.0, 5.0],
            "CD20": [5.0, 700.0, 5.0],
            "CD68": [5.0, 650.0, 5.0],
            "CD163": [5.0, 620.0, 5.0],
            "FOXP3": [5.0, 500.0, 5.0],
            "CD45RO": [5.0, 550.0, 5.0],
            "PD-1": [5.0, 540.0, 5.0],
            "SMA": [5.0, 5.0, 2500.0],
            "CD31": [5.0, 5.0, 2400.0],
            "Ki67": [10.0, 90.0, 10.0],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_cellvit_prior_epithelial_is_split():
    from stages.assign_cells import cellvit_prior_probs

    p = cellvit_prior_probs(5)
    assert p["cancer"] == pytest.approx(0.5)
    assert p["healthy"] == pytest.approx(0.5)
    assert p["immune"] == pytest.approx(0.0)


def test_compute_state_thresholds_missing_ki67_is_inf():
    from stages.assign_cells import compute_state_thresholds

    df = pd.DataFrame({"Xt": [1.0], "Yt": [2.0]})
    thresholds = compute_state_thresholds(df, 75)
    assert thresholds["Ki67"] == float("inf")


def test_assign_state_dead_override():
    from stages.assign_cells import assign_state

    row = pd.Series({"Ki67": 1000.0})
    thresholds = {"Ki67": 1.0}
    assert assign_state(row, thresholds, type_cellvit=4) == "dead"


def test_compute_type_probabilities_rule_prefers_expected_classes():
    from stages.assign_cells import compute_type_probabilities

    df = _make_features_df()
    log = logging.getLogger("test-rule")
    probs, used, _ = compute_type_probabilities(
        df, classifier="rule", allow_astir_fallback=False, log=log
    )

    assert used == "rule"
    top = probs.idxmax(axis=1).tolist()
    assert top[0] == "cancer"
    assert top[1] == "immune"
    assert top[2] == "healthy"


def test_collapse_astir_probabilities_groups_fine_types():
    from stages.assign_cells import _collapse_astir_probabilities

    fine = pd.DataFrame(
        {
            "epithelial": [0.50, 0.10],
            "cd4_t": [0.20, 0.20],
            "b_cell": [0.10, 0.30],
            "macrophage": [0.05, 0.05],
            "endothelial": [0.15, 0.35],
        },
        index=["a", "b"],
    )

    collapsed = _collapse_astir_probabilities(fine)
    assert collapsed.loc["a", "cancer"] == pytest.approx(0.50)
    assert collapsed.loc["a", "immune"] == pytest.approx(0.35)
    assert collapsed.loc["a", "healthy"] == pytest.approx(0.15)
    assert collapsed.loc["b", "cancer"] == pytest.approx(0.10)
    assert collapsed.loc["b", "immune"] == pytest.approx(0.55)
    assert collapsed.loc["b", "healthy"] == pytest.approx(0.35)


def test_compute_type_probabilities_astir_mock(monkeypatch):
    import stages.assign_cells as mod

    df = _make_features_df()

    def _fake_astir(expr_df, marker_dict, logger=None):
        return pd.DataFrame(
            {
                "cancer": [0.7, 0.1, 0.2],
                "immune": [0.2, 0.8, 0.2],
                "healthy": [0.1, 0.1, 0.6],
            },
            index=expr_df.index,
        )

    monkeypatch.setattr(mod, "predict_cell_type_probabilities", _fake_astir)

    probs, used, _ = mod.compute_type_probabilities(
        df,
        classifier="astir",
        allow_astir_fallback=False,
        log=logging.getLogger("test-astir"),
    )
    assert used == "astir"
    assert probs.iloc[0]["cancer"] == pytest.approx(0.7)


def test_compute_type_probabilities_codex_prefers_expected_classes():
    from stages.assign_cells import compute_type_probabilities

    df = _make_features_df()
    probs, used, _ = compute_type_probabilities(
        df,
        classifier="codex",
        allow_astir_fallback=False,
        log=logging.getLogger("test-codex"),
    )
    assert used == "codex"
    top = probs.idxmax(axis=1).tolist()
    assert top[0] == "cancer"
    assert top[1] == "immune"
    assert top[2] == "healthy"
    fine_top = probs.attrs["model_fine_top"].tolist()
    assert fine_top[0] == "epithelial"


def test_preprocess_codex_matrix_matches_per_marker_zscore():
    from stages.assign_cells import _preprocess_codex_matrix

    df = pd.DataFrame(
        {
            "marker_a": [0.0, 10.0, 1000.0],
            "marker_b": [5.0, 5.0, 5.0],
        }
    )

    normalized = _preprocess_codex_matrix(df)
    expected_a = (df["marker_a"] - df["marker_a"].mean()) / df["marker_a"].std(ddof=0)

    assert normalized["marker_a"].to_numpy() == pytest.approx(expected_a.to_numpy())
    assert normalized["marker_b"].to_numpy() == pytest.approx([0.0, 0.0, 0.0])


def test_compute_type_probabilities_astir_fallback(monkeypatch):
    import stages.assign_cells as mod

    df = _make_features_df()

    def _raise(*_args, **_kwargs):
        raise mod.AstirUnavailableError("astir missing")

    monkeypatch.setattr(mod, "predict_cell_type_probabilities", _raise)

    probs, used, _ = mod.compute_type_probabilities(
        df,
        classifier="astir",
        allow_astir_fallback=True,
        log=logging.getLogger("test-fallback"),
    )
    assert used == "rule_fallback"
    assert set(probs.columns) == {"cancer", "immune", "healthy"}


def test_compute_type_probabilities_astir_no_fallback_raises(monkeypatch):
    import stages.assign_cells as mod

    df = _make_features_df()

    def _raise(*_args, **_kwargs):
        raise mod.AstirUnavailableError("astir missing")

    monkeypatch.setattr(mod, "predict_cell_type_probabilities", _raise)

    with pytest.raises(mod.AstirUnavailableError):
        mod.compute_type_probabilities(
            df,
            classifier="astir",
            allow_astir_fallback=False,
            log=logging.getLogger("test-no-fallback"),
        )


def test_match_cells_mismatch_downgrades_confidence_to_medium():
    from stages.assign_cells import build_csv_index, match_cells

    df = pd.DataFrame(
        {
            "Xt": [100.0],
            "Yt": [100.0],
            "Ki67": [5.0],
            "p_model_cancer": [0.90],
            "p_model_immune": [0.05],
            "p_model_healthy": [0.05],
        }
    )
    tree = build_csv_index(df, "Xt", "Yt")
    cells = [
        {
            "centroid": [100, 100],
            "contour": _small_rect_contour(100, 100),
            "type_cellvit": 2,  # immune prior; model prefers cancer
        }
    ]

    out = match_cells(
        cells,
        tree,
        df,
        state_thresholds={"Ki67": 50.0},
        x0=0,
        y0=0,
        max_dist=5.0,
        coord_scale=1.0,
        model_weight=0.85,
    )
    assert out[0]["cell_type"] == "cancer"
    assert out[0]["is_mismatch"] is True
    assert out[0]["cell_type_confidence"] == "medium"
    assert out[0]["cell_state"] == "quiescent"


def test_match_cells_unmatched_uses_prior_and_low_confidence():
    from stages.assign_cells import build_csv_index, match_cells

    df = pd.DataFrame(
        {
            "Xt": [500.0],
            "Yt": [500.0],
            "Ki67": [100.0],
            "p_model_cancer": [0.33],
            "p_model_immune": [0.33],
            "p_model_healthy": [0.34],
        }
    )
    tree = build_csv_index(df, "Xt", "Yt")
    cells = [
        {
            "centroid": [10, 10],
            "contour": _small_rect_contour(10, 10),
            "type_cellvit": 5,  # epithelial split prior, tie resolves to cancer
        }
    ]

    out = match_cells(
        cells,
        tree,
        df,
        state_thresholds={"Ki67": 50.0},
        x0=0,
        y0=0,
        max_dist=5.0,
        coord_scale=1.0,
    )
    assert out[0]["cell_type"] == "cancer"
    assert out[0]["cell_type_confidence"] == "low"
    assert out[0]["cell_state"] == "quiescent"


def test_rasterize_cells_returns_rgba():
    from stages.assign_cells import CELL_TYPE_COLORS, rasterize_cells

    cells = [
        {
            "centroid": [30, 30],
            "contour": _small_rect_contour(30, 30, half=6),
            "cell_type": "cancer",
            "cell_state": "quiescent",
        }
    ]
    out = rasterize_cells(cells, patch_size=64, color_key="cell_type", color_map=CELL_TYPE_COLORS)
    assert out.shape == (64, 64, 4)
    assert out.dtype == np.uint8
    assert out[30, 30, 3] > 0


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


def test_cli_rule_mode_creates_outputs_and_summary(tmp_path):
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    out_dir = tmp_path / "out"

    cell_data = {
        "patch": "0_0",
        "cells": [
            {
                "centroid": [64, 64],
                "contour": _small_rect_contour(64, 64),
                "bbox": [[54, 54], [74, 74]],
                "type_cellvit": 1,
                "type_prob": 0.9,
            }
        ],
    }
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "Xt,Yt,Pan-CK,E-cadherin,CD45,CD3e,CD4,CD8a,CD20,CD68,CD163,FOXP3,CD45RO,PD-1,SMA,CD31,Ki67\n"
        "64,64,5000,2500,5,5,5,5,5,5,5,5,5,5,5,5,10\n"
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "patch_size": 256,
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
        "--classifier",
        "rule",
        "--csv-mpp",
        "1.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr

    assert (out_dir / "cell_types" / "0_0.png").exists()
    assert (out_dir / "cell_states" / "0_0.png").exists()
    assignments_csv = out_dir / "cell_assignments.csv"
    assert assignments_csv.exists()

    summary = json.loads((out_dir / "cell_summary.json").read_text())
    assert summary["classifier_requested"] == "rule"
    assert summary["classifier_used"] == "rule"
    assert summary["cell_types"].get("cancer", 0) >= 1
    assert "confidence" in summary
    assert summary["cell_assignments_csv"] == str(assignments_csv)

    assignments = pd.read_csv(assignments_csv)
    expected_columns = {
        "patch_id",
        "type_cellvit",
        "type_cellvit_prior",
        "type_astir",
        "type_astir_fine",
        "cell_type",
        "cell_state",
        "cell_type_confidence",
        "is_mismatch",
        "p_model_cancer",
        "p_model_immune",
        "p_model_healthy",
        "p_final_cancer",
        "p_final_immune",
        "p_final_healthy",
        "Pan-CK",
        "E-cadherin",
        "CD45",
        "Ki67",
    }
    assert expected_columns.issubset(assignments.columns)
    assert assignments.loc[0, "patch_id"] == "0_0"
    assert assignments.loc[0, "cell_type"] == "cancer"
    assert assignments.loc[0, "type_cellvit_prior"] == "cancer"


def test_cli_auto_extract_mode_rule_generates_feature_csv(tmp_path):
    cellvit_dir = tmp_path / "cellvit"
    multiplex_dir = tmp_path / "multiplex"
    out_dir = tmp_path / "out"
    cellvit_dir.mkdir()
    multiplex_dir.mkdir()

    cell_data = {
        "patch": "0_0",
        "cells": [
            {
                "centroid": [64, 64],
                "contour": _small_rect_contour(64, 64),
                "bbox": [[54, 54], [74, 74]],
                "type_cellvit": 1,
                "type_prob": 0.9,
            }
        ],
    }
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    patch = np.zeros((3, 256, 256), dtype=np.uint16)
    patch[0, :, :] = 5000  # Pan-CK
    patch[1, :, :] = 2400  # E-cadherin
    patch[2, :, :] = 10  # Ki67
    np.save(multiplex_dir / "0_0.npy", patch)

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "patch_size": 256,
        "channels": ["Pan-CK", "E-cadherin", "Ki67"],
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index_data))

    cmd = [
        *_assign_cells_cmd(),
        "--cellvit-dir",
        str(cellvit_dir),
        "--multiplex-dir",
        str(multiplex_dir),
        "--index",
        str(index_path),
        "--out",
        str(out_dir),
        "--classifier",
        "rule",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr

    generated = out_dir / "cellvit_mx_features.csv"
    assert generated.exists()

    summary = json.loads((out_dir / "cell_summary.json").read_text())
    assert summary["feature_source"] == "cellvit_mx_auto"
    assert summary["classifier_used"] == "rule"


def test_cli_astir_mode_with_fallback_runs(tmp_path):
    cellvit_dir = tmp_path / "cellvit"
    cellvit_dir.mkdir()
    out_dir = tmp_path / "out"

    cell_data = {
        "patch": "0_0",
        "cells": [
            {
                "centroid": [64, 64],
                "contour": _small_rect_contour(64, 64),
                "bbox": [[54, 54], [74, 74]],
                "type_cellvit": 2,
                "type_prob": 0.9,
            }
        ],
    }
    (cellvit_dir / "0_0.json").write_text(json.dumps(cell_data))

    features_path = tmp_path / "features.csv"
    features_path.write_text(
        "Xt,Yt,Pan-CK,E-cadherin,CD45,CD3e,CD4,CD8a,CD20,CD68,CD163,FOXP3,CD45RO,PD-1,SMA,CD31,Ki67\n"
        "64,64,10,10,3000,1000,900,900,800,700,650,600,550,500,5,5,90\n"
    )

    index_data = {
        "patches": [{"i": 0, "j": 0, "x0": 0, "y0": 0, "x1": 256, "y1": 256}],
        "patch_size": 256,
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
        "--classifier",
        "astir",
        "--allow-astir-fallback",
        "--csv-mpp",
        "1.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr

    summary = json.loads((out_dir / "cell_summary.json").read_text())
    assert summary["classifier_requested"] == "astir"
    assert summary["classifier_used"] in {"astir", "rule_fallback"}
    assert "mismatch_rate" in summary
