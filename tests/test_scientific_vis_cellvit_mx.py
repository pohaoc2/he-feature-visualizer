from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import tools.scientific_vis_cellvit_mx as patch_vis


_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _tool_cmd() -> list[str]:
    return [sys.executable, "-m", "tools.scientific_vis_cellvit_mx"]


def _rgba_overlay(shape: tuple[int, int], rgb: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    out[8:24, 8:24, :3] = np.array(rgb, dtype=np.uint8)
    out[8:24, 8:24, 3] = 180
    return out


def test_add_state_legend_contains_expected_labels() -> None:
    fig, ax = plt.subplots()
    patch_vis._add_state_legend(ax)

    legend = ax.get_legend()
    assert legend is not None
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels == ["proliferative", "quiescent", "dead"]

    plt.close(fig)


def test_state_palette_is_distinct_from_type_palette() -> None:
    assert patch_vis.CELL_STATE_COLORS["proliferative"][:3] != patch_vis.CELL_TYPE_COLORS["healthy"][:3]
    assert patch_vis.CELL_STATE_COLORS["quiescent"][:3] != patch_vis.CELL_TYPE_COLORS["immune"][:3]
    assert patch_vis.CELL_STATE_COLORS["dead"][:3] != patch_vis.MODEL_FINE_COLORS["treg"][:3]


def test_state_colors_cover_all_states() -> None:
    for state in ("proliferative", "quiescent", "dead", "other"):
        assert state in patch_vis.CELL_STATE_COLORS, f"missing state: {state}"
        r, g, b, a = patch_vis.CELL_STATE_COLORS[state]
        assert 0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255
    # All four non-other states should be unique RGB triplets
    non_other = {k: v[:3] for k, v in patch_vis.CELL_STATE_COLORS.items() if k != "other"}
    assert len(set(non_other.values())) == len(non_other), "duplicate state colors"


def test_patch_comparison_figure_smoke(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subdir in ("he", "multiplex", "cellvit", "cell_types", "cell_states"):
        (processed / subdir).mkdir(parents=True, exist_ok=True)

    patch_id = "0_0"
    Image.fromarray(np.full((64, 64, 3), 220, dtype=np.uint8), mode="RGB").save(
        processed / "he" / f"{patch_id}.png"
    )

    mx = np.zeros((3, 64, 64), dtype=np.uint16)
    mx[0, 10:26, 10:26] = 4500
    mx[1, 36:52, 36:52] = 4200
    mx[2, :, :] = 300
    np.save(processed / "multiplex" / f"{patch_id}.npy", mx)

    cellvit_payload = {
        "patch": patch_id,
        "cells": [
            {
                "centroid": [16, 16],
                "contour": [[10, 10], [22, 10], [22, 22], [10, 22]],
                "type_cellvit": 1,
            },
            {
                "centroid": [44, 44],
                "contour": [[38, 38], [50, 38], [50, 50], [38, 50]],
                "type_cellvit": 2,
            },
        ],
    }
    (processed / "cellvit" / f"{patch_id}.json").write_text(
        json.dumps(cellvit_payload),
        encoding="utf-8",
    )

    Image.fromarray(_rgba_overlay((64, 64), (220, 50, 50)), mode="RGBA").save(
        processed / "cell_types" / f"{patch_id}.png"
    )
    Image.fromarray(_rgba_overlay((64, 64), (100, 149, 237)), mode="RGBA").save(
        processed / "cell_states" / f"{patch_id}.png"
    )

    (processed / "index.json").write_text(
        json.dumps(
            {
                "patch_size": 64,
                "channels": ["Pan-CK", "CD45", "SMA"],
            }
        ),
        encoding="utf-8",
    )

    assignments = pd.DataFrame(
        [
            {
                "patch_id": patch_id,
                "type_cellvit": 1,
                "type_cellvit_prior": "cancer",
                "type_codex": "cancer",
                "type_codex_fine": "epithelial",
                "cell_type": "cancer",
                "cell_state": "quiescent",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 16.0,
                "centroid_y_local": 16.0,
                "centroid_x_global": 16.0,
                "centroid_y_global": 16.0,
                "p_model_cancer": 0.92,
                "p_model_immune": 0.04,
                "p_model_healthy": 0.04,
                "p_final_cancer": 0.93,
                "p_final_immune": 0.04,
                "p_final_healthy": 0.03,
                "Pan-CK": 4500.0,
                "CD45": 40.0,
                "SMA": 15.0,
            },
            {
                "patch_id": patch_id,
                "type_cellvit": 2,
                "type_cellvit_prior": "immune",
                "type_codex": "immune",
                "type_codex_fine": "cd4_t",
                "cell_type": "immune",
                "cell_state": "proliferative",
                "cell_type_confidence": "medium",
                "is_mismatch": False,
                "centroid_x_local": 44.0,
                "centroid_y_local": 44.0,
                "centroid_x_global": 44.0,
                "centroid_y_global": 44.0,
                "p_model_cancer": 0.05,
                "p_model_immune": 0.85,
                "p_model_healthy": 0.10,
                "p_final_cancer": 0.05,
                "p_final_immune": 0.88,
                "p_final_healthy": 0.07,
                "Pan-CK": 30.0,
                "CD45": 4200.0,
                "SMA": 25.0,
            },
        ]
    )
    assignments.to_csv(processed / "cell_assignments.csv", index=False)
    (processed / "cell_summary.json").write_text(
        json.dumps({"classifier_used": "codex"}),
        encoding="utf-8",
    )

    out_prefix = processed / "patch_compare"
    cmd = [
        *_tool_cmd(),
        "--processed",
        str(processed),
        "--patch",
        patch_id,
        "--assignments-csv",
        str(processed / "cell_assignments.csv"),
        "--formats",
        "png",
        "--out-prefix",
        str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()


def test_patch_comparison_figure_supports_codex_fine_overlay(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for subdir in ("he", "multiplex", "cellvit", "cell_types", "cell_states"):
        (processed / subdir).mkdir(parents=True, exist_ok=True)

    patch_id = "0_0"
    Image.fromarray(np.full((64, 64, 3), 220, dtype=np.uint8), mode="RGB").save(
        processed / "he" / f"{patch_id}.png"
    )

    mx = np.zeros((3, 64, 64), dtype=np.uint16)
    mx[0, 10:26, 10:26] = 4500
    mx[1, 28:44, 28:44] = 4200
    mx[2, 40:56, 40:56] = 3500
    np.save(processed / "multiplex" / f"{patch_id}.npy", mx)

    cellvit_payload = {
        "patch": patch_id,
        "cells": [
            {
                "centroid": [16, 16],
                "contour": [[10, 10], [22, 10], [22, 22], [10, 22]],
                "type_cellvit": 1,
            },
            {
                "centroid": [36, 36],
                "contour": [[30, 30], [42, 30], [42, 42], [30, 42]],
                "type_cellvit": 2,
            },
            {
                "centroid": [48, 48],
                "contour": [[42, 42], [54, 42], [54, 54], [42, 54]],
                "type_cellvit": 3,
            },
        ],
    }
    (processed / "cellvit" / f"{patch_id}.json").write_text(
        json.dumps(cellvit_payload),
        encoding="utf-8",
    )

    Image.fromarray(_rgba_overlay((64, 64), (220, 50, 50)), mode="RGBA").save(
        processed / "cell_types" / f"{patch_id}.png"
    )
    Image.fromarray(_rgba_overlay((64, 64), (100, 149, 237)), mode="RGBA").save(
        processed / "cell_states" / f"{patch_id}.png"
    )

    (processed / "index.json").write_text(
        json.dumps({"patch_size": 64, "channels": ["Pan-CK", "CD45", "SMA"]}),
        encoding="utf-8",
    )

    assignments = pd.DataFrame(
        [
            {
                "patch_id": patch_id,
                "type_cellvit": 1,
                "type_cellvit_prior": "cancer",
                "type_codex": "cancer",
                "type_codex_fine": "epithelial",
                "cell_type": "cancer",
                "cell_state": "quiescent",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 16.0,
                "centroid_y_local": 16.0,
                "centroid_x_global": 16.0,
                "centroid_y_global": 16.0,
                "p_model_cancer": 0.92,
                "p_model_immune": 0.04,
                "p_model_healthy": 0.04,
                "p_final_cancer": 0.93,
                "p_final_immune": 0.04,
                "p_final_healthy": 0.03,
                "Pan-CK": 4500.0,
                "CD45": 40.0,
                "SMA": 15.0,
            },
            {
                "patch_id": patch_id,
                "type_cellvit": 2,
                "type_cellvit_prior": "immune",
                "type_codex": "immune",
                "type_codex_fine": "cd4_t",
                "cell_type": "immune",
                "cell_state": "proliferative",
                "cell_type_confidence": "medium",
                "is_mismatch": False,
                "centroid_x_local": 36.0,
                "centroid_y_local": 36.0,
                "centroid_x_global": 36.0,
                "centroid_y_global": 36.0,
                "p_model_cancer": 0.05,
                "p_model_immune": 0.85,
                "p_model_healthy": 0.10,
                "p_final_cancer": 0.05,
                "p_final_immune": 0.88,
                "p_final_healthy": 0.07,
                "Pan-CK": 30.0,
                "CD45": 4200.0,
                "SMA": 25.0,
            },
            {
                "patch_id": patch_id,
                "type_cellvit": 3,
                "type_cellvit_prior": "healthy",
                "type_codex": "healthy",
                "type_codex_fine": "sma_stromal",
                "cell_type": "healthy",
                "cell_state": "quiescent",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 48.0,
                "centroid_y_local": 48.0,
                "centroid_x_global": 48.0,
                "centroid_y_global": 48.0,
                "p_model_cancer": 0.04,
                "p_model_immune": 0.06,
                "p_model_healthy": 0.90,
                "p_final_cancer": 0.05,
                "p_final_immune": 0.08,
                "p_final_healthy": 0.87,
                "Pan-CK": 20.0,
                "CD45": 40.0,
                "SMA": 3500.0,
            },
        ]
    )
    assignments.to_csv(processed / "cell_assignments.csv", index=False)
    (processed / "cell_summary.json").write_text(
        json.dumps({"classifier_used": "codex"}),
        encoding="utf-8",
    )

    out_prefix = processed / "patch_compare_codex"
    cmd = [
        *_tool_cmd(),
        "--processed",
        str(processed),
        "--patch",
        patch_id,
        "--assignments-csv",
        str(processed / "cell_assignments.csv"),
        "--summary-json",
        str(processed / "cell_summary.json"),
        "--formats",
        "png",
        "--out-prefix",
        str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()
