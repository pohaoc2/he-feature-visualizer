from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import tools.scientific_vis_patch_grid as grid_vis

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _tool_cmd() -> list[str]:
    return [sys.executable, "-m", "tools.scientific_vis_patch_grid"]


def _make_processed(tmp_path: Path, n_patches: int = 3) -> tuple[Path, list[str]]:
    """Create a minimal processed directory with n_patches synthetic patches."""
    processed = tmp_path / "processed"
    for sub in ("he", "multiplex", "cellvit"):
        (processed / sub).mkdir(parents=True, exist_ok=True)

    patch_ids = [f"{i * 64}_{0}" for i in range(n_patches)]
    rows = []
    for pid in patch_ids:
        Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB").save(
            processed / "he" / f"{pid}.png"
        )
        mx = np.zeros((4, 64, 64), dtype=np.uint16)
        mx[0, 5:20, 5:20] = 3000   # DNA channel
        mx[2, 30:50, 30:50] = 4000  # CD31 channel
        np.save(processed / "multiplex" / f"{pid}.npy", mx)
        cellvit_payload = {
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
            ]
        }
        (processed / "cellvit" / f"{pid}.json").write_text(
            json.dumps(cellvit_payload), encoding="utf-8"
        )
        rows += [
            {
                "patch_id": pid,
                "cell_type": "cancer",
                "cell_state": "quiescent",
                "cellvit_mapped_type": "cancer",
                "type_cellvit": 1,
                "type_codex": "cancer",
                "type_codex_fine": "epithelial",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 16.0,
                "centroid_y_local": 16.0,
                "p_model_cancer": 0.9, "p_model_immune": 0.05, "p_model_healthy": 0.05,
                "p_final_cancer": 0.9, "p_final_immune": 0.05, "p_final_healthy": 0.05,
            },
            {
                "patch_id": pid,
                "cell_type": "immune",
                "cell_state": "proliferative",
                "cellvit_mapped_type": "immune",
                "type_cellvit": 2,
                "type_codex": "immune",
                "type_codex_fine": "cd4_t",
                "cell_type_confidence": "medium",
                "is_mismatch": False,
                "centroid_x_local": 44.0,
                "centroid_y_local": 44.0,
                "p_model_cancer": 0.05, "p_model_immune": 0.85, "p_model_healthy": 0.10,
                "p_final_cancer": 0.05, "p_final_immune": 0.88, "p_final_healthy": 0.07,
            },
        ]

    (processed / "index.json").write_text(
        json.dumps({"patch_size": 64, "channels": ["DNA", "Pan-CK", "CD31", "SMA"]}),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(processed / "cell_assignments.csv", index=False)
    return processed, patch_ids


def test_available_patches_returns_valid_ids(tmp_path: Path) -> None:
    processed, patch_ids = _make_processed(tmp_path)
    assignments_df = pd.read_csv(processed / "cell_assignments.csv")
    found = grid_vis._available_patches(processed, assignments_df)
    assert set(found) == set(patch_ids)


def test_available_patches_excludes_missing_cellvit(tmp_path: Path) -> None:
    processed, patch_ids = _make_processed(tmp_path, n_patches=2)
    # Remove one CellViT file
    (processed / "cellvit" / f"{patch_ids[0]}.json").unlink()
    assignments_df = pd.read_csv(processed / "cell_assignments.csv")
    found = grid_vis._available_patches(processed, assignments_df)
    assert patch_ids[0] not in found
    assert patch_ids[1] in found


def test_resolve_marker_soft_hit_and_miss(tmp_path: Path) -> None:
    processed, _ = _make_processed(tmp_path, n_patches=1)
    mx_arr = np.zeros((4, 64, 64), dtype=np.float32)
    index_path = processed / "index.json"

    img, name = grid_vis._resolve_marker_soft(index_path, mx_arr, "CD31")
    assert img is not None
    assert "CD31" in name

    img_miss, name_miss = grid_vis._resolve_marker_soft(index_path, mx_arr, "Ki67")
    assert img_miss is None
    assert name_miss == "Ki67"


def test_patch_grid_smoke(tmp_path: Path) -> None:
    processed, _ = _make_processed(tmp_path, n_patches=3)
    out_prefix = processed / "patch_grid"
    cmd = [
        *_tool_cmd(),
        "--processed", str(processed),
        "--random", "2",
        "--seed", "0",
        "--formats", "png",
        "--out-prefix", str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()


def test_patch_grid_clamps_to_available(tmp_path: Path) -> None:
    """Requesting more patches than available should not raise — it clamps."""
    processed, _ = _make_processed(tmp_path, n_patches=2)
    out_prefix = processed / "patch_grid_clamped"
    cmd = [
        *_tool_cmd(),
        "--processed", str(processed),
        "--random", "99",
        "--seed", "1",
        "--formats", "png",
        "--out-prefix", str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()


def test_patch_grid_placeholder_when_cd31_missing(tmp_path: Path) -> None:
    """If CD31 is not in the channel list the grid should still save."""
    processed, _ = _make_processed(tmp_path, n_patches=1)
    # Overwrite index without CD31
    (processed / "index.json").write_text(
        json.dumps({"patch_size": 64, "channels": ["DNA", "Pan-CK"]}),
        encoding="utf-8",
    )
    out_prefix = processed / "patch_grid_no_cd31"
    cmd = [
        *_tool_cmd(),
        "--processed", str(processed),
        "--random", "1",
        "--formats", "png",
        "--out-prefix", str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()
