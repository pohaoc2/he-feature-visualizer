from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import tools.scientific_vis_astir_report as report
from utils.cell_assignment_reports import load_cell_assignments


_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _tool_cmd() -> list[str]:
    return [sys.executable, "-m", "tools.scientific_vis_astir_report"]


def _make_assignments() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patch_id": "0_0",
                "type_cellvit": 1,
                "type_cellvit_prior": "cancer",
                "type_astir": "cancer",
                "type_astir_fine": "epithelial",
                "cell_type": "cancer",
                "cell_state": "quiescent",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 16.0,
                "centroid_y_local": 16.0,
                "p_model_cancer": 0.91,
                "p_model_immune": 0.05,
                "p_model_healthy": 0.04,
                "p_final_cancer": 0.92,
                "p_final_immune": 0.04,
                "p_final_healthy": 0.04,
                "Pan-CK": 4200.0,
                "CD45": 20.0,
                "SMA": 15.0,
            },
            {
                "patch_id": "0_0",
                "type_cellvit": 1,
                "type_cellvit_prior": "cancer",
                "type_astir": "immune",
                "type_astir_fine": "b_cell",
                "cell_type": "cancer",
                "cell_state": "quiescent",
                "cell_type_confidence": "low",
                "is_mismatch": True,
                "centroid_x_local": 20.0,
                "centroid_y_local": 20.0,
                "p_model_cancer": 0.40,
                "p_model_immune": 0.45,
                "p_model_healthy": 0.15,
                "p_final_cancer": 0.46,
                "p_final_immune": 0.41,
                "p_final_healthy": 0.13,
                "Pan-CK": 1800.0,
                "CD45": 900.0,
                "SMA": 25.0,
            },
            {
                "patch_id": "0_1",
                "type_cellvit": 2,
                "type_cellvit_prior": "immune",
                "type_astir": "immune",
                "type_astir_fine": "cd4_t",
                "cell_type": "immune",
                "cell_state": "proliferative",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 18.0,
                "centroid_y_local": 18.0,
                "p_model_cancer": 0.03,
                "p_model_immune": 0.92,
                "p_model_healthy": 0.05,
                "p_final_cancer": 0.03,
                "p_final_immune": 0.93,
                "p_final_healthy": 0.04,
                "Pan-CK": 25.0,
                "CD45": 3900.0,
                "SMA": 20.0,
            },
            {
                "patch_id": "1_0",
                "type_cellvit": 3,
                "type_cellvit_prior": "healthy",
                "type_astir": "healthy",
                "type_astir_fine": "sma_stromal",
                "cell_type": "healthy",
                "cell_state": "quiescent",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "centroid_x_local": 22.0,
                "centroid_y_local": 22.0,
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
            {
                "patch_id": "1_0",
                "type_cellvit": 3,
                "type_cellvit_prior": "healthy",
                "type_astir": "healthy",
                "type_astir_fine": "endothelial",
                "cell_type": "healthy",
                "cell_state": "quiescent",
                "cell_type_confidence": "low",
                "is_mismatch": False,
                "centroid_x_local": 28.0,
                "centroid_y_local": 28.0,
                "p_model_cancer": 0.20,
                "p_model_immune": 0.21,
                "p_model_healthy": 0.59,
                "p_final_cancer": 0.22,
                "p_final_immune": 0.23,
                "p_final_healthy": 0.55,
                "Pan-CK": 40.0,
                "CD45": 80.0,
                "SMA": 2100.0,
            },
        ]
    )


def test_build_report_figure_marks_fallback_and_omits_missing_buckets(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "he").mkdir(parents=True, exist_ok=True)
    for patch_id in ("0_0", "0_1", "1_0"):
        Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB").save(
            processed / "he" / f"{patch_id}.png"
        )

    assignments_path = processed / "cell_assignments.csv"
    _make_assignments().to_csv(assignments_path, index=False)
    assignments = load_cell_assignments(assignments_path)
    summary = {"classifier_used": "rule_fallback"}

    fig, selected = report.build_report_figure(assignments, summary, processed)

    assert "rule_fallback" in fig._suptitle.get_text()
    healthy_examples = selected[selected["cell_type"] == "healthy"]
    assert set(healthy_examples["example_kind"]) == {"match", "ambiguous"}


def test_build_report_figure_uses_codex_fine_labels(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "he").mkdir(parents=True, exist_ok=True)
    for patch_id in ("0_0", "0_1", "1_0"):
        Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB").save(
            processed / "he" / f"{patch_id}.png"
        )

    assignments_path = processed / "cell_assignments.csv"
    _make_assignments().to_csv(assignments_path, index=False)
    assignments = load_cell_assignments(assignments_path)
    summary = {"classifier_used": "codex"}

    fig, _ = report.build_report_figure(assignments, summary, processed)

    assert "mode=codex" in fig._suptitle.get_text()
    titles = [ax.get_title() for ax in fig.axes]
    assert "CellViT vs CODEX" in titles
    assert "Cancer examples" in titles
    assert "Immune examples" in titles
    assert "Healthy examples" in titles
    assert titles.count("Match") == 3

    all_text = "\n".join(text.get_text() for ax in fig.axes for text in ax.texts)
    assert "CODEX: epithelial" in all_text
    assert "Fine -> final collapse:" in all_text
    assert "immune <-" in all_text
    assert "macrophage" in all_text


def test_astir_report_cli_smoke(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "he").mkdir(parents=True, exist_ok=True)
    for patch_id in ("0_0", "0_1", "1_0"):
        Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB").save(
            processed / "he" / f"{patch_id}.png"
        )

    assignments_path = processed / "cell_assignments.csv"
    _make_assignments().to_csv(assignments_path, index=False)
    summary_path = processed / "cell_summary.json"
    summary_path.write_text(json.dumps({"classifier_used": "astir"}), encoding="utf-8")

    out_prefix = processed / "astir_report"
    cmd = [
        *_tool_cmd(),
        "--processed",
        str(processed),
        "--assignments-csv",
        str(assignments_path),
        "--summary-json",
        str(summary_path),
        "--formats",
        "png",
        "--out-prefix",
        str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()


def test_codex_report_cli_smoke(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    (processed / "he").mkdir(parents=True, exist_ok=True)
    for patch_id in ("0_0", "0_1", "1_0"):
        Image.fromarray(np.full((64, 64, 3), 180, dtype=np.uint8), mode="RGB").save(
            processed / "he" / f"{patch_id}.png"
        )

    assignments_path = processed / "cell_assignments.csv"
    _make_assignments().to_csv(assignments_path, index=False)
    summary_path = processed / "cell_summary.json"
    summary_path.write_text(json.dumps({"classifier_used": "codex"}), encoding="utf-8")

    out_prefix = processed / "codex_report"
    cmd = [
        *_tool_cmd(),
        "--processed",
        str(processed),
        "--assignments-csv",
        str(assignments_path),
        "--summary-json",
        str(summary_path),
        "--formats",
        "png",
        "--out-prefix",
        str(out_prefix),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_PROJECT_ROOT)
    assert result.returncode == 0, result.stderr
    assert out_prefix.with_suffix(".png").exists()
