from __future__ import annotations

from pathlib import Path

import pandas as pd

import utils.cell_assignment_reports as reports


def _make_assignments_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "cell_id": "c_match",
                "patch_id": "p_bad",
                "cell_type": "cancer",
                "type_codex": "cancer",
                "type_codex_fine": "epithelial",
                "type_cellvit_prior": "cancer",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "p_final_cancer": 0.92,
                "p_final_immune": 0.05,
                "p_final_healthy": 0.03,
                "p_model_cancer": 0.90,
                "p_model_immune": 0.06,
                "p_model_healthy": 0.04,
                "Pan-CK": 5000.0,
                "CD45": 20.0,
                "SMA": 10.0,
            },
            {
                "cell_id": "c_ambiguous",
                "patch_id": "p_bad",
                "cell_type": "cancer",
                "type_codex": "cancer",
                "type_codex_fine": "epithelial",
                "type_cellvit_prior": "cancer",
                "cell_type_confidence": "low",
                "is_mismatch": False,
                "p_final_cancer": 0.42,
                "p_final_immune": 0.36,
                "p_final_healthy": 0.22,
                "p_model_cancer": 0.44,
                "p_model_immune": 0.35,
                "p_model_healthy": 0.21,
                "Pan-CK": 2600.0,
                "CD45": 100.0,
                "SMA": 40.0,
            },
            {
                "cell_id": "c_disagree",
                "patch_id": "p_bad",
                "cell_type": "cancer",
                "type_codex": "immune",
                "type_codex_fine": "b_cell",
                "type_cellvit_prior": "cancer",
                "cell_type_confidence": "medium",
                "is_mismatch": True,
                "p_final_cancer": 0.51,
                "p_final_immune": 0.41,
                "p_final_healthy": 0.08,
                "p_model_cancer": 0.25,
                "p_model_immune": 0.70,
                "p_model_healthy": 0.05,
                "Pan-CK": 1800.0,
                "CD45": 1600.0,
                "SMA": 25.0,
            },
            {
                "cell_id": "i_match",
                "patch_id": "p_bad",
                "cell_type": "immune",
                "type_codex": "immune",
                "type_codex_fine": "cd4_t",
                "type_cellvit_prior": "immune",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "p_final_cancer": 0.03,
                "p_final_immune": 0.93,
                "p_final_healthy": 0.04,
                "p_model_cancer": 0.02,
                "p_model_immune": 0.95,
                "p_model_healthy": 0.03,
                "Pan-CK": 30.0,
                "CD45": 4200.0,
                "SMA": 50.0,
            },
            {
                "cell_id": "h_match",
                "patch_id": "p_good",
                "cell_type": "healthy",
                "type_codex": "healthy",
                "type_codex_fine": "sma_stromal",
                "type_cellvit_prior": "healthy",
                "cell_type_confidence": "high",
                "is_mismatch": False,
                "p_final_cancer": 0.05,
                "p_final_immune": 0.10,
                "p_final_healthy": 0.85,
                "p_model_cancer": 0.04,
                "p_model_immune": 0.12,
                "p_model_healthy": 0.84,
                "Pan-CK": 15.0,
                "CD45": 20.0,
                "SMA": 3600.0,
            },
            {
                "cell_id": "h_low",
                "patch_id": "p_good",
                "cell_type": "healthy",
                "type_codex": "healthy",
                "type_codex_fine": "endothelial",
                "type_cellvit_prior": "healthy",
                "cell_type_confidence": "low",
                "is_mismatch": False,
                "p_final_cancer": 0.22,
                "p_final_immune": 0.25,
                "p_final_healthy": 0.53,
                "p_model_cancer": 0.20,
                "p_model_immune": 0.28,
                "p_model_healthy": 0.52,
                "Pan-CK": 40.0,
                "CD45": 80.0,
                "SMA": 2100.0,
            },
        ]
    )


def test_load_cell_assignments_adds_margin_and_bool(tmp_path: Path) -> None:
    df = _make_assignments_df().copy()
    df["is_mismatch"] = df["is_mismatch"].map({True: "true", False: "false"})
    csv_path = tmp_path / "cell_assignments.csv"
    df.to_csv(csv_path, index=False)

    loaded = reports.load_cell_assignments(csv_path)

    assert loaded["is_mismatch"].dtype == bool
    assert "cellvit_mapped_type" in loaded.columns
    assert "final_margin" in loaded.columns
    assert loaded.loc[loaded["cell_id"] == "c_match", "final_margin"].item() == 0.87
    assert (
        loaded.loc[loaded["cell_id"] == "c_match", "cellvit_mapped_type"].item()
        == "other"
    )


def test_summarize_patch_assignments_computes_expected_metrics() -> None:
    summary = reports.summarize_patch_assignments(_make_assignments_df())

    assert list(summary["patch_id"]) == ["p_bad", "p_good"]
    assert summary.loc[summary["patch_id"] == "p_bad", "mismatch_count"].item() == 1
    assert (
        summary.loc[summary["patch_id"] == "p_bad", "low_confidence_count"].item() == 1
    )
    assert summary.loc[summary["patch_id"] == "p_good", "n_cells"].item() == 2


def test_rank_representative_patches_prefers_mismatch_then_low_confidence() -> None:
    ranked = reports.rank_representative_patches(_make_assignments_df(), top_n=2)
    assert ranked == ["p_bad", "p_good"]


def test_select_representative_cells_returns_expected_buckets() -> None:
    selected = reports.select_representative_cells(_make_assignments_df())

    cancer = selected[selected["cell_type"] == "cancer"]
    assert set(cancer["example_kind"]) == {"match", "ambiguous", "disagreement"}
    assert cancer.loc[cancer["example_kind"] == "match", "cell_id"].item() == "c_match"
    assert (
        cancer.loc[cancer["example_kind"] == "ambiguous", "cell_id"].item()
        == "c_ambiguous"
    )
    assert (
        cancer.loc[cancer["example_kind"] == "disagreement", "cell_id"].item()
        == "c_disagree"
    )

    healthy = selected[selected["cell_type"] == "healthy"]
    assert set(healthy["example_kind"]) == {"match", "ambiguous"}


def test_choose_marker_for_patch_prefers_most_separating_marker() -> None:
    df = _make_assignments_df()
    marker = reports.choose_marker_for_patch(df, ["Pan-CK", "CD45", "SMA"])
    assert marker == "CD45"


def test_model_label_column_prefers_fine_labels_for_codex() -> None:
    df = _make_assignments_df()

    assert reports.model_display_name("codex") == "CODEX"
    assert (
        reports.model_label_column(df, "codex", prefer_fine=True) == "type_codex_fine"
    )


def test_collapse_display_lines_for_fine_model_labels() -> None:
    lines = reports.collapse_display_lines("type_codex_fine")

    assert "cancer <- epithelial" in lines
    assert "healthy <- endothelial, sma_stromal" in lines
    assert reports.collapse_display_lines("type_codex") == []
