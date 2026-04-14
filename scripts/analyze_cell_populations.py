"""Quantitative summary of cell type, state, morphology, and marker differences."""

from __future__ import annotations

import argparse
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from utils.marker_aliases import resolve_first_present_column


HE_MPP = 0.325  # µm/px for CRC33 H&E (from OME metadata used elsewhere in the repo)
CELL_TYPE_ORDER = ["cancer", "immune", "healthy"]
STATE_ORDER = ["nonproliferative", "proliferative", "dead"]
FINE_TYPE_ORDER = [
    "epithelial",
    "cd4_t",
    "cd8_t",
    "treg",
    "b_cell",
    "macrophage",
    "endothelial",
    "sma_stromal",
]
CANONICAL_MARKERS = [
    "Hoechst",
    "CD31",
    "CD45",
    "CD68",
    "CD4",
    "FOXP3",
    "CD8a",
    "CD45RO",
    "CD20",
    "PD-L1",
    "CD3e",
    "CD163",
    "E-cadherin",
    "PD-1",
    "Ki67",
    "Pan-CK",
    "SMA",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize cell type/state counts, morphology, marker differences, "
            "and patch-level heterogeneity from Stage 3 outputs."
        )
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def load_data(data_dir: Path) -> pd.DataFrame:
    assignments_path = data_dir / "cell_assignments.csv"
    shape_path = data_dir / "cell_shape_features.csv"

    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing: {assignments_path}")
    if not shape_path.exists():
        raise FileNotFoundError(f"Missing: {shape_path}")

    assignments = pd.read_csv(assignments_path)
    shapes = pd.read_csv(shape_path)

    required_assignment_cols = {"CellID", "PatchID", "cell_type", "cell_state"}
    missing = sorted(required_assignment_cols.difference(assignments.columns))
    if missing:
        raise KeyError(f"Missing required columns in {assignments_path}: {missing}")

    merge_keys = ["CellID", "PatchID"]
    shape_cols = merge_keys + [
        col
        for col in ("area_px", "perimeter_px", "circularity")
        if col in shapes.columns
    ]
    df = assignments.merge(shapes.loc[:, shape_cols], on=merge_keys, how="left")
    def _normalize_state_label(value: object) -> str:
        state = str(value).strip().lower()
        if state in {"nonprolif", "nonproliferative"}:
            return "nonproliferative"
        if state.startswith("q") and state.endswith("cent"):
            return "nonproliferative"
        if state == "apoptotic":
            return "dead"
        return str(value)

    df["cell_state_norm"] = df["cell_state"].map(_normalize_state_label)
    if "area_px" in df.columns:
        df["area_um2"] = df["area_px"].astype(float) * (HE_MPP**2)
    return df


def select_markers(columns: pd.Index) -> list[tuple[str, str]]:
    resolved: list[tuple[str, str]] = []
    seen: set[str] = set()
    for marker in CANONICAL_MARKERS:
        column = resolve_first_present_column(columns, marker)
        if column is None or column in seen:
            continue
        seen.add(column)
        resolved.append((marker, column))
    return resolved


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_pvalue(pvalue: float) -> str:
    if math.isnan(pvalue):
        return "nan"
    if pvalue == 0.0:
        return "<1e-300"
    if pvalue < 1e-4:
        return f"{pvalue:.2e}"
    return f"{pvalue:.4f}"


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    result = stats.mannwhitneyu(x, y, alternative="two-sided")
    return (2.0 * float(result.statistic) / (len(x) * len(y))) - 1.0


def markdown_table(df: pd.DataFrame, decimals: int = 3) -> str:
    work = df.copy()
    for col in work.columns:
        col_name = str(col)
        if pd.api.types.is_float_dtype(work[col]):
            work[col] = work[col].map(
                lambda x: ""
                if pd.isna(x)
                else (
                    format_pvalue(float(x))
                    if "p_value" in col_name
                    else f"{float(x):.{decimals}f}"
                )
            )
    headers = [str(col) for col in work.columns]
    rows = work.astype(str).values.tolist()
    divider = ["---"] * len(headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(divider) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def count_summary(df: pd.DataFrame, column: str, order: list[str] | None = None) -> pd.DataFrame:
    counts = df[column].value_counts(dropna=False)
    if order is not None:
        counts = counts.reindex(order).dropna()
    out = counts.rename("count").reset_index().rename(columns={"index": column})
    out["percent"] = (100.0 * out["count"] / len(df)).round(2)
    return out


def cross_tab(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(df["cell_type"], df["cell_state_norm"])
    counts = counts.reindex(index=CELL_TYPE_ORDER, columns=STATE_ORDER, fill_value=0)
    counts["total"] = counts.sum(axis=1)

    row_pct = pd.crosstab(
        df["cell_type"], df["cell_state_norm"], normalize="index"
    ) * 100.0
    row_pct = row_pct.reindex(index=CELL_TYPE_ORDER, columns=STATE_ORDER, fill_value=0.0)
    row_pct["total"] = row_pct.sum(axis=1)
    return counts.reset_index(), row_pct.reset_index()


def morphology_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_type in CELL_TYPE_ORDER:
        sub = df[df["cell_type"] == cell_type]
        rows.append(
            {
                "cell_type": cell_type,
                "n_cells": len(sub),
                "median_area_um2": float(sub["area_um2"].median()),
                "mean_area_um2": float(sub["area_um2"].mean()),
                "q25_area_um2": float(sub["area_um2"].quantile(0.25)),
                "q75_area_um2": float(sub["area_um2"].quantile(0.75)),
                "median_circularity": float(sub["circularity"].median()),
                "mean_circularity": float(sub["circularity"].mean()),
                "q25_circularity": float(sub["circularity"].quantile(0.25)),
                "q75_circularity": float(sub["circularity"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def pairwise_morphology(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("area_um2", "circularity"):
        for left, right in combinations(CELL_TYPE_ORDER, 2):
            x = df.loc[df["cell_type"] == left, metric].dropna().astype(float).to_numpy()
            y = df.loc[df["cell_type"] == right, metric].dropna().astype(float).to_numpy()
            result = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "metric": metric,
                    "group_a": left,
                    "group_b": right,
                    "n_a": len(x),
                    "n_b": len(y),
                    "median_a": float(np.median(x)),
                    "median_b": float(np.median(y)),
                    "median_diff_a_minus_b": float(np.median(x) - np.median(y)),
                    "cliffs_delta": cliffs_delta(x, y),
                    "p_value": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)


def add_marker_zscores(df: pd.DataFrame, markers: list[tuple[str, str]]) -> pd.DataFrame:
    zdf = df.copy()
    for _, column in markers:
        series = zdf[column].astype(float)
        lower = float(series.quantile(0.01))
        upper = float(series.quantile(0.99))
        clipped = series.clip(lower=lower, upper=upper)
        std = float(clipped.std())
        zdf[f"{column}__z"] = 0.0 if std == 0.0 or math.isnan(std) else (clipped - float(clipped.mean())) / std
    return zdf


def marker_group_means(
    zdf: pd.DataFrame,
    markers: list[tuple[str, str]],
    group_col: str,
    group_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, column in markers:
        row: dict[str, object] = {"marker": label}
        for group in group_order:
            sub = zdf[zdf[group_col] == group]
            row[group] = float(sub[f"{column}__z"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def marker_enrichment_vs_rest(
    zdf: pd.DataFrame,
    markers: list[tuple[str, str]],
    group_col: str,
    group_order: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group in group_order:
        sub = zdf[zdf[group_col] == group]
        rest = zdf[zdf[group_col] != group]
        if len(sub) == 0 or len(rest) == 0:
            continue
        for label, column in markers:
            rows.append(
                {
                    "group": group,
                    "marker": label,
                    "n_group": len(sub),
                    "n_rest": len(rest),
                    "mean_z_group": float(sub[f"{column}__z"].mean()),
                    "mean_z_rest": float(rest[f"{column}__z"].mean()),
                    "mean_z_diff_vs_rest": float(
                        sub[f"{column}__z"].mean() - rest[f"{column}__z"].mean()
                    ),
                    "median_raw_group": float(sub[column].median()),
                    "median_raw_rest": float(rest[column].median()),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["group", "mean_z_diff_vs_rest"], ascending=[True, False], kind="stable"
    ).reset_index(drop=True)


def marker_enrichment_state_within_type(
    zdf: pd.DataFrame,
    markers: list[tuple[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_type in CELL_TYPE_ORDER:
        prolif = zdf[
            (zdf["cell_type"] == cell_type) & (zdf["cell_state_norm"] == "proliferative")
        ]
        nonprolif = zdf[
            (zdf["cell_type"] == cell_type)
            & (zdf["cell_state_norm"] == "nonproliferative")
        ]
        if len(prolif) == 0 or len(nonprolif) == 0:
            continue
        for label, column in markers:
            rows.append(
                {
                    "cell_type": cell_type,
                    "marker": label,
                    "n_proliferative": len(prolif),
                    "n_nonproliferative": len(nonprolif),
                    "mean_z_proliferative": float(prolif[f"{column}__z"].mean()),
                    "mean_z_nonproliferative": float(nonprolif[f"{column}__z"].mean()),
                    "mean_z_diff_prolif_minus_nonproliferative": float(
                        prolif[f"{column}__z"].mean() - nonprolif[f"{column}__z"].mean()
                    ),
                    "median_raw_proliferative": float(prolif[column].median()),
                    "median_raw_nonproliferative": float(nonprolif[column].median()),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["cell_type", "mean_z_diff_prolif_minus_nonproliferative"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def morphology_by_state_within_type(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cell_type in CELL_TYPE_ORDER:
        prolif = df[
            (df["cell_type"] == cell_type) & (df["cell_state_norm"] == "proliferative")
        ]
        nonprolif = df[
            (df["cell_type"] == cell_type)
            & (df["cell_state_norm"] == "nonproliferative")
        ]
        if len(prolif) == 0 or len(nonprolif) == 0:
            continue
        for metric in ("area_um2", "circularity"):
            x = prolif[metric].dropna().astype(float).to_numpy()
            y = nonprolif[metric].dropna().astype(float).to_numpy()
            result = stats.mannwhitneyu(x, y, alternative="two-sided")
            rows.append(
                {
                    "cell_type": cell_type,
                    "metric": metric,
                    "n_proliferative": len(x),
                    "n_nonproliferative": len(y),
                    "median_proliferative": float(np.median(x)),
                    "median_nonproliferative": float(np.median(y)),
                    "median_diff_prolif_minus_nonproliferative": float(
                        np.median(x) - np.median(y)
                    ),
                    "cliffs_delta": cliffs_delta(x, y),
                    "p_value": float(result.pvalue),
                }
            )
    return pd.DataFrame(rows)


def patch_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    patch = df.groupby("patch_id").agg(
        n_cells=("CellID", "size"),
        cancer_frac=("cell_type", lambda s: float((s == "cancer").mean())),
        immune_frac=("cell_type", lambda s: float((s == "immune").mean())),
        healthy_frac=("cell_type", lambda s: float((s == "healthy").mean())),
        proliferative_frac=("cell_state_norm", lambda s: float((s == "proliferative").mean())),
    )
    frac_values = patch[["cancer_frac", "immune_frac", "healthy_frac"]].to_numpy()
    safe = np.where(frac_values > 0, frac_values, 1.0)
    patch["shannon_type_diversity"] = -(frac_values * np.log(safe)).sum(axis=1)
    patch["n_present_types"] = (patch[["cancer_frac", "immune_frac", "healthy_frac"]] > 0).sum(
        axis=1
    )
    patch["dominant_type"] = (
        patch[["cancer_frac", "immune_frac", "healthy_frac"]]
        .idxmax(axis=1)
        .str.replace("_frac", "", regex=False)
    )

    summary = pd.DataFrame(
        [
            {
                "metric": "n_patches",
                "value": float(len(patch)),
            },
            {
                "metric": "median_cells_per_patch",
                "value": float(patch["n_cells"].median()),
            },
            {
                "metric": "mean_cells_per_patch",
                "value": float(patch["n_cells"].mean()),
            },
            {
                "metric": "pct_patches_with_2plus_types",
                "value": float((patch["n_present_types"] >= 2).mean() * 100.0),
            },
            {
                "metric": "pct_patches_with_all_3_types",
                "value": float((patch["n_present_types"] == 3).mean() * 100.0),
            },
            {
                "metric": "mean_shannon_type_diversity",
                "value": float(patch["shannon_type_diversity"].mean()),
            },
            {
                "metric": "corr_cancer_frac_vs_proliferative_frac",
                "value": float(patch["cancer_frac"].corr(patch["proliferative_frac"])),
            },
            {
                "metric": "corr_immune_frac_vs_proliferative_frac",
                "value": float(patch["immune_frac"].corr(patch["proliferative_frac"])),
            },
            {
                "metric": "corr_healthy_frac_vs_proliferative_frac",
                "value": float(patch["healthy_frac"].corr(patch["proliferative_frac"])),
            },
        ]
    )
    dominant = (
        patch["dominant_type"].value_counts(normalize=True).reindex(CELL_TYPE_ORDER).fillna(0.0)
        * 100.0
    ).reset_index()
    dominant.columns = ["dominant_type", "percent_of_patches"]
    return summary, dominant, patch.reset_index()


def mismatch_and_confidence(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mismatch = (
        pd.crosstab(df["cell_type"], df["is_mismatch"], normalize="index")
        .reindex(index=CELL_TYPE_ORDER, fill_value=0.0)
        * 100.0
    ).reset_index()
    confidence = (
        pd.crosstab(df["cell_type"], df["cell_type_confidence"], normalize="index")
        .reindex(index=CELL_TYPE_ORDER, fill_value=0.0)
        * 100.0
    ).reset_index()
    return mismatch, confidence


def top_rows(df: pd.DataFrame, group_col: str, value_col: str, n: int = 5) -> pd.DataFrame:
    return (
        df.groupby(group_col, group_keys=False)
        .head(n)
        .reset_index(drop=True)
    )


def build_report(
    *,
    data_dir: Path,
    markers: list[tuple[str, str]],
    cell_type_counts: pd.DataFrame,
    fine_type_counts: pd.DataFrame,
    state_counts: pd.DataFrame,
    type_state_counts: pd.DataFrame,
    type_state_pct: pd.DataFrame,
    morphology: pd.DataFrame,
    morphology_pairs: pd.DataFrame,
    marker_type_top: pd.DataFrame,
    marker_state_top: pd.DataFrame,
    marker_state_within_type_top: pd.DataFrame,
    morphology_state_within_type: pd.DataFrame,
    patch_metrics: pd.DataFrame,
    dominant_patch_types: pd.DataFrame,
    mismatch: pd.DataFrame,
    confidence: pd.DataFrame,
) -> str:
    marker_names = ", ".join(label for label, _ in markers)
    lines: list[str] = []
    lines.append("# Cell Population Quantitative Analysis")
    lines.append("")
    lines.append(f"Dataset: `{data_dir}`")
    lines.append("")
    lines.append(
        "State labels were normalized so legacy non-proliferative aliases are "
        "reported as `nonproliferative`."
    )
    lines.append("")
    lines.append("## 1. Cell type and cell state counts")
    lines.append("")
    lines.append(markdown_table(cell_type_counts, decimals=2))
    lines.append("")
    lines.append(markdown_table(fine_type_counts, decimals=2))
    lines.append("")
    lines.append(markdown_table(state_counts, decimals=2))
    lines.append("")
    lines.append("### Type x state counts")
    lines.append("")
    lines.append(markdown_table(type_state_counts, decimals=2))
    lines.append("")
    lines.append("### Type x state row percentages")
    lines.append("")
    lines.append(markdown_table(type_state_pct, decimals=2))
    lines.append("")
    lines.append("## 2. Area and circularity differences between cell types")
    lines.append("")
    lines.append(markdown_table(morphology, decimals=3))
    lines.append("")
    lines.append(markdown_table(morphology_pairs, decimals=3))
    lines.append("")
    lines.append("## 3. MX marker differences between cell types")
    lines.append("")
    lines.append(
        "Markers were winsorized to the 1st-99th percentile and z-scored globally "
        "before computing mean enrichment versus the rest of the slide."
    )
    lines.append("")
    lines.append(f"Markers analyzed: {marker_names}")
    lines.append("")
    lines.append(markdown_table(marker_type_top, decimals=3))
    lines.append("")
    lines.append("## 4. MX marker differences between cell states")
    lines.append("")
    lines.append(
        "The first table is overall by state. The second table compares proliferative "
        "vs nonproliferative within each broad cell type, which is less confounded by cell composition."
    )
    lines.append("")
    lines.append(markdown_table(marker_state_top, decimals=3))
    lines.append("")
    lines.append(markdown_table(marker_state_within_type_top, decimals=3))
    lines.append("")
    lines.append("## 5. Other analyses worth looking at")
    lines.append("")
    lines.append("### Proliferation-associated morphology shifts within each cell type")
    lines.append("")
    lines.append(markdown_table(morphology_state_within_type, decimals=3))
    lines.append("")
    lines.append("### Patch-level heterogeneity")
    lines.append("")
    lines.append(markdown_table(patch_metrics, decimals=3))
    lines.append("")
    lines.append(markdown_table(dominant_patch_types, decimals=3))
    lines.append("")
    lines.append("### CellViT / MX fusion QC")
    lines.append("")
    lines.append(markdown_table(mismatch, decimals=2))
    lines.append("")
    lines.append(markdown_table(confidence, decimals=2))
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- This is a single-slide analysis, so p-values quantify within-slide separation, "
        "not cohort-level reproducibility."
    )
    lines.append(
        "- State-level marker differences are partly driven by composition shifts; the "
        "within-cell-type proliferative vs nonproliferative table is the cleaner comparison."
    )
    lines.append(
        "- `dead` cells are rare, so their marker summary is descriptive only."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    out_dir = (args.out_dir or (data_dir / "analysis")).resolve()
    ensure_dir(out_dir)

    df = load_data(data_dir)
    markers = select_markers(df.columns)
    zdf = add_marker_zscores(df, markers)

    cell_type_counts = count_summary(df, "cell_type", CELL_TYPE_ORDER)
    fine_type_counts = count_summary(df, "type_codex_fine", FINE_TYPE_ORDER)
    state_counts = count_summary(df, "cell_state_norm", STATE_ORDER)
    type_state_counts, type_state_pct = cross_tab(df)
    morphology = morphology_summary(df)
    morphology_pairs = pairwise_morphology(df)

    marker_type = marker_enrichment_vs_rest(zdf, markers, "cell_type", CELL_TYPE_ORDER)
    marker_state = marker_enrichment_vs_rest(zdf, markers, "cell_state_norm", STATE_ORDER)
    marker_state_within_type = marker_enrichment_state_within_type(zdf, markers)
    morphology_state = morphology_by_state_within_type(df)
    patch_metrics, dominant_patch_types, patch_level = patch_level_summary(df)
    mismatch, confidence = mismatch_and_confidence(df)
    marker_type_matrix = marker_group_means(zdf, markers, "cell_type", CELL_TYPE_ORDER)
    marker_state_matrix = marker_group_means(zdf, markers, "cell_state_norm", STATE_ORDER)

    outputs: dict[str, pd.DataFrame] = {
        "cell_type_counts.csv": cell_type_counts,
        "fine_type_counts.csv": fine_type_counts,
        "cell_state_counts.csv": state_counts,
        "cell_type_by_state_counts.csv": type_state_counts,
        "cell_type_by_state_row_pct.csv": type_state_pct,
        "morphology_by_cell_type.csv": morphology,
        "morphology_pairwise_tests.csv": morphology_pairs,
        "marker_mean_z_by_cell_type.csv": marker_type_matrix,
        "marker_mean_z_by_cell_state.csv": marker_state_matrix,
        "marker_enrichment_by_cell_type.csv": marker_type,
        "marker_enrichment_by_cell_state.csv": marker_state,
        "marker_enrichment_state_within_type.csv": marker_state_within_type,
        "morphology_state_within_type.csv": morphology_state,
        "patch_level_metrics.csv": patch_metrics,
        "patch_level_dominant_types.csv": dominant_patch_types,
        "patch_level_full.csv": patch_level,
        "cellvit_mismatch_by_cell_type_pct.csv": mismatch,
        "confidence_by_cell_type_pct.csv": confidence,
    }
    for name, table in outputs.items():
        table.to_csv(out_dir / name, index=False)

    report = build_report(
        data_dir=data_dir,
        markers=markers,
        cell_type_counts=cell_type_counts,
        fine_type_counts=fine_type_counts,
        state_counts=state_counts,
        type_state_counts=type_state_counts,
        type_state_pct=type_state_pct,
        morphology=morphology,
        morphology_pairs=morphology_pairs,
        marker_type_top=top_rows(marker_type, "group", "mean_z_diff_vs_rest"),
        marker_state_top=top_rows(marker_state, "group", "mean_z_diff_vs_rest"),
        marker_state_within_type_top=top_rows(
            marker_state_within_type,
            "cell_type",
            "mean_z_diff_prolif_minus_nonproliferative",
        ),
        morphology_state_within_type=morphology_state,
        patch_metrics=patch_metrics,
        dominant_patch_types=dominant_patch_types,
        mismatch=mismatch,
        confidence=confidence,
    )
    (out_dir / "cell_population_report.md").write_text(report, encoding="utf-8")

    print(f"Wrote analysis outputs to {out_dir}")


if __name__ == "__main__":
    main()
