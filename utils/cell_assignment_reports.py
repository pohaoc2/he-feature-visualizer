"""Shared helpers for CellViT/Astir comparison reports."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


FINAL_PROB_COLUMNS: tuple[str, str, str] = (
    "p_final_cancer",
    "p_final_immune",
    "p_final_healthy",
)

CONFIDENCE_SCORES = {"low": 0, "medium": 1, "high": 2}


def map_cellvit_type(type_cellvit: object, type_cellvit_prior: object | None = None) -> str:
    """Map raw CellViT classes into the shared 3-class comparison ontology."""
    try:
        value = int(type_cellvit)
    except Exception:
        return "other"
    if value == 1:
        return "cancer"
    if value == 2:
        return "immune"
    if value == 3:
        return "healthy"
    if value == 5 and isinstance(type_cellvit_prior, str):
        return type_cellvit_prior if type_cellvit_prior in {"cancer", "immune", "healthy"} else "other"
    return "other"


def _coerce_is_mismatch(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes"})


def _final_margin(df: pd.DataFrame) -> pd.Series:
    probs = df.reindex(columns=FINAL_PROB_COLUMNS, fill_value=0.0).astype(float)
    top2 = probs.apply(lambda row: row.nlargest(2).to_list(), axis=1)
    return top2.apply(
        lambda vals: float(vals[0] - vals[1]) if len(vals) >= 2 else float(vals[0])
    )


def load_cell_assignments(csv_path: Path) -> pd.DataFrame:
    """Load Stage 3 per-cell assignments and add normalized helper columns."""
    df = pd.read_csv(csv_path)
    if "is_mismatch" in df.columns:
        df["is_mismatch"] = _coerce_is_mismatch(df["is_mismatch"])
    else:
        df["is_mismatch"] = False
    if "cell_type_confidence" not in df.columns:
        df["cell_type_confidence"] = "low"
    df["cellvit_mapped_type"] = [
        map_cellvit_type(type_cellvit, prior)
        for type_cellvit, prior in zip(
            df.get("type_cellvit", pd.Series(index=df.index, dtype=int)),
            df.get("type_cellvit_prior", pd.Series(index=df.index, dtype=object)),
            strict=False,
        )
    ]
    df["final_margin"] = _final_margin(df)
    df["confidence_score"] = (
        df["cell_type_confidence"].astype(str).str.lower().map(CONFIDENCE_SCORES).fillna(0)
    )
    return df


def summarize_patch_assignments(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate patch-level metrics used for representative patch selection."""
    work = df.copy()
    if "final_margin" not in work.columns:
        work["final_margin"] = _final_margin(work)
    if "confidence_score" not in work.columns:
        work["confidence_score"] = (
            work["cell_type_confidence"]
            .astype(str)
            .str.lower()
            .map(CONFIDENCE_SCORES)
            .fillna(0)
        )

    summary = (
        work.groupby("patch_id", dropna=False)
        .agg(
            n_cells=("patch_id", "size"),
            mismatch_count=("is_mismatch", "sum"),
            low_confidence_count=(
                "cell_type_confidence",
                lambda s: int(s.astype(str).str.lower().eq("low").sum()),
            ),
            mean_confidence_score=("confidence_score", "mean"),
            mean_final_margin=("final_margin", "mean"),
        )
        .reset_index()
    )
    summary["mismatch_rate"] = summary["mismatch_count"] / summary["n_cells"].where(
        summary["n_cells"] > 0, 1
    )
    return summary.sort_values(
        by=["mismatch_rate", "low_confidence_count", "n_cells", "patch_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def rank_representative_patches(df: pd.DataFrame, top_n: int) -> list[str]:
    """Return patch ids ordered by representative priority."""
    summary = summarize_patch_assignments(df)
    return summary["patch_id"].head(int(top_n)).astype(str).tolist()


def _pick_first(
    df: pd.DataFrame,
    *,
    ascending: list[bool],
    used: set[object],
) -> pd.Series | None:
    if df.empty:
        return None
    order_cols = ["confidence_score", "final_margin", "patch_id"]
    if "cell_id" in df.columns:
        order_cols.append("cell_id")
    else:
        order_cols.append("_row_id")
    ordered = df.sort_values(order_cols, ascending=ascending + [True, True], kind="stable")
    for _, row in ordered.iterrows():
        key = row["cell_id"] if "cell_id" in row else row["_row_id"]
        if key in used:
            continue
        used.add(key)
        return row
    return None


def select_representative_cells(df: pd.DataFrame, per_class: int = 3) -> pd.DataFrame:
    """Pick representative examples per final cell type."""
    del per_class  # v1 locks to three semantic buckets at most.

    work = df.copy().reset_index(drop=True)
    if "final_margin" not in work.columns:
        work["final_margin"] = _final_margin(work)
    if "confidence_score" not in work.columns:
        work["confidence_score"] = (
            work["cell_type_confidence"]
            .astype(str)
            .str.lower()
            .map(CONFIDENCE_SCORES)
            .fillna(0)
        )
    work["_row_id"] = work.index

    selected_rows: list[dict[str, object]] = []
    for cell_type in sorted(work["cell_type"].dropna().astype(str).unique()):
        group = work[work["cell_type"].astype(str) == cell_type]
        used: set[object] = set()

        match = _pick_first(
            group[group["is_mismatch"] == False],  # noqa: E712 - explicit Series comparison
            ascending=[False, False],
            used=used,
        )
        if match is not None:
            row = match.to_dict()
            row["example_kind"] = "match"
            selected_rows.append(row)

        ambiguous = _pick_first(group, ascending=[True, True], used=used)
        if ambiguous is not None:
            row = ambiguous.to_dict()
            row["example_kind"] = "ambiguous"
            selected_rows.append(row)

        disagreement = _pick_first(
            group[group["is_mismatch"] == True],  # noqa: E712 - explicit Series comparison
            ascending=[False, False],
            used=used,
        )
        if disagreement is not None:
            row = disagreement.to_dict()
            row["example_kind"] = "disagreement"
            selected_rows.append(row)

    return pd.DataFrame(selected_rows).drop(columns=["_row_id"], errors="ignore")


def choose_marker_for_patch(df_patch: pd.DataFrame, marker_columns: list[str]) -> str | None:
    """Return the available marker with the strongest class separation."""
    if df_patch.empty:
        return None

    best_marker: str | None = None
    best_score = float("-inf")
    for marker in marker_columns:
        if marker not in df_patch.columns:
            continue
        medians = (
            df_patch.groupby("cell_type", dropna=True)[marker]
            .median()
            .dropna()
            .astype(float)
        )
        if medians.empty:
            continue
        score = float(medians.max() - medians.min())
        if score > best_score or (
            score == best_score and best_marker is not None and marker < best_marker
        ):
            best_score = score
            best_marker = marker
    return best_marker
