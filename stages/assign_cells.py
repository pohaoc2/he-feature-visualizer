"""
assign_cells.py — Stage 3 of the histopathology pipeline.

Astir-first cell typing (cancer/immune/healthy) with CellViT prior fusion,
plus state assignment (dead/proliferative/quiescent).

Dual mode:
  1) Existing mode: read --features-csv directly.
  2) CellViT+MX mode: when --features-csv is omitted, build it from
     CellViT contours + per-patch multiplex arrays, then continue with the
     same CSV-driven assignment path.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import scipy.spatial
from PIL import Image

from stages.extract_cell_features import extract_cell_features_to_csv
from utils.astir_adapter import AstirUnavailableError, predict_cell_type_probabilities
from utils.marker_aliases import marker_candidates, resolve_first_present_column

# ---------------------------------------------------------------------------
# Labels and colors
# ---------------------------------------------------------------------------

CELL_TYPES: tuple[str, str, str] = ("cancer", "immune", "healthy")

CELL_TYPE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "cancer": (220, 50, 50, 200),
    "immune": (50, 100, 220, 200),
    "healthy": (50, 180, 50, 200),
    "other": (150, 150, 150, 150),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int, int]] = {
    "proliferative": (0, 255, 0, 200),
    "quiescent": (100, 149, 237, 200),
    "dead": (139, 0, 139, 200),
    "other": (80, 80, 80, 150),
}

# ---------------------------------------------------------------------------
# CellViT priors and marker config
# ---------------------------------------------------------------------------

CELLVIT_TYPE_PRIORS: dict[int, dict[str, float]] = {
    0: {"cancer": 1 / 3, "immune": 1 / 3, "healthy": 1 / 3},  # background/unknown
    1: {"cancer": 1.0, "immune": 0.0, "healthy": 0.0},  # Neoplastic
    2: {"cancer": 0.0, "immune": 1.0, "healthy": 0.0},  # Inflammatory
    3: {"cancer": 0.0, "immune": 0.0, "healthy": 1.0},  # Connective
    4: {"cancer": 1 / 3, "immune": 1 / 3, "healthy": 1 / 3},  # Dead (state override)
    5: {"cancer": 0.5, "immune": 0.0, "healthy": 0.5},  # Epithelial split prior
}

ASTIR_FINE_TYPE_MARKERS: dict[str, list[str]] = {
    "epithelial": ["PanCK", "Ecadherin"],
    "cd4_t": ["CD45", "CD3e", "CD4", "CD45RO"],
    "cd8_t": ["CD45", "CD3e", "CD8a", "CD45RO"],
    "treg": ["CD45", "CD3e", "CD4", "FOXP3"],
    "b_cell": ["CD45", "CD20"],
    "macrophage": ["CD45", "CD68", "CD163"],
    "endothelial": ["CD31"],
    "sma_stromal": ["SMA"],
}

ASTIR_FINE_TO_FINAL: dict[str, str] = {
    "epithelial": "cancer",
    "cd4_t": "immune",
    "cd8_t": "immune",
    "treg": "immune",
    "b_cell": "immune",
    "macrophage": "immune",
    "endothelial": "healthy",
    "sma_stromal": "healthy",
}

CODEX_FINE_TYPE_MARKERS = ASTIR_FINE_TYPE_MARKERS

CODEX_FINE_TO_FINAL_WEIGHTS: dict[str, dict[str, float]] = {
    "epithelial": {"cancer": 1.0},
    "cd4_t": {"immune": 1.0},
    "cd8_t": {"immune": 1.0},
    "treg": {"immune": 1.0},
    "b_cell": {"immune": 1.0},
    "macrophage": {"immune": 1.0},
    "endothelial": {"healthy": 1.0},
    "sma_stromal": {"healthy": 1.0},
}

# Legacy name retained for compatibility with existing tests/tools.
ASTIR_TYPE_MARKERS = ASTIR_FINE_TYPE_MARKERS

NON_TYPING_MARKERS: tuple[str, ...] = ("Hoechst", "AF1", "Argo550", "PD-L1")

CONFIDENCE_THRESHOLDS: tuple[float, float] = (0.25, 0.12)
DEFAULT_ASTIR_WEIGHT = 0.85
CODEX_ZSCORE_EPS = 1e-6
CODEX_CLUSTER_PENALTY = 0.35
CODEX_CLUSTER_TEMPERATURE = 0.75

# Legacy compatibility surfaces kept for older tests/tools.
LEGACY_TYPE_MARKERS: dict[str, list[str]] = {
    "tumor": ["Keratin", "NaKATPase", "CDX2"],
    "immune": [
        "CD45",
        "CD3",
        "CD4",
        "CD8a",
        "CD20",
        "CD45RO",
        "CD68",
        "CD163",
        "FOXP3",
        "PD1",
    ],
    "stromal": ["SMA", "CD31", "Desmin", "Collagen"],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _argmax_label(score_map: dict[str, float]) -> str:
    # Stable tie-breaking by CELL_TYPES order.
    return max(CELL_TYPES, key=lambda k: (score_map.get(k, 0.0), -CELL_TYPES.index(k)))


def _collapse_astir_type_label(label: object) -> str:
    key = str(label).strip()
    if key in CELL_TYPES:
        return key
    return ASTIR_FINE_TO_FINAL.get(key, "other")


def _collapse_astir_probabilities(probs: pd.DataFrame) -> pd.DataFrame:
    """Collapse fine-grained Astir outputs into the shared 3-class ontology."""
    collapsed = pd.DataFrame(0.0, index=probs.index, columns=list(CELL_TYPES))
    for col in probs.columns:
        target = _collapse_astir_type_label(col)
        if target not in CELL_TYPES:
            continue
        collapsed[target] = collapsed[target] + pd.to_numeric(
            probs[col], errors="coerce"
        ).fillna(0.0).clip(lower=0.0)
    return collapsed


def _collapse_weighted_fine_probabilities(
    probs: pd.DataFrame,
    fine_to_final: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """Collapse fine-grained probabilities into the shared 3-class ontology."""
    collapsed = pd.DataFrame(0.0, index=probs.index, columns=list(CELL_TYPES))
    for fine_type, weights in fine_to_final.items():
        if fine_type not in probs.columns:
            continue
        series = pd.to_numeric(probs[fine_type], errors="coerce").fillna(0.0).clip(lower=0.0)
        for final_type, weight in weights.items():
            if final_type in collapsed.columns and weight != 0.0:
                collapsed[final_type] = collapsed[final_type] + float(weight) * series

    row_sum = collapsed.sum(axis=1)
    safe_den = row_sum.where(row_sum > 0.0, 1.0)
    return collapsed.div(safe_den, axis=0)


def _stable_softmax(frame: pd.DataFrame, temperature: float = 1.0) -> pd.DataFrame:
    """Return row-wise softmax probabilities for a score frame."""
    if frame.empty:
        return frame.copy()
    temp = float(max(temperature, 1e-6))
    values = frame.to_numpy(dtype=float) / temp
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom <= 0.0] = 1.0
    return pd.DataFrame(exp / denom, index=frame.index, columns=frame.columns)


def _preprocess_codex_matrix(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare per-cell multiplex features for CODEX-style clustering.

    The current CODEX path follows the Frontiers benchmarking paper's
    recommended normalization for cell-type identification: per-marker
    Z normalization across all cells.
    """
    out = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(lower=0.0)
    if out.empty:
        return out.astype(float)

    arr = out.to_numpy(dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std <= CODEX_ZSCORE_EPS] = 1.0
    arr = (arr - mean) / std
    return pd.DataFrame(arr, index=out.index, columns=out.columns)


def _compute_codex_cluster_scores(
    centers: pd.DataFrame,
    fine_type_markers: dict[str, list[str]],
) -> pd.DataFrame:
    """Score cluster centers against fine cell-type signatures."""
    positive = centers.clip(lower=0.0)
    scores = pd.DataFrame(index=centers.index, columns=list(fine_type_markers), dtype=float)

    for fine_type, markers in fine_type_markers.items():
        present = [marker for marker in markers if marker in centers.columns]
        if not present:
            scores[fine_type] = 0.0
            continue
        signal = centers[present].mean(axis=1)
        other_markers = [marker for marker in centers.columns if marker not in present]
        if other_markers:
            penalty = positive[other_markers].mean(axis=1)
        else:
            penalty = pd.Series(0.0, index=centers.index, dtype=float)
        scores[fine_type] = signal - CODEX_CLUSTER_PENALTY * penalty

    return scores.fillna(0.0)


def _compute_codex_probabilities(
    expr_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Cluster multiplex features and annotate clusters into fine cell types."""
    if expr_df.empty:
        out = pd.DataFrame(0.0, index=expr_df.index, columns=list(CELL_TYPES))
        out.attrs["model_fine_probs"] = pd.DataFrame(index=expr_df.index)
        out.attrs["model_fine_top"] = pd.Series(dtype=object, index=expr_df.index)
        return out

    normalized = _preprocess_codex_matrix(expr_df)
    n_cells = len(normalized)
    n_clusters = min(max(1, len(CODEX_FINE_TYPE_MARKERS)), n_cells)

    if n_clusters == 1:
        cluster_ids = np.zeros(n_cells, dtype=int)
        centers = pd.DataFrame([normalized.mean(axis=0)], columns=normalized.columns)
    else:
        try:
            from sklearn.cluster import KMeans
        except Exception as exc:  # pragma: no cover - sklearn is expected via astir deps
            raise RuntimeError("CODEX classifier requires scikit-learn.") from exc

        km = KMeans(n_clusters=n_clusters, random_state=0, n_init=20)
        cluster_ids = km.fit_predict(normalized.to_numpy(dtype=float))
        centers = pd.DataFrame(km.cluster_centers_, columns=normalized.columns)

    centers.index = pd.Index(range(len(centers)), name="cluster_id")
    cluster_scores = _compute_codex_cluster_scores(centers, CODEX_FINE_TYPE_MARKERS)
    cluster_probs = _stable_softmax(cluster_scores, temperature=CODEX_CLUSTER_TEMPERATURE)

    cell_cluster_ids = pd.Series(cluster_ids, index=expr_df.index, name="cluster_id")
    fine_probs = cluster_probs.loc[cell_cluster_ids.to_numpy()].set_index(expr_df.index)
    fine_top = fine_probs.idxmax(axis=1).astype(str)
    final_probs = _collapse_weighted_fine_probabilities(
        fine_probs,
        CODEX_FINE_TO_FINAL_WEIGHTS,
    )

    logger.info(
        "CODEX clustering assigned %d cells across %d clusters; fine types=%s",
        n_cells,
        n_clusters,
        dict(fine_top.value_counts()),
    )
    final_probs.attrs["model_fine_probs"] = fine_probs
    final_probs.attrs["model_fine_top"] = fine_top
    final_probs.attrs["model_cluster_id"] = cell_cluster_ids
    return final_probs


def _mean_available(series_list: list[pd.Series]) -> pd.Series:
    if not series_list:
        raise ValueError("_mean_available called with empty list")
    if len(series_list) == 1:
        return series_list[0]
    return pd.concat(series_list, axis=1).mean(axis=1)


def _get_marker_value(row: pd.Series, marker: str) -> float:
    """Return marker value from row using canonical name + aliases."""
    for candidate in marker_candidates(marker):
        try:
            value = row.get(candidate, None) if hasattr(row, "get") else None
            if value is None and hasattr(row, candidate):
                value = getattr(row, candidate)
            if value is None:
                continue
            return _safe_float(value, default=0.0)
        except Exception:
            continue
    return 0.0


def _percentile_rank(series: pd.Series) -> pd.Series:
    """Return percentile-rank normalized values in [0,1]."""
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    ranks = vals.rank(method="average", pct=True)
    return ranks.fillna(0.0).astype(float)


def _normalized_marker_series(df: pd.DataFrame, marker: str) -> pd.Series:
    col = resolve_first_present_column(df.columns, marker)
    if col is None:
        return pd.Series(0.0, index=df.index, dtype=float)
    return _percentile_rank(df[col])


def _compute_rule_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based fallback probabilities over (cancer, immune, healthy)."""
    cd45 = _normalized_marker_series(df, "CD45")
    panck = _normalized_marker_series(df, "PanCK")
    ecad = _normalized_marker_series(df, "Ecadherin")
    cd31 = _normalized_marker_series(df, "CD31")
    sma = _normalized_marker_series(df, "SMA")

    immune_t_markers = ["CD3e", "CD4", "CD8a", "CD45RO", "FOXP3", "PD1"]
    immune_mb_markers = ["CD20", "CD68", "CD163"]

    immune_t = _mean_available([_normalized_marker_series(df, m) for m in immune_t_markers])
    immune_mb = _mean_available(
        [_normalized_marker_series(df, m) for m in immune_mb_markers]
    )

    immune_score = 0.55 * cd45 + 0.30 * immune_t + 0.15 * immune_mb
    cancer_score = 0.55 * panck + 0.25 * ecad + 0.20 * (1.0 - cd45)
    healthy_score = 0.60 * _mean_available([sma, cd31]) + 0.40 * _mean_available(
        [1.0 - panck, 1.0 - cd45]
    )

    probs = pd.DataFrame(
        {
            "cancer": cancer_score.clip(lower=0.0),
            "immune": immune_score.clip(lower=0.0),
            "healthy": healthy_score.clip(lower=0.0),
        },
        index=df.index,
    )
    denom = probs.sum(axis=1).replace(0.0, 1.0)
    return probs.div(denom, axis=0)


def _build_astir_input(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Build Astir marker matrix and return marker coverage metadata."""
    required = sorted({m for markers in ASTIR_FINE_TYPE_MARKERS.values() for m in markers})
    data: dict[str, pd.Series] = {}
    coverage: dict[str, dict[str, object]] = {}

    for marker in required:
        resolved = resolve_first_present_column(df.columns, marker)
        if resolved is None:
            data[marker] = pd.Series(0.0, index=df.index, dtype=float)
            coverage[marker] = {"available": False, "source_column": None}
        else:
            data[marker] = pd.to_numeric(df[resolved], errors="coerce").fillna(0.0)
            coverage[marker] = {"available": True, "source_column": resolved}

    return pd.DataFrame(data, index=df.index), coverage


def compute_type_probabilities(
    df: pd.DataFrame,
    classifier: str,
    allow_astir_fallback: bool,
    log: logging.Logger,
) -> tuple[pd.DataFrame, str, dict[str, dict[str, object]]]:
    """Return model probabilities and runtime classifier mode."""
    expr_df, coverage = _build_astir_input(df)

    if classifier == "rule":
        return _compute_rule_probabilities(df), "rule", coverage
    if classifier == "codex":
        return _compute_codex_probabilities(expr_df, log), "codex", coverage

    try:
        fine_probs = predict_cell_type_probabilities(
            expr_df,
            marker_dict=ASTIR_FINE_TYPE_MARKERS,
            logger=log,
        )
        probs = _collapse_astir_probabilities(fine_probs)
        fine_top = (
            fine_probs.idxmax(axis=1).astype(str)
            if len(fine_probs.columns) > 0
            else pd.Series("other", index=fine_probs.index, dtype=object)
        )
        probs.attrs["model_fine_probs"] = fine_probs
        probs.attrs["model_fine_top"] = fine_top
        return probs, "astir", coverage
    except AstirUnavailableError as exc:
        if allow_astir_fallback:
            log.warning("Astir unavailable (%s). Falling back to rule classifier.", exc)
            return _compute_rule_probabilities(df), "rule_fallback", coverage
        raise


def build_csv_index(df: pd.DataFrame, x_col: str, y_col: str) -> scipy.spatial.KDTree:
    """Build a KDTree from the cell coordinate columns of a DataFrame."""
    coords = df[[x_col, y_col]].to_numpy(dtype=float)
    return scipy.spatial.KDTree(coords)


def compute_state_thresholds(
    df: pd.DataFrame,
    state_percentile: float = 75.0,
) -> dict[str, float]:
    """Compute state thresholds from feature table."""
    out: dict[str, float] = {}
    ki67_col = resolve_first_present_column(df.columns, "Ki67")
    if ki67_col is None:
        out["Ki67"] = float("inf")
    else:
        out["Ki67"] = float(np.nanpercentile(df[ki67_col].to_numpy(dtype=float), state_percentile))
    return out


def assign_type(row: pd.Series, thresholds: dict[str, float] | None) -> str:
    """Legacy type assignment kept for compatibility with older tests/tools."""
    def _get(marker: str) -> float:
        return _get_marker_value(row, marker)

    try:
        for type_name, markers in LEGACY_TYPE_MARKERS.items():
            for marker in markers:
                if _get(marker) >= thresholds.get(marker, float("inf")):  # type: ignore[union-attr]
                    return type_name
    except Exception:
        return "other"
    return "other"


def compute_thresholds(
    df: pd.DataFrame,
    default_state_percentile: float = 95.0,
    ecad_low_percentile: float = 25.0,
    ecad_high_percentile: float = 50.0,
) -> dict[str, float]:
    """Legacy threshold helper retained for compatibility.

    New Astir-first flow uses `compute_state_thresholds`.
    """
    out: dict[str, float] = {}

    for marker in ("Ki67", "PCNA", "Vimentin", "Keratin"):
        col = resolve_first_present_column(df.columns, marker)
        if col is None:
            out[marker] = float("inf")
        else:
            vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            out[marker] = float(np.nanpercentile(vals, default_state_percentile))

    ecad_col = resolve_first_present_column(df.columns, "Ecadherin")
    if ecad_col is None:
        out["Ecadherin"] = float("-inf")
        out["Ecadherin_high"] = float("inf")
    else:
        vals = pd.to_numeric(df[ecad_col], errors="coerce").to_numpy(dtype=float)
        out["Ecadherin"] = float(np.nanpercentile(vals, ecad_low_percentile))
        out["Ecadherin_high"] = float(np.nanpercentile(vals, ecad_high_percentile))

    return out


def assign_state(row: pd.Series, thresholds: dict[str, float] | None, type_cellvit: int = 0) -> str:
    """State assignment with Astir-first mode and legacy compatibility mode."""
    legacy_mode = bool(
        isinstance(thresholds, dict)
        and any(k in thresholds for k in ("PCNA", "Vimentin", "Ecadherin_high", "Keratin"))
    )

    try:
        if int(type_cellvit) == 4:
            return "apoptotic" if legacy_mode else "dead"

        if thresholds is None:
            return "other"

        if legacy_mode:
            ki67 = _get_marker_value(row, "Ki67")
            pcna = _get_marker_value(row, "PCNA")
            vimentin = _get_marker_value(row, "Vimentin")
            ecad = _get_marker_value(row, "Ecadherin")
            keratin = _get_marker_value(row, "Keratin")

            if ki67 >= thresholds.get("Ki67", float("inf")) or pcna >= thresholds.get(
                "PCNA", float("inf")
            ):
                return "proliferative"
            if vimentin >= thresholds.get("Vimentin", float("inf")) and ecad < thresholds.get(
                "Ecadherin", float("-inf")
            ):
                return "emt"
            if keratin >= thresholds.get("Keratin", float("inf")) and ecad >= thresholds.get(
                "Ecadherin_high", float("inf")
            ):
                return "quiescent"
            if ecad >= thresholds.get("Ecadherin_high", float("inf")) and keratin < thresholds.get(
                "Keratin", float("inf")
            ):
                return "healthy"
            return "other"

        ki67 = _get_marker_value(row, "Ki67")
        if ki67 >= thresholds.get("Ki67", float("inf")):
            return "proliferative"
        return "quiescent"
    except Exception:
        return "other"


def cellvit_prior_probs(type_cellvit: int) -> dict[str, float]:
    return CELLVIT_TYPE_PRIORS.get(
        int(type_cellvit),
        {"cancer": 1 / 3, "immune": 1 / 3, "healthy": 1 / 3},
    )


def confidence_from_probs(fused_probs: dict[str, float], mismatch: bool) -> str:
    vals = sorted((float(v) for v in fused_probs.values()), reverse=True)
    margin = vals[0] - vals[1] if len(vals) >= 2 else vals[0]

    high_th, med_th = CONFIDENCE_THRESHOLDS
    if margin >= high_th:
        conf = "high"
    elif margin >= med_th:
        conf = "medium"
    else:
        conf = "low"

    if mismatch and conf != "low":
        conf = "medium" if conf == "high" else "low"
    return conf


def match_cells(
    cells: list[dict],
    kdtree,
    df: pd.DataFrame,
    state_thresholds: dict[str, float],
    x0: int,
    y0: int,
    max_dist: float = 15.0,
    coord_scale: float = 1.0,
    model_weight: float = DEFAULT_ASTIR_WEIGHT,
) -> list[dict]:
    """Assign cell type/state by nearest-row lookup + probabilistic fusion."""
    model_weight = float(np.clip(model_weight, 0.0, 1.0))
    prior_weight = 1.0 - model_weight

    for cell in cells:
        try:
            centroid = cell.get("centroid", [0, 0])
            lx = float(centroid[0])
            ly = float(centroid[1])
            gx = (x0 + lx) * coord_scale
            gy = (y0 + ly) * coord_scale
            cell["centroid_x_local"] = lx
            cell["centroid_y_local"] = ly
            cell["centroid_x_global"] = gx
            cell["centroid_y_global"] = gy

            dist, idx = kdtree.query([gx, gy])
            type_cellvit = int(cell.get("type_cellvit", 0))
            prior_probs = cellvit_prior_probs(type_cellvit)
            prior_type = _argmax_label(prior_probs)

            if dist <= max_dist:
                row = df.iloc[idx]
                model_probs = {c: _safe_float(row.get(f"p_model_{c}", 0.0)) for c in CELL_TYPES}
                model_type = _argmax_label(model_probs)
                model_type_fine = str(row.get("type_astir_fine", model_type))
                fused_probs = {
                    c: model_weight * model_probs[c] + prior_weight * prior_probs[c]
                    for c in CELL_TYPES
                }
                final_type = _argmax_label(fused_probs)
                mismatch = model_type != prior_type
                confidence = confidence_from_probs(fused_probs, mismatch)
                state = assign_state(row, state_thresholds, type_cellvit)
                matched_row_idx: int | None = int(idx)
            else:
                model_probs = {c: 0.0 for c in CELL_TYPES}
                model_type = "unmatched"
                model_type_fine = "unmatched"
                fused_probs = prior_probs.copy()
                final_type = prior_type
                mismatch = False
                confidence = "low"
                state = "dead" if type_cellvit == 4 else "quiescent"
                matched_row_idx = None

            cell["cell_type"] = final_type
            cell["cell_type_confidence"] = confidence
            cell["cell_state"] = state
            cell["type_astir"] = model_type
            cell["type_astir_fine"] = model_type_fine
            cell["type_cellvit_prior"] = prior_type
            cell["is_mismatch"] = mismatch
            cell["matched_row_idx"] = matched_row_idx
            cell["match_distance"] = float(dist)
            for cls in CELL_TYPES:
                cell[f"p_final_{cls}"] = float(fused_probs[cls])
                cell[f"p_model_{cls}"] = float(model_probs[cls])

        except Exception:
            type_cellvit = int(cell.get("type_cellvit", 0))
            prior_probs = cellvit_prior_probs(type_cellvit)
            prior_type = _argmax_label(prior_probs)
            cell["cell_type"] = prior_type
            cell["cell_type_confidence"] = "low"
            cell["cell_state"] = "dead" if type_cellvit == 4 else "quiescent"
            cell["type_astir"] = "error"
            cell["type_astir_fine"] = "error"
            cell["type_cellvit_prior"] = prior_type
            cell["is_mismatch"] = False
            cell["matched_row_idx"] = None
            cell["match_distance"] = float("inf")
            for cls in CELL_TYPES:
                cell[f"p_final_{cls}"] = float(prior_probs[cls])
                cell[f"p_model_{cls}"] = 0.0

    return cells


def build_assignment_record(
    patch_id: str,
    x0: int,
    y0: int,
    cell: dict,
    row: pd.Series,
) -> dict[str, object]:
    """Build one analysis-ready row for downstream visualization tools."""
    record = {str(k): row[k] for k in row.index}
    record.update(
        {
            "patch_id": patch_id,
            "patch_x0": int(x0),
            "patch_y0": int(y0),
            "type_cellvit": int(cell.get("type_cellvit", 0)),
            "type_cellvit_prior": str(cell.get("type_cellvit_prior", "other")),
            "type_astir": str(cell.get("type_astir", "other")),
            "type_astir_fine": str(cell.get("type_astir_fine", cell.get("type_astir", "other"))),
            "cell_type": str(cell.get("cell_type", "other")),
            "cell_state": str(cell.get("cell_state", "other")),
            "cell_type_confidence": str(cell.get("cell_type_confidence", "low")),
            "is_mismatch": bool(cell.get("is_mismatch", False)),
            "centroid_x_local": _safe_float(cell.get("centroid_x_local", 0.0)),
            "centroid_y_local": _safe_float(cell.get("centroid_y_local", 0.0)),
            "centroid_x_global": _safe_float(cell.get("centroid_x_global", 0.0)),
            "centroid_y_global": _safe_float(cell.get("centroid_y_global", 0.0)),
            "match_distance": _safe_float(cell.get("match_distance", float("inf"))),
        }
    )
    for cls in CELL_TYPES:
        record[f"p_model_{cls}"] = _safe_float(cell.get(f"p_model_{cls}", 0.0))
        record[f"p_final_{cls}"] = _safe_float(cell.get(f"p_final_{cls}", 0.0))
    return record


def rasterize_cells(
    cells: list[dict],
    patch_size: int,
    color_key: str,
    color_map: dict[str, tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw filled cell contours into a uint8 RGBA image."""
    canvas = np.zeros((patch_size, patch_size, 4), dtype=np.uint8)

    for cell in cells:
        contour = cell.get("contour", [])
        if len(contour) < 3:
            continue

        label = str(cell.get(color_key, "other"))
        rgba = color_map.get(label, color_map.get("other", (150, 150, 150, 150)))

        pts = np.array(contour, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts], rgba)

    return canvas


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def _resolve_coord_cols(df: pd.DataFrame) -> tuple[str, str]:
    """Return (x_col, y_col): prefer Xt/Yt, fallback to X/Y."""
    if "Xt" in df.columns and "Yt" in df.columns:
        return "Xt", "Yt"
    if "X" in df.columns and "Y" in df.columns:
        return "X", "Y"
    raise ValueError(
        "CSV must contain coordinate columns 'Xt'/'Yt' or 'X'/'Y'. "
        f"Found columns: {list(df.columns)}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="Stage 3: Astir-first type/state assignment and contour rasterization."
    )
    parser.add_argument(
        "--cellvit-dir",
        required=True,
        help="Directory of {x0}_{y0}.json files from CellViT.",
    )
    parser.add_argument(
        "--features-csv",
        default=None,
        help=(
            "Optional precomputed features CSV with Xt/Yt + marker columns. "
            "If omitted, features are extracted from CellViT + --multiplex-dir."
        ),
    )
    parser.add_argument(
        "--multiplex-dir",
        default=None,
        help=(
            "Directory of {x0}_{y0}.npy multiplex patches (C,H,W). Required when "
            "--features-csv is omitted."
        ),
    )
    parser.add_argument(
        "--auto-features-csv",
        default=None,
        help=(
            "Output path for auto-extracted CellViT+MX features CSV. "
            "Default: <out>/cellvit_mx_features.csv."
        ),
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Optional channel names in multiplex patch order for auto-extraction. "
            "If omitted, uses index.json channels."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help=(
            "Optional channel metadata CSV used only if --channels and index.json "
            "channels are unavailable during auto-extraction."
        ),
    )
    parser.add_argument(
        "--index", required=True, help="processed/index.json (patch grid)."
    )
    parser.add_argument(
        "--out", default="processed/", help="Output directory (default: processed/)."
    )
    parser.add_argument(
        "--max-dist",
        type=float,
        default=15.0,
        help="Max nearest-neighbor distance in CSV pixel units (default: 15.0).",
    )
    parser.add_argument(
        "--coord-scale",
        type=float,
        default=1.0,
        help="Scale applied to H&E global coordinates before KDTree query.",
    )
    parser.add_argument(
        "--csv-mpp",
        type=float,
        default=0.65,
        help="Only used for external --features-csv: convert µm coordinates to px via divide.",
    )
    parser.add_argument(
        "--classifier",
        choices=["astir", "codex", "rule"],
        default="astir",
        help="Cell-type classifier: astir (default), codex clustering, or rule fallback.",
    )
    parser.add_argument(
        "--allow-astir-fallback",
        action="store_true",
        help="If Astir is unavailable, fallback to rule classifier instead of failing.",
    )
    parser.add_argument(
        "--astir-weight",
        type=float,
        default=DEFAULT_ASTIR_WEIGHT,
        help="Weight of model probabilities in fused type score (default: 0.85).",
    )
    parser.add_argument(
        "--state-percentile",
        type=float,
        default=75.0,
        help="Percentile for Ki67 proliferative threshold (default: 75).",
    )
    parser.add_argument(
        "--type-percentile",
        type=float,
        default=95.0,
        help="Deprecated in Astir-first mode; accepted for CLI compatibility.",
    )
    parser.add_argument(
        "--thresholds-config",
        default=None,
        help="Deprecated in Astir-first mode; accepted for CLI compatibility.",
    )
    args = parser.parse_args()

    if args.thresholds_config is not None:
        log.warning("--thresholds-config is ignored in Astir-first mode.")
    if args.type_percentile != 95.0:
        log.warning("--type-percentile is ignored in Astir-first mode.")

    cellvit_dir = pathlib.Path(args.cellvit_dir)
    features_csv: pathlib.Path | None = (
        pathlib.Path(args.features_csv) if args.features_csv else None
    )
    index_path = pathlib.Path(args.index)
    out_dir = pathlib.Path(args.out)
    max_dist = float(args.max_dist)
    coord_scale = float(args.coord_scale)
    astir_weight = float(np.clip(args.astir_weight, 0.0, 1.0))

    # Output subdirectories
    types_dir = out_dir / "cell_types"
    states_dir = out_dir / "cell_states"
    types_dir.mkdir(parents=True, exist_ok=True)
    states_dir.mkdir(parents=True, exist_ok=True)

    auto_extracted = False
    if features_csv is None:
        if not args.multiplex_dir:
            parser.error(
                "Either --features-csv must be provided, or --multiplex-dir "
                "must be set for CellViT+MX auto-extraction."
            )
        auto_extracted = True
        features_csv = (
            pathlib.Path(args.auto_features_csv)
            if args.auto_features_csv
            else out_dir / "cellvit_mx_features.csv"
        )
        log.info(
            "No --features-csv provided. Extracting CellViT+MX features to: %s",
            features_csv,
        )
        extract_cell_features_to_csv(
            cellvit_dir=cellvit_dir,
            multiplex_dir=pathlib.Path(args.multiplex_dir),
            index_path=index_path,
            out_csv=features_csv,
            channels=args.channels,
            metadata_csv=(
                pathlib.Path(args.metadata_csv) if args.metadata_csv else None
            ),
            coord_scale=coord_scale,
            logger=log,
        )

    # ------------------------------------------------------------------
    # 1. Load feature table and compute probabilities
    # ------------------------------------------------------------------
    log.info("Loading features CSV: %s", features_csv)
    df = pd.read_csv(features_csv)
    log.info("  %d rows loaded.", len(df))

    x_col, y_col = _resolve_coord_cols(df)
    log.info("  Using coordinate columns: (%s, %s)", x_col, y_col)

    if auto_extracted and args.csv_mpp != 1.0:
        log.warning(
            "Ignoring --csv-mpp=%.4f for auto-extracted features (already in query space).",
            args.csv_mpp,
        )
    elif args.csv_mpp != 1.0:
        log.info("  Converting CSV coords from µm to px (÷ %.4f) …", args.csv_mpp)
        df = df.copy()
        df[x_col] = df[x_col] / args.csv_mpp
        df[y_col] = df[y_col] / args.csv_mpp

    probs, classifier_used, marker_coverage = compute_type_probabilities(
        df,
        classifier=args.classifier,
        allow_astir_fallback=args.allow_astir_fallback,
        log=log,
    )
    fine_probs = probs.attrs.get("model_fine_probs")
    fine_top = probs.attrs.get("model_fine_top")
    if isinstance(fine_probs, pd.DataFrame):
        for fine_type in fine_probs.columns:
            df[f"p_model_{fine_type}"] = fine_probs[fine_type].to_numpy(dtype=float)
    if isinstance(fine_top, pd.Series):
        df["type_astir_fine"] = fine_top.astype(str).to_numpy()
    elif "type_astir_fine" not in df.columns:
        df["type_astir_fine"] = probs.idxmax(axis=1).astype(str).to_numpy()
    for cls in CELL_TYPES:
        df[f"p_model_{cls}"] = probs[cls].to_numpy(dtype=float)

    state_thresholds = compute_state_thresholds(df, state_percentile=args.state_percentile)
    log.info("State thresholds: %s", state_thresholds)

    # ------------------------------------------------------------------
    # 2. Build KDTree and load patch index
    # ------------------------------------------------------------------
    log.info("Building KDTree …")
    kdtree = build_csv_index(df, x_col, y_col)
    log.info("  KDTree built on %d points.", kdtree.n)

    log.info("Loading patch index: %s", index_path)
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)
    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    # ------------------------------------------------------------------
    # 3. Iterate patches
    # ------------------------------------------------------------------
    patch_size = int(index.get("patch_size", 256))
    processed = 0
    skipped = 0
    total_cells = 0

    assignments_path = out_dir / "cell_assignments.csv"
    assignment_records: list[dict[str, object]] = []
    per_patch_summary: dict[str, dict] = {}
    global_type_counts: Counter = Counter()
    global_state_counts: Counter = Counter()
    global_confidence_counts: Counter = Counter()
    global_mismatch_count = 0
    global_conflict_pairs: Counter = Counter()

    for patch_meta in patches:
        x0 = int(patch_meta["x0"])
        y0 = int(patch_meta["y0"])
        patch_id = f"{x0}_{y0}"

        json_path = cellvit_dir / f"{patch_id}.json"
        if not json_path.exists():
            log.warning("Missing CellViT file: %s — skipping.", json_path)
            skipped += 1
            continue

        with json_path.open(encoding="utf-8") as fh:
            cells: list[dict] = json.load(fh).get("cells", [])

        cells = match_cells(
            cells=cells,
            kdtree=kdtree,
            df=df,
            state_thresholds=state_thresholds,
            x0=x0,
            y0=y0,
            max_dist=max_dist,
            coord_scale=coord_scale,
            model_weight=astir_weight,
        )
        total_cells += len(cells)

        for c in cells:
            global_confidence_counts[c.get("cell_type_confidence", "low")] += 1
            if bool(c.get("is_mismatch", False)):
                global_mismatch_count += 1
                key = f"model={c.get('type_astir')},cellvit={c.get('type_cellvit_prior')}"
                global_conflict_pairs[key] += 1
            matched_row_idx = c.get("matched_row_idx")
            if matched_row_idx is None:
                continue
            assignment_records.append(
                build_assignment_record(
                    patch_id=patch_id,
                    x0=x0,
                    y0=y0,
                    cell=c,
                    row=df.iloc[int(matched_row_idx)],
                )
            )

        type_img = rasterize_cells(cells, patch_size, "cell_type", CELL_TYPE_COLORS)
        state_img = rasterize_cells(cells, patch_size, "cell_state", CELL_STATE_COLORS)

        Image.fromarray(type_img, mode="RGBA").save(types_dir / f"{patch_id}.png")
        Image.fromarray(state_img, mode="RGBA").save(states_dir / f"{patch_id}.png")

        type_counts = Counter(c.get("cell_type", "other") for c in cells)
        state_counts = Counter(c.get("cell_state", "other") for c in cells)
        per_patch_summary[patch_id] = {
            "n_cells": len(cells),
            "x0": x0,
            "y0": y0,
            "cell_types": dict(type_counts),
            "cell_states": dict(state_counts),
        }
        global_type_counts += type_counts
        global_state_counts += state_counts

        processed += 1
        if processed % 50 == 0:
            log.info("  Progress: %d patches processed, %d skipped …", processed, skipped)

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    mismatch_rate = float(global_mismatch_count / total_cells) if total_cells else 0.0
    assignment_columns = list(
        dict.fromkeys(
            [
        "patch_id",
        "patch_x0",
        "patch_y0",
        "type_cellvit",
        "type_cellvit_prior",
        "type_astir",
        "type_astir_fine",
        "cell_type",
        "cell_state",
        "cell_type_confidence",
        "is_mismatch",
        "centroid_x_local",
        "centroid_y_local",
        "centroid_x_global",
        "centroid_y_global",
        "match_distance",
        *list(df.columns),
        *(f"p_model_{cls}" for cls in CELL_TYPES),
        *(f"p_final_{cls}" for cls in CELL_TYPES),
            ]
        )
    )
    pd.DataFrame.from_records(assignment_records, columns=assignment_columns).to_csv(
        assignments_path,
        index=False,
    )

    summary = {
        "n_patches": processed,
        "n_cells": total_cells,
        "feature_source": "cellvit_mx_auto" if auto_extracted else "features_csv",
        "features_csv": str(features_csv),
        "cell_assignments_csv": str(assignments_path),
        "classifier_requested": args.classifier,
        "classifier_used": classifier_used,
        "astir_first": classifier_used == "astir",
        "astir_weight": astir_weight,
        "coord_scale": coord_scale,
        "state_percentile": args.state_percentile,
        "cell_types": dict(global_type_counts),
        "cell_states": dict(global_state_counts),
        "confidence": dict(global_confidence_counts),
        "mismatch_count": int(global_mismatch_count),
        "mismatch_rate": mismatch_rate,
        "conflict_pairs": dict(global_conflict_pairs),
        "marker_coverage": marker_coverage,
        "non_typing_markers": list(NON_TYPING_MARKERS),
        "per_patch": per_patch_summary,
    }

    summary_path = out_dir / "cell_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary written to %s", summary_path)
    log.info("Assignments CSV written to %s", assignments_path)

    log.info("Done.")
    log.info("  Patches processed : %d", processed)
    log.info("  Patches skipped   : %d", skipped)
    log.info("  Total cells       : %d", total_cells)
    log.info("  Cell types  (global): %s", dict(global_type_counts))
    log.info("  Cell states (global): %s", dict(global_state_counts))
    log.info("  Confidence (global): %s", dict(global_confidence_counts))
    log.info("  Mismatch rate      : %.4f", mismatch_rate)
    log.info("  Cell type PNGs    → %s", types_dir)
    log.info("  Cell state PNGs   → %s", states_dir)


if __name__ == "__main__":
    main()
