"""Astir integration helpers for probabilistic cell-type prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd


class AstirUnavailableError(RuntimeError):
    """Raised when Astir is unavailable or fails to run."""


def _to_dataframe(
    probs: Any,
    index: pd.Index,
    expected_classes: list[str],
) -> pd.DataFrame:
    """Convert model output to probability DataFrame with expected class columns."""
    if isinstance(probs, pd.DataFrame):
        out = probs.copy()
    elif isinstance(probs, np.ndarray):
        out = pd.DataFrame(probs, index=index)
    else:
        out = pd.DataFrame(np.asarray(probs), index=index)

    # Normalize column names to lowercase keys for resilient matching.
    normalized_columns = {str(c).strip().lower(): c for c in out.columns}
    resolved: dict[str, pd.Series] = {}
    for cls in expected_classes:
        key = cls.lower()
        if key in normalized_columns:
            resolved[cls] = pd.to_numeric(
                out[normalized_columns[key]], errors="coerce"
            ).fillna(0.0)
        else:
            # Missing class columns are treated as zero confidence.
            resolved[cls] = pd.Series(0.0, index=out.index)

    df = pd.DataFrame(resolved, index=index).fillna(0.0)
    row_sum = df.sum(axis=1)
    safe_den = row_sum.where(row_sum > 0.0, 1.0)
    return df.div(safe_den, axis=0)


def predict_cell_type_probabilities(
    expr_df: pd.DataFrame,
    marker_dict: dict[str, list[str]],
    *,
    random_seed: int = 0,
    max_epochs: int = 200,
    use_gpu: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Run Astir and return per-cell type probabilities.

    Returns a DataFrame with columns in ``marker_dict`` key order.
    """
    log = logger or logging.getLogger(__name__)
    classes = list(marker_dict.keys())

    try:
        from astir import Astir  # type: ignore
    except Exception as exc:
        raise AstirUnavailableError(
            "Astir dependency is unavailable. Install package 'astir'."
        ) from exc

    np.random.seed(random_seed)

    # Astir versions differ in accepted marker_dict schema; try common variants.
    marker_variants: list[Any] = [
        marker_dict,
        {"cell_type": marker_dict},
        {"cell_types": marker_dict},
    ]

    last_exc: Exception | None = None
    model = None
    for variant in marker_variants:
        try:
            model = Astir(input_expr=expr_df, marker_dict=variant)
            break
        except Exception as exc:  # pragma: no cover - version-specific path
            last_exc = exc

    if model is None:
        raise AstirUnavailableError(
            "Failed to initialize Astir model with known marker_dict formats."
        ) from last_exc

    fit_variants = [
        {"max_epochs": max_epochs, "use_gpu": use_gpu},
        {"max_epochs": max_epochs},
        {},
    ]
    fit_ok = False
    for kwargs in fit_variants:
        try:
            model.fit_type(**kwargs)
            fit_ok = True
            break
        except TypeError:
            continue
        except Exception as exc:  # pragma: no cover - model-runtime path
            last_exc = exc
            break

    if not fit_ok:
        raise AstirUnavailableError("Astir fit_type() failed.") from last_exc

    try:
        probs = model.get_celltype_probabilities()
    except Exception as exc:  # pragma: no cover - model-runtime path
        raise AstirUnavailableError(
            "Astir get_celltype_probabilities() failed."
        ) from exc

    out = _to_dataframe(probs, index=expr_df.index, expected_classes=classes)
    log.info("Astir produced probabilities for %d cells.", len(out))
    return out
