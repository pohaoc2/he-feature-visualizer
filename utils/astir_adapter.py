"""Astir integration helpers for probabilistic cell-type prediction."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

_ASTIR_NORMALIZE_BOUNDS: tuple[int, int] = (1, 99)


class AstirUnavailableError(RuntimeError):
    """Raised when Astir is unavailable or fails to run."""


def _to_dataframe(
    probs: Any,
    index: pd.Index,
    expected_classes: list[str],
    *,
    logger: logging.Logger | None = None,
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

    df = pd.DataFrame(resolved, index=index).fillna(0.0).clip(lower=0.0)
    expected_keys = {cls.lower() for cls in expected_classes}
    extra_columns = [
        col for col in out.columns if str(col).strip().lower() not in expected_keys
    ]
    if extra_columns:
        extra_mass = (
            out.loc[:, extra_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .clip(lower=0.0)
            .sum(axis=1)
        )
        if logger is not None:
            logger.info(
                "Astir returned auxiliary probability columns %s; "
                "preserving target-class mass without renormalizing "
                "(mean auxiliary mass=%.3f).",
                extra_columns,
                float(extra_mass.mean()),
            )
        return df

    row_sum = df.sum(axis=1)
    safe_den = row_sum.where(row_sum > 0.0, 1.0)
    return df.div(safe_den, axis=0)


def _sanitize_expr_df(
    expr_df: pd.DataFrame,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Coerce Astir inputs to a numeric non-negative matrix."""
    out = expr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    negative_count = int((out < 0.0).sum().sum())
    if negative_count > 0:
        logger.warning(
            "Clipping %d negative Astir intensities to zero before normalization.",
            negative_count,
        )
        out = out.clip(lower=0.0)
    return out


def _normalize_type_dataset(model: Any, logger: logging.Logger) -> None:
    """Apply Astir's documented preprocessing to the type dataset."""
    lower, upper = _ASTIR_NORMALIZE_BOUNDS
    normalize_model = getattr(model, "normalize", None)
    last_exc: Exception | None = None

    if callable(normalize_model):
        for args, kwargs in (
            ((), {"percentile_lower": lower, "percentile_upper": upper}),
            ((lower, upper), {}),
            ((), {}),
        ):
            try:
                normalize_model(*args, **kwargs)
                logger.info(
                    "Applied Astir.normalize() to the model input with bounds [%d, %d].",
                    lower,
                    upper,
                )
                return
            except Exception as exc:  # pragma: no cover - version-specific path
                last_exc = exc

    type_dset = getattr(model, "_type_dset", None)
    if type_dset is None:
        get_type_dataset = getattr(model, "get_type_dataset", None)
        if callable(get_type_dataset):
            try:
                type_dset = get_type_dataset()
            except Exception as exc:  # pragma: no cover - version-specific path
                last_exc = exc

    normalize_dset = getattr(type_dset, "normalize", None)
    if callable(normalize_dset):
        for args, kwargs in (
            ((), {"percentile_lower": lower, "percentile_upper": upper}),
            ((lower, upper), {}),
            ((), {}),
        ):
            try:
                normalize_dset(*args, **kwargs)
                logger.info(
                    "Applied Astir type-dataset normalization with bounds [%d, %d].",
                    lower,
                    upper,
                )
                return
            except TypeError:
                continue
            except Exception as exc:  # pragma: no cover - version-specific path
                last_exc = exc
                break

    raise AstirUnavailableError(
        "Astir normalization failed before fit_type()."
    ) from last_exc


def _repair_type_dataset_after_normalize(model: Any, logger: logging.Logger) -> None:
    """Repair Astir dataset tensors after normalization.

    Some Astir builds normalize onto CPU tensors without refreshing cached
    dataset statistics, which later causes CPU/CUDA mismatches in ``fit_type``.
    """
    try:
        import torch
    except Exception:  # pragma: no cover - torch is an Astir dependency
        return

    type_dset = getattr(model, "_type_dset", None)
    if type_dset is None:
        get_type_dataset = getattr(model, "get_type_dataset", None)
        if callable(get_type_dataset):
            try:
                type_dset = get_type_dataset()
            except Exception:  # pragma: no cover - version-specific path
                type_dset = None
    if type_dset is None:
        return

    exprs = getattr(type_dset, "_exprs", None)
    if not torch.is_tensor(exprs):
        return

    target_device = getattr(type_dset, "_device", exprs.device)
    target_dtype = getattr(type_dset, "_dtype", exprs.dtype)
    repaired = False

    if exprs.device != target_device or exprs.dtype != target_dtype:
        type_dset._exprs = exprs.to(device=target_device, dtype=target_dtype)
        repaired = True

    for attr in ("_marker_mat", "_design"):
        value = getattr(type_dset, attr, None)
        if torch.is_tensor(value) and (
            value.device != target_device or value.dtype != target_dtype
        ):
            setattr(type_dset, attr, value.to(device=target_device, dtype=target_dtype))
            repaired = True

    type_dset._exprs_mean = type_dset._exprs.mean(0) + 1e-6
    type_dset._exprs_std = type_dset._exprs.std(0) + 1e-6

    if repaired:
        logger.info(
            "Repaired Astir type dataset tensors on %s after normalization.",
            target_device,
        )


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

    expr_df = _sanitize_expr_df(expr_df, log)
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
            candidate = Astir(input_expr=expr_df, marker_dict=variant)
            # Some Astir versions accept the constructor call but silently fail
            # to build a cell-type dataset when the marker_dict wrapper key is wrong.
            if getattr(candidate, "_type_dset", None) is None:
                last_exc = RuntimeError("Astir marker_dict variant did not build a type dataset.")
                continue
            model = candidate
            break
        except Exception as exc:  # pragma: no cover - version-specific path
            last_exc = exc

    if model is None:
        raise AstirUnavailableError(
            "Failed to initialize Astir model with known marker_dict formats."
        ) from last_exc

    _normalize_type_dataset(model, log)
    _repair_type_dataset_after_normalize(model, log)

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

    out = _to_dataframe(
        probs,
        index=expr_df.index,
        expected_classes=classes,
        logger=log,
    )
    log.info("Astir produced probabilities for %d cells.", len(out))
    return out
