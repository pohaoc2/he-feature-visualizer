"""Group definitions and marker-threshold helpers for the Minerva-style viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupSpec:
    """Display metadata and marker rules for one overlay group."""

    label: str
    color: tuple[int, int, int, int]
    markers: tuple[str, ...]


GROUP_SPECS: dict[str, GroupSpec] = {
    "immune": GroupSpec(
        label="Immune",
        color=(68, 170, 255, 210),
        markers=("CD45", "CD3", "CD4", "CD8a", "CD20", "CD68", "CD163", "FOXP3", "CD45RO"),
    ),
    "tissue": GroupSpec(
        label="Tissue/Stromal",
        color=(255, 188, 74, 210),
        markers=("aSMA", "Vimentin", "Collagen", "Desmin"),
    ),
    "cancer": GroupSpec(
        label="Cancer",
        color=(255, 95, 109, 210),
        markers=("Keratin", "CDX2", "Ecadherin"),
    ),
    "proliferative": GroupSpec(
        label="Proliferative",
        color=(132, 225, 110, 210),
        markers=("Ki67", "Ki67_570", "PCNA"),
    ),
    "vasculature": GroupSpec(
        label="Vasculature",
        color=(229, 75, 75, 210),
        markers=("CD31",),
    ),
}


def all_group_markers(groups: Mapping[str, GroupSpec] = GROUP_SPECS) -> list[str]:
    """Return sorted union of marker names used by all groups."""
    markers = {marker for spec in groups.values() for marker in spec.markers}
    return sorted(markers)


def compute_marker_thresholds(
    df: pd.DataFrame,
    percentile: float = 95.0,
    groups: Mapping[str, GroupSpec] = GROUP_SPECS,
) -> dict[str, float]:
    """Compute percentile thresholds for markers that exist in the dataframe."""
    p = float(percentile)
    if p < 0.0 or p > 100.0:
        raise ValueError("percentile must be between 0 and 100")

    thresholds: dict[str, float] = {}
    for marker in all_group_markers(groups):
        if marker not in df.columns:
            continue
        values = pd.to_numeric(df[marker], errors="coerce").dropna()
        if values.empty:
            continue
        thresholds[marker] = float(np.percentile(values.values, p))
    return thresholds


def build_group_flags(
    df: pd.DataFrame,
    thresholds: Mapping[str, float],
    groups: Mapping[str, GroupSpec] = GROUP_SPECS,
) -> pd.DataFrame:
    """Return boolean columns (`grp_<group_id>`) for each group definition."""
    flags: dict[str, pd.Series] = {}

    for group_id, spec in groups.items():
        mask = pd.Series(False, index=df.index)
        for marker in spec.markers:
            threshold = thresholds.get(marker)
            if threshold is None or marker not in df.columns:
                continue
            marker_values = pd.to_numeric(df[marker], errors="coerce")
            mask = mask | (marker_values > threshold).fillna(False)
        flags[f"grp_{group_id}"] = mask.astype(bool)

    return pd.DataFrame(flags, index=df.index)


def build_group_meta(
    df: pd.DataFrame,
    groups: Mapping[str, GroupSpec] = GROUP_SPECS,
) -> list[dict]:
    """Build serializable group metadata used by the frontend."""
    out: list[dict] = []
    for group_id, spec in groups.items():
        count = int(df.get(f"grp_{group_id}", pd.Series(dtype=bool)).sum())
        out.append(
            {
                "id": group_id,
                "label": spec.label,
                "color": list(spec.color),
                "markers": list(spec.markers),
                "count": count,
            }
        )
    return out
