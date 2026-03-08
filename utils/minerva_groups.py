"""Group definitions and marker-threshold helpers for the Minerva-style viewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarkerComponent:
    """One colored marker channel in a group composite preset."""

    id: str
    label: str
    color: tuple[int, int, int, int]
    markers: tuple[str, ...]


@dataclass(frozen=True)
class GroupSpec:
    """Display metadata and marker rules for one overlay group."""

    label: str
    color: tuple[int, int, int, int]
    markers: tuple[str, ...]
    components: tuple[MarkerComponent, ...]


GROUP_SPECS: dict[str, GroupSpec] = {
    "immune": GroupSpec(
        label="Immune",
        color=(68, 170, 255, 210),
        markers=(
            "CD45",
            "CD3",
            "CD4",
            "CD8a",
            "CD20",
            "CD68",
            "CD163",
            "FOXP3",
            "CD45RO",
        ),
        components=(
            MarkerComponent("cd45", "CD45", (74, 234, 255, 220), ("CD45",)),
            MarkerComponent("cd3", "CD3", (63, 149, 255, 220), ("CD3",)),
            MarkerComponent("cd4", "CD4", (139, 209, 255, 220), ("CD4",)),
            MarkerComponent("cd8a", "CD8a", (28, 84, 255, 220), ("CD8a",)),
            MarkerComponent("cd20", "CD20", (180, 117, 255, 220), ("CD20",)),
            MarkerComponent("cd68", "CD68", (95, 255, 160, 220), ("CD68",)),
            MarkerComponent("cd163", "CD163", (52, 206, 124, 220), ("CD163",)),
            MarkerComponent("foxp3", "FOXP3", (255, 132, 226, 220), ("FOXP3",)),
            MarkerComponent("cd45ro", "CD45RO", (255, 197, 102, 220), ("CD45RO",)),
        ),
    ),
    "tissue": GroupSpec(
        label="Tissue/Stromal",
        color=(255, 188, 74, 210),
        markers=("Hoechst1", "DNA1", "Hoechst0", "PanCk", "PanCK", "Keratin", "aSMA"),
        components=(
            MarkerComponent(
                "dna1", "DNA1", (87, 140, 255, 220), ("Hoechst1", "DNA1", "Hoechst0")
            ),
            MarkerComponent(
                "panck", "PanCk", (255, 191, 74, 220), ("PanCk", "PanCK", "Keratin")
            ),
            MarkerComponent("asma", "aSMA", (87, 224, 149, 220), ("aSMA",)),
        ),
    ),
    "cancer": GroupSpec(
        label="Cancer",
        color=(255, 95, 109, 210),
        markers=("Keratin", "CDX2", "Ecadherin"),
        components=(
            MarkerComponent("keratin", "Keratin", (255, 95, 109, 220), ("Keratin",)),
            MarkerComponent("cdx2", "CDX2", (255, 208, 92, 220), ("CDX2",)),
            MarkerComponent(
                "ecadherin", "Ecadherin", (106, 221, 255, 220), ("Ecadherin",)
            ),
        ),
    ),
    "proliferative": GroupSpec(
        label="Proliferative",
        color=(132, 225, 110, 210),
        markers=("Ki67", "Ki67_570", "PCNA"),
        components=(
            MarkerComponent("ki67", "Ki67", (132, 225, 110, 220), ("Ki67",)),
            MarkerComponent("ki67_570", "Ki67_570", (93, 250, 175, 220), ("Ki67_570",)),
            MarkerComponent("pcna", "PCNA", (255, 166, 102, 220), ("PCNA",)),
        ),
    ),
    "vasculature": GroupSpec(
        label="Vasculature",
        color=(229, 75, 75, 210),
        markers=("CD31",),
        components=(MarkerComponent("cd31", "CD31", (229, 75, 75, 220), ("CD31",)),),
    ),
}


def all_group_markers(groups: Mapping[str, GroupSpec] = GROUP_SPECS) -> list[str]:
    """Return sorted union of marker names used by all groups."""
    markers: set[str] = set()
    for spec in groups.values():
        markers.update(spec.markers)
        for component in spec.components:
            markers.update(component.markers)
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
    component_sources: Mapping[str, Mapping[str, str | None]] | None = None,
    groups: Mapping[str, GroupSpec] = GROUP_SPECS,
) -> list[dict]:
    """Build serializable group metadata used by the frontend."""
    out: list[dict] = []
    for group_id, spec in groups.items():
        count = int(df.get(f"grp_{group_id}", pd.Series(dtype=bool)).sum())
        sources = component_sources.get(group_id, {}) if component_sources else {}
        components = []
        for component in spec.components:
            source_marker = sources.get(component.id)
            components.append(
                {
                    "id": component.id,
                    "label": component.label,
                    "color": list(component.color),
                    "markers": list(component.markers),
                    "source_marker": source_marker,
                    "available": source_marker is not None,
                }
            )
        out.append(
            {
                "id": group_id,
                "label": spec.label,
                "color": list(spec.color),
                "markers": list(spec.markers),
                "count": count,
                "components": components,
            }
        )
    return out


def resolve_component_sources(
    thresholds: Mapping[str, float],
    groups: Mapping[str, GroupSpec] = GROUP_SPECS,
) -> dict[str, dict[str, str | None]]:
    """Pick one available marker column for each group component."""
    available = set(thresholds.keys())
    out: dict[str, dict[str, str | None]] = {}
    for group_id, spec in groups.items():
        group_map: dict[str, str | None] = {}
        for component in spec.components:
            source = next(
                (marker for marker in component.markers if marker in available), None
            )
            group_map[component.id] = source
        out[group_id] = group_map
    return out
