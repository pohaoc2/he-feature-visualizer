from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.group_visualizer import (
    GROUP_DEFINITIONS,
    MarkerChannel,
    apply_colormap,
    alpha_blend,
    combine_channels,
    load_marker_channel_map,
    max_project_multicolor,
    normalize_marker_name,
    resolve_group_channels,
    tint_map,
)


def test_normalize_marker_name_handles_punctuation_and_case() -> None:
    assert normalize_marker_name("CDX-2") == "cdx2"
    assert normalize_marker_name("  Antigen Ki67 (2) ") == "antigenki672"
    assert normalize_marker_name("Pan-cytokeratin") == "pancytokeratin"


def test_load_marker_channel_map_uses_first_occurrence(tmp_path) -> None:
    csv_path = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "Channel_Number": [1, 2, 7],
            "Marker_Name": ["CD45", "CD4", "CD45"],
        }
    ).to_csv(csv_path, index=False)

    marker_map = load_marker_channel_map(csv_path)

    assert marker_map[normalize_marker_name("CD45")].channel_index == 0
    assert marker_map[normalize_marker_name("CD4")].channel_index == 1


def test_resolve_group_channels_picks_available_markers() -> None:
    marker_map = {
        normalize_marker_name("CD45"): MarkerChannel("CD45", 18),
        normalize_marker_name("CD8a"): MarkerChannel("CD8a", 23),
        normalize_marker_name("CD31"): MarkerChannel("CD31", 35),
        normalize_marker_name("Antigen Ki67"): MarkerChannel("Antigen Ki67", 13),
    }

    resolved = resolve_group_channels(marker_map, groups=GROUP_DEFINITIONS)

    immune_markers = [ch.marker_name for ch in resolved["immune"].channels]
    assert "CD45" in immune_markers
    assert "CD8a" in immune_markers

    vasculature_markers = [ch.marker_name for ch in resolved["vasculature"].channels]
    assert vasculature_markers == ["CD31"]

    proliferative_markers = [
        ch.marker_name for ch in resolved["proliferative"].channels
    ]
    assert proliferative_markers == ["Antigen Ki67"]


def test_resolve_group_channels_deduplicates_aliases_for_same_channel() -> None:
    marker_map = {
        normalize_marker_name("Pan-cytokeratin"): MarkerChannel("Pan-cytokeratin", 14),
        normalize_marker_name("E-cadherin"): MarkerChannel("E-cadherin", 29),
        normalize_marker_name("CDX-2"): MarkerChannel("CDX-2", 31),
    }
    resolved = resolve_group_channels(marker_map, groups=GROUP_DEFINITIONS)

    cancer = resolved["cancer"].channels
    names = [c.marker_name for c in cancer]
    idxs = [c.channel_index for c in cancer]

    assert names == ["Pan-cytokeratin", "E-cadherin", "CDX-2"]
    assert len(idxs) == len(set(idxs))


def test_resolve_group_channels_collapses_proliferative_marker_family() -> None:
    marker_map = {
        normalize_marker_name("Antigen Ki67"): MarkerChannel("Antigen Ki67", 13),
        normalize_marker_name("Antigen Ki67 (2)"): MarkerChannel(
            "Antigen Ki67 (2)", 38
        ),
        normalize_marker_name("PCNA"): MarkerChannel("PCNA", 37),
    }
    resolved = resolve_group_channels(marker_map, groups=GROUP_DEFINITIONS)

    prolif = resolved["proliferative"].channels
    assert [m.marker_name for m in prolif] == ["Antigen Ki67"]


def test_combine_channels_max_and_sum_without_percentile() -> None:
    chw = np.array(
        [
            [[0, 65535], [32768, 0]],
            [[65535, 0], [32768, 65535]],
        ],
        dtype=np.uint16,
    )

    combined_max = combine_channels(chw, mode="max")
    combined_sum = combine_channels(chw, mode="sum")

    assert combined_max.shape == (2, 2)
    assert float(combined_max[0, 0]) == pytest.approx(1.0)
    assert float(combined_max[1, 0]) == pytest.approx(32768 / 65535, abs=1e-6)

    assert combined_sum.shape == (2, 2)
    assert float(combined_sum[0, 0]) == pytest.approx(1.0)
    assert float(combined_sum[1, 0]) == pytest.approx(1.0)


def test_tint_map_and_alpha_blend() -> None:
    gray = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)
    tint = tint_map(gray, (200, 100, 50))

    assert tint.shape == (2, 2, 3)
    assert tint.dtype == np.uint8
    assert tuple(tint[0, 0]) == (0, 0, 0)

    base = np.zeros((2, 2, 3), dtype=np.uint8)
    overlay = np.full((2, 2, 3), 100, dtype=np.uint8)
    blended = alpha_blend(base, overlay, 0.25)
    assert int(blended[0, 0, 0]) == 25


def test_multicolor_max_projection_returns_per_marker_colors() -> None:
    chw = np.array(
        [
            [[65535, 0], [0, 0]],
            [[0, 65535], [0, 0]],
            [[0, 0], [65535, 0]],
        ],
        dtype=np.uint16,
    )
    markers = ("A", "B", "C")

    rgb, marker_colors = max_project_multicolor(chw, markers, colormap="tab10")

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert len(marker_colors) == 3
    assert marker_colors[0][0] == "A"
    assert tuple(rgb[0, 0]) != tuple(rgb[0, 1])


def test_apply_colormap_outputs_rgb_uint8() -> None:
    gray = np.array([[0.0, 1.0]], dtype=np.float32)
    rgb = apply_colormap(gray, colormap="viridis")
    assert rgb.shape == (1, 2, 3)
    assert rgb.dtype == np.uint8
