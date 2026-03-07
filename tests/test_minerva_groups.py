import pandas as pd
import pytest

from utils.minerva_groups import (
    GROUP_SPECS,
    build_group_flags,
    build_group_meta,
    compute_marker_thresholds,
    resolve_component_sources,
)


def test_compute_marker_thresholds_uses_requested_percentile():
    df = pd.DataFrame(
        {
            "CD45": [0.0, 10.0, 20.0, 30.0],
            "CD31": [1.0, 2.0, 3.0, 4.0],
            "Xt": [0.0, 1.0, 2.0, 3.0],
            "Yt": [0.0, 1.0, 2.0, 3.0],
        }
    )

    thresholds = compute_marker_thresholds(df, percentile=50)

    assert thresholds["CD45"] == 15.0
    assert thresholds["CD31"] == 2.5
    assert "Ki67" not in thresholds


def test_build_group_flags_assigns_multiple_groups():
    df = pd.DataFrame(
        {
            "CD45": [100.0, 1.0, 1.0, 1.0],
            "CD31": [1.0, 1.0, 100.0, 1.0],
            "Keratin": [1.0, 100.0, 1.0, 1.0],
            "Ki67": [1.0, 1.0, 1.0, 100.0],
            "aSMA": [1.0, 1.0, 1.0, 100.0],
            "Xt": [1.0, 2.0, 3.0, 4.0],
            "Yt": [5.0, 6.0, 7.0, 8.0],
        }
    )
    thresholds = {"CD45": 50.0, "CD31": 50.0, "Keratin": 50.0, "Ki67": 50.0, "aSMA": 50.0}

    flags = build_group_flags(df, thresholds)

    assert flags["grp_immune"].tolist() == [True, False, False, False]
    assert flags["grp_cancer"].tolist() == [False, True, False, False]
    assert flags["grp_vasculature"].tolist() == [False, False, True, False]
    assert flags["grp_proliferative"].tolist() == [False, False, False, True]
    assert flags["grp_tissue"].tolist() == [False, True, False, True]


def test_build_group_meta_counts_rows_per_group():
    df = pd.DataFrame(
        {
            "grp_immune": [True, False, True],
            "grp_tissue": [False, True, False],
            "grp_cancer": [False, False, True],
            "grp_proliferative": [False, False, False],
            "grp_vasculature": [True, True, False],
        }
    )

    component_sources = resolve_component_sources({"CD45": 1.0, "CD31": 1.0, "Keratin": 1.0, "aSMA": 1.0})
    meta = build_group_meta(df, component_sources=component_sources)
    counts = {item["id"]: item["count"] for item in meta}

    assert set(counts) == set(GROUP_SPECS)
    assert counts["immune"] == 2
    assert counts["tissue"] == 1
    assert counts["cancer"] == 1
    assert counts["proliferative"] == 0
    assert counts["vasculature"] == 2
    tissue = next(item for item in meta if item["id"] == "tissue")
    assert tissue["components"][0]["id"] == "dna1"
    assert tissue["components"][1]["id"] == "panck"
    assert tissue["components"][2]["id"] == "asma"


def test_resolve_component_sources_uses_alias_fallbacks():
    thresholds = {
        "Hoechst0": 1.0,
        "Keratin": 2.0,
        "aSMA": 3.0,
        "CD45": 4.0,
    }
    resolved = resolve_component_sources(thresholds)

    assert resolved["tissue"]["dna1"] == "Hoechst0"
    assert resolved["tissue"]["panck"] == "Keratin"
    assert resolved["tissue"]["asma"] == "aSMA"
    assert resolved["immune"]["cd45"] == "CD45"


def test_compute_marker_thresholds_rejects_invalid_percentile():
    df = pd.DataFrame({"CD45": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="between 0 and 100"):
        compute_marker_thresholds(df, percentile=101)


def test_build_group_flags_handles_missing_marker_columns():
    df = pd.DataFrame({"Xt": [0.0, 1.0], "Yt": [0.0, 1.0]})
    flags = build_group_flags(df, thresholds={})
    assert flags.shape == (2, len(GROUP_SPECS))
    assert flags.to_numpy().sum() == 0
