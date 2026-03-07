import asyncio
import json

import pandas as pd
import pytest
from fastapi import HTTPException

import server_minerva


@pytest.fixture(autouse=True)
def _clear_server_state():
    """Reset global server state between tests."""
    server_minerva.G.clear()
    yield
    server_minerva.G.clear()


def test_resolve_coord_cols_prefers_xt_yt():
    df = pd.DataFrame({"Xt": [1.0], "Yt": [2.0], "X": [3.0], "Y": [4.0]})
    assert server_minerva.resolve_coord_cols(df) == ("Xt", "Yt")


def test_resolve_coord_cols_falls_back_to_xy():
    df = pd.DataFrame({"X": [3.0], "Y": [4.0]})
    assert server_minerva.resolve_coord_cols(df) == ("X", "Y")


def test_resolve_coord_cols_raises_without_coords():
    df = pd.DataFrame({"col": [1.0]})
    with pytest.raises(ValueError, match="Xt/Yt or X/Y"):
        server_minerva.resolve_coord_cols(df)


def test_dzi_level_info_contains_all_levels():
    levels, max_level = server_minerva.dzi_level_info(img_w=1000, img_h=300)
    assert max_level == 10
    assert levels[0]["w"] == 1
    assert levels[max_level]["w"] == 1000
    assert levels[max_level]["h"] == 300


def test_cells_returns_empty_for_he_group():
    response = asyncio.run(
        server_minerva.cells(group="he", x0=0, y0=0, x1=100, y1=100, max_cells=500)
    )
    payload = json.loads(response.body)
    assert payload == {"cells": [], "total_in_view": 0, "sampled": False}


def test_cells_filters_by_group_and_viewport():
    server_minerva.G["df"] = pd.DataFrame(
        {
            "Xt": [10.0, 20.0, 500.0],
            "Yt": [10.0, 20.0, 500.0],
            "grp_immune": [True, False, True],
        }
    )
    server_minerva.G["coord_cols"] = ("Xt", "Yt")
    server_minerva.G["group_ids"] = {"immune"}

    response = asyncio.run(
        server_minerva.cells(group="immune", x0=0, y0=0, x1=100, y1=100, max_cells=500)
    )
    payload = json.loads(response.body)

    assert payload["total_in_view"] == 1
    assert payload["sampled"] is False
    assert len(payload["cells"]) == 1
    assert payload["cells"][0]["x"] == 10.0
    assert payload["cells"][0]["y"] == 10.0


def test_cells_applies_sampling_flag():
    n = 20
    server_minerva.G["df"] = pd.DataFrame(
        {
            "Xt": [float(i) for i in range(n)],
            "Yt": [float(i) for i in range(n)],
            "grp_immune": [True] * n,
        }
    )
    server_minerva.G["coord_cols"] = ("Xt", "Yt")
    server_minerva.G["group_ids"] = {"immune"}

    response = asyncio.run(
        server_minerva.cells(group="immune", x0=-1, y0=-1, x1=100, y1=100, max_cells=5)
    )
    payload = json.loads(response.body)

    assert payload["total_in_view"] == n
    assert payload["sampled"] is True
    assert len(payload["cells"]) == 5


def test_cells_raises_for_unknown_group():
    server_minerva.G["df"] = pd.DataFrame({"Xt": [0.0], "Yt": [0.0], "grp_immune": [True]})
    server_minerva.G["coord_cols"] = ("Xt", "Yt")
    server_minerva.G["group_ids"] = {"immune"}

    with pytest.raises(HTTPException, match="Unknown group"):
        asyncio.run(
            server_minerva.cells(group="unknown", x0=0, y0=0, x1=100, y1=100, max_cells=500)
        )
