"""Unit tests for Astir adapter preprocessing and probability handling."""

from __future__ import annotations

import logging
import sys
import types

import pandas as pd
import torch

from utils import astir_adapter as mod


class _FakeTypeDataset:
    def __init__(self) -> None:
        self.normalize_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def normalize(self, *args, **kwargs) -> None:
        self.normalize_calls.append((args, kwargs))


def _install_fake_astir(monkeypatch, probs: pd.DataFrame):
    class FakeAstir:
        instances: list["FakeAstir"] = []

        def __init__(self, input_expr, marker_dict):
            self.input_expr = input_expr.copy()
            self.marker_dict = marker_dict
            self.fit_kwargs = None
            if "cell_type" in marker_dict:
                self._type_dset = _FakeTypeDataset()
            else:
                # Simulate versions that accept the constructor but fail to
                # build the type dataset unless the marker dict is wrapped.
                self._type_dset = None
            type(self).instances.append(self)

        def normalize(self, *_args, **_kwargs) -> None:
            raise RuntimeError("state dataset missing")

        def fit_type(self, **kwargs) -> None:
            self.fit_kwargs = kwargs

        def get_celltype_probabilities(self) -> pd.DataFrame:
            out = probs.copy()
            out.index = self.input_expr.index
            return out

    monkeypatch.setitem(sys.modules, "astir", types.SimpleNamespace(Astir=FakeAstir))
    return FakeAstir


def test_predict_cell_type_probabilities_normalizes_type_dataset(monkeypatch):
    probs = pd.DataFrame(
        {
            "cancer": [0.10, 0.20],
            "immune": [0.20, 0.10],
            "healthy": [0.10, 0.20],
            "Other": [0.60, 0.50],
        }
    )
    fake_astir = _install_fake_astir(monkeypatch, probs)

    expr_df = pd.DataFrame(
        {
            "PanCK": [10.0, -5.0],
            "CD45": [8.0, 9.0],
            "SMA": [2.0, 3.0],
        },
        index=["cell-1", "cell-2"],
    )

    out = mod.predict_cell_type_probabilities(
        expr_df,
        marker_dict={
            "cancer": ["PanCK"],
            "immune": ["CD45"],
            "healthy": ["SMA"],
        },
        logger=logging.getLogger("test-astir-adapter"),
    )

    working_model = next(inst for inst in fake_astir.instances if inst._type_dset is not None)
    assert working_model.input_expr.loc["cell-2", "PanCK"] == 0.0
    assert working_model.fit_kwargs is not None
    assert working_model._type_dset.normalize_calls == [
        ((), {"percentile_lower": 1, "percentile_upper": 99})
    ]
    # Extra "Other" mass should be preserved, not renormalized into the three targets.
    assert out.loc["cell-1", "cancer"] == 0.10
    assert out.loc["cell-1", "immune"] == 0.20
    assert out.loc["cell-1", "healthy"] == 0.10


def test_predict_cell_type_probabilities_normalizes_three_class_output(monkeypatch):
    probs = pd.DataFrame(
        {
            "cancer": [1.0],
            "immune": [1.0],
            "healthy": [0.0],
        }
    )
    _install_fake_astir(monkeypatch, probs)

    out = mod.predict_cell_type_probabilities(
        pd.DataFrame({"PanCK": [1.0], "CD45": [1.0], "SMA": [1.0]}, index=["cell-1"]),
        marker_dict={
            "cancer": ["PanCK"],
            "immune": ["CD45"],
            "healthy": ["SMA"],
        },
        logger=logging.getLogger("test-astir-adapter"),
    )

    assert out.loc["cell-1", "cancer"] == 0.5
    assert out.loc["cell-1", "immune"] == 0.5
    assert out.loc["cell-1", "healthy"] == 0.0


def test_repair_type_dataset_after_normalize_refreshes_stats():
    class FakeDataset:
        def __init__(self) -> None:
            self._device = torch.device("cpu")
            self._dtype = torch.float64
            self._exprs = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            self._marker_mat = torch.ones((2, 2), dtype=torch.float32)
            self._design = torch.ones((2, 1), dtype=torch.float32)
            self._exprs_mean = torch.tensor([99.0, 99.0], dtype=torch.float32)
            self._exprs_std = torch.tensor([88.0, 88.0], dtype=torch.float32)

    model = types.SimpleNamespace(_type_dset=FakeDataset())
    mod._repair_type_dataset_after_normalize(
        model,
        logging.getLogger("test-astir-adapter"),
    )

    dset = model._type_dset
    assert dset._exprs.dtype == torch.float64
    assert dset._marker_mat.dtype == torch.float64
    assert dset._design.dtype == torch.float64
    assert torch.allclose(dset._exprs_mean, torch.tensor([2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(
        dset._exprs_std,
        torch.tensor([1.4142135623730951, 1.4142135623730951], dtype=torch.float64),
    )
