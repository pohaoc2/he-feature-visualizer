"""Unit tests for Astir adapter stubs and remaining helpers."""

from __future__ import annotations

import logging
import types

import pandas as pd
import pytest
import torch

from utils import astir_adapter as mod


def test_predict_cell_type_probabilities_raises_runtime_error():
    """predict_cell_type_probabilities is now a stub that raises RuntimeError."""
    with pytest.raises(RuntimeError, match="ASTIR has been removed"):
        mod.predict_cell_type_probabilities(
            pd.DataFrame({"PanCK": [1.0], "CD45": [1.0]}),
            marker_dict={"cancer": ["PanCK"], "immune": ["CD45"]},
        )


def test_astir_unavailable_error_importable():
    """AstirUnavailableError is still importable for backward compat."""
    from utils.astir_adapter import AstirUnavailableError
    assert issubclass(AstirUnavailableError, RuntimeError)


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
