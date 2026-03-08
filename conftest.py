import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def _stub_zarr_array_for_unit_tests(monkeypatch):
    """Keep unit tests deterministic under zarr 3.x in-memory array behavior.

    Some CI environments can stall on ``zarr.array(np_array)`` for tiny synthetic
    data used in tests. We only need array-like semantics (shape/ndim/getitem),
    so map ``zarr.array`` to ``np.asarray`` during tests.
    """

    monkeypatch.setattr(zarr, "array", lambda data, *args, **kwargs: np.asarray(data))
