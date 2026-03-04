from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_marker_thresholds(df: pd.DataFrame, *, markers: Iterable[str], pct: float = 95) -> dict[str, float]:
    thr: dict[str, float] = {}
    for m in markers:
        if m in df.columns:
            s = pd.to_numeric(df[m], errors="coerce").dropna()
            if len(s) == 0:
                continue
            thr[m] = float(np.percentile(s, pct))
    return thr


def assign_cell_state(df: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    """
    Returns a Series of categorical states.

    Current states (minimal v1):
    - proliferating: Ki67 or PCNA high
    - emt: Vimentin high AND Ecadherin low
    - other: default
    """
    idx = df.index
    state = pd.Series("other", index=idx, dtype="object")

    # proliferating
    ki67_thr = thresholds.get("Ki67")
    pcna_thr = thresholds.get("PCNA")
    prolif = pd.Series(False, index=idx)
    if ki67_thr is not None and "Ki67" in df.columns:
        ki67 = pd.to_numeric(df["Ki67"], errors="coerce")
        if ki67.nunique(dropna=True) > 1:
            prolif |= ki67 >= ki67_thr
    if pcna_thr is not None and "PCNA" in df.columns:
        pcna = pd.to_numeric(df["PCNA"], errors="coerce")
        if pcna.nunique(dropna=True) > 1:
            prolif |= pcna >= pcna_thr
    state[prolif.fillna(False)] = "proliferating"

    # EMT / migration proxy
    vim_thr = thresholds.get("Vimentin")
    e_thr = thresholds.get("Ecadherin")
    emt = pd.Series(False, index=idx)
    if vim_thr is not None and e_thr is not None and ("Vimentin" in df.columns) and ("Ecadherin" in df.columns):
        vim = pd.to_numeric(df["Vimentin"], errors="coerce")
        ecad = pd.to_numeric(df["Ecadherin"], errors="coerce")
        emt = (vim >= vim_thr) & (ecad <= e_thr)

    # only assign EMT if not already proliferating
    state[emt.fillna(False) & ~prolif.fillna(False)] = "emt"

    return state


STATE_COLORS = {
    "proliferating": (0, 255, 0, 200),
    "emt": (255, 165, 0, 200),
    "other": (0, 0, 0, 0),
}


def state_to_rgba(states: pd.Series) -> np.ndarray:
    out = np.zeros((len(states), 4), dtype=np.uint8)
    for k, rgba in STATE_COLORS.items():
        out[states.values == k] = rgba
    return out

