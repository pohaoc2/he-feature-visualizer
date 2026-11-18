"""Per-cell population and state for Lin 2023 cells (pure functions; file reading lives in the experiment scripts).

Population: `celltype_celesta_fig3d.classify_celesta_fig3d` (CELESTA final call + Lin 112 um H&E tissue class at the
cell centroid). State, in the order he-feature-visualizer's assign_cells applies it: CellViT dead (type 4) first, then
Ki-67 mean intensity above the Lin gate -> prolif, else nonprolif.
"""
from __future__ import annotations

import numpy as np

from .celltype_celesta_fig3d import LIN_L, classify_celesta_fig3d, lookup  # noqa: F401  (re-exported)

POP_NAMES = ("healthy", "cancer", "immune")      # index = population code of celltype_celesta_fig3d
STATE_NAMES = ("prolif", "nonprolif", "dead")    # index = state code here
PROLIF, NONPROLIF, DEAD = 0, 1, 2
LIN_GRID_UM = 112.0                              # Lin 2023 H&E patch classifier grid (class-images)
MPP = 0.325                                      # CRC33_01 H&E / mask um per px (OME PhysicalSizeX)
CELLVIT_DEAD = 4                                 # CellViT PanNuke class "dead" (hfv type_cellvit)


def lin_class_at(x, y, cmap, grid_px: float = LIN_GRID_UM / MPP):
    """Lin tissue-class code at pixel (x, y): the class-map cell containing it (clipped to the map)."""
    r = np.clip((np.asarray(y, float) / grid_px).astype(np.int64), 0, cmap.shape[0] - 1)
    c = np.clip((np.asarray(x, float) / grid_px).astype(np.int64), 0, cmap.shape[1] - 1)
    return np.asarray(cmap)[r, c]


def dead_from_cellvit(lin_ids, hfv_label, hfv_type_cellvit, dead_code: int = CELLVIT_DEAD):
    """Lin cells that contain the centroid of a CellViT dead object.

    hfv_label: Lin mask label under each CellViT centroid (0 = background). Returns (dead bool per Lin cell, counts).
    """
    hfv_label = np.asarray(hfv_label, np.int64)
    is_dead = np.asarray(hfv_type_cellvit) == dead_code
    hit = np.unique(hfv_label[is_dead & (hfv_label > 0)])
    dead = np.isin(np.asarray(lin_ids, np.int64), hit)
    return dead, dict(cellvit_dead=int(is_dead.sum()), cellvit_dead_on_background=int((is_dead & (hfv_label == 0)).sum()),
                      lin_cells_dead=int(dead.sum()))


def cell_state(ki67, gate: float, dead):
    """dead first, then Ki-67 > gate -> prolif, else nonprolif (int8 codes of STATE_NAMES)."""
    st = np.where(np.asarray(ki67, float) > float(gate), PROLIF, NONPROLIF).astype(np.int8)
    st[np.asarray(dead, bool)] = DEAD
    return st


def assign(final, lin_cls, ki67, gate: float, dead):
    """-> (population int8, typing source int8 (celltype_celesta_fig3d.SRC_*), state int8)."""
    ty, src = classify_celesta_fig3d(final, lin_cls)
    return ty, src, cell_state(ki67, gate, dead)
