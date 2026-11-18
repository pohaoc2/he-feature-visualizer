"""Render Lin cells into the 7 cell channels with sim-exp-validation's renderer (the sim2real code path).

Each Lin cell becomes one renderer agent, as morphology_renderer.agent_stamps builds them for ARCADE cells:
  outline  = NucleusShapes.pick(population, target nucleus area, key) from a library of Lin nuclei
             (agent_stamps' default, frozen per-snapshot selection; key = CellID, so a cell crossing a tile edge
             gets the same outline in both tiles);
  position = the Lin centroid (the measured position plays the role of paired_positions' off-lattice anchor);
then morphology_renderer.render_crop(shape="outline", placement="jitter", allow_contact=True) paints one 256 px tile
and enforces the type / state partitions (_enforce_partition). allow_contact=True because real nuclei touch; the
simulation side places separated footprints.

Import sim_exp_validation from a checkout of the sim-exp-validation repo (`src/` on sys.path, or `SEV_SRC`).
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .labels import POP_NAMES, STATE_NAMES

TILE = 256
REACH = 24   # half the 48 px stamp: render_crop skips agents farther than this outside the tile


def renderer(src: str | None = None):
    """-> (rasterize, morphology_renderer, channels) modules of sim-exp-validation."""
    src = src or os.environ.get("SEV_SRC")
    if src and src not in sys.path:
        sys.path.insert(0, src)
    from sim_exp_validation import channels, morphology_renderer, rasterize
    return rasterize, morphology_renderer, channels


def target_area(strict_area, loose_area):
    """Target nucleus area per cell: the strict erosion; the loose bound where strict erosion left nothing; >= 1."""
    a = np.asarray(strict_area, float).copy()
    zero = a <= 0
    a[zero] = np.asarray(loose_area, float)[zero]
    return np.maximum(a, 1.0)


class CellTable:
    """Per-cell arrays sorted by x for fast tile queries: id, x, y, ty (POP code), st (STATE code), area (px)."""

    def __init__(self, cid, x, y, ty, st, area):
        o = np.argsort(np.asarray(x, float), kind="stable")
        self.id = np.asarray(cid, np.int64)[o]
        self.x = np.asarray(x, float)[o]
        self.y = np.asarray(y, float)[o]
        self.ty = np.asarray(ty, np.int64)[o]
        self.st = np.asarray(st, np.int64)[o]
        self.area = np.asarray(area, float)[o]

    def near(self, x0, y0, size=TILE, reach=REACH):
        a, b = np.searchsorted(self.x, [x0 - reach, x0 + size + reach])
        i = np.arange(a, b)
        return i[(self.y[i] >= y0 - reach) & (self.y[i] < y0 + size + reach)]


def tile_agents(table: CellTable, x0: int, y0: int, library):
    """Renderer agents and positions for the tile whose top-left global pixel is (x0, y0)."""
    agents, positions = [], {}
    for i in table.near(x0, y0):
        cid = int(table.id[i])
        typ = POP_NAMES[table.ty[i]]
        outline = library.pick(typ, float(table.area[i]), str(cid))
        agents.append(dict(id=cid, u=0, v=0, type=typ, state=STATE_NAMES[table.st[i]], outline=outline.astype(bool)))
        positions[cid] = dict(x=int(round(table.x[i])), y=int(round(table.y[i])), attempts=0)
    return agents, positions


def render_tile(table: CellTable, x0: int, y0: int, library, mr):
    """-> ((7, 256, 256) float32 channels in channels.CHANNEL_NAMES[:7] order, renderer instance records)."""
    agents, positions = tile_agents(table, x0, y0, library)
    return mr.render_crop(agents, positions, y0, x0, shape="outline", placement="jitter", allow_contact=True)


def ink_tiles(ink_fraction: dict) -> set:
    """Tiles with any ink pixel (darkdiag ink.json: style_pool PIGMENT_LEVEL 60 rule, MAX_PIGMENT_FRACTION 0)."""
    return {k for k, v in ink_fraction.items() if float(v) > 0}
