"""Nuclei from Lin's cell mask, and a NucleusShapes outline library built from them.

Lin's cell mask is the UNMICST2 nucleus mask dilated by 5 px (DILATION_PX). Eroding each cell by the same disk
recovers the nucleus exactly for an isolated, convex-enough cell (dilation then erosion = closing). Where two cells
touch, the dilation stopped at the shared border, so erosion also eats into the nucleus from that side:
  strict : a pixel stays only if every pixel within the disk carries the same label (neighbours count as outside);
           this is "erode the cell by 5 px" and under-recovers nuclei of touching cells;
  loose  : a pixel stays if no background lies within the disk (neighbours count as inside); this over-recovers
           touching cells (keeps up to the shared border).
The true nucleus of a touching cell lies between the two; `strict` is used, `loose` is reported as the bound.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

DILATION_PX = 5
BOX = 48                                     # NucleusShapes stamp size (morphology_renderer.agent_stamps requires 48x48)
TYPE_NAMES = ("cancer", "healthy", "immune")  # same order as sim-exp-validation artifacts/nucleus_shapes_orion_crc33.npz


def disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y <= r * r).astype(np.uint8)


def erode_labels(lab: np.ndarray, r: int = DILATION_PX):
    """-> (strict, loose) label images (0 outside the eroded nuclei). See module docstring."""
    lab = np.asarray(lab)
    if lab.size and int(lab.max()) >= 2 ** 24:
        raise ValueError("labels must stay below 2**24 to be exact in float32")
    k = disk(r)
    f = lab.astype(np.float32)
    lo = cv2.erode(f, k, borderType=cv2.BORDER_REPLICATE)
    hi = cv2.dilate(f, k, borderType=cv2.BORDER_REPLICATE)
    fg = lab > 0
    strict = np.where(fg & (lo == hi), lab, 0)
    keep = cv2.erode(fg.astype(np.uint8), k, borderType=cv2.BORDER_REPLICATE) > 0
    loose = np.where(fg & keep, lab, 0)
    return strict, loose


def touching_labels(lab: np.ndarray) -> np.ndarray:
    """Labels with at least one 4-neighbour pixel of another non-zero label."""
    lab = np.asarray(lab)
    f = lab.astype(np.float32)
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    big = np.float32(2 ** 24)
    hi = cv2.dilate(f, cross, borderType=cv2.BORDER_REPLICATE)
    lo = cv2.erode(np.where(lab > 0, f, big), cross, borderType=cv2.BORDER_REPLICATE)
    fg = lab > 0
    t = fg & ((hi > f) | (lo < f))
    return np.unique(lab[t])


def stamp(mask: np.ndarray, box: int = BOX):
    """Largest 4-connected part of a boolean object crop, centred by bounding box in a box x box canvas.

    -> (stamp bool (box, box), area) or (None, 0) if empty or the part's bounding box exceeds the canvas
    (rule of sim-exp-validation build_nucleus_shape_library.py).
    """
    mask = np.asarray(mask, bool)
    lab, n = ndimage.label(mask)
    if n == 0:
        return None, 0
    if n > 1:
        mask = lab == (1 + int(np.argmax(ndimage.sum(mask, lab, range(1, n + 1)))))
    ys, xs = np.nonzero(mask)
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    if h > box or w > box:
        return None, 0
    out = np.zeros((box, box), bool)
    y0, x0 = (box - h) // 2, (box - w) // 2
    out[y0:y0 + h, x0:x0 + w] = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return out, int(out.sum())


def save_library(path, masks, types, areas, source_tile, cell_id, type_names=TYPE_NAMES):
    """Write a NucleusShapes library (keys of artifacts/nucleus_shapes_orion_crc33.npz, plus cell_id)."""
    np.savez_compressed(path, masks=np.asarray(masks, bool), type=np.asarray(types, np.int64),
                        area=np.asarray(areas, np.int64), type_names=np.array(type_names),
                        source_tile=np.array(source_tile), cell_id=np.asarray(cell_id, np.int64))
