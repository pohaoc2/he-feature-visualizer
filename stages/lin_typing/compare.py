"""Number-free agreement measures between a binary nucleus channel and the H&E or another channel."""
from __future__ import annotations

import numpy as np


def dice(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    s = int(a.sum()) + int(b.sum())
    return float("nan") if s == 0 else 2.0 * int((a & b).sum()) / s


def auc(pos, neg) -> float:
    """P(value inside > value outside) + 0.5 P(tie) (Mann-Whitney U / (n_pos n_neg)), from ranks."""
    pos, neg = np.asarray(pos, float).ravel(), np.asarray(neg, float).ravel()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    sv = v[order]
    # average ranks over ties
    edges = np.flatnonzero(np.diff(sv)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(v)]])
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    u = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def hematoxylin(rgb):
    """Hematoxylin optical density (Ruifrok & Johnston 2001 colour deconvolution, skimage.color.rgb2hed)."""
    from skimage.color import rgb2hed
    return rgb2hed(np.asarray(rgb)[..., :3])[..., 0]


def nucleus_auc(h, mask) -> float:
    """AUC of hematoxylin OD inside vs outside a nucleus mask (0.5 = the mask ignores the stain)."""
    m = np.asarray(mask, bool)
    return auc(h[m], h[~m])
