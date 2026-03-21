#!/usr/bin/env python3
"""Randomly sample patches from a processed directory and plot an 8-column grid.

No index.json required — patches are discovered from the ``he/`` directory.

Columns: H&E | Hoechst | Cell mask | Cell type | Cell state | Vasculature | O₂ | Glucose

Below each patch grid:
  - Cell type column  → colour legend (cancer / immune / healthy)
  - Cell state column → colour legend (proliferative / nonprolif / dead)
  - Oxygen column     → horizontal colorbar spanning the column width
  - Glucose column    → horizontal colorbar spanning the column width
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path("/tmp/matplotlib")
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from PIL import Image

from utils.normalize import percentile_norm

# ── Colour maps ───────────────────────────────────────────────────────────────

HOECHST_CMAP = LinearSegmentedColormap.from_list(
    "hoechst33342",
    [
        (0.00, (0.00, 0.00, 0.00)),
        (0.35, (0.04, 0.10, 0.55)),
        (0.65, (0.10, 0.35, 0.90)),
        (0.85, (0.35, 0.65, 1.00)),
        (1.00, (0.80, 0.93, 1.00)),
    ],
)

# Keys must be substrings of the subdirectory names under cell_types/ and cell_states/
CELL_TYPE_COLORS: dict[str, tuple[int, int, int]] = {
    "cancer":   (220,  50,  50),
    "immune":   ( 50, 100, 220),
    "healthy":  ( 50, 180,  50),
}

CELL_STATE_COLORS: dict[str, tuple[int, int, int]] = {
    "prolif":    (230,  50, 180),
    "nonprolif": (240, 140,  30),
    "dead":      (110,  40, 160),
}

# Legend display names (same order as dicts above)
CELL_TYPE_LABELS  = ["cancer", "immune", "healthy"]
CELL_STATE_LABELS = ["prolif.", "nonprolif", "dead"]

COL_TITLES = [
    "H&E",
    "Hoechst",
    "Cell mask",
    "Cell type",
    "Cell state",
    "Vasculature",
    "Oxygen (O₂)",
    "Glucose",
]

_COL_TYPE  = 3
_COL_STATE = 4
_COL_O2    = 6
_COL_GLC   = 7

# ── Image helpers ─────────────────────────────────────────────────────────────


def _load_rgb(path: Path) -> np.ndarray | None:
    return np.array(Image.open(path).convert("RGB")) if path.exists() else None


def _load_rgba(path: Path) -> np.ndarray | None:
    return np.array(Image.open(path).convert("RGBA")) if path.exists() else None


def _load_gray(path: Path) -> np.ndarray | None:
    return np.array(Image.open(path).convert("L")) if path.exists() else None


def _placeholder(shape: tuple[int, int], text: str, ax: plt.Axes) -> None:
    ax.imshow(np.full((*shape, 3), 220, dtype=np.uint8))
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=7, color="#555555", transform=ax.transAxes)


def _class_overlay(
    he_rgb: np.ndarray,
    class_dir: Path,
    pid: str,
    color_map: dict[str, tuple[int, int, int]],
    alpha: float = 0.75,
) -> np.ndarray | None:
    """Composite per-class binary masks for one patch onto the H&E background.

    Subdirectories under *class_dir* are matched by substring: key ``"cancer"``
    matches directory ``cell_type_cancer``, etc.  Each matched mask (0/255 PNG)
    is painted with the corresponding solid colour at *alpha* opacity.

    Returns the composited RGB uint8 image, or ``None`` if no masks exist.
    """
    if not class_dir.is_dir():
        return None

    h, w = he_rgb.shape[:2]
    base    = he_rgb.astype(np.float32)
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    painted = np.zeros((h, w), dtype=bool)

    for key, color in color_map.items():
        for subdir in sorted(class_dir.iterdir()):
            if not subdir.is_dir() or key not in subdir.name:
                continue
            mask_path = subdir / f"{pid}.png"
            if not mask_path.exists():
                break
            mask = np.array(Image.open(mask_path).convert("L")) > 128
            painted |= mask
            overlay[mask] = color
            break

    if not painted.any():
        return None

    a = painted[:, :, np.newaxis].astype(np.float32) * alpha
    return (a * overlay + (1 - a) * base).clip(0, 255).astype(np.uint8)


# ── Patch discovery ───────────────────────────────────────────────────────────


def _discover_patches(processed: Path) -> list[str]:
    he_dir = processed / "he"
    if not he_dir.is_dir():
        raise FileNotFoundError(f"No 'he/' subdirectory found in {processed}")
    return sorted(p.stem for p in he_dir.glob("*.png"))


# ── Legend / colorbar renderers ───────────────────────────────────────────────


def _draw_legend(
    ax: plt.Axes,
    colors: dict[str, tuple[int, int, int]],
    labels: list[str],
) -> None:
    handles = [
        Patch(facecolor=np.array(colors[k]) / 255.0, edgecolor="none", label=lbl)
        for k, lbl in zip(colors, labels)
    ]
    ax.axis("off")
    ax.legend(
        handles=handles,
        loc="center",
        ncol=len(handles),
        fontsize=7,
        frameon=False,
        handlelength=1.2,
        handleheight=1.0,
        columnspacing=0.8,
    )


def _draw_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap: str,
    label: str,
    lo_label: str,
    hi_label: str,
) -> None:
    sm = ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_ticks([0, 1])
    cb.set_ticklabels([lo_label, hi_label], fontsize=6)
    cb.ax.tick_params(length=2, pad=2)
    cb.set_label(label, fontsize=7, labelpad=3)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random patch grid (no index.json needed): "
                    "H&E | Hoechst | Cell mask | Cell type | Cell state | "
                    "Vasculature | O₂ | Glucose"
    )
    parser.add_argument("--processed", required=True, help="Processed directory.")
    parser.add_argument(
        "--random", dest="n_patches", type=int, required=True,
        help="Number of patches to sample randomly.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--out-prefix", default=None,
        help="Output path prefix (default: <processed>/patch_grid).",
    )
    parser.add_argument(
        "--formats", default="png",
        help="Comma-separated output formats (default: png).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Raster DPI (default: 150).")
    args = parser.parse_args()

    processed = Path(args.processed)
    out_prefix = Path(args.out_prefix) if args.out_prefix else processed / "patch_grid"
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    all_patches = _discover_patches(processed)
    if not all_patches:
        raise RuntimeError(f"No PNG files found in {processed / 'he'}")

    n = min(args.n_patches, len(all_patches))
    rng = random.Random(args.seed)
    selected = rng.sample(all_patches, n)
    print(f"Plotting {n} patches: {selected}")

    has_multiplex = (processed / "multiplex").is_dir()

    # ── Layout ────────────────────────────────────────────────────────────────
    col_w, row_h = 2.5, 2.5
    annot_h = 0.45
    n_cols = len(COL_TITLES)

    fig = plt.figure(figsize=(n_cols * col_w, n * row_h + annot_h),
                     constrained_layout=True)
    gs = gridspec.GridSpec(
        n + 1, n_cols,
        figure=fig,
        height_ratios=[*([1] * n), annot_h / row_h],
    )

    axes = np.array(
        [[fig.add_subplot(gs[r, c]) for c in range(n_cols)] for r in range(n)]
    )

    _annot_used = {_COL_TYPE, _COL_STATE, _COL_O2, _COL_GLC}
    leg_type_ax  = fig.add_subplot(gs[n, _COL_TYPE])
    leg_state_ax = fig.add_subplot(gs[n, _COL_STATE])
    cax_o2       = fig.add_subplot(gs[n, _COL_O2])
    cax_glc      = fig.add_subplot(gs[n, _COL_GLC])
    for c in range(n_cols):
        if c not in _annot_used:
            fig.add_subplot(gs[n, c]).set_visible(False)

    for col, title in enumerate(COL_TITLES):
        axes[0, col].set_title(title, fontsize=9, pad=3)

    # ── Per-patch rendering ───────────────────────────────────────────────────
    for row_idx, pid in enumerate(selected):
        he = _load_rgb(processed / "he" / f"{pid}.png")
        if he is None:
            raise FileNotFoundError(f"Missing H&E patch: {pid}.png")
        h, w = he.shape[:2]
        ax = axes[row_idx]

        # C1: H&E
        ax[0].imshow(he)
        ax[0].set_ylabel(pid, fontsize=6, labelpad=3)

        # C2: Hoechst
        if has_multiplex:
            mx_path = processed / "multiplex" / f"{pid}.npy"
            if mx_path.exists():
                mx = np.load(mx_path)
                ax[1].imshow(percentile_norm(mx[0].astype(np.float32)),
                             cmap=HOECHST_CMAP, vmin=0.0, vmax=1.0)
            else:
                _placeholder((h, w), "Hoechst\nnot found", ax[1])
        else:
            _placeholder((h, w), "Hoechst\nnot available", ax[1])

        # C3: Cell mask
        mask = _load_gray(processed / "cell_masks" / f"{pid}.png")
        if mask is not None:
            ax[2].imshow(mask, cmap="gray")
        else:
            _placeholder((h, w), "Cell mask\nnot found", ax[2])

        # C4: Cell type — composited from per-class binary masks
        ct = _class_overlay(he, processed / "cell_types", pid, CELL_TYPE_COLORS)
        if ct is not None:
            ax[3].imshow(ct)
        else:
            _placeholder((h, w), "Cell type\nnot found", ax[3])

        # C5: Cell state — composited from per-class binary masks
        cs = _class_overlay(he, processed / "cell_states", pid, CELL_STATE_COLORS)
        if cs is not None:
            ax[4].imshow(cs)
        else:
            _placeholder((h, w), "Cell state\nnot found", ax[4])

        # C6: Vasculature (RGBA composited onto H&E)
        vasc = _load_rgba(processed / "vasculature" / f"{pid}.png")
        if vasc is not None:
            ov    = vasc[:, :, :3].astype(np.float32)
            alpha = vasc[:, :, 3:4].astype(np.float32) / 255.0
            blended = (alpha * ov + (1 - alpha) * he.astype(np.float32)).clip(0, 255).astype(np.uint8)
            ax[5].imshow(blended)
        else:
            _placeholder((h, w), "Vasculature\nnot found", ax[5])

        # C7: Oxygen
        oxy = _load_rgba(processed / "oxygen" / f"{pid}.png")
        if oxy is not None:
            ax[6].imshow(oxy)
        else:
            _placeholder((h, w), "Oxygen\nnot found", ax[6])

        # C8: Glucose
        glc = _load_rgba(processed / "glucose" / f"{pid}.png")
        if glc is not None:
            ax[7].imshow(glc)
        else:
            _placeholder((h, w), "Glucose\nnot found", ax[7])

        for a in ax:
            a.set_xticks([])
            a.set_yticks([])

    # ── Annotation row ────────────────────────────────────────────────────────
    _draw_legend(leg_type_ax,  CELL_TYPE_COLORS,  CELL_TYPE_LABELS)
    _draw_legend(leg_state_ax, CELL_STATE_COLORS, CELL_STATE_LABELS)
    _draw_colorbar(fig, cax_o2,  "RdYlBu", "O₂ proxy",     "hypoxic",  "oxygenated")
    _draw_colorbar(fig, cax_glc, "hot",    "Glucose proxy", "depleted", "high")

    # ── Save ──────────────────────────────────────────────────────────────────
    formats = [f.strip() for f in args.formats.split(",") if f.strip()] or ["png"]
    for fmt in formats:
        out_path = out_prefix.with_suffix(f".{fmt}")
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
