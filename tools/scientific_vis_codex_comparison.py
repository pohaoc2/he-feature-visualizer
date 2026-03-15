#!/usr/bin/env python3
"""CellViT vs CODEX cell type comparison figure."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    _mpl_cache = Path("/tmp/matplotlib")
    _mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_cache)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils.cell_assignment_reports import load_cell_assignments
from utils.normalize import percentile_norm

CELL_TYPES: tuple[str, str, str] = ("cancer", "immune", "healthy")

TYPE_COLORS = {
    "cancer":  "#DC3232",
    "immune":  "#3264DC",
    "healthy": "#32B432",
}

MARKER_COLORS = {
    "cancer":  "#DC3232",
    "immune":  "#3264DC",
    "healthy": "#32B432",
}

CROP_HALF = 20   # 40×40 px crop


# ── Data helpers ──────────────────────────────────────────────────────────────

def _filter_assignable(df: pd.DataFrame) -> pd.DataFrame:
    """Exclude rows where cell_type is not one of the three canonical classes."""
    return df[df["cell_type"].isin(CELL_TYPES)].copy()


def _compute_codex_margin(df: pd.DataFrame) -> pd.Series:
    """Return per-row CODEX margin: p_model_winner - max(p_model_others).

    Caller must ensure df contains only rows where cell_type ∈ {cancer, immune, healthy}.
    """
    margins = []
    for _, row in df.iterrows():
        ct = str(row["cell_type"])
        winner = float(row.get(f"p_model_{ct}", 0.0))
        others = [
            float(row.get(f"p_model_{other}", 0.0))
            for other in CELL_TYPES
            if other != ct
        ]
        margins.append(winner - max(others))
    return pd.Series(margins, index=df.index, dtype=float)


def _select_examples(df: pd.DataFrame) -> dict[str, pd.Series | None]:
    """Select one cell per bucket (agree, medium, disagree) with de-duplication.

    df must already have a 'codex_margin' column.
    Returns dict with keys 'agree', 'medium', 'disagree' (value is None if bucket empty).
    """
    used: set[object] = set()
    results: dict[str, pd.Series | None] = {}

    # agree: is_mismatch=False, highest codex_margin
    non_mismatch = df[~df["is_mismatch"]].sort_values("codex_margin", ascending=False)
    results["agree"] = None
    for idx, row in non_mismatch.iterrows():
        if idx not in used:
            used.add(idx)
            results["agree"] = row
            break

    # medium: is_mismatch=False, lowest codex_margin (least confident agreement), not already used
    # Reverse the descending-sorted slice rather than re-sorting ascending.
    results["medium"] = None
    for idx, row in non_mismatch.iloc[::-1].iterrows():
        if idx not in used:
            used.add(idx)
            results["medium"] = row
            break

    # disagree: is_mismatch=True, highest codex_margin
    results["disagree"] = None
    for idx, row in df[df["is_mismatch"]].sort_values("codex_margin", ascending=False).iterrows():
        if idx not in used:
            used.add(idx)
            results["disagree"] = row
            break

    return results


# ── Rendering helpers ──────────────────────────────────────────────────────────

def _load_he_crop(
    processed_dir: Path,
    patch_id: str,
    cx: float,
    cy: float,
) -> np.ndarray | None:
    """Load H&E patch and return 40×40 px crop centered at (cx, cy), clamped to bounds."""
    he_path = processed_dir / "he" / f"{patch_id}.png"
    if not he_path.exists():
        return None
    img = np.array(Image.open(he_path).convert("RGB"))
    h, w = img.shape[:2]
    x0 = int(max(0, min(cx - CROP_HALF, w - 2 * CROP_HALF)))
    y0 = int(max(0, min(cy - CROP_HALF, h - 2 * CROP_HALF)))
    x1 = min(x0 + 2 * CROP_HALF, w)
    y1 = min(y0 + 2 * CROP_HALF, h)
    return img[y0:y1, x0:x1]


BUCKET_LABELS = {"agree": "Agree", "medium": "Medium", "disagree": "Disagree"}


def _plot_marker_bar(
    ax: plt.Axes,
    row: pd.Series,
    norm_vals: dict[str, np.ndarray],
    markers: dict[str, str],
    row_index: int,
) -> None:
    """Draw three horizontal marker bars plus CellViT/CODEX/Margin text block.

    markers: dict with keys 'cancer_marker', 'immune_marker', 'healthy_marker'.
    norm_vals: dict[marker_name -> float32 array over all cells], indexed by row_index.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    bar_y_positions = [0.78, 0.52, 0.26]
    bar_h = 0.18
    role_keys = ["cancer_marker", "immune_marker", "healthy_marker"]
    colors = [MARKER_COLORS["cancer"], MARKER_COLORS["immune"], MARKER_COLORS["healthy"]]

    for bar_y, role, color in zip(bar_y_positions, role_keys, colors):
        marker_name = markers.get(role, "")
        label = marker_name or role

        if marker_name in norm_vals:
            val = float(norm_vals[marker_name][row_index])
            ax.barh(bar_y, val, height=bar_h, color=color, align="center", left=0)
            ax.text(val + 0.02, bar_y, f"{val:.2f}", va="center", fontsize=6)
        else:
            ax.text(0.5, bar_y, "n/a", va="center", ha="center",
                    fontsize=7, color="#888888")

        ax.text(-0.02, bar_y, label, va="center", ha="right", fontsize=6)

    # vertical threshold line at 0.5
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=0.6, alpha=0.7)

    # text block
    cvit = str(row.get("cellvit_mapped_type", "?"))
    codex = str(row.get("type_astir", "?"))
    margin = float(row.get("codex_margin", float("nan")))
    text = f"CellViT: {cvit}\nCODEX:   {codex}\nMargin:  {margin:.2f}"
    ax.text(0.5, 0.04, text, va="bottom", ha="center", fontsize=6,
            family="monospace", transform=ax.transAxes)


def _plot_confusion_heatmap(ax: plt.Axes, df: pd.DataFrame) -> None:
    """3×3 confusion matrix: CellViT rows × CODEX (type_astir) cols.

    Cells show raw count (bold) + row-normalised % below.
    """
    classes = list(CELL_TYPES)
    counts = np.zeros((3, 3), dtype=int)
    for i, rv in enumerate(classes):
        for j, cv in enumerate(classes):
            counts[i, j] = int(
                ((df["cellvit_mapped_type"] == rv) & (df["type_astir"] == cv)).sum()
            )

    row_totals = counts.sum(axis=1, keepdims=True)
    row_norm = np.where(row_totals > 0, counts / row_totals, 0.0)

    im = ax.imshow(counts, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(3):
        for j in range(3):
            ax.text(j, i - 0.12, str(counts[i, j]),
                    ha="center", va="center", fontsize=14, fontweight="bold",
                    color="white" if counts[i, j] > counts.max() * 0.6 else "black")
            ax.text(j, i + 0.22, f"{row_norm[i, j]:.0%}",
                    ha="center", va="center", fontsize=9,
                    color="white" if counts[i, j] > counts.max() * 0.6 else "black")

    ax.set_xticks(range(3))
    ax.set_xticklabels(classes, fontsize=13)
    ax.set_yticks(range(3))
    ax.set_yticklabels(classes, fontsize=13)
    ax.set_xlabel("CODEX label", fontsize=13)
    ax.set_ylabel("CellViT label", fontsize=13)
    ax.set_title("CellViT vs CODEX Confusion", fontsize=15)


def _plot_summary_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    markers: dict[str, str],
) -> None:
    """Text summary panel: total cells, mismatch rate, per-class counts, markers."""
    ax.axis("off")
    total = len(df)
    n_mismatch = int(df["is_mismatch"].sum())
    rate = n_mismatch / total if total else 0.0

    lines = [
        "Model: CODEX",
        f"Total cells: {total:,}",
        f"Mismatch rate: {rate:.1%}",
        "",
    ]
    for ct in CELL_TYPES:
        sub = df[df["cell_type"] == ct]
        non_mm = sub[~sub["is_mismatch"]]
        n_mm = int(sub["is_mismatch"].sum())
        # Split non-mismatch into agree (high margin ≥0.5) vs medium (low margin <0.5)
        if "codex_margin" in non_mm.columns:
            n_ag = int((non_mm["codex_margin"] >= 0.5).sum())
            n_med = int((non_mm["codex_margin"] < 0.5).sum())
        else:
            n_ag = len(non_mm)
            n_med = 0
        lines.append(f"{ct}: {n_ag} agree / {n_med} medium / {n_mm} disagree")

    lines += [
        "",
        f"Cancer marker:  {markers.get('cancer_marker', 'Pan-CK')} -> cancer",
        f"Immune marker:  {markers.get('immune_marker', 'CD45')} -> immune",
        f"Healthy marker: {markers.get('healthy_marker', 'SMA')} -> healthy",
    ]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=10, family="monospace",
            linespacing=1.6)
