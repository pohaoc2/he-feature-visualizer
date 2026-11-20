"""hfv_retrain_20261118: Fig 2 reference variant (coordinator 2026-11-20, follow-up 2).

COPY of PixCell cb62fa7 src/paper_figures/fig_combined_method_perf.py (the original stays unchanged and its output,
figures_rerender/.../fig2_architecture_performance.png, is kept). Changes against the original, nothing else:
  1. _load_summaries takes the ablation tree (ABL_ROOT/{paired,unpaired}_ablation) and the orion root as arguments
     instead of the hard-coded ROOT/inference_output/concat_ablation_1000 and ROOT/data/orion-crc33, so the same
     drawing code can be fed the primary-reference per-tile scores (eval/realref.py tree: PQ / DICE against CellViT-256
     on the real H&E) or the secondary ones (PixCell's own scoring: PQ / DICE against the Lin input cell mask).
  2. build_combined_method_perf_figure takes reference_label and writes it (English) right-aligned in the label strip
     above panel D, so each PNG says which PQ / DICE reference panels D, E and G use.
  3. __main__ is an argparse CLI (paths, --bands, --reference primary|secondary, --out).
--bands BANDS_JSON updates tools.ablation_report.shared.TRADEOFF_REFERENCE_BANDS in place exactly as eval/figrender.py
run does (the Vanilla fud / pq / dice bands of panel D), so the primary version uses the Vanilla PQ / DICE against the
same primary reference (realref fig2 cache.json, metrics["a2_off_shelf"]).

  PYTHONPATH=<PixCell cb62fa7 checkout> python fig2_refvariant.py --abl-root ABL --orion-root ORION --svg SVG \
      --cache CACHE_JSON --mapping UNPAIRED_MAPPING_JSON --images-root IMG_ABL --bands BANDS_JSON \
      --reference primary|secondary --out OUT_PNG
  (IMG_ABL = the tree whose PNGs the F / H qualitative strips show; the same generated images in both versions.)

Original docstring follows.

Combined Fig 2: training architecture + conditioning ablation + performance.

Five panels, four rows, full-width composite consistent with the other
`concat/` figures (220 DPI raster stack, ~15.8 in target width):

    Row 1:  A = training/architecture schematic (stage_2_svg.svg)
            B = training loss curve (top of the right column)
            C = conditioning-encoder ablation (ΔLPIPS bars, below B)
    Row 2:  D = metric trade-offs across channel conditions (performance)
    Row 3:  E = paired channel-group effect-size heatmap
            F = paired qualitative strip
    Row 4:  G = unpaired channel-group effect-size heatmap
            H = unpaired qualitative strip

Text-size consistency (vis_guidance: never rescale a rendered panel) is held by
rendering every matplotlib panel directly at its final pixel size and only ever
*resizing image rasters* (the SVG schematic and the tile strip), never the
text-bearing plots.
"""
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tools.ablation_report.data import DatasetSummary
from tools.ablation_report.figures import (
    build_channel_effect_heatmaps_figure,
    build_metric_trends_figure,
)
from matplotlib.font_manager import FontProperties

from src.paper_figures.fig_si_a1_a2_unified import (
    DISPLAY_VARIANT_ORDER,
    SECTION1_FONT_FAMILY,
    _draw_section4_sensitivity,
    _load_cache,
    _plot_loss_curves,
    _section1_legend_handles,
)
from src.paper_figures.fig_channel_ablation_strip import build_channel_ablation_strip_split
from src.paper_figures.style import FONT_SIZE_TICK

_RENDER_DPI = 220
_LABEL_STRIP_PX = 56
COMPOSITE_WIDTH_IN = 15.8
COMPOSITE_HEATMAP_FONT_SCALE = 1.28

_SCHEMATIC_ASPECT = 2084 / 1762          # native stage_2_svg.svg width/height
_ROW1_GAP_PX = 50                        # white gutter between panels A and B
# Row-1 height is derived from this aspect: h1 = (W - gap) / (aspect_B + aspect_A),
# and panels A and B/C are both drawn at h1, so A stays exactly as tall as the
# B+C stack whatever value this takes. Raised from 1.15 to shorten row 1 by ~11%
# (see _DE_HEATMAP_H_IN): the composite was 39 pt taller than \textheight once
# its caption was set, and shrinking the two tallest bands is the only fix that
# does not shrink fig 2's type relative to every other figure.
_PANELB_TARGET_ASPECT = 1.53             # B = loss + ΔLPIPS + legend, wider than tall
_PANELB_VARIANTS = list(DISPLAY_VARIANT_ORDER)
_PANEL_LETTER_FS = 18                    # matches FONT_PANEL_LETTER across paper figs
# Panel B internal gridspec (loss curve / ΔLPIPS bars / legend). Kept as
# constants so the "C" letter can be placed at the top of the ΔLPIPS sub-plot.
_PANELB_GS_RATIOS = (1.0, 0.92, 0.10)
_PANELB_GS_HSPACE = 0.55
_PANELB_GS_TOP = 0.955
_PANELB_GS_BOTTOM = 0.075
_DE_ROW_GAP_IN = 0.05
_DE_HEATMAP_W_IN = 6.55
# Height of each E/G effect-size heatmap. Width-proportional would be 2.93 in;
# 2.50 in shortens the two qualitative rows with it, because the F/H tile strips
# are rendered to the heatmap's height. Heatmap cells are aspect="auto", so this
# only makes the cells shorter -- no label changes size.
_DE_HEATMAP_H_IN = 2.50
_DE_QUAL_W_IN = COMPOSITE_WIDTH_IN - _DE_HEATMAP_W_IN - _DE_ROW_GAP_IN


def _fig_to_rgb(fig: plt.Figure, *, tight: bool = True) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_RENDER_DPI, bbox_inches="tight" if tight else None)
    buf.seek(0)
    arr = np.array(Image.open(buf).convert("RGB"))
    plt.close(fig)
    return arr


def _load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _svg_to_rgb_on_white(svg_path: Path, height_px: int) -> np.ndarray:
    import cairosvg

    png = cairosvg.svg2png(url=str(svg_path), output_height=height_px)
    im = Image.open(io.BytesIO(png)).convert("RGBA")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    bg.alpha_composite(im)
    return np.array(bg.convert("RGB"))


def _resize_to_width(img: np.ndarray, width_px: int) -> np.ndarray:
    if img.shape[1] == width_px:
        return img
    h = round(img.shape[0] * width_px / img.shape[1])
    return np.array(Image.fromarray(img).resize((width_px, h), Image.LANCZOS))


def _pad_to_width(img: np.ndarray, width_px: int) -> np.ndarray:
    """Centre a panel on a white canvas of the target width (no rescaling)."""
    if img.shape[1] >= width_px:
        return img
    pad = width_px - img.shape[1]
    left = pad // 2
    right = pad - left
    return np.pad(img, ((0, 0), (left, right), (0, 0)), constant_values=255)


def _add_label_strip(img: np.ndarray, px: int = _LABEL_STRIP_PX) -> np.ndarray:
    strip = np.full((px, img.shape[1], 3), 255, dtype=np.uint8)
    return np.concatenate([strip, img], axis=0)


def _detect_gutter_frac(img_d: np.ndarray) -> float:
    """Fraction-of-width of the white gutter between D's two heatmaps."""
    h = img_d.shape[0]
    band = img_d[int(h * 0.25):int(h * 0.75)]
    dens = (band < 245).any(axis=2).mean(axis=0)   # colored-pixel fraction per column
    w = len(dens)
    filled = dens > 0.5
    runs: list[tuple[int, int]] = []
    start = None
    for i, v in enumerate(filled):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(filled)))
    runs = [r for r in runs if r[1] - r[0] > w * 0.05]
    two = sorted(sorted(runs, key=lambda r: r[1] - r[0], reverse=True)[:2])
    return (two[0][1] + two[1][0]) / 2 / w


def _detect_strip_divider_frac(img_e: np.ndarray, lo: float = 0.46, hi: float = 0.54) -> float:
    """Fraction-of-width of the dashed paired|unpaired divider, searched only in the
    central band so the black cell-mask columns can't be mistaken for it."""
    h = img_e.shape[0]
    band = img_e[int(h * 0.15):int(h * 0.95)]
    dashes = (band < 160).all(axis=2).mean(axis=0)
    w = img_e.shape[1]
    a, b = int(w * lo), int(w * hi)
    return (a + int(np.argmax(dashes[a:b]))) / w


def _place_strip_aligned(img_e: np.ndarray, full_width: int, gutter_frac: float) -> np.ndarray:
    """Uniformly scale + right/left-pad the tile strip so its dashed divider lands on
    `gutter_frac` (D's gutter). No cropping, no aspect distortion."""
    f_e = _detect_strip_divider_frac(img_e)
    s_max = min(gutter_frac * full_width / f_e, full_width * (1 - gutter_frac) / (1 - f_e))
    s = int(min(full_width, s_max))
    scaled = _resize_to_width(img_e, s)
    off = max(0, int(round(gutter_frac * full_width - f_e * s)))
    off = min(off, full_width - s)
    canvas = np.full((scaled.shape[0], full_width, 3), 255, dtype=np.uint8)
    canvas[:, off:off + s] = scaled
    return canvas


def _build_panelB_raster(cache: dict, *, width_px: int, height_px: int) -> np.ndarray:
    """Panel B, rendered at its final pixel size: training loss (top), ΔLPIPS
    encoder-ablation bars (middle), shared variant color legend (bottom)."""
    fig = plt.figure(figsize=(width_px / _RENDER_DPI, height_px / _RENDER_DPI))
    gs = fig.add_gridspec(
        3, 1, height_ratios=list(_PANELB_GS_RATIOS), hspace=_PANELB_GS_HSPACE,
        left=0.15, right=0.965, top=_PANELB_GS_TOP, bottom=_PANELB_GS_BOTTOM,
    )
    curves = cache.get("training_curves", {})
    _plot_loss_curves(fig.add_subplot(gs[0]), curves, _PANELB_VARIANTS, "Training loss")
    _draw_section4_sensitivity(fig.add_subplot(gs[1]), cache)
    legend_ax = fig.add_subplot(gs[2])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=_section1_legend_handles(_PANELB_VARIANTS),
        loc="center",
        ncol=len(_PANELB_VARIANTS),
        frameon=False,
        prop=FontProperties(family=SECTION1_FONT_FAMILY, size=FONT_SIZE_TICK),
        handlelength=2.0,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    arr = _fig_to_rgb(fig, tight=False)
    return _resize_to_width(arr, width_px)  # guard against rounding drift only


def _panelB_subrow_top_frac(row_idx: int) -> float:
    """Fraction from the TOP of the B raster down to the top of B-gridspec row
    `row_idx` (0=loss, 1=ΔLPIPS bars, 2=legend)."""
    ratios = _PANELB_GS_RATIOS
    n = len(ratios)
    span = _PANELB_GS_TOP - _PANELB_GS_BOTTOM
    unit = span / (sum(ratios) + _PANELB_GS_HSPACE * (sum(ratios) / n) * (n - 1))
    gap = _PANELB_GS_HSPACE * (sum(ratios) / n) * unit
    top_from_bottom = _PANELB_GS_TOP
    for i in range(row_idx):
        top_from_bottom -= ratios[i] * unit + gap
    return 1.0 - top_from_bottom


def _build_row1(svg_path: Path, cache: dict, full_width: int) -> tuple[np.ndarray, int, int]:
    """Panels A (schematic) + B (loss + ΔLPIPS + legend), sharing height, summing to full_width."""
    h1 = round((full_width - _ROW1_GAP_PX) / (_PANELB_TARGET_ASPECT + _SCHEMATIC_ASPECT))
    w_a = round(_SCHEMATIC_ASPECT * h1)
    w_b = full_width - w_a - _ROW1_GAP_PX

    img_a = _svg_to_rgb_on_white(svg_path, height_px=h1)
    img_a = np.array(Image.fromarray(img_a).resize((w_a, h1), Image.LANCZOS))
    img_b = _build_panelB_raster(cache, width_px=w_b, height_px=h1)
    img_b = img_b[:h1] if img_b.shape[0] >= h1 else _pad_row_height(img_b, h1)

    gutter = np.full((h1, _ROW1_GAP_PX, 3), 255, dtype=np.uint8)
    return np.concatenate([img_a, gutter, img_b], axis=1), w_a, h1


def _pad_row_height(img: np.ndarray, h: int) -> np.ndarray:
    pad = h - img.shape[0]
    return np.pad(img, ((0, pad), (0, 0), (0, 0)), constant_values=255)


def _pad_to_height(img: np.ndarray, height_px: int, *, valign: str = "center") -> np.ndarray:
    if img.shape[0] >= height_px:
        return img
    pad = height_px - img.shape[0]
    if valign == "top":
        top = 0
    elif valign == "bottom":
        top = pad
    else:
        top = pad // 2
    bottom = pad - top
    return np.pad(img, ((top, bottom), (0, 0), (0, 0)), constant_values=255)


def _build_dataset_de_row(
    summary: DatasetSummary,
    *,
    paired_root: Path,
    unpaired_root: Path,
    paired_orion: Path,
    unpaired_orion: Path,
    layout_root: Path,
    unpaired_mapping_json: Path,
) -> tuple[np.ndarray, int]:
    """One Fig. 2 D/E row: heatmap at left, matching split qualitative at right.

    Returns the composited row and the pixel x of the heatmap|qualitative
    boundary (left edge of the qualitative strip), used to place the (i)/(ii)
    sub-panel labels.
    """
    img_heat = _fig_to_rgb(
        build_channel_effect_heatmaps_figure(
            [summary],
            width_inches=_DE_HEATMAP_W_IN,
            font_scale=COMPOSITE_HEATMAP_FONT_SCALE,
            height_inches=_DE_HEATMAP_H_IN,
        )
    )
    img_qual = _fig_to_rgb(
        build_channel_ablation_strip_split(
            split=summary.slug,
            paired_root=paired_root,
            unpaired_root=unpaired_root,
            paired_orion=paired_orion,
            unpaired_orion=unpaired_orion,
            layout_root=layout_root,
            unpaired_mapping_json=unpaired_mapping_json,
            target_height_in=img_heat.shape[0] / _RENDER_DPI,
            n_tiles=3,
        ),
        tight=False,
    )

    target_h = img_heat.shape[0]
    if img_qual.shape[0] != target_h:
        img_qual = _pad_to_height(img_qual, target_h, valign="center")
    gap = np.full((target_h, round(_DE_ROW_GAP_IN * _RENDER_DPI), 3), 255, dtype=np.uint8)
    divider_px = img_heat.shape[1] + gap.shape[1]
    return np.concatenate([img_heat, gap, img_qual], axis=1), divider_px


def build_combined_method_perf_figure(
    summaries: list[DatasetSummary],
    *,
    svg_path: Path,
    sensitivity_cache_path: Path,
    channel_ablation_png: Path,
    paired_root: Path | None = None,
    unpaired_root: Path | None = None,
    paired_orion: Path | None = None,
    unpaired_orion: Path | None = None,
    layout_root: Path | None = None,
    unpaired_mapping_json: Path | None = None,
    reference_label: str | None = None,
) -> plt.Figure:
    # Full-width text-bearing panels rendered at native point sizes.
    img_c = _fig_to_rgb(build_metric_trends_figure(summaries, width_inches=COMPOSITE_WIDTH_IN))
    if paired_root is None:
        paired_root = channel_ablation_png.parents[3] / "inference_output" / "concat_ablation_1000" / "paired_ablation" / "ablation_results"
    if unpaired_root is None:
        unpaired_root = channel_ablation_png.parents[3] / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "ablation_results"
    if paired_orion is None:
        paired_orion = channel_ablation_png.parents[3] / "data" / "orion-crc33"
    if unpaired_orion is None:
        unpaired_orion = paired_orion
    if layout_root is None:
        layout_root = paired_orion
    if unpaired_mapping_json is None:
        unpaired_mapping_json = channel_ablation_png.parents[3] / "inference_output" / "concat_ablation_1000" / "unpaired_ablation" / "metadata" / "unpaired_mapping.json"

    summary_by_slug = {summary.slug: summary for summary in summaries}
    img_d, div_d = _build_dataset_de_row(
        summary_by_slug["paired"],
        paired_root=paired_root,
        unpaired_root=unpaired_root,
        paired_orion=paired_orion,
        unpaired_orion=unpaired_orion,
        layout_root=layout_root,
        unpaired_mapping_json=unpaired_mapping_json,
    )
    img_e, div_e = _build_dataset_de_row(
        summary_by_slug["unpaired"],
        paired_root=paired_root,
        unpaired_root=unpaired_root,
        paired_orion=paired_orion,
        unpaired_orion=unpaired_orion,
        layout_root=layout_root,
        unpaired_mapping_json=unpaired_mapping_json,
    )

    # Common full width = widest text panel; narrower text panels are padded
    # (centred on white), never stretched, so their fonts stay consistent.
    full_width = max(img_c.shape[1], img_d.shape[1], img_e.shape[1])
    d_left = max(0, (full_width - img_d.shape[1]) // 2)
    e_left = max(0, (full_width - img_e.shape[1]) // 2)
    img_c = _pad_to_width(img_c, full_width)
    img_d = _pad_to_width(img_d, full_width)
    img_e = _pad_to_width(img_e, full_width)

    cache = _load_cache(sensitivity_cache_path)
    row1, w_a, h1 = _build_row1(svg_path, cache, full_width)

    # --- stack rows, each preceded by a white label strip ---
    rows = [row1, img_c, img_d, img_e]
    rows = [_add_label_strip(r) for r in rows]
    heights = [r.shape[0] for r in rows]

    fig_w = full_width / _RENDER_DPI
    fig_h = sum(heights) / _RENDER_DPI
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(len(rows), 1, height_ratios=heights, hspace=0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    b_x = (w_a + _ROW1_GAP_PX) / full_width
    # Panel B stacks the loss curve (B) over the ΔLPIPS bars (C); C is labelled
    # at the top of the ΔLPIPS sub-plot, left-aligned under B. The metric
    # trade-offs panel is D; the paired/unpaired heatmap + qualitative strips
    # become E/F and G/H, labelled at their respective left edges.
    c_y = (1.0 - _panelB_subrow_top_frac(1)) * (h1 / rows[0].shape[0]) + 0.012
    row_label_specs = [
        [("A", 0.006, 0.992), ("B", b_x + 0.006, 0.992), ("C", b_x + 0.006, c_y)],
        [("D", 0.006, 0.992)],
        [("E", d_left / full_width + 0.006, 0.992), ("F", (d_left + div_d) / full_width + 0.006, 0.992)],
        [("G", e_left / full_width + 0.006, 0.992), ("H", (e_left + div_e) / full_width + 0.006, 0.992)],
    ]
    for i, (img, specs) in enumerate(zip(rows, row_label_specs)):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img, interpolation="none")
        ax.axis("off")
        for letter, x, y in specs:
            ax.text(
                x,
                y,
                letter,
                transform=ax.transAxes,
                fontsize=_PANEL_LETTER_FS,
                fontweight="bold",
                va="top",
                ha="left",
                color="black",
            )
        if i == 1 and reference_label:
            ax.text(
                0.994,
                0.992,
                reference_label,
                transform=ax.transAxes,
                fontsize=FONT_SIZE_TICK + 2,
                va="top",
                ha="right",
                color="black",
                fontfamily=SECTION1_FONT_FAMILY,
            )
    return fig


REFERENCE_LABELS = {
    "primary": "PQ and DICE in D, E, G: reference = CellViT on the real H&E (primary)",
    "secondary": "PQ and DICE in D, E, G: reference = Lin input cell mask (secondary)",
}


def _load_summaries(abl_root: Path, orion_root: Path):
    from tools.ablation_report.data import load_dataset_summary
    from tools.stage3.style_mapping import load_style_mapping

    style = load_style_mapping(None)
    paired = load_dataset_summary(
        slug="paired", title="Paired",
        metrics_root=abl_root / "paired_ablation" / "ablation_results",
        dataset_root=abl_root / "paired_ablation",
        reference_root=orion_root, style_mapping=style,
    )
    unpaired = load_dataset_summary(
        slug="unpaired", title="Unpaired",
        metrics_root=abl_root / "unpaired_ablation" / "ablation_results",
        dataset_root=abl_root / "unpaired_ablation",
        reference_root=orion_root, style_mapping=style,
    )
    return [paired, unpaired]


if __name__ == "__main__":
    import argparse
    import json

    from tools.ablation_report.figures import save_figure_png
    from src.paper_figures.style import apply_style

    ap = argparse.ArgumentParser()
    ap.add_argument("--abl-root", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--orion-root", type=Path, required=True)
    ap.add_argument("--svg", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument("--bands", type=Path, required=True)
    ap.add_argument("--reference", choices=sorted(REFERENCE_LABELS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tools.ablation_report.shared as shared
    for k, v in json.load(open(a.bands))["bands"].items():
        old = dict(shared.TRADEOFF_REFERENCE_BANDS[k])
        shared.TRADEOFF_REFERENCE_BANDS[k].clear()
        shared.TRADEOFF_REFERENCE_BANDS[k].update({"label": old["label"], **v})
        print(f"band {k}: {old} -> {shared.TRADEOFF_REFERENCE_BANDS[k]}")

    apply_style()
    summaries = _load_summaries(a.abl_root, a.orion_root)
    fig = build_combined_method_perf_figure(
        summaries,
        svg_path=a.svg,
        sensitivity_cache_path=a.cache,
        channel_ablation_png=a.out,  # only used for default paths, all of which are given below
        paired_root=a.images_root / "paired_ablation" / "ablation_results",
        unpaired_root=a.images_root / "unpaired_ablation" / "ablation_results",
        paired_orion=a.orion_root,
        unpaired_orion=a.orion_root,
        layout_root=a.orion_root,
        unpaired_mapping_json=a.mapping,
        reference_label=REFERENCE_LABELS[a.reference],
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png(fig, a.out)
    for s in summaries:
        print(s.slug, "tiles", getattr(s, "tile_count", "?"))
    print(f"wrote {a.out} ({a.reference} reference)")
