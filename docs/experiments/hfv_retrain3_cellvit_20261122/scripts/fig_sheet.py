"""hfv_retrain3_cellvit_20261122 stage report figure: contact sheet of the official-CellViT training inputs (local).

  python fig_sheet.py DATA OUT
DATA/sheet/<slide>__<t>/{he, cell_masks, cell_type_*}.png and picks.json (sheet_pick.py). Per tile: real H&E | official-
CellViT mask (the binary cell_masks channel) | population channels (healthy / cancer / immune; nuclei on no Lin cell grey).
Two blocks of three columns (CRC33_01 left, CRC36 right), three tiles each; 180 mm, sim-exp-validation figstyle.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/pohaochiu/Documents/UW/bagherilab/sim-exp-validation/docs/experiments/mechanism_sink_20261026/report_20261110")
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
picks = json.loads((D / "sheet/picks.json").read_text())
POP = {"healthy": "#4ea72e", "cancer": "#d62728", "immune": "#1f77b4"}
UNM = "#9a9a9a"


def rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)


def load(d, n):
    return np.asarray(Image.open(d / f"{n}.png"))


s = 1.0 * fs.MM * 27          # tile side (in)
gap, bgap, top, left = 0.03, 0.2, 0.32, 0.14
W = fs.W2
fig = plt.figure(figsize=(W, top + 3 * s + 2 * gap + 0.32))
cols = ["H&E", "official-CellViT mask", "population channels"]
for b, (slide, name) in enumerate((("crc33", "CRC33_01"), ("crc36", "CRC36"))):
    x0 = left + b * (3 * s + 2 * gap + bgap)
    fs.letter(fig, x0 - 0.04, 0.02, "ab"[b])
    fig.text((x0 + 1.5 * s + gap) / W, 1 - 0.05 / fig.get_size_inches()[1], f"{name} (training tiles)",
             ha="center", va="top", fontsize=fs.FS_T)
    for r, t in enumerate(picks[slide]["tiles"]):
        d = D / "sheet" / f"{slide}__{t}"
        m = load(d, "cell_masks") > 0
        pop = np.full(m.shape + (3,), 255, np.uint8)
        pop[m] = rgb(UNM)
        for p, c in POP.items():
            pop[load(d, f"cell_type_{p}") > 0] = rgb(c)
        imgs = [np.asarray(Image.open(d / "he.png").convert("RGB")), np.where(m, 0, 255).astype(np.uint8), pop]
        for c, im in enumerate(imgs):
            ax = fs.axes(fig, x0 + c * (s + gap), top + r * (s + gap), s, s)
            ax.imshow(im, cmap="gray" if im.ndim == 2 else None, vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(fs.LW_AX * .6); sp.set_color(fs.GREY)
            if r == 0:
                ax.set_title(cols[c], fontsize=fs.FS_K, pad=2)
            if c == 0:
                ax.set_ylabel(t, fontsize=fs.FS_K * .9, labelpad=1)
h = [Patch(facecolor=c, edgecolor="none", label=p) for p, c in POP.items()] + \
    [Patch(facecolor=UNM, edgecolor="none", label="nucleus on no Lin cell")]
fig.legend(handles=h, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handlelength=1, columnspacing=1.2)
fs.save(fig, OUT)
print("wrote", OUT)
