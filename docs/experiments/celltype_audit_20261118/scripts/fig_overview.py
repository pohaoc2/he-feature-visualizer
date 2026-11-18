"""Figure 2 of celltype_audit_20261118 (Hyak, cellvit env + pylib imagecodecs): CRC33_01 overview at pyramid level 4.
  a  H&E (Lin registered H&E = the generator's training H&E), training tiles outlined area
  b  Lin H&E CNN tissue class map (TUM / NORM / STR / LYM / other), training-tile area only
  c  hfv 'cancer' objects per 112 um grid cell (the generator's cancer channel), same grid as b
  d  Lin gates + rule (a) (tcfix_rule: NORM epithelium healthy; TUM non-immune cancer unless SMA/CD31+) cancer cells per grid cell
Only image-level aggregates leave Hyak (this PNG/PDF).  fig_overview.py W OUT.png
"""
import csv
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, sys.argv[1] + "/scripts")
W, OUT = sys.argv[1:3]
from figstyle import FS_K, W2, letter, save  # noqa: E402

he = tifffile.TiffFile(f"{W}/lin/CRC33_01__19510_C33_US_SCAN_OR_001__160715-2-registered.ome.tif").series[0].levels[4].asarray()
cmap = np.array(__import__("PIL.Image", fromlist=["Image"]).open(f"{W}/lin/CRC33_01_lin_he_class.tiff"))
P = 112.0 / 0.325
H0, W0 = 52740, 36354
idx = json.load(open(f"{W}/raw/processed_crc33/index.json"))
cover = np.zeros(cmap.shape, bool)
cnt_tile = np.zeros(cmap.shape)
for p in idx["patches"]:
    r = int((p["y0"] + 128) / P); c = int((p["x0"] + 128) / P)
    if r < cmap.shape[0] and c < cmap.shape[1]:
        cover[r, c] = True
hc = np.zeros(cmap.shape)
with open(f"{W}/raw/processed_crc33/cell_assignments.csv") as f:
    for row in csv.DictReader(f):
        if row["cell_type"] == "cancer":
            r = min(int(float(row["centroid_y_global"]) / P), cmap.shape[0] - 1)
            c = min(int(float(row["centroid_x_global"]) / P), cmap.shape[1] - 1)
            hc[r, c] += 1
TH = json.load(open(f"{W}/out/lin_gates_crc33.json"))["markers"]
lc = np.zeros(cmap.shape)
with open(f"{W}/lin/CRC33_01__P37_S76_01-CRC33_01.csv") as f:
    for row in csv.DictReader(f):
        x, y = float(row["X_centroid"]), float(row["Y_centroid"])
        r = min(int(y / P), cmap.shape[0] - 1); c = min(int(x / P), cmap.shape[1] - 1)
        imm = float(row["CD45"]) > TH["CD45"]["final_thr"]
        epi = float(row["Pan-CK"]) > TH["Pan-CK"]["final_thr"] or float(row["E-cadherin"]) > TH["E-cadherin"]["final_thr"]
        sv = float(row["SMA"]) > TH["SMA"]["final_thr"] or float(row["CD31"]) > TH["CD31"]["final_thr"]
        k = cmap[r, c]
        if not imm and ((epi and k != 1) or (not epi and k == 3 and not sv)):
            lc[r, c] += 1
grp = np.full(cmap.shape, 4)
grp[cmap == 3] = 0; grp[cmap == 1] = 1; grp[cmap == 7] = 2; grp[cmap == 8] = 3; grp[cmap == 0] = 5
grp = np.where(cover, grp, 5).astype(float)
ext = (0, W0 * .325 / 1000, H0 * .325 / 1000, 0)
fig = plt.figure(figsize=(W2, 3.2))
names = ["H&E (training slide)", "Lin H&E tissue class", "hfv cancer objects", "Lin gates + rule (a): cancer"]
for i in range(4):
    ax = fig.add_axes([.01 + i * .238, .1, .225, .8])
    if i == 0:
        ax.imshow(he, extent=ext)
    elif i == 1:
        cm = ListedColormap(["#c0392b", "#e6a23c", "#a9c1d3", "#2e7d32", "#d9d9d9", "#ffffff"])
        ax.imshow(grp, cmap=cm, vmin=0, vmax=5, extent=ext, interpolation="nearest")
        for j, (n, col) in enumerate(zip(["TUM", "NORM", "STR", "LYM", "other"], cm.colors)):
            ax.text(.03, .36 - j * .07, n, transform=ax.transAxes, fontsize=FS_K - .5, color="k",
                    bbox=dict(boxstyle="square,pad=.15", fc=col, ec="none"))
    else:
        v = (hc if i == 2 else lc).astype(float)
        v[~cover] = np.nan
        vmax = np.nanpercentile(np.r_[hc[cover], lc[cover]], 99)
        im = ax.imshow(v, cmap="magma_r", vmin=0, vmax=vmax, extent=ext, interpolation="nearest")
        if i == 3:
            cax = fig.add_axes([.94, .25, .008, .5])
            cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=FS_K - 1); cb.set_label("cells / grid cell", fontsize=FS_K - .5)
    ax.set_title(names[i], fontsize=FS_K + .3, pad=2)
    ax.set_xticks([]); ax.set_yticks([])
    if i == 0:
        ax.plot([1, 3], [ext[2] - .8] * 2, color="k", lw=1.2)
        ax.text(2, ext[2] - 1.0, "2 mm", ha="center", va="bottom", fontsize=FS_K - .5)
    letter(fig, .02 + i * .238 * W2, .05, "abcd"[i])
fig.text(.5, .03, "Cells per 112 um grid cell (Lin CNN grid); grey/white = outside the 10,379 training tiles or background",
         ha="center", fontsize=FS_K - .5)
save(fig, OUT)
print("tiles covered grid cells", int(cover.sum()), "TUM", int(((cmap == 3) & cover).sum()), "NORM", int(((cmap == 1) & cover).sum()))
