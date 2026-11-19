"""lintype_render_20261118 step 3 (cellvit env; on Hyak, reads per-cell data and H&E tiles, writes small json + figure).

  summary.py A N HFV
Inputs : N/out/{labels.json, nuclei_render.json, cells_labels.npz, cells_nuclei.npz, render_tiles.npz, compare_tiles.npz,
         tiles_keep.txt}, N/data/{old_channels.tar, new_exp_channels.tar, nucleus_shapes_orion_crc33.npz (sim-exp-validation
         artifact, CellViT outlines)}, A/raw/processed_crc33/cell_assignments.csv (CellViT object areas).
Outputs: N/out/summary.json, N/out/fig_lintype_render.{png,pdf}.
Figure (180 mm):
  a  three sample tiles (Lin TUM / NORM / STR at the tile centre): H&E | current channels (CellViT objects, hfv typing) |
     eroded Lin nuclei on the H&E | new channels (Lin cells, CELESTA + Fig. 3d typing, rendered with borrowed outlines)
  b  mean area fraction per cell channel over the kept tiles, current vs new
  c  nucleus area (um^2): CellViT objects, Lin cell mask, Lin eroded strict / loose, the two outline libraries
  d  hematoxylin AUC (inside vs outside the mask) over the sample tiles, per mask
  e  population of each matched CellViT object, current (rows) vs new (columns), row-normalised
"""
import csv
import io
import json
import sys
import tarfile

import cv2
import numpy as np

A, N, HFV = sys.argv[1:4]
sys.path.insert(0, f"{N}/scripts")
sys.path.insert(0, HFV)
from figstyle import FS_K, GREY, INK, W2, axes, letter, plt, save  # noqa: E402
from stages.lin_typing import labels as LB  # noqa: E402

HE = "/gscratch/cheme/chiu/crc16-20261024/darkdiag_20261031/s3/he"
NAMES = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
         "cell_state_nonprolif", "cell_state_dead"]
SHORT = ["mask", "healthy", "cancer", "immune", "prolif", "nonprolif", "dead"]
PC = {"healthy": (169, 193, 211), "cancer": (192, 57, 43), "immune": (46, 125, 50)}
OLD_C, NEW_C = "#0b76a0", "#4ea72e"
um2 = LB.MPP ** 2
L = json.load(open(f"{N}/out/labels.json"))
NR = json.load(open(f"{N}/out/nuclei_render.json"))
keep = open(f"{N}/out/tiles_keep.txt").read().split()
S = dict(labels=L, nuclei_render=NR)


def png(b):
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)


# ---- whole-slide channel fractions, current vs new (kept tiles) ---------------------------------------------------
rt = np.load(f"{N}/out/render_tiles.npz")
new_frac = dict(zip(rt["tile"].tolist(), rt["frac"]))
old = {}
with tarfile.open(f"{N}/data/old_channels.tar") as tf:
    for m in tf:
        p = m.name.split("/")
        if len(p) == 4 and p[1] == "exp_channels" and p[3].endswith(".png"):
            t = p[3][:-4]
            if t in new_frac:
                old.setdefault(t, {})[p[2]] = float((png(tf.extractfile(m).read()) > 0).mean())
tiles = [t for t in keep if t in old and len(old[t]) == 7]
assert len(tiles) == len(keep), (len(tiles), len(keep))
O = np.array([[old[t][n] for n in NAMES] for t in tiles])
Nw = np.array([new_frac[t] for t in tiles])
S["channels_keep"] = {n: dict(old_mean=round(float(O[:, i].mean()), 4), new_mean=round(float(Nw[:, i].mean()), 4),
                              pearson_r_tiles=round(float(np.corrcoef(O[:, i], Nw[:, i])[0, 1]), 3)
                              if O[:, i].std() > 0 and Nw[:, i].std() > 0 else None)
                      for i, n in enumerate(NAMES)}

import os  # noqa: E402
if os.path.exists(f"{N}/out/render_tiles_strict.npz"):
    rs = np.load(f"{N}/out/render_tiles_strict.npz")
    fs = dict(zip(rs["tile"].tolist(), rs["frac"]))
    Ns = np.array([fs[t] for t in tiles])
    for i, n in enumerate(NAMES):
        S["channels_keep"][n]["new_strict_mean"] = round(float(Ns[:, i].mean()), 4)
    S["nuclei_render_strict"] = json.load(open(f"{N}/out/nuclei_render_strict.json"))

# ---- nucleus areas -----------------------------------------------------------------------------------------------
cn = np.load(f"{N}/out/cells_nuclei.npz")
cl = np.load(f"{N}/out/cells_labels.npz")
k = cl["in_keep"]
ps = 256
keep_set = set(keep)
cv_all, cv_keep = [], []
with open(f"{A}/raw/processed_crc33/cell_assignments.csv") as f:
    for r in csv.DictReader(f):
        x, y = float(r["centroid_x_global"]), float(r["centroid_y_global"])
        cv_all.append(float(r["Area_cellvit_px"]))
        cv_keep.append(f"{int(x // ps) * ps}_{int(y // ps) * ps}" in keep_set)
cv_all, cv_keep = np.array(cv_all), np.array(cv_keep)
cv_area = cv_all[cv_keep]
# one-to-one pairs: a CellViT object (kept tile) whose contour holds exactly one Lin centroid and whose centroid lies on a
# Lin cell that holds no other CellViT centroid -> paired nucleus areas
lk = np.load(f"{A}/out/lookup.npz")
row_of = {int(v): i for i, v in enumerate(cl["cell_id"])}
one = cv_keep & (lk["hfv_merge"] == 1) & (lk["hfv_label"] > 0)
li = np.array([row_of.get(int(v), -1) for v in lk["hfv_label"][one]])
okp = li >= 0
li = li[okp]
cva = cv_all[one][okp]
okp2 = lk["lin2hfv"][li] == 1
li, cva = li[okp2], cva[okp2]
S["paired_area"] = dict(n=int(len(li)), touching_frac=round(float(cn["touching"][li].mean()), 4))
for nm in ("strict_px", "loose_px", "cell_px"):
    v = cn[nm][li].astype(float)
    good = v > 0
    lr = np.log2(v[good] / cva[good])
    S["paired_area"][nm] = dict(median_ratio_to_cellvit=round(float(2 ** np.median(lr)), 3),
                                median_abs_log2_ratio=round(float(np.median(np.abs(lr))), 3), zero=int((~good).sum()))
    for tch in (False, True):
        g2 = good & (cn["touching"][li] == tch)
        lr2 = np.log2(v[g2] / cva[g2])
        S["paired_area"][nm]["touching" if tch else "isolated"] = round(float(2 ** np.median(lr2)), 3) if g2.any() else None
libn = np.load(NR["library"]["path"])
libo = np.load(f"{N}/data/nucleus_shapes_orion_crc33.npz")
AREAS = {"CellViT objects": cv_area, "Lin cell mask": cn["cell_px"][k], "Lin eroded (strict)": cn["strict_px"][k],
         "Lin eroded (loose)": cn["loose_px"][k], "library: CellViT outlines": libo["area"],
         "library: Lin nuclei": libn["area"]}
q = lambda v: [round(float(t), 1) for t in np.percentile(v * um2, [10, 25, 50, 75, 90])]
S["area_um2_quantiles_10_25_50_75_90"] = {kk: q(v) for kk, v in AREAS.items()}
S["area_n"] = {kk: int(len(v)) for kk, v in AREAS.items()}

# ---- hematoxylin AUC and Dice (sample tiles) ---------------------------------------------------------------------
ct = np.load(f"{N}/out/compare_tiles.npz")
json.dump(S, open(f"{N}/out/summary.json", "w"), indent=1)

# ---- figure ------------------------------------------------------------------------------------------------------
FH = 6.55
fig = plt.figure(figsize=(W2, FH))


def clean(ax):
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)


def overlay(ch, alpha=1.0):
    img = np.full((256, 256, 3), 255, np.uint8)
    for n in ("healthy", "cancer", "immune"):
        img[ch[f"cell_type_{n}"]] = PC[n]
    return img


# a: example tiles
cmap = np.array(__import__("PIL.Image", fromlist=["Image"]).open(f"{A}/lin/CRC33_01_lin_he_class.tiff"))
cmp_tiles = ct["tile"].tolist()
cov = dict(zip(cmp_tiles, ct["cov_new"]))
med = float(np.median(ct["cov_new"]))
pick = []
for want in (3, 1, 7):   # TUM, NORM, STR
    for t in cmp_tiles:
        x0, y0 = map(int, t.split("_"))
        if LB.lin_class_at(x0 + 128, y0 + 128, cmap) == want and cov[t] >= med and t not in pick:
            pick.append(t)
            break
S["example_tiles"] = pick
newtar = tarfile.open(f"{N}/data/new_exp_channels.tar")
oldtar = tarfile.open(f"{N}/data/old_channels.tar")
ero = np.unpackbits(ct["ero_loose"])[:len(cmp_tiles) * 65536].reshape(len(cmp_tiles), 256, 256).astype(bool)
if True:
    size, gap, x_in, y_in = 1.3, 0.06, 0.25, 0.12
    cols = ["H&E", "current channels\n(CellViT objects, hfv typing)", "Lin nuclei\n(cell mask eroded 5 px)",
            "new channels (Lin cells, CELESTA +\nFig. 3d typing, borrowed outlines)"]
    letter(fig, 0.02, 0.02, "a")
    for r, t in enumerate(pick):
        x0, y0 = map(int, t.split("_"))
        rgb = cv2.cvtColor(cv2.imread(f"{HE}/{t}.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        oc = {n: png(oldtar.extractfile(f"orion-crc33/exp_channels/{n}/{t}.png").read()) > 0 for n in NAMES}
        nc = {n: png(newtar.extractfile(f"orion-crc33/exp_channels/{n}/{t}.png").read()) > 0 for n in NAMES}
        s_ = ero[cmp_tiles.index(t)]
        edge = (cv2.dilate(s_.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~s_
        he_e = rgb.copy(); he_e[edge] = (0, 0, 0)
        for j, img in enumerate([rgb, overlay(oc), he_e, overlay(nc)]):
            ax = axes(fig, x_in + j * (size + gap), y_in + 0.17 + r * (size + gap), size, size)
            ax.imshow(img, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for s2 in ax.spines.values():
                s2.set_linewidth(.4); s2.set_color(GREY)
            if r == 0:
                ax.set_title(cols[j], fontsize=FS_K, pad=2)
            if j == 0:
                ax.set_ylabel(["Lin TUM", "Lin NORM", "Lin STR"][r], fontsize=FS_K)
    # legend for population colours
    lx = x_in + 4 * (size + gap) + 0.02
    for i, n in enumerate(("cancer", "healthy", "immune")):
        ax = axes(fig, lx, y_in + 0.3 + i * 0.16, 0.1, 0.1)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=np.array(PC[n]) / 255)); ax.axis("off")
        fig.text((lx + 0.14) / W2, 1 - (y_in + 0.35 + i * 0.16) / FH, n, fontsize=FS_K, va="center")

top2 = y_in + 0.17 + 3 * (size + gap) + 0.3
PH = 1.1
# b: channel fractions
letter(fig, 0.02, top2 - 0.12, "b")
ax = axes(fig, 0.42, top2, 1.45, PH)
xx = np.arange(len(NAMES))
ax.bar(xx - 0.2, O.mean(0), 0.4, color=OLD_C, label="current")
ax.bar(xx + 0.2, Nw.mean(0), 0.4, color=NEW_C, label="new")
ax.set_xticks(xx, SHORT, rotation=45, ha="right"); ax.set_ylabel("mean area fraction of tile")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); clean(ax)
# c: areas
letter(fig, 1.98, top2 - 0.12, "c")
ax = axes(fig, 3.12, top2, 1.0, PH)
labs = list(AREAS)
ax.boxplot([AREAS[kk] * um2 for kk in labs], vert=False, showfliers=False, widths=.6,
           medianprops=dict(color=INK, lw=.8), boxprops=dict(lw=.5), whiskerprops=dict(lw=.5), capprops=dict(lw=.5))
ax.set_yticks(range(1, len(labs) + 1), labs); ax.invert_yaxis(); ax.set_xlabel("area (µm²)"); clean(ax)
# d: hematoxylin AUC
letter(fig, 4.2, top2 - 0.12, "d")
ax = axes(fig, 4.55, top2, 1.0, PH)
dk = [("auc_old", "CellViT"), ("auc_linmask", "Lin cell mask"), ("auc_strict", "Lin strict"), ("auc_loose", "Lin loose"),
      ("auc_new", "rendered")]
ax.boxplot([ct[kk][~np.isnan(ct[kk])] for kk, _ in dk], showfliers=False, widths=.6,
           medianprops=dict(color=INK, lw=.8), boxprops=dict(lw=.5), whiskerprops=dict(lw=.5), capprops=dict(lw=.5))
ax.axhline(.5, color=GREY, lw=.5, ls=":")
ax.set_xticks(range(1, len(dk) + 1), [d[1] for d in dk], rotation=45, ha="right")
ax.set_ylabel("hematoxylin AUC\n(inside vs outside mask)"); clean(ax)
# e: confusion
letter(fig, 5.68, top2 - 0.12, "e")
ax = axes(fig, 6.2, top2, 0.82, 0.82)
cm = np.array(L["old_vs_new"]["pop_confusion_rows_old_cols_new"], float)
cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
ax.imshow(cmn, cmap="Greys", vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cmn[i, j]:.0%}", ha="center", va="center", fontsize=FS_K, color="white" if cmn[i, j] > .5 else INK)
ax.set_xticks(range(3), LB.POP_NAMES, rotation=45, ha="right"); ax.set_yticks(range(3), LB.POP_NAMES)
ax.set_xlabel("new"); ax.set_ylabel("current")
json.dump(S, open(f"{N}/out/summary.json", "w"), indent=1)
save(fig, f"{N}/out/fig_lintype_render.png")
print(json.dumps({kk: S[kk] for kk in ("channels_keep", "area_um2_quantiles_10_25_50_75_90", "example_tiles")}, indent=1))
