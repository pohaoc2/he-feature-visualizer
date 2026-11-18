"""Figure 1 of celltype_audit_20261118 (180 mm, Nature Communications style via figstyle.py), from data/audit.json only.
  a  class fractions by Lin H&E tissue class: hfv training labels vs Lin-gate rules (old, old + tissue step, rule a)
  b  keratin-weak tumour cells in TUM typed healthy; normal epithelium in NORM typed cancer (per labeling)
  c  hfv final label by CellViT nucleus type (how much the multiplex can change the H&E prior)
  d  labels of cells under ink vs elsewhere (hfv objects; Lin cells)
  e  proliferating fraction by tissue class: hfv state (Ki-67 >= slide 75th percentile) vs Lin Ki-67 gate
  fig_audit.py data/audit.json figures/fig1_audit.png
"""
import json
import sys

import numpy as np

from figstyle import C, FS_K, GREY, INK, W2, axes, letter, plt, save

A = json.load(open(sys.argv[1]))
ORDER = ["cancer", "immune", "healthy"]
COL = {"cancer": "#c0392b", "immune": "#2e7d32", "healthy": "#a9c1d3"}
LAB = [("hfv", "hfv (training labels)"), ("old", "Lin gates, old rule"), ("old_t", "+ tissue step"), ("rule_a", "rule (a)")]


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def stack(ax, rows, names, title):
    for i, r in enumerate(rows):
        left = 0.0
        for k in ORDER:
            v = r.get(k, 0.0)
            ax.barh(i, v, left=left, color=COL[k], height=.72, lw=0)
            if v >= .12:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=FS_K - .8,
                        color="white" if k != "healthy" else INK)
            left += v
    ax.set_yticks(range(len(rows)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
    ax.set_title(title, fontsize=FS_K, pad=2)
    clean(ax)


H = 5.2
fig = plt.figure(figsize=(W2, H))
# a
letter(fig, .02, .05, "a")
a = A["a_class_fractions"]
for j, g in enumerate(["NORM", "TUM", "STR", "LYM"]):
    ax = axes(fig, 1.05 + j * 1.52, .25, 1.32, .95)
    stack(ax, [a[g][k] for k, _ in LAB], [n for _, n in LAB], f"{g}: {a[g]['n_hfv']:,} hfv / {a[g]['n_lin']:,} Lin cells")
    if j > 0:
        ax.set_yticklabels([])
    ax.set_xlabel("Fraction of cells", labelpad=1)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COL[k]) for k in ORDER], labels=ORDER, loc="upper left",
           bbox_to_anchor=(.01, 1 - 1.45 / H), ncol=3, handlelength=1, fontsize=FS_K, columnspacing=1)
# b
letter(fig, .02, 1.75, "b")
ax = axes(fig, .42, 2.05, 2.0, 1.05)
b, c = A["b_keratin_weak_TUM"]["typed_healthy"], A["c_normal_epithelium_NORM"]["typed_cancer"]
keys = ["hfv_matched", "old", "old_t", "rule_a"]
x = np.arange(len(keys))
for off, vals, col, lab in [(-.2, b, C["old"], "keratin-weak TUM cells typed healthy"),
                            (.2, c, C["turn"], "NORM epithelium typed cancer")]:
    ax.bar(x + off, [vals[k] for k in keys], .38, color=col, label=lab)
    for xi, k in zip(x, keys):
        ax.text(xi + off, vals[k] + .02, f"{vals[k]:.2f}", ha="center", va="bottom", fontsize=FS_K - 1)
ax.set_xticks(x, ["hfv", "old", "+ tissue", "rule (a)"])
ax.set_ylim(0, 1.12)
ax.set_yticks([0, .5, 1])
ax.set_ylabel("Fraction")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=FS_K - .3, handlelength=1)
clean(ax)
# c
letter(fig, 2.75, 1.75, "c")
f = A["hfv_fusion"]
ks = [k for k in ["neoplastic", "epithelial", "inflammatory", "connective", "background", "dead"] if k in f]
ax = axes(fig, 3.75, 2.05, 1.1, 1.05)
stack(ax, [f[k]["final"] for k in ks], [f"{k} ({f[k]['n']:,})" for k in ks], "hfv label by CellViT nucleus type")
ax.set_xlabel("Fraction of objects", labelpad=1)
# c2: prior-only fallback share
ax = axes(fig, 5.0, 2.05, .5, 1.05)
ax.barh(range(len(ks)), [f[k]["prior_only"] for k in ks], color=GREY, height=.72)
ax.set_yticks(range(len(ks)), [""] * len(ks))
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks([0, 1], ["0", "1"])
ax.set_title("multiplex\nignored", fontsize=FS_K, pad=2)
clean(ax)
# e (top right of row 2): proliferating fraction by tissue
letter(fig, 5.75, 1.75, "e")
ax = axes(fig, 6.1, 2.05, .95, 1.05)
gs = ["NORM", "TUM", "STR", "LYM"]
y = np.arange(len(gs))
ax.barh(y - .2, [a[g]["prolif_hfv"] for g in gs], .38, color=INK, label="hfv state")
ax.barh(y + .2, [a[g]["prolif_lin"] for g in gs], .38, color=GREY, label="Lin Ki-67 gate")
ax.set_yticks(y, gs)
ax.invert_yaxis()
ax.set_xlabel("Proliferating", labelpad=1)
ax.legend(loc="lower right", bbox_to_anchor=(1.05, 1.0), fontsize=FS_K - .5, handlelength=1)
clean(ax)
# d: ink
letter(fig, .02, 3.55, "d")
d = A["d_ink"]
hv, lv = d["hfv"], d["lin"]
rows = [hv["inked_cells"]["type"], hv["clean_tiles"]["type"], lv["inked_cells"]["old"], lv["not_inked"]["old"]]
names = [f"hfv objects under ink ({hv['inked_cells']['n']:,})", f"hfv objects, ink-free tiles ({hv['clean_tiles']['n']:,})",
         f"Lin cells under ink ({lv['inked_cells']['n']:,})", f"Lin cells, not under ink ({lv['not_inked']['n']:,})"]
ax = axes(fig, 1.75, 3.8, 1.6, .95)
stack(ax, rows, names, "Cell type (ink = H&E gray < 60)")
ax = axes(fig, 3.55, 3.8, .8, .95)
pv = [hv["inked_cells"]["state"].get("prolif", 0), hv["clean_tiles"]["state"].get("prolif", 0),
      lv["inked_cells"]["ki67p"], lv["not_inked"]["ki67p"]]
ax.barh(range(4), pv, color=[INK, GREY, INK, GREY], height=.72)
for i, v in enumerate(pv):
    ax.text(v + .005, i, f"{v:.3f}", va="center", fontsize=FS_K - 1)
ax.set_yticks(range(4), [""] * 4)
ax.invert_yaxis()
ax.set_xlim(0, .35)
ax.set_title("Proliferating", fontsize=FS_K, pad=2)
clean(ax)
# d right: what the ink objects are
ax = axes(fig, 5.25, 3.8, 1.6, .95)
it = hv["inked_cells"]["tissue"]
tk = sorted(it, key=lambda k: -it[k])[:5]
ax.barh(range(len(tk)), [it[k] for k in tk], color=GREY, height=.72)
ax.set_yticks(range(len(tk)), tk)
ax.invert_yaxis()
ax.set_xlim(0, .6)
ax.set_title("Lin tissue class under hfv ink objects", fontsize=FS_K, pad=2)
ax.set_xlabel("Fraction", labelpad=1)
clean(ax)
save(fig, sys.argv[2])
