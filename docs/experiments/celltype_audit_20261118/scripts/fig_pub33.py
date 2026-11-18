"""Figure 3 of celltype_audit_20261118 (180 mm), from data/compare33.json only (cells in the training tiles).
  a  population fractions in Lin TUM / NORM / all: Lin gates + tissue step, rule (a), CELESTA, Astir (unknown kept apart)
  b  the cells rule (a) moves to cancer (TUM, non-immune, non-epithelial, SMA-/CD31-): what CELESTA and Astir call them
  c  Cohen's kappa in TUM between each rule and each published method, unknown merged into healthy or into cancer
     (95 % interval: 200 bootstrap resamples of 1024-px squares)
  fig_pub33.py data/compare33.json figures/fig3_celesta_astir.png
"""
import json
import sys

import numpy as np

from figstyle import C, FS_K, GREY, INK, W2, axes, letter, plt, save

R = json.load(open(sys.argv[1]))
S = R["tiles"]
TY = [t for t in ["old", "rule_a", "celesta", "celesta_whole", "astir"] if t in R["typings"]]
NM = {"old": "Lin gates + tissue step", "rule_a": "rule (a)", "celesta": "CELESTA (tiled)", "celesta_whole": "CELESTA (whole)",
      "astir": "Astir"}
PC = {"cancer": "#c0392b", "immune": "#2e7d32", "healthy": "#a9c1d3", "unknown": "#d9d9d9"}
RC = {"E": "#c0392b", "S": "#a9c1d3", "I": "#2e7d32", "U": "#d9d9d9"}
RN = {"E": "epithelial", "S": "stromal", "I": "immune", "U": "unknown"}


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def stack(ax, rows, keys, cols, ylabels, title):
    for i, r in enumerate(rows):
        left = 0.0
        for kk in keys:
            v = r.get(kk, 0.0)
            ax.barh(i, v, left=left, color=cols[kk], height=.72, lw=0)
            if v >= .12:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=FS_K - .8,
                        color="white" if kk in ("cancer", "immune", "E", "I") else INK)
            left += v
    ax.set_yticks(range(len(rows)), ylabels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], ["0", "0.5", "1"])
    ax.set_title(title, fontsize=FS_K, pad=2)
    clean(ax)


H = 3.55
fig = plt.figure(figsize=(W2, H))
letter(fig, .02, .05, "a")
for j, g in enumerate(["TUM", "NORM", "all"]):
    ax = axes(fig, 1.05 + j * 1.1, .3, 1.0, .2 * len(TY) + .1)
    stack(ax, [S[g][t]["pop"] for t in TY], ["cancer", "immune", "healthy", "unknown"], PC, [NM[t] for t in TY],
          f"{g if g != 'all' else 'all tissue'} ({S[g]['n']:,} cells)")
    if j:
        ax.set_yticklabels([])
    ax.set_xlabel("Fraction of cells", labelpad=1)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PC[kk]) for kk in PC], labels=list(PC), loc="upper left",
           bbox_to_anchor=(4.35 / W2, 1 - .3 / H), ncol=1, handlelength=1, fontsize=FS_K)
# b
letter(fig, .02, 1.75, "b")
mv = S["rule_a_moved"]
pub = [t for t in TY if t in mv]
ax = axes(fig, 1.05, 1.95, 1.4, .2 * (len(pub) + 1) + .1)
stack(ax, [{"E": 1.0}] + [mv[t]["raw"] for t in pub], ["E", "I", "S", "U"], RC, ["rule (a)"] + [NM[t] for t in pub],
      f"Cells rule (a) moves to cancer in TUM (n = {mv['n']:,})")
ax.set_xlabel("Fraction of these cells", labelpad=1)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=RC[kk]) for kk in ["E", "I", "S", "U"]],
           labels=[RN[kk] for kk in ["E", "I", "S", "U"]], loc="upper left", bbox_to_anchor=(2.55 / W2, 1 - 1.95 / H),
           ncol=1, handlelength=1, fontsize=FS_K)
d, te = mv["descriptors"], S["TUM_gate_epithelial"]["descriptors"]
ax.text(0, -.62, f"these cells: area {d['area_um2']} um2, Pan-CK {d['panck']:.0f}, Ki-67+ {d['ki67p']:.3f}\n"
        f"TUM gate epithelium: area {te['area_um2']} um2, Pan-CK {te['panck']:.0f}, Ki-67+ {te['ki67p']:.3f}",
        transform=ax.transAxes, fontsize=FS_K - .5, va="top", ha="left", color=INK)
# c
letter(fig, 3.55, 1.7, "c")
ax = axes(fig, 5.25, 1.95, 1.6, 1.0)
pairs = [(a, b) for a in ["old", "rule_a"] for b in TY if b not in ("old", "rule_a")]
for i, (a, b) in enumerate(pairs):
    for off, nm, col in [(-.15, "U_to_healthy", C["old"]), (.15, "U_to_cancer", C["turn"])]:
        v, lo, hi = S["TUM"]["kappa"][f"{a}~{b}"][nm]
        ax.errorbar(v, i + off, xerr=[[v - lo], [hi - v]], fmt="o", color=col, ms=2.5, lw=.8, capsize=0,
                    label=("unknown -> healthy" if nm == "U_to_healthy" else "unknown -> cancer") if i == 0 else None)
ax.set_yticks(range(len(pairs)), [f"{NM[a].replace('Lin gates + tissue step', 'gates + tissue')} vs {NM[b]}" for a, b in pairs])
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xlabel("Cohen's kappa in TUM", labelpad=1)
ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=FS_K - .3, handlelength=1, ncol=1)
clean(ax)
save(fig, sys.argv[2])
