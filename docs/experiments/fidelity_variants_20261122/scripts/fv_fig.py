"""fidelity_variants_20261122 figure (RESULTS.md): 180 mm, shared figstyle (sim-exp-validation report_20261110).

  python fv_fig.py STATS_JSON OUT_PNG
a-c  paired differences between variants (Concat - Grouped, Concat - Grouped+bypass, Grouped - Grouped+bypass) for the
     three rule quantities: fidelity PQ (reference = the input mask, official CellViT post-processing), hematoxylin (a), (c)
d-f  each trained variant minus the real-H&E reference line, same quantities
filled = CRC36 (300 tiles), open = CRC33 (100 tiles); bars = 95 % tile-bootstrap intervals (10,000, seed 42)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "sim-exp-validation/docs/experiments/mechanism_sink_20261026/report_20261110"))
import figstyle as FS  # noqa: E402

plt = FS.plt
S = json.loads(Path(sys.argv[1]).read_text())
SL = (("CRC36", True), ("CRC33", False))
Q = (("pq_official", "fidelity PQ\n(official post-processing)"), ("a", "mask pixels with\nhematoxylin ≥ t  (a)"),
     ("c", "hematoxylin contrast\ninside − outside mask  (c)"))
PAIRS = (("cat-grp", "Concat −\nGrouped"), ("cat-gbp", "Concat −\nGrp+bypass"), ("grp-gbp", "Grouped −\nGrp+bypass"))
ARMS = (("grp", "Grouped", FS.C["turn"]), ("cat", "Concat", FS.C["old"]), ("gbp", "Grp+bypass", FS.C["shape"]))
OFF = .14


def eb(ax, x, d, c, filled):
    m, (lo, hi) = d["mean"], d["ci95"]
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, mfc=c if filled else "white", mec=c, ms=FS.MS,
                lw=FS.LW, capsize=0, zorder=3)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=FS.GREY, lw=FS.LW_AX, ls=":", zorder=1)


fig = plt.figure(figsize=(FS.W2, 3.55))
W, H, X0, DX = 1.55, 1.05, 0.55, 2.35
for j, (q, ql) in enumerate(Q):
    ax = FS.axes(fig, X0 + j * DX, 0.15, W, H)
    FS.letter(fig, X0 + j * DX - 0.5, 0.05, "abc"[j])
    for i, (p, _) in enumerate(PAIRS):
        for k, (s, f) in enumerate(SL):
            eb(ax, i + (k - .5) * 2 * OFF, S["slides"][s]["quantities"][q]["diff"][p], FS.INK, f)
    ax.set_xticks(range(3), [x[1] for x in PAIRS])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel("paired difference")
    ax.set_title(ql, fontsize=FS.FS_K, color=FS.INK, pad=3)
    clean(ax)
    if j == 0:
        ax.plot([], [], "o", color=FS.INK, label="CRC36")
        ax.plot([], [], "o", color=FS.INK, mfc="white", label="CRC33")
        ax.legend(loc="lower left", handletextpad=.2, borderaxespad=.1)
    ax = FS.axes(fig, X0 + j * DX, 2.0, W, H)
    FS.letter(fig, X0 + j * DX - 0.5, 1.9, "def"[j])
    for i, (a, _, c) in enumerate(ARMS):
        for k, (s, f) in enumerate(SL):
            eb(ax, i + (k - .5) * 2 * OFF, S["slides"][s]["quantities"][q]["diff"][f"{a}-real"], c, f)
    ax.set_xticks(range(3), [x[1] for x in ARMS])
    ax.set_xlim(-.5, 2.5)
    ax.set_ylabel("variant − real H&E")
    clean(ax)
FS.save(fig, sys.argv[2])
