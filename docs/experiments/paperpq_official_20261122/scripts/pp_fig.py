"""paperpq_official_20261122 figure (English, Nat Commun 180 mm; sim-exp-validation figstyle.py).

  python pp_fig.py DATA OUT.png
a paper-style PQ, b hematoxylin readout (a), c hematoxylin readout (c), d LPIPS to real H&E: mean and 95 % tile-bootstrap
interval per generator; official-consistent pipeline filled, runner-consistent pipeline open; real H&E reference lines.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / "Documents/UW/bagherilab/sim-exp-validation/docs/experiments/mechanism_sink_20261026/report_20261110"))
import figstyle as F  # noqa: E402

DATA, OUT = Path(sys.argv[1]), Path(sys.argv[2])
S = json.load(open(DATA / "stats.json"))
ARMS = [("P", "a2_off_shelf", "PixCell\noff-the-shelf"), ("P", "a1_concat", "Concat"), ("P", "production", "Grouped"),
        ("P", "a2_bypass_full_tme", "Grouped\n+ bypass"), ("P", "a1_per_channel", "Per-channel"),
        ("R", "a1_concat", "Concat"), ("R", "production", "Grouped"), ("R", "a2_bypass_full_tme", "Grouped\n+ bypass"),
        ("R", "a1_per_channel", "Per-channel")]
X = [0, 1, 2, 3, 4, 5.6, 6.6, 7.6, 8.6]
COL = {"off": "#0b76a0", "run": "#e97132"}
PANELS = [("pq", "Paper-style PQ\n(input mask = reference)"), ("a", "Hematoxylin readout (a)\nmask pixels stained"),
          ("c", "Hematoxylin readout (c)\nnucleus-background contrast"), ("lpips", "LPIPS to real H&E\n(lower = closer)")]
fig = F.plt.figure(figsize=(F.W2, 95 * F.MM))
pw, ph, gx, x0, y0 = 1.42, 1.85, 0.42, 0.42, 0.18
for i, (q, lab) in enumerate(PANELS):
    ax = F.axes(fig, x0 + i * (pw + gx), y0, pw, ph)
    F.letter(fig, x0 + i * (pw + gx) - 0.38, 0.05, "abcd"[i])
    for pipe, dx in (("off", -0.15), ("run", 0.15)):
        for x, (fam, v, _) in zip(X, ARMS):
            d = S["arms"][f"{fam}|{pipe}|{v}"].get(q)
            if d is None:
                continue
            ax.errorbar(x + dx, d["mean"], yerr=[[d["mean"] - d["lo"]], [d["hi"] - d["mean"]]], fmt="o", ms=2.6,
                        color=COL[pipe], mfc=COL[pipe] if pipe == "off" else "white", mew=0.7, elinewidth=0.7, capsize=0)
        r = S["arms"].get(f"P|{pipe}|real", {}).get(q)
        if r is not None:
            ax.axhline(r["mean"], color=COL[pipe], lw=0.6, ls="-" if pipe == "off" else "--", zorder=0)
    ax.axvline(4.8, color=F.LIGHT, lw=0.6)
    ax.set_xticks(X)
    ax.set_xticklabels([a[2] for a in ARMS], rotation=90, fontsize=F.FS_K - 0.6)
    ax.set_title(lab, fontsize=F.FS, pad=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.6, 9.2)
    if i == 0:
        yt = ax.get_ylim()[1]
        ax.text(2, yt, "CellViT-mask\ngenerators", ha="center", va="bottom", fontsize=F.FS_K - 0.6, color=F.GREY)
        ax.text(7.1, yt, "shape-library-mask\ngenerators", ha="center", va="bottom", fontsize=F.FS_K - 0.6, color=F.GREY)
h = [F.plt.Line2D([], [], color=COL["off"], marker="o", ls="", ms=3, label="official-consistent pipeline"),
     F.plt.Line2D([], [], color=COL["run"], marker="o", mfc="white", ls="", ms=3, label="runner-consistent pipeline"),
     F.plt.Line2D([], [], color=F.GREY, lw=0.6, label="real H&E (solid: official, dashed: runner)")]
fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
F.save(fig, OUT)
