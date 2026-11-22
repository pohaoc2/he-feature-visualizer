"""cellvit_official_20261122 figure (RESULTS.md): 180 mm, shared figstyle (sim-exp-validation report_20261110).

  python cvo_fig.py STATS_JSON OUT_PNG
a  stage 3: the six PREREG §9.5 decision intervals (N - K), runner (open) vs official (filled) CellViT post-processing
b  stage 3 readout 3: delta per arm, runner vs official (window-bootstrap 95 % intervals, app 8 convention)
c  Fig 2 main PQ (reference = real-H&E CellViT, binarised as in the paper) per variant: published, runner rerun, official
d  Fig 2 secondary PQ (reference = Lin mask) per variant on the retrained sets, runner vs official; reference line = real H&E
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "sim-exp-validation/docs/experiments/mechanism_sink_20261026/report_20261110"))
import figstyle as FS  # noqa: E402

plt = FS.plt
S = json.loads(Path(sys.argv[1]).read_text())
s3, f2 = S["stage3"], S["fig2"]
CK, CN, CR = FS.C["old"], FS.C["new"], FS.INK
OFF = .17


def eb(ax, x, m, lo, hi, c, filled, ms=FS.MS):
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=c, mfc=c if filled else "white", mec=c, ms=ms,
                lw=FS.LW, capsize=0, zorder=3)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)


fig = plt.figure(figsize=(FS.W2, 4.75))
# a ------------------------------------------------------------------------------------------------------------------
ax = FS.axes(fig, 0.45, 0.2, 2.55, 1.5)
FS.letter(fig, 0.02, 0.05, "a")
dd = [(s3["readout1"][f"{st}:{q}"], f"{qn}\n{sn}") for q, qn in (("fid_pq", "fidelity PQ"), ("fid_f1", "fidelity F1"))
      for st, sn in (("crc36ho", "CRC36"), ("crc33sec", "CRC33"))]
for i, (d, _) in enumerate(dd):
    for k, (p, f) in enumerate((("old", False), ("new", True))):
        v = d[p]["N-K"]
        eb(ax, i + (k - .5) * 2 * OFF, v["mean"], *v["ci95"], FS.INK, f)
for j, (a, an) in enumerate((("NP-KP", "Δ own\nstyle"), ("ND-KD", "Δ other\nstyle"))):
    d = s3["readout3_n_minus_k"][f"{a}:delta"]
    for k, (p, f) in enumerate((("old", False), ("new", True))):
        eb(ax, 4 + j + (k - .5) * 2 * OFF, d[p]["point"], *d[p]["ci95"], FS.INK, f)
    dd.append((None, an))
ax.axhline(0, color=FS.GREY, lw=FS.LW_AX, ls=":")
ax.axvline(3.5, color=FS.LIGHT, lw=FS.LW_AX)
ax.set_xticks(range(6), [x[1] for x in dd])
ax.set_ylabel("model N − model K")
ax.plot([], [], "o", color=FS.INK, mfc="white", label="runner post-processing")
ax.plot([], [], "o", color=FS.INK, label="official post-processing")
ax.legend(loc="lower left", handletextpad=.2)
clean(ax)
# b ------------------------------------------------------------------------------------------------------------------
ax = FS.axes(fig, 3.55, 0.2, 3.4, 1.5)
FS.letter(fig, 3.2, 0.05, "b")
arms = [("R_L1", "real\n(L1 units)", CR), ("NP", "N own", CN), ("ND", "N other", CN),
        ("R_s3b", "real\n(s3b units)", CR), ("KP", "K own", CK), ("KD", "K other", CK)]
for i, (a, _, c) in enumerate(arms):
    for k, (p, f) in enumerate((("old", False), ("new", True))):
        m, lo, hi = s3["readout3_arms"][a]["delta"][p]
        eb(ax, i + (k - .5) * 2 * OFF, m, lo, hi, c, f)
ax.axhline(0, color=FS.GREY, lw=FS.LW_AX, ls=":")
ax.set_xticks(range(len(arms)), [a[1] for a in arms])
ax.set_ylabel("Δ (cancer − healthy,\nneoplastic call rate)")
clean(ax)
# c ------------------------------------------------------------------------------------------------------------------
ax = FS.axes(fig, 0.45, 2.45, 3.75, 1.45)
FS.letter(fig, 0.02, 2.3, "c")
VN = {"van": "Vanilla", "grp": "Grouped", "cat": "Concat", "gbp": "G+bypass", "pch": "Per-ch."}
groups = [("P_van1000", ["van"], "paper\n1000"), ("P_100", ["van", "cat", "gbp", "pch"], "paper models\nCRC33 100 tiles"),
          ("36", list(VN), "retrained\nCRC36 300 tiles"), ("33", list(VN), "retrained\nCRC33 100 tiles")]
x, ticks, labs, spans = 0, [], [], []
for g, vs, gn in groups:
    x0 = x
    for v in vs:
        d = f2[g][v]
        pub = d["published"] if isinstance(d["published"], float) else d["published"]["pq_main"]
        ax.plot(x - 2 * OFF, pub, "x", color=FS.GREY, ms=FS.MS, mew=.8, zorder=3)
        for k, (p, f) in enumerate((("runner", False), ("official", True))):
            v_ = d["pq_main"][p]
            eb(ax, x + k * 2 * OFF - OFF + OFF, v_["mean"], *v_["ci95"], FS.INK, f)
        ticks.append(x + OFF)
        labs.append(VN[v])
        x += 1
    spans.append(((x0 + x - 1) / 2 + OFF, gn))
    x += .6
ax.set_xticks(ticks, labs, fontsize=FS.FS_K - .6, rotation=90)
for xc, gn in spans:
    ax.text(xc, -0.36, gn, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS.FS_K, color=FS.GREY)
ax.set_ylabel("PQ vs real-H&E CellViT")
ax.set_ylim(0, .9)
ax.plot([], [], "x", color=FS.GREY, label="published")
ax.plot([], [], "o", color=FS.INK, mfc="white", label="runner (rerun)")
ax.plot([], [], "o", color=FS.INK, label="official")
ax.legend(loc="upper right", ncol=3, handletextpad=.2, columnspacing=.8)
clean(ax)
# d ------------------------------------------------------------------------------------------------------------------
ax = FS.axes(fig, 4.75, 2.45, 2.2, 1.45)
FS.letter(fig, 4.4, 2.3, "d")
x, ticks, labs, spans = 0, [], [], []
for g, gn in (("36", "CRC36 300"), ("33", "CRC33 100")):
    x0 = x
    for v in VN:
        for k, (p, f) in enumerate((("runner", False), ("official", True))):
            v_ = f2[g][v]["pq_lin"][p]
            eb(ax, x + (k - .5) * 2 * OFF, v_["mean"], *v_["ci95"], FS.INK, f, ms=FS.MS * .85)
        ticks.append(x)
        labs.append({"van": "Van", "grp": "Grp", "cat": "Cat", "gbp": "G+b", "pch": "Per-ch."}[v])
        x += 1
    for k, (p, ls) in enumerate((("runner", "--"), ("official", "-"))):
        r = f2[g]["ref_lin"][p]["mean"]
        ax.plot([x0 - .4, x - .6], [r, r], color=FS.GREY, lw=FS.LW_AX, ls=ls, zorder=1)
    spans.append(((x0 + x - 1) / 2, gn))
    x += .6
ax.set_xticks(ticks, labs, fontsize=FS.FS_K - .6, rotation=90)
for xc, gn in spans:
    ax.text(xc, -0.36, gn, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=FS.FS_K, color=FS.GREY)
ax.set_ylabel("PQ vs Lin mask")
ax.plot([], [], color=FS.GREY, lw=FS.LW_AX, ls="--", label="real H&E, runner")
ax.plot([], [], color=FS.GREY, lw=FS.LW_AX, ls="-", label="real H&E, official")
ax.set_ylim(-.02, .82)
ax.legend(loc="upper center", bbox_to_anchor=(.42, 1.04), handlelength=1.4, handletextpad=.3)
clean(ax)
FS.save(fig, sys.argv[2])
