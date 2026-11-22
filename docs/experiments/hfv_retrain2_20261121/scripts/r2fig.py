"""hfv_retrain2 stage 3 figure (RESULTS_stage3.md): 180 mm, shared figstyle (sim-exp-validation report_20261110).

  python r2fig.py STATS_R1 APP8_JSON OUT_PNG
a  mask fidelity = model - real-H&E reference line (secondary PQ against the model's own training-style mask; detection F1
   against the Lin centroids), per set, model K and model N, 95 % tile-bootstrap intervals
b  readout 3: delta (canc_neo - heal_neo) per arm, real H&E on both unit sets, 95 % window-bootstrap intervals
c  the six decision intervals of PREREG §9.5 as N - K paired differences
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "sim-exp-validation/docs/experiments/mechanism_sink_20261026/report_20261110"))
import figstyle as FS  # noqa: E402

plt = FS.plt
r1 = json.loads(Path(sys.argv[1]).read_text())
a8 = json.loads(Path(sys.argv[2]).read_text())
CK, CN = FS.C["old"], FS.C["new"]
SETS = [("crc36ho", "CRC36 held-out"), ("crc33sec", "CRC33 Fig 2")]

fig = plt.figure(figsize=(FS.W2, 2.05))
# a
ax = FS.axes(fig, 0.42, 0.22, 2.05, 1.45)
FS.letter(fig, 0.02, 0.05, "a")
labels, x = [], 0
for q, qn in (("fid_pq", "PQ"), ("fid_f1", "F1")):
    for s, sn in SETS:
        for k, (m, c) in enumerate((("K", CK), ("N", CN))):
            v = r1["sets"][s]["per_model"][m][q]
            ax.errorbar(x + (k - .5) * .28, v["mean"], yerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]],
                        fmt="o", color=c, ms=FS.MS, lw=FS.LW, capsize=0)
        labels.append(f"{qn}\n{sn.split()[0]}")
        x += 1
ax.axhline(0, color=FS.GREY, lw=FS.LW_AX, ls=":")
ax.set_xticks(range(len(labels)), labels)
ax.set_ylabel("model − real H&E")
ax.spines[["top", "right"]].set_visible(False)
ax.plot([], [], "o", color=CK, label="model K (R0 mask)")
ax.plot([], [], "o", color=CN, label="model N (L1 mask)")
ax.legend(loc="best", handletextpad=.2)
# b
ax = FS.axes(fig, 2.95, 0.22, 1.75, 1.45)
FS.letter(fig, 2.62, 0.05, "b")
arms = [("R_L1", "real\n(L1 units)", FS.INK), ("NP", "N own", CN), ("ND", "N other", CN),
        ("R_s3b", "real\n(s3b units)", FS.INK), ("KP", "K own", CK), ("KD", "K other", CK)]
for i, (a, an, c) in enumerate(arms):
    p, ci = a8["point"][a]["delta"], a8["ci95"][a]["delta"]
    ax.errorbar(i, p, yerr=[[p - ci[0]], [ci[1] - p]], fmt="o", color=c, ms=FS.MS, lw=FS.LW, capsize=0)
ax.axhline(0, color=FS.GREY, lw=FS.LW_AX, ls=":")
ax.set_xticks(range(len(arms)), [a[1] for a in arms])
ax.set_ylabel("Δ (cancer − healthy,\nneoplastic call rate)")
ax.spines[["top", "right"]].set_visible(False)
# c
ax = FS.axes(fig, 5.55, 0.22, 1.4, 1.45)
FS.letter(fig, 4.98, 0.05, "c")
rows = [(r1["sets"][s]["n_minus_k"][q], f"{qn} {sn.split()[0]}") for q, qn in (("fid_pq", "PQ"), ("fid_f1", "F1"))
        for s, sn in SETS]
rows += [(dict(mean=a8["n_minus_k"][f"{k}:delta"]["point"], ci95=a8["n_minus_k"][f"{k}:delta"]["ci95"]), f"Δ {nm}")
         for k, nm in (("NP-KP", "own"), ("ND-KD", "other"))]
for i, (v, nm) in enumerate(rows[::-1]):
    ax.errorbar(v["mean"], i, xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt="o", color=FS.INK,
                ms=FS.MS, lw=FS.LW, capsize=0)
ax.axvline(0, color=FS.GREY, lw=FS.LW_AX, ls=":")
ax.set_yticks(range(len(rows)), [r[1] for r in rows[::-1]])
ax.set_xlabel("N − K")
ax.spines[["top", "right"]].set_visible(False)
FS.save(fig, sys.argv[3])
print(sys.argv[3])
