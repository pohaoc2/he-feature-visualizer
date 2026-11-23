"""hfv_retrain3_cellvit_20261122 RESULTS figures (local; English text, Nat Commun 180 mm, sim-exp-validation figstyle).

  python fig_r3.py STATS_FIG2 STATS_ABL EXAMPLES OUTDIR
STATS_FIG2  r3stats.py output (crc36ho primary, crc33sec secondary)
STATS_ABL   evalstats.py output of the Concat ablation (abl_crc36ho, abl_crc33sec)
EXAMPLES    folder <tile>/{real, mask, a1_concat, production, a2_bypass_full_tme, a1_per_channel, a2_off_shelf}.png
fig_r3_main: a examples (CRC36 held-out); b per-model mean with 95 % interval of paper-style PQ, DICE (input mask =
reference; dashed line = the real H&E's own official CellViT on the same mask), LPIPS, style_hed for both sets; c
hematoxylin readouts generated - real H&E on the input mask (CRC36 held-out); d,e added-group effect of the Concat
ablation (PixCell summarize_added_group_effect_stats; positive = adding the group helps), paired and unpaired, CRC36
held-out; a dot marks an interval that excludes 0. fig_r3_abl33: the same heatmaps for the CRC33 secondary set.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/Users/pohaochiu/Documents/UW/bagherilab/sim-exp-validation/docs/experiments/mechanism_sink_20261026/report_20261110")
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

F2, AB, EX, OUT = (Path(x) for x in sys.argv[1:5])
f2, ab = json.loads(F2.read_text()), json.loads(AB.read_text())
V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
LAB = {"a1_concat": "Concat", "production": "Grouped", "a2_bypass_full_tme": "Grouped +\nbypass",
       "a1_per_channel": "Per-channel", "a2_off_shelf": "PixCell\noriginal"}
COL = {"a1_concat": "#0b76a0", "production": "#e97132", "a2_bypass_full_tme": "#7570B3", "a1_per_channel": "#4ea72e",
       "a2_off_shelf": "#9a9a9a"}
SETS = [("crc36ho", "CRC36 held-out", "o"), ("crc33sec", "CRC33 Fig 2 tiles", "s")]
MET = [("pq", "paper-style PQ\n(higher = better)"), ("dice", "DICE\n(higher = better)"),
       ("lpips", "LPIPS\n(lower = better)"), ("style_hed", "style_hed\n(lower = better)")]
RD = [("a", "(a) mask pixels\nstained"), ("b", "(b) stained pixels\non mask"), ("c", "(c) contrast\ninside - outside"),
      ("f1", "pixel F1")]
GROUPS = ["cell_types", "cell_state", "microenv", "vasculature"]
GLAB = {"cell_types": "cell types", "cell_state": "cell state", "microenv": "O$_2$ / glucose",
        "vasculature": "vasculature"}
AM = [("pq", "PQ"), ("dice", "DICE"), ("lpips", "LPIPS"), ("style_hed", "style_hed"), ("fud", "FUD")]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2, pad=1)


def heat(ax, st, title):
    z = np.full((len(GROUPS), len(AM)), np.nan)
    sig = np.zeros_like(z, bool)
    for i, g in enumerate(GROUPS):
        for j, (m, _) in enumerate(AM):
            e = st["added_effect"].get(f"{g}/{m}")
            if e is None:
                continue
            z[i, j] = e.get("pixcell_mean", e.get("mean"))
            sig[i, j] = bool(e.get("excludes_0")) and m != "fud"   # FUD: one number per image set, no interval
    zn = z / np.nanmax(np.abs(z), axis=0, keepdims=True)   # per-metric scale: colour = sign and relative size
    ax.imshow(zn, cmap="RdBu_r", norm=TwoSlopeNorm(0, -1, 1), aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            if np.isfinite(z[i, j]):
                ax.text(j, i, f"{z[i, j]:+.3f}", ha="center", va="center", fontsize=fs.FS_K * .85,
                        color="white" if abs(zn[i, j]) > .6 else fs.INK)
                if sig[i, j]:
                    ax.plot(j + .36, i - .3, "o", ms=1.6, color=fs.INK)
    ax.set_xticks(range(len(AM)), [a for _, a in AM])
    ax.set_yticks(range(len(GROUPS)), [GLAB[g] for g in GROUPS])
    ax.set_title(title, fontsize=fs.FS_T, pad=3)
    ax.tick_params(length=0, pad=1)


def main():
    tiles = sorted(p.name for p in EX.iterdir() if p.is_dir())[:3]
    W = fs.W2
    s = min((W - 0.55 - 6 * 0.03) / 7, 0.62)
    xa = (W - 7 * s - 6 * 0.03) / 2 + 0.1
    hA = 0.25 + 3 * s + 2 * 0.03
    hB, hC, hD = 1.25, 1.05, 1.0
    H = hA + 0.3 + hB + 0.35 + hC + 0.35 + hD + 0.25
    fig = plt.figure(figsize=(W, H))
    # a examples
    fs.letter(fig, 0.02, 0.02, "a")
    cols = ["real H&E", "input mask"] + [LAB[v].replace("\n", " ") for v in V5]
    for r, t in enumerate(tiles):
        for c, n in enumerate(["real", "mask"] + V5):
            ax = fs.axes(fig, xa + c * (s + 0.03), 0.25 + r * (s + 0.03), s, s)
            ax.imshow(np.asarray(Image.open(EX / t / f"{n}.png").convert("RGB")), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(fs.LW_AX * .5); sp.set_color(fs.GREY)
            if r == 0:
                ax.set_title(cols[c], fontsize=fs.FS_K, pad=2)
            if c == 0:
                ax.set_ylabel(t, fontsize=fs.FS_K * .85, labelpad=1)
    # b per-model means
    y0 = hA + 0.3
    fs.letter(fig, 0.02, y0 - 0.22, "b")
    pw = (W - 0.55 - 3 * 0.42) / 4
    for k, (m, lab) in enumerate(MET):
        ax = fs.axes(fig, 0.45 + k * (pw + 0.42), y0 + 0.05, pw, hB - 0.45)
        for si, (sk, sl, mk) in enumerate(SETS):
            if sk not in f2:
                continue
            for i, v in enumerate(V5):
                e = f2[sk]["models"][f"{v}/{m}"]
                x = i + (si - .5) * .3
                ax.errorbar(x, e["mean"], yerr=[[e["mean"] - e["ci95"][0]], [e["ci95"][1] - e["mean"]]], fmt=mk,
                            ms=2.6, color=COL[v], mfc=COL[v] if si == 0 else "white", mew=.7, elinewidth=.7, capsize=0)
            if m in ("pq", "dice"):
                ref = f2[sk]["real_ref"][m]["mean"]
                ax.axhline(ref, color=fs.INK, lw=.6, ls="--" if si == 0 else ":", zorder=0)
        ax.set_xticks(range(5), [LAB[v] for v in V5], rotation=90, fontsize=fs.FS_K * .9)
        ax.set_title(lab, fontsize=fs.FS_K, pad=2)
        ax.set_xlim(-.6, 4.6)
        if m == "dice":   # legend: marker = evaluation set, line = real H&E reference
            from matplotlib.lines import Line2D
            hs = [Line2D([], [], ls="", marker="o", ms=2.6, color=fs.GREY, mfc=fs.GREY, mew=.7),
                  Line2D([], [], ls="", marker="s", ms=2.6, color=fs.GREY, mfc="white", mew=.7),
                  Line2D([], [], color=fs.INK, lw=.6, ls="--"), Line2D([], [], color=fs.INK, lw=.6, ls=":")]
            ax.legend(hs, ["CRC36 held-out (n = 300)", "CRC33 Fig 2 tiles (n = 100)",
                           "real H&E, CRC36", "real H&E, CRC33"], loc="lower left", fontsize=fs.FS_K * .75,
                      frameon=False, handlelength=1.4, borderaxespad=.1, labelspacing=.25)
        style(ax)
    # c readouts
    y1 = y0 + hB + 0.35
    fs.letter(fig, 0.02, y1 - 0.22, "c")
    for k, (m, lab) in enumerate(RD):
        ax = fs.axes(fig, 0.45 + k * (pw + 0.42), y1 + 0.05, pw, hC - 0.45)
        for i, v in enumerate(V5):
            e = f2["crc36ho"]["readouts"][f"{v}/{m}"]
            ax.errorbar(i, e["mean"], yerr=[[e["mean"] - e["ci95"][0]], [e["ci95"][1] - e["mean"]]], fmt="o", ms=2.6,
                        color=COL[v], elinewidth=.7, capsize=0)
        ax.axhline(0, color=fs.INK, lw=.6, zorder=0)
        ax.set_xticks(range(5), [LAB[v] for v in V5], rotation=90, fontsize=fs.FS_K * .9)
        ax.set_title(lab, fontsize=fs.FS_K, pad=2)
        ax.set_xlim(-.6, 4.6)
        if k == 0:
            ax.set_ylabel("generated - real H&E", fontsize=fs.FS_K)
        style(ax)
    # d, e ablation
    y2 = y1 + hC + 0.35
    hw = (W - 0.9 - 0.8) / 2
    for b, mode in enumerate(("paired", "unpaired")):
        fs.letter(fig, 0.02 + b * (hw + 0.95), y2 - 0.05, "de"[b])
        ax = fs.axes(fig, 0.9 + b * (hw + 0.7), y2 + 0.12, hw, hD - 0.3)
        heat(ax, ab["abl_crc36ho"][mode], f"Concat ablation, {mode} UNI (CRC36 held-out, n = "
                                          f"{ab['abl_crc36ho'][mode]['n_tiles']})")
    fs.save(fig, OUT / "fig_r3_main.png")
    if "abl_crc33sec" in ab:
        fig = plt.figure(figsize=(W, 1.25))
        for b, mode in enumerate(("paired", "unpaired")):
            fs.letter(fig, 0.02 + b * (hw + 0.95), 0.02, "ab"[b])
            ax = fs.axes(fig, 0.9 + b * (hw + 0.7), 0.25, hw, 0.75)
            heat(ax, ab["abl_crc33sec"][mode], f"Concat ablation, {mode} UNI (CRC33, n = "
                                               f"{ab['abl_crc33sec'][mode]['n_tiles']})")
        fs.save(fig, OUT / "fig_r3_abl33.png")
    print("wrote", OUT)


main()
