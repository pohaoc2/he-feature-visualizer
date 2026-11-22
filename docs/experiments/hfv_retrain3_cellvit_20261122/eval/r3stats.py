"""hfv_retrain3_cellvit_20261122 Fig 2 statistics (PREREG §5.3-5.4), CPU, scores nothing itself.

  python r3stats.py OUT.json NAME:GROUP_DIR:EXTRA_CSV [...]
GROUP_DIR  scored Fig 2 group (cache.json + fig2_per_tile.csv from score3.slurm: evalscore3 + fig2tiles)
EXTRA_CSV  r3extra.py output for the same group (hematoxylin readouts, real-H&E reference PQ / DICE)
Per evaluation set, one shared resample index set (numpy default_rng(42), 10,000 resamples of tiles with replacement,
95 % percentile interval) for every quantity, so every difference is paired (hfv_retrain_20261118 evalstats.py boot()).
  models       per model: per-tile mean with interval of PQ / DICE / LPIPS / style_hed; FUD from cache.json (one number)
  real_ref     real H&E's own official CellViT vs the input mask: PQ / DICE with interval
  readouts     per model: a / b / c / f1 generated - real H&E (same mask), per-tile paired difference with interval
  pairs        all 10 model pairs (A - B) for PQ / DICE / LPIPS / style_hed / dlpips_text
               (dlpips_text of a variant = LPIPS(PixCell original) - LPIPS(variant), pairs only among the 4 variants)
  dlpips_code  per variant: sensitivity_eval LPIPS (mean over the 4 groups) with interval, and per group
  ranking      PREREG §5.4 on PQ, DICE (higher better) and LPIPS, style_hed (lower better): A before B iff at least
               one A - B interval lies wholly on A's better side and none lies wholly on B's better side; both ways =
               trade-off; neither = tie. Reported for every set; only the primary set (crc36ho) is the Fig 2 rule.
"""
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

B, SEED = 10000, 42
V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
NAME = {"a1_concat": "Concat", "production": "Grouped", "a2_bypass_full_tme": "Grouped + bypass",
        "a1_per_channel": "Per-channel", "a2_off_shelf": "PixCell original"}
M4 = {"pq": 1, "dice": 1, "lpips": -1, "style_hed": -1}   # +1 higher better, -1 lower better
RD = ["a", "b", "c", "f1"]


def boot(arrs, idx):
    out = {}
    for k, x in arrs.items():
        x = np.asarray(x, dtype=np.float64)
        ok = np.isfinite(x)
        if not ok.all():   # a readout undefined on a tile (empty mask): drop that tile from this quantity only
            bm = np.array([np.nanmean(x[i]) for i in idx])
        else:
            bm = x[idx].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        out[k] = dict(mean=float(np.nanmean(x)), ci95=[float(lo), float(hi)], n=int(ok.sum()),
                      excludes_0=bool(lo > 0 or hi < 0))
    return out


def one(g, extra):
    cache = json.load(open(g / "cache.json"))
    rows = list(csv.DictReader(open(g / "fig2_per_tile.csv")))
    tiles = sorted({r["tile"] for r in rows if r["variant"] == "a2_off_shelf"})
    val = {(r["variant"], r["tile"]): r for r in rows}
    ex = {(r["model"], r["tile"]): r for r in csv.DictReader(open(extra))}
    miss = [t for t in tiles if ("real", t) not in ex or any((v, t) not in val for v in V5)]
    if miss:
        sys.exit(f"STOP {g}: {len(miss)} tiles missing a model or the real-H&E row, e.g. {miss[:3]}")
    idx = np.random.default_rng(SEED).integers(0, len(tiles), size=(B, len(tiles)))
    f = lambda d, k: float(d[k]) if d[k] not in ("", "nan") else np.nan
    per = {(v, m): np.array([f(val[(v, t)], m) for t in tiles]) for v in V5 for m in M4}
    for v in V5[:4]:
        per[(v, "dlpips_text")] = per[("a2_off_shelf", "lpips")] - per[(v, "lpips")]
    res = dict(n_tiles=len(tiles))
    res["models"] = boot({f"{v}/{m}": per[(v, m)] for v in V5 for m in M4}, idx)
    for v in V5:
        res["models"][f"{v}/fud"] = dict(mean=cache["metrics"][v].get("fud"))
    res["real_ref"] = boot({k: [f(ex[("real", t)], f"{k}_real") for t in tiles] for k in ("pq", "dice")}, idx)
    rd = {}
    for v in V5:
        for k in RD:
            rd[f"{v}/{k}"] = [f(ex[(v, t)], k) - f(ex[("real", t)], k) for t in tiles]
    for k in RD:
        rd[f"real/{k}"] = [f(ex[("real", t)], k) for t in tiles]
    res["readouts"] = boot(rd, idx)
    pr = {}
    for a, b in itertools.combinations(V5, 2):
        for m in M4:
            pr[f"{a}-{b}/{m}"] = per[(a, m)] - per[(b, m)]
        if b != "a2_off_shelf":
            pr[f"{a}-{b}/dlpips_text"] = per[(a, "dlpips_text")] - per[(b, "dlpips_text")]
    res["pairs"] = boot(pr, idx)
    res["vs_vanilla_dlpips_text"] = boot({v: per[(v, "dlpips_text")] for v in V5[:4]}, idx)
    sens = {}
    for v in V5[:4]:
        pg = cache["sensitivity"][v]["per_group"]
        lens = {k: len(x["per_tile"]) for k, x in pg.items()}
        if set(lens.values()) != {len(tiles)}:
            sys.exit(f"STOP {g} {v}: sensitivity per-tile counts {lens} != {len(tiles)}")
        sens[v] = boot({"mean_over_groups": np.mean([pg[k]["per_tile"] for k in pg], axis=0),
                        **{k: pg[k]["per_tile"] for k in pg}}, idx)
        sens[v]["pixcell_mean"] = cache["sensitivity"][v]["mean"]
    res["dlpips_code"] = sens
    rel = {}
    for a, b in itertools.combinations(V5, 2):
        aw, bw = [], []
        for m, s in M4.items():
            lo, hi = res["pairs"][f"{a}-{b}/{m}"]["ci95"]
            if s * lo > 0 and s * hi > 0:
                aw.append(m)          # whole interval on A's better side
            elif s * lo < 0 and s * hi < 0:
                bw.append(m)          # whole interval on B's better side
        if aw and bw:
            first, kind = None, "trade-off"
        elif aw or bw:
            first, kind = (a, b) if aw else (b, a), "ordered"
        else:
            first, kind = None, "tie"
        rel[f"{a}|{b}"] = dict(kind=kind, first=None if first is None else first[0],
                               second=None if first is None else first[1], better_A=aw, better_B=bw)
    res["ranking"] = rel
    res["ranking_counts"] = {v: dict(before=sum(r["first"] == v for r in rel.values()),
                                     after=sum(r["second"] == v for r in rel.values())) for v in V5}
    return res


out = {}
for task in sys.argv[2:]:
    name, g, extra = task.split(":", 2)
    out[name] = dict(dir=g, **one(Path(g), Path(extra)))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print("wrote", sys.argv[1])
