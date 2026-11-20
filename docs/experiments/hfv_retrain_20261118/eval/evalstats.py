"""hfv_retrain_20261118 evaluation statistics (CPU; runs on the scored outputs, scores nothing itself).

  PYTHONPATH=<PixCell cb62fa7> python evalstats.py OUT.json fig2:NAME:DIR abl:NAME:DIR ...
fig2 DIR   scored Fig 2 group (cache.json from metrics_io compute + sensitivity_eval score, fig2_per_tile.csv)
abl DIR    scored ablation group (DIR/{paired,unpaired}_ablation/ablation_results/<tile>/metrics.json, fud_scores.json)
loo DIR:ROOT  scored sub-channel LOO group; ROOT = the orion root whose exp_channels/cell_masks is the PQ reference
           (the Lin pack masks, or a realref.py masks root for the real-H&E reference)
All intervals are 95 % percentile bootstrap intervals over tiles (tiles resampled with replacement, the same resample for
every quantity that is a difference between models or conditions on the same tiles; numpy default_rng(42), 10,000
resamples). An interval that contains 0 reads "cannot distinguish" (README ablation rule 2). FUD is one number per image
set (Fréchet distance between two feature distributions), so it gets no per-tile interval here.
Fig 2:
  metrics          per variant: FUD (cache.json), mean and std of LPIPS / PQ / DICE / HED (cache.json, metrics_io)
  vs_vanilla       per variant, per metric: mean over tiles of (variant - Vanilla) with interval; for LPIPS this sign
                   flipped is the README text definition of ΔLPIPS: LPIPS(Vanilla, real) - LPIPS(variant, real),
                   positive = the variant is closer to the real H&E than Vanilla
  dlpips_code      per variant: PixCell's ΔLPIPS (sensitivity_eval: LPIPS between the fully conditioned image and the
                   image with one channel group zeroed, mean over the 4 groups) with an interval over tiles, and per group
Ablation (per mode paired / unpaired):
  condition_means  PixCell's tools/ablation_report/data.py load_condition_means_from_paths (+ fud_scores.json)
  added_effect     PixCell's summarize_added_group_effect_stats definition (the Fig 2e/g heatmap): for group g, the mean
                   over every context subset S of the other groups of metric(S + g) - metric(S), sign so that positive =
                   better. Per tile the same mean of differences is taken, so the PixCell value is the mean over tiles,
                   and the interval is the bootstrap over tiles of that per-tile value. FUD: PixCell value only.
Sub-channel LOO:
  pq_drop          per sub-channel: per tile PQ(all_baseline) - PQ(sub-channel left out), the quantity of PixCell's
                   tools/stage3/aggregate_subchannel_layout_metrics.py (same run_cellvit / _load_gt_instance_mask /
                   _compute_pq calls, same tile and sub-channel skipping), mean with interval; positive = leaving the
                   sub-channel out lowers PQ
  pixel            per sub-channel: the per-tile subchannel_loo_diff_stats.json measures that PixCell's
                   aggregate_subchannel_loo.py averages (mean_diff, delta_e_mean, delta_e_p99, pct_pixels_above_10),
                   mean with interval
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

from tools.ablation_report.data import load_condition_means_from_paths, load_fud_scores, \
    summarize_added_group_effect_stats
from tools.ablation_report.shared import DEFAULT_METRIC_ORDER

B, SEED = 10000, 42
V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
M4 = ["lpips", "pq", "dice", "style_hed"]


def boot(per_tile_arrays):
    """per_tile_arrays: dict name -> 1-D array over the same tiles; one shared resample index set"""
    n = len(next(iter(per_tile_arrays.values())))
    idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
    out = {}
    for k, x in per_tile_arrays.items():
        x = np.asarray(x, dtype=np.float64)
        bm = x[idx].mean(axis=1)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        out[k] = dict(mean=float(x.mean()), ci95=[float(lo), float(hi)], n=int(n),
                      excludes_0=bool(lo > 0 or hi < 0))
    return out


def fig2(d):
    cache = json.load(open(d / "cache.json"))
    rows = list(csv.DictReader(open(d / "fig2_per_tile.csv")))
    tiles = sorted({r["tile"] for r in rows if r["variant"] == "a2_off_shelf"})
    val = {(r["variant"], r["tile"]): r for r in rows}
    res = dict(n_tiles=len(tiles), metrics={v: cache["metrics"][v] for v in V5})
    arrs = {}
    for v in V5[:4]:
        for m in M4:
            arrs[f"{v}/{m}"] = [float(val[(v, t)][m]) - float(val[("a2_off_shelf", t)][m]) for t in tiles]
        arrs[f"{v}/dlpips_text"] = [-x for x in arrs[f"{v}/lpips"]]
    res["vs_vanilla"] = boot(arrs)
    sens = {}
    for v in V5[:4]:
        s = cache["sensitivity"][v]
        pg = s["per_group"]
        lens = {g: len(x["per_tile"]) for g, x in pg.items()}
        if set(lens.values()) != {len(tiles)}:
            sys.exit(f"STOP {d.name} {v}: sensitivity per-tile counts {lens} != {len(tiles)} tiles")
        per_tile_mean = np.mean([pg[g]["per_tile"] for g in pg], axis=0)
        b = boot({"mean_over_groups": per_tile_mean, **{g: pg[g]["per_tile"] for g in pg}})
        b["pixcell_mean"] = s["mean"]
        sens[v] = b
    res["dlpips_code"] = sens
    return res


def abl(d):
    out = {}
    for mode in ("paired", "unpaired"):
        R = d / f"{mode}_ablation" / "ablation_results"
        paths = sorted(R.glob("*/metrics.json"))
        cm, n, keys = load_condition_means_from_paths(paths)
        fud = load_fud_scores(R)
        for k, v in fud.items():
            cm.setdefault(k, {})["fud"] = float(v)
        keys_all = [k for k in DEFAULT_METRIC_ORDER if k in keys or (k == "fud" and fud)]
        pixcell = summarize_added_group_effect_stats(cm, keys_all)
        per_tile = {}   # (group, metric) -> list over tiles
        for p in paths:
            pc = json.load(open(p)).get("per_condition") or {}
            tm = {c: {m: float(r[m]) for m in keys if r.get(m) is not None} for c, r in pc.items() if isinstance(r, dict)}
            eff = summarize_added_group_effect_stats(tm, keys)
            for g, mm in eff.items():
                for m, (mean, _sd) in mm.items():
                    per_tile.setdefault((g, m), []).append(mean)
        bad = {f"{g}/{m}": len(x) for (g, m), x in per_tile.items() if len(x) != n}
        if bad:
            sys.exit(f"STOP {d.name} {mode}: per-tile added effects missing for some tiles {bad} (n {n})")
        ci = boot({f"{g}/{m}": x for (g, m), x in per_tile.items()})
        for g, mm in pixcell.items():
            for m, (mean, sd) in mm.items():
                k = f"{g}/{m}"
                ci.setdefault(k, {})
                ci[k]["pixcell_mean"], ci[k]["pixcell_std_over_contexts"] = mean, sd
        out[mode] = dict(n_tiles=n, condition_means=cm, added_effect=ci)
    return out


def loo(d, root):
    from tools.compute_ablation_metrics import _compute_pq, _load_gt_instance_mask, run_cellvit
    from tools.stage3.aggregate_subchannel_layout_metrics import SUBCHANNELS
    drop = {s: [] for s in SUBCHANNELS}
    pix = {}
    n_tiles = 0
    for td in sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("_")):
        base = td / "all_baseline.png"
        if not base.is_file():
            continue
        try:
            bp = run_cellvit(base)
            gt = _load_gt_instance_mask(root, td.name, shape=bp.shape)
            bpq = _compute_pq(gt, bp)[2]
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        used = False
        for s in SUBCHANNELS:
            ip = td / s / "generated_he.png"
            if not ip.is_file():
                continue
            try:
                lpq = _compute_pq(gt, run_cellvit(ip))[2]
            except (FileNotFoundError, RuntimeError, ValueError):
                continue
            drop[s].append(float(bpq - lpq))
            used = True
        n_tiles += used
        st = td / "subchannel_loo_diff_stats.json"
        if st.is_file():
            for s, mm in json.load(open(st)).items():
                for m in ("mean_diff", "delta_e_mean", "delta_e_p99", "pct_pixels_above_10"):
                    if mm.get(m) is not None:
                        pix.setdefault(f"{s}/{m}", []).append(float(mm[m]))
    lens = {len(x) for x in drop.values()}
    if len(lens) != 1:
        sys.exit(f"STOP {d.name}: sub-channels have different tile counts {[(s, len(x)) for s, x in drop.items()]}")
    plens = {len(x) for x in pix.values()}
    out = dict(n_tiles=n_tiles, reference_root=str(root), pq_drop=boot(drop))
    out["pixel"] = {} if not pix else (boot(pix) if len(plens) == 1 else
                                        {k: boot({k: x})[k] for k, x in pix.items()})
    return out


res = {}
for task in sys.argv[2:]:
    kind, name, d = task.split(":", 2)
    if kind == "loo":
        d, root = d.split(":", 1)
        res[name] = dict(kind=kind, dir=d, **loo(Path(d), Path(root)))
    else:
        res[name] = dict(kind=kind, dir=d, **(fig2(Path(d)) if kind == "fig2" else abl(Path(d))))
json.dump(res, open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]}")
