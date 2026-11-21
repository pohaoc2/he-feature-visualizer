"""hfv_retrain_20261118 追加 3: validation and selection for the decodability rerun (README 追加 3 decisions 3-4, written
before any probe result).

  python a3summ.py DEC_DIR PAPER_RESULTS_DIR OUT_JSON R2NEW_DIR

DEC_DIR            hfv_a3_20261120/dec (targets/, probe/, paper/)
PAPER_RESULTS_DIR  S3 share_space/05_results/results/a1_probe_mlp_spatial copy (uni_16, ctranspath_07, virchow2_16, resnet50_07,
                   uni_16_crc34 with mlp_spatial_probe_results.csv)
4(a) old targets rebuilt from the old channels vs the paper's mask_targets_T1_spatial.npy (rows matched by tile id):
     max |difference| per target, NaN positions compared separately.
4(c) old targets + re-extracted features on the full paper set vs the paper CSV: per-target r2_within_mean difference.
     Rule (no threshold): for UNI and CTransPath the max_train_rows setting (500,000 or 50,000) with the smaller median
     |difference| over the 10 targets is used for the kept-set runs; ties -> 500,000 (the program default).
4(d) old vs new targets on the same kept tiles and features: r2_within_mean per target, side by side.
R2NEW_DIR receives one folder per paper name (uni_16, ctranspath_07, virchow2_16, resnet50_07, uni_16_crc34) holding the
selected new-target run (mlp_spatial_probe_results.{csv,json}, manifest.json, cv_splits.json if present) for the render.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

dec, paper, out_json, r2new = map(Path, sys.argv[1:5])
res = {}


def rows(p):
    return {r["target"]: r for r in csv.DictReader(open(p))}


def ids(p):
    return [l.strip() for l in open(p) if l.strip()]


# 4(a)
pt = np.load(dec / "paper" / "targets16_mask_targets_T1_spatial.npy")
pid = {t: i for i, t in enumerate(ids(dec / "paper" / "targets16_tile_ids.txt"))}
names = json.load(open(dec / "paper" / "targets16_target_names_T1_spatial.json"))
mine = np.load(dec / "targets" / "old16_800" / "mask_targets_T1_spatial.npy")
mid = ids(dec / "targets" / "old16_800" / "tile_ids.txt")
common = [i for i, t in enumerate(mid) if t in pid]
ref = pt[[pid[mid[i]] for i in common]]
mine = mine[common]
a = {}
for j, n in enumerate(names):
    x, y = mine[..., j], ref[..., j]
    nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum())
    ok = ~np.isnan(x) & ~np.isnan(y)
    a[n] = {"max_abs_diff": float(np.abs(x[ok] - y[ok]).max()) if ok.any() else None, "nan_mismatch": nan_mismatch}
res["4a_old_targets_vs_paper"] = {"n_tiles_rebuilt": len(mid), "n_tiles_in_paper_file": len(common), "per_target": a}

# 4(c)
enc = {"uni": "uni_16", "ctp": "ctranspath_07", "vir": "virchow2_16", "res": "resnet50_07"}
full = {"uni": "old800", "ctp": "old800", "vir": "old200", "res": "old200"}
res["4c_reproduction"], chosen = {}, {}
for k, pname in enc.items():
    pap = rows(paper / pname / "mlp_spatial_probe_results.csv")
    settings = ["_r500000", "_r50000"] if k in ("uni", "ctp") else [""]
    per = {}
    for s in settings:
        f = dec / "probe" / f"{k}_{full[k]}{s}" / "mlp_spatial_probe_results.csv"
        if not f.is_file():
            per[s or "_handover"] = None
            continue
        me = rows(f)
        d = {t: float(me[t]["r2_within_mean"]) - float(pap[t]["r2_within_mean"]) for t in pap}
        per[s or "_handover"] = {"diff": d, "median_abs_diff": float(np.median(np.abs(list(d.values())))),
                                 "mine": {t: float(me[t]["r2_within_mean"]) for t in pap},
                                 "paper": {t: float(pap[t]["r2_within_mean"]) for t in pap}}
    res["4c_reproduction"][pname] = per
    if k in ("uni", "ctp"):
        ok = {s: v["median_abs_diff"] for s, v in per.items() if v}
        chosen[k] = (min(ok, key=lambda s: (ok[s], s != "_r500000")) if ok else None)
    else:
        chosen[k] = ""
res["chosen_setting"] = chosen

# 4(d) and R2NEW
res["4d_old_vs_new_same_tiles"] = {}
kept = {"uni": "715", "ctp": "715", "vir": "184", "res": "184"}
r2new.mkdir(parents=True, exist_ok=False)
for k, pname in enc.items():
    s = chosen[k]
    if s is None:
        continue
    o = dec / "probe" / f"{k}_old{kept[k]}{s}" / "mlp_spatial_probe_results.csv"
    n = dec / "probe" / f"{k}_new{kept[k]}{s}"
    if not (o.is_file() and (n / "mlp_spatial_probe_results.csv").is_file()):
        res["4d_old_vs_new_same_tiles"][pname] = None
        continue
    ro, rn = rows(o), rows(n / "mlp_spatial_probe_results.csv")
    res["4d_old_vs_new_same_tiles"][pname] = {"run": n.name, "old": {t: float(ro[t]["r2_within_mean"]) for t in ro},
                                              "new": {t: float(rn[t]["r2_within_mean"]) for t in rn},
                                              "new_sd": {t: float(rn[t]["r2_within_sd"]) for t in rn}}
    dst = r2new / pname
    dst.mkdir()
    for f in ("mlp_spatial_probe_results.csv", "mlp_spatial_probe_results.json", "manifest.json", "cv_splits.json"):
        if (n / f).is_file():
            shutil.copy2(n / f, dst / f)
c34 = dec / "probe" / "c34_new_r500000"
if (c34 / "mlp_spatial_probe_results.csv").is_file():
    dst = r2new / "uni_16_crc34"
    dst.mkdir()
    for f in ("mlp_spatial_probe_results.csv", "mlp_spatial_probe_results.json", "manifest.json"):
        if (c34 / f).is_file():
            shutil.copy2(c34 / f, dst / f)
    rn, pap = rows(c34 / "mlp_spatial_probe_results.csv"), rows(paper / "uni_16_crc34" / "mlp_spatial_probe_results.csv")
    res["c34_new_vs_paper"] = {t: {"new": float(rn[t]["r2_within_mean"]), "paper": float(pap[t]["r2_within_mean"]),
                                   "new_delta_shuffle": rn[t].get("delta_shuffle"),
                                   "paper_delta_shuffle": pap[t].get("delta_shuffle")} for t in pap}
json.dump(res, open(out_json, "w"), indent=1)
print(json.dumps({"chosen": chosen, "4a": res["4a_old_targets_vs_paper"]}, indent=1))
for pname, per in res["4c_reproduction"].items():
    for s, v in per.items():
        if v:
            print(pname, s, "median|diff|", round(v["median_abs_diff"], 4))
