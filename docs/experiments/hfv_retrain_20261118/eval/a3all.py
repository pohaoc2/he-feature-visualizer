"""hfv_retrain_20261118 追加 3b: probes with all training rows (user decision 2026-11-21, README 追加 3b, written before
any all-rows result).

  python a3all.py rows DEC_DIR OUT_JSON           finite training rows per fold for every run that enters r2all; STOP (exit 1)
                                                  if a reused run's cap was smaller than its largest fold (cap would have bound)
  python a3all.py summ DEC_DIR ROWS_JSON OUT_JSON R2ALL_DIR

DEC_DIR: targets/<t>/{mask_targets_T1_spatial.npy,tile_ids.txt}, cv/*.json, paper/cv_*.json, paper/results/<name>/ (paper CSV),
probe/<run>/ (追加 3 runs from dec_results.tar), proball/<run>/ (this job's runs).
r2all: uni_16 = probe/uni_new715_r500000, ctranspath_07 = probe/ctp_new715_r500000, resnet50_07 = probe/res_new184 (reused,
       each must satisfy max finite rows per fold <= its cap), virchow2_16 = proball/vir_new184_all,
       uni_16_crc34 = proball/c34_new_all.
Summary (descriptive, no threshold): per target r2_within_mean (sd over folds) for all rows vs capped vs paper; C34 also
delta_shuffle.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

# run -> (targets dir, cv json relative to DEC_DIR, cap used by that run or None = all rows)
RUNS = {
    "probe/uni_new715_r500000": ("new16_715", "cv/cv_uni_16_kept.json", 500000),
    "probe/ctp_new715_r500000": ("new7_715", "cv/cv_ctranspath_07_kept.json", 500000),
    "probe/res_new184": ("new7_184", "cv/cv_resnet50_07_kept.json", 10000),
    "probe/res_old200": ("old7_200", "paper/cv_resnet50_07.json", 10000),
    "probe/vir_new184": ("new16_184", "cv/cv_virchow2_16_kept.json", 10000),
    "probe/vir_old200": ("old16_200", "paper/cv_virchow2_16.json", 10000),
    "probe/c34_new_r500000": ("new16_c34", "cv/cv_uni_16_crc34_kept.json", 500000),
    "proball/vir_new184_all": ("new16_184", "cv/cv_virchow2_16_kept.json", None),
    "proball/vir_old184_all": ("old16_184", "cv/cv_virchow2_16_kept.json", None),
    "proball/vir_old200_all": ("old16_200", "paper/cv_virchow2_16.json", None),
    "proball/c34_new_all": ("new16_c34", "cv/cv_uni_16_crc34_kept.json", None),
}
REUSED = ("probe/uni_new715_r500000", "probe/ctp_new715_r500000", "probe/res_new184")
R2ALL = {"uni_16": "probe/uni_new715_r500000", "ctranspath_07": "probe/ctp_new715_r500000",
         "resnet50_07": "probe/res_new184", "virchow2_16": "proball/vir_new184_all", "uni_16_crc34": "proball/c34_new_all"}


def rows_mode(dec, out):
    res, bad = {}, []
    for run, (tg, cv, cap) in RUNS.items():
        Y = np.load(dec / "targets" / tg / "mask_targets_T1_spatial.npy", mmap_mode="r")
        names = json.load(open(dec / "targets" / tg / "target_names_T1_spatial.json"))
        if Y.ndim != 3 or Y.shape[2] != len(names):
            raise SystemExit(f"STOP unexpected target shape {tg} {Y.shape}")
        splits = json.load(open(dec / cv))["splits"]
        per = []
        for s in splits:
            tr = np.asarray(s["train_idx"], dtype=np.int64)
            fin = np.isfinite(np.asarray(Y[tr], dtype=np.float32)).sum(axis=(0, 1))
            per.append({n: int(v) for n, v in zip(names, fin)})
        mx = max(max(p.values()) for p in per)
        res[run] = {"targets": tg, "cv": cv, "cap": cap, "n_patches": int(Y.shape[1]), "max_finite_train_rows": mx,
                    "min_finite_train_rows": min(min(p.values()) for p in per), "cap_binds": bool(cap is not None and mx > cap),
                    "per_fold": per}
        print(run, "cap", cap, "rows", res[run]["min_finite_train_rows"], "-", mx, "binds" if res[run]["cap_binds"] else "")
        if run in REUSED and res[run]["cap_binds"]:
            bad.append(run)
    json.dump(res, open(out, "w"), indent=1)
    if bad:
        raise SystemExit(f"STOP reused runs whose cap bound: {bad}")


def csvrows(p):
    return {r["target"]: r for r in csv.DictReader(open(p))}


def summ_mode(dec, rows_json, out, r2all):
    res = {"rows": json.load(open(rows_json))}
    r2all.mkdir(parents=True, exist_ok=False)
    for name, run in R2ALL.items():
        src = dec / run
        if not (src / "mlp_spatial_probe_results.csv").is_file():
            raise SystemExit(f"STOP missing {src}/mlp_spatial_probe_results.csv")
        (r2all / name).mkdir()
        for f in ("mlp_spatial_probe_results.csv", "mlp_spatial_probe_results.json", "manifest.json", "cv_splits.json"):
            if (src / f).is_file():
                shutil.copy2(src / f, r2all / name / f)
    cmp = {"c34": ("proball/c34_new_all", "probe/c34_new_r500000", "uni_16_crc34"),
           "virchow2_new184": ("proball/vir_new184_all", "probe/vir_new184", None),
           "virchow2_old184": ("proball/vir_old184_all", "probe/vir_old184", None),
           "virchow2_old200": ("proball/vir_old200_all", "probe/vir_old200", "virchow2_16"),
           "uni_c33_new715": ("probe/uni_new715_r500000", "probe/uni_new715_r50000", None)}
    for key, (a, b, pap) in cmp.items():
        ra, rb = csvrows(dec / a / "mlp_spatial_probe_results.csv"), csvrows(dec / b / "mlp_spatial_probe_results.csv")
        rp = csvrows(dec / "paper" / "results" / pap / "mlp_spatial_probe_results.csv") if pap else {}
        tab = {}
        for t in ra:
            tab[t] = {"all": float(ra[t]["r2_within_mean"]), "all_sd": float(ra[t]["r2_within_sd"]),
                      "capped": float(rb[t]["r2_within_mean"]), "capped_sd": float(rb[t]["r2_within_sd"]),
                      "paper": float(rp[t]["r2_within_mean"]) if t in rp else None}
            for k, r in (("all", ra), ("capped", rb), ("paper", rp)):
                if t in r and r[t].get("delta_shuffle") not in (None, ""):
                    tab[t][f"{k}_delta_shuffle"] = float(r[t]["delta_shuffle"])
        res[key] = {"all_run": a, "capped_run": b, "paper": pap, "per_target": tab}
        print(f"== {key}: all={a} capped={b} paper={pap}")
        for t, v in tab.items():
            print(f"  {t:18s} all {v['all']:+.3f} ({v['all_sd']:.3f})  capped {v['capped']:+.3f}  paper "
                  f"{'' if v['paper'] is None else format(v['paper'], '+.3f')}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "rows":
        rows_mode(Path(sys.argv[2]), Path(sys.argv[3]))
    elif m == "summ":
        summ_mode(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]))
    else:
        raise SystemExit(__doc__)
