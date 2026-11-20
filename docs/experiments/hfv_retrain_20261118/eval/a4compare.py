"""hfv_retrain_20261118 follow-up 1: descriptive comparison of the Fig 3e specificity square (targeted Pearson r between
the edit strength alpha and each attribute's primary metric, PixCell figures.ATTR_TO_PRIMARY_METRIC) between the paper's
sweep (old Concat, 30 tiles; specificity_full.csv recomputed from the paper's sweep metrics with PixCell cb62fa7
run_specificity, which reproduced the published panel) and the rerun with the retrained Concat (28 tiles).

  python a4compare.py OLD_specificity_full.csv NEW_specificity_full.csv OUT.json
No test and no threshold: per-cell r, abs_ratio (|targeted slope| / |random slope|), the diagonal (edited attribute vs
its own metric), and for each row whether the diagonal is the largest |r| in that row.
"""
import csv
import json
import sys

ATTRS = ["eccentricity_mean", "nuclear_area_mean", "nuclei_density",
         "texture_e_contrast", "texture_h_contrast", "texture_h_energy"]
PRIMARY = {"eccentricity_mean": "morpho.eccentricity_mean", "nuclear_area_mean": "morpho.nuclear_area_mean",
           "nuclei_density": "morpho.nuclei_density", "texture_e_contrast": "appearance.texture_e_contrast",
           "texture_h_contrast": "appearance.texture_h_contrast", "texture_h_energy": "appearance.texture_h_energy"}


def square(path):
    rows = {(r["edited_attr"], r["measured_metric"]): r for r in csv.DictReader(open(path))}
    out = {}
    for e in ATTRS:
        out[e] = {}
        for m in ATTRS:
            r = rows.get((e, PRIMARY[m]))
            out[e][m] = None if r is None else {"r": float(r["targeted_pearson_r"]), "abs_ratio": float(r["abs_ratio"])}
    return out


old, new = square(sys.argv[1]), square(sys.argv[2])
res = {"old": old, "new": new, "rows": {}}
for e in ATTRS:
    def diag_max(sq):
        vals = {m: abs(sq[e][m]["r"]) for m in ATTRS if sq[e][m] is not None}
        return max(vals, key=vals.get) == e if vals else None
    res["rows"][e] = {"diag_r_old": old[e][e]["r"], "diag_r_new": new[e][e]["r"],
                      "diag_abs_ratio_old": old[e][e]["abs_ratio"], "diag_abs_ratio_new": new[e][e]["abs_ratio"],
                      "diag_is_row_max_old": diag_max(old), "diag_is_row_max_new": diag_max(new)}
json.dump(res, open(sys.argv[3], "w"), indent=1)
print("edited attr | diag r old -> new | abs_ratio old -> new | diag is row max old / new")
for e, v in res["rows"].items():
    print(f"{e:20s} {v['diag_r_old']:+.3f} -> {v['diag_r_new']:+.3f} | {v['diag_abs_ratio_old']:.2f} -> "
          f"{v['diag_abs_ratio_new']:.2f} | {v['diag_is_row_max_old']} / {v['diag_is_row_max_new']}")
print("\nnew square (rows = edited attr, cols = primary metric of):", " ".join(a[:10] for a in ATTRS))
for e in ATTRS:
    print(f"{e:20s}", " ".join(f"{new[e][m]['r']:+.2f}" if new[e][m] else "  nan" for m in ATTRS))
print("\nold square:")
for e in ATTRS:
    print(f"{e:20s}", " ".join(f"{old[e][m]['r']:+.2f}" if old[e][m] else "  nan" for m in ATTRS))
