"""celltype_audit_20261118 follow-up tables (envsmc; descriptive): CellViT nucleus type by Lin tissue class, hfv objects
under ink vs Lin mask, and the Ki-67 gate on CRC33_01 alone.
  extra.py W OUT_JSON
"""
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

W, OUT = sys.argv[1:3]
GROUP = {0: "other", 1: "NORM", 2: "other", 3: "TUM", 4: "other", 5: "other", 6: "other", 7: "STR", 8: "LYM"}
NAMES = {0: "background", 1: "neoplastic", 2: "inflammatory", 3: "connective", 4: "dead", 5: "epithelial"}
lk = np.load(f"{W}/out/lookup.npz")
cmap = np.array(Image.open(f"{W}/lin/CRC33_01_lin_he_class.tiff"))
P = 112.0 / 0.325
h = pd.read_csv(f"{W}/raw/processed_crc33/cell_assignments.csv",
                usecols=["patch_id", "type_cellvit", "cell_type", "centroid_x_global", "centroid_y_global"])
r = np.clip((h.centroid_y_global / P).astype(int), 0, cmap.shape[0] - 1)
c = np.clip((h.centroid_x_global / P).astype(int), 0, cmap.shape[1] - 1)
h["grp"] = pd.Series(cmap[r, c]).map(GROUP).to_numpy()
h["ct"] = h.type_cellvit.map(NAMES)
h["inked"] = lk["hfv_gray"] < 60
h["label"] = lk["hfv_label"]
res = dict(
    cellvit_type_by_tissue=pd.crosstab(h.grp, h.ct, normalize="index").round(4).to_dict(orient="index"),
    hfv_cancer_by_tissue_and_cellvit=pd.crosstab(h.grp[h.cell_type == "cancer"], h.ct[h.cell_type == "cancer"]).to_dict(orient="index"),
    hfv_cancer_share_by_tissue=(h[h.cell_type == "cancer"].grp.value_counts(normalize=True).round(4).to_dict()),
    hfv_inked=dict(n=int(h.inked.sum()), on_lin_background=round(float((h.label[h.inked] == 0).mean()), 4),
                   cellvit=h.ct[h.inked].value_counts(normalize=True).round(4).to_dict()),
    hfv_not_inked_on_lin_background=round(float((h.label[~h.inked] == 0).mean()), 4),
    tum_grid_cells_total=int((cmap == 3).sum()), norm_grid_cells_total=int((cmap == 1).sum()))
G = json.load(open(f"{W}/out/lin_gates_crc33.json"))["markers"]
res["crc33_01_frac_at_pooled_gate"] = {m: round(G[m]["final_frac_by_piece"][0], 4) for m in G}
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps(res, indent=1))
