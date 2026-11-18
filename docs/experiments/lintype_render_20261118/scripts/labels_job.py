"""lintype_render_20261118 step 1 (envsmc: pandas, PIL): per-cell population / state for every Lin CRC33_01 cell.

  labels_job.py A N HFV      A = he_celltype_audit_20261118 dir (inputs, read only), N = this run's dir, HFV = repo copy
Writes N/out/cells_labels.npz (per cell, stays on Hyak) and N/out/labels.json (counts only).

Population : CELESTA (tiled, default settings; A/celesta33/out/tile_*.csv) + Lin Fig. 3d fill-in
             (stages/lin_typing/celltype_celesta_fig3d.py), Lin class at the cell centroid.
State      : CellViT dead (type_cellvit 4; centroid on a Lin label, A/out/lookup.npz) first, then Ki-67 mean intensity
             > Lin gate pooled over CRC33_01 + 02 (A/out/lin_gates_crc33.json markers.Ki67.final_thr) -> prolif.
Tiles      : training tiles of A/raw/processed_crc33/index.json minus ink tiles (darkdiag ink.json fraction > 0).
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

A, N, HFV = sys.argv[1:4]
sys.path.insert(0, HFV)
from stages.lin_typing import labels as LB          # noqa: E402
from stages.lin_typing import celltype_celesta_fig3d as F  # noqa: E402
from stages.lin_typing.render import ink_tiles      # noqa: E402

INK = "/gscratch/cheme/chiu/crc16-20261024/darkdiag_20261031/ink.json"   # read only

df = pd.read_csv(f"{A}/lin/CRC33_01__P37_S76_01-CRC33_01.csv", usecols=["CellID", "X_centroid", "Y_centroid", "Area", "Ki67"])
cid = df.CellID.to_numpy(np.int64)
x, y = df.X_centroid.to_numpy(float), df.Y_centroid.to_numpy(float)
cmap = np.array(Image.open(f"{A}/lin/CRC33_01_lin_he_class.tiff"))
k = LB.lin_class_at(x, y, cmap)

cel = pd.concat([pd.read_csv(f, usecols=["CellID", "final"]) for f in sorted(glob.glob(f"{A}/celesta33/out/tile_*.csv"))])
assert cel.CellID.is_unique, "CELESTA tiles overlap"
cel = cel.sort_values("CellID")
final = F.lookup(cel.CellID.to_numpy(np.int64), cel.final.to_numpy(object), cid)

gate = float(json.load(open(f"{A}/out/lin_gates_crc33.json"))["markers"]["Ki67"]["final_thr"])
lk = np.load(f"{A}/out/lookup.npz")
assert (lk["lin_id"] == cid).all()
hfv = pd.read_csv(f"{A}/raw/processed_crc33/cell_assignments.csv",
                  usecols=["type_cellvit", "cell_type", "cell_state", "centroid_x_global", "centroid_y_global"])
assert np.allclose(hfv.centroid_x_global.to_numpy(), lk["hfv_x"]) and np.allclose(hfv.centroid_y_global.to_numpy(), lk["hfv_y"])
dead, dead_n = LB.dead_from_cellvit(cid, lk["hfv_label"], hfv.type_cellvit.to_numpy())
ty, src, st = LB.assign(final, k, df.Ki67.to_numpy(float), gate, dead)

idx = json.load(open(f"{A}/raw/processed_crc33/index.json"))
ps = int(idx["patch_size"]); assert int(idx["stride"]) == ps
ink = ink_tiles(json.load(open(INK)))
tiles = {f"{int(p['x0'])}_{int(p['y0'])}" for p in idx["patches"]}
assert ink <= tiles
keep = tiles - ink
tid = np.array([f"{a}_{b}" for a, b in zip((x // ps).astype(int) * ps, (y // ps).astype(int) * ps)])
in_keep = np.isin(tid, sorted(keep))
in_ink = np.isin(tid, sorted(ink))
np.savez_compressed(f"{N}/out/cells_labels.npz", cell_id=cid, x=x, y=y, area_cell_px=df.Area.to_numpy(float),
                    lin_cls=k.astype(np.int8), ty=ty, src=src, st=st, dead=dead, in_keep=in_keep, in_ink=in_ink,
                    celesta=np.array([v if isinstance(v, str) else "" for v in final]))
with open(f"{N}/out/tiles_keep.txt", "w") as f:
    f.write("\n".join(sorted(keep)) + "\n")


def counts(v, names, m):
    return {names[i]: int(((v == i) & m).sum()) for i in range(len(names))}


def frac(v, names, m):
    n = max(int(m.sum()), 1)
    return {names[i]: round(int(((v == i) & m).sum()) / n, 4) for i in range(len(names))}


allm = np.ones(len(cid), bool)
R = dict(n_cells=int(len(cid)), n_tiles=len(tiles), n_ink_tiles=len(ink), n_keep_tiles=len(keep),
         cells_in_keep_tiles=int(in_keep.sum()), cells_in_ink_tiles=int(in_ink.sum()), ki67_gate=gate,
         celesta_nocall=int(sum(not isinstance(v, str) for v in final)), dead=dead_n,
         pop={"all": counts(ty, LB.POP_NAMES, allm), "keep": counts(ty, LB.POP_NAMES, in_keep)},
         pop_frac={"all": frac(ty, LB.POP_NAMES, allm), "keep": frac(ty, LB.POP_NAMES, in_keep)},
         state_frac={"all": frac(st, LB.STATE_NAMES, allm), "keep": frac(st, LB.STATE_NAMES, in_keep)},
         source={"keep": {F.SRC_N[s]: counts(ty, LB.POP_NAMES, in_keep & (src == s)) for s in range(len(F.SRC_N))}},
         fill_by_lin_class={"keep": {F.LIN_L[c]: int((in_keep & (src >= F.SRC_UNKNOWN) & (k == c)).sum())
                                     for c in range(len(F.LIN_L))}},
         fill_low_classes_keep=int((in_keep & (src >= F.SRC_UNKNOWN) & np.isin(k, F.FILL_LOW)).sum()),
         cells_by_lin_class_keep={F.LIN_L[c]: counts(ty, LB.POP_NAMES, in_keep & (k == c)) for c in range(len(F.LIN_L))})

# old (hfv, CellViT objects) vs new (Lin cells): objects / cells in the kept tiles, and per hfv object whose centroid
# lies on a Lin cell, its old label vs the new label of that cell
hx, hy = lk["hfv_x"], lk["hfv_y"]
htid = np.array([f"{a}_{b}" for a, b in zip((hx // ps).astype(int) * ps, (hy // ps).astype(int) * ps)])
h_keep = np.isin(htid, sorted(keep))
OLD_T = {"healthy": 0, "cancer": 1, "immune": 2}
OLD_S = {"proliferative": 0, "nonproliferative": 1, "dead": 2}
ot = hfv.cell_type.map(OLD_T).to_numpy()
os_ = hfv.cell_state.map(OLD_S).to_numpy()
pos = pd.Series(np.arange(len(cid)), index=cid)
lab = lk["hfv_label"]
on = h_keep & (lab > 0)
li = pos.reindex(lab[on]).to_numpy()
okm = ~np.isnan(li)
li = li[okm].astype(int)


def confusion(a, b, n=3):
    m = np.zeros((n, n), int)
    np.add.at(m, (a, b), 1)
    return m.tolist()


R["old_vs_new"] = dict(
    old_objects_keep=int(h_keep.sum()), old_pop_frac_keep=frac(ot, LB.POP_NAMES, h_keep),
    old_state_frac_keep=frac(os_, LB.STATE_NAMES, h_keep),
    matched_objects=int(okm.sum()), unmatched_objects_on_background=int((h_keep & (lab == 0)).sum()),
    pop_confusion_rows_old_cols_new=confusion(ot[on][okm], ty[li]),
    state_confusion_rows_old_cols_new=confusion(os_[on][okm], st[li]),
    pop_agreement=round(float((ot[on][okm] == ty[li]).mean()), 4),
    state_agreement=round(float((os_[on][okm] == st[li]).mean()), 4),
    norm_epithelial_cells_keep=int((in_keep & (k == F.NORM) & (src == F.SRC_EPI)).sum()))
json.dump(R, open(f"{N}/out/labels.json", "w"), indent=1)
print(json.dumps({kk: R[kk] for kk in ("n_cells", "n_keep_tiles", "cells_in_keep_tiles", "pop_frac", "state_frac", "dead")}))
print("agreement", R["old_vs_new"]["pop_agreement"], R["old_vs_new"]["state_agreement"])
