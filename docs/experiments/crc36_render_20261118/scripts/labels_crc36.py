"""crc36_render_20261118 (envsmc): per-cell population / state for every Lin CRC36 cell.

  labels_crc36.py N HFV
Population : CELESTA (k-d tiled, defaults; N/celesta/out/tile_*.csv) + Lin Fig. 3d fill-in
             (stages/lin_typing/celltype_celesta_fig3d.py), Lin class at the cell centroid (N/lin/classmaps).
State      : Ki-67 mean intensity > CRC36 Lin gate (N/out/lin_gates_crc36.json, Lin's published C36 fraction) -> prolif.
             Dead: CellViT dead (type 4) objects from the Colab CellViT run on the CRC36 H&E tiles, mapped to the Lin cell
             under the CellViT centroid (N/out/dead_lin_ids.npy, written by cellvit_dead.py) - the CRC33_01 rule. Until
             that file exists no cell is dead (first pass; channels are re-rendered once it exists).
Tiles      : CRC36 tissue tiles (hfv patchify rule) minus ink tiles (N/out/tiles_crc36.json).
Writes N/out/cells_labels.npz (per cell, Hyak only), N/out/labels.json, N/out/tiles_keep.txt.
"""
import glob, json, sys
import numpy as np
import pandas as pd
from PIL import Image

N, HFV = sys.argv[1:3]
sys.path.insert(0, HFV)
from stages.lin_typing import labels as LB                  # noqa: E402
from stages.lin_typing import celltype_celesta_fig3d as F   # noqa: E402

df = pd.read_csv(f"{N}/lin/data__CRC36__P37_S79-CRC36.csv", usecols=["CellID", "X_centroid", "Y_centroid", "Area", "Ki67"])
cid = df.CellID.to_numpy(np.int64)
x, y = df.X_centroid.to_numpy(float), df.Y_centroid.to_numpy(float)
cmap = np.array(Image.open(f"{N}/lin/classmaps/CRC36_lin_he_class.tiff"))
k = LB.lin_class_at(x, y, cmap)
cel = pd.concat([pd.read_csv(f, usecols=["CellID", "final"]) for f in sorted(glob.glob(f"{N}/celesta/out/tile_*.csv"))])
assert len(glob.glob(f"{N}/celesta/out/tile_*.csv")) == 16, "STOP: CELESTA tiles missing"
assert cel.CellID.is_unique, "CELESTA tiles overlap"
cel = cel.sort_values("CellID")
final = F.lookup(cel.CellID.to_numpy(np.int64), cel.final.to_numpy(object), cid)
G = json.load(open(f"{N}/out/lin_gates_crc36.json"))
gate = float(G["markers"]["Ki67"]["final_thr"])
import os
DEAD = f"{N}/out/dead_lin_ids.npy"
dead = np.isin(cid, np.load(DEAD)) if os.path.exists(DEAD) else np.zeros(len(cid), bool)
ty, src, st = LB.assign(final, k, df.Ki67.to_numpy(float), gate, dead)

T = json.load(open(f"{N}/out/tiles_crc36.json"))
ps = 256
tiles = set(T["tiles"])
ink = {t for t, v in T["ink"].items() if v > 0}
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
cel_n = pd.Series([v if isinstance(v, str) else "no-call" for v in final]).value_counts().to_dict()
R = dict(n_cells=int(len(cid)), n_tiles=len(tiles), n_ink_tiles=len(ink), n_keep_tiles=len(keep),
         cells_in_tiles=int(np.isin(tid, sorted(tiles)).sum()), cells_in_keep_tiles=int(in_keep.sum()),
         cells_in_ink_tiles=int(in_ink.sum()), ki67_gate=gate, ki67_gate_source=G["markers"]["Ki67"]["final_source"],
         gates={m: dict(thr=round(v["final_thr"], 2), source=v["final_source"], frac=round(v["final_frac"], 4),
                        lin_published=v["lin_frac"]) for m, v in G["markers"].items() if v.get("typing")},
         identity=G.get("identity"), celesta_final=cel_n,
         celesta_nocall=int(sum(not isinstance(v, str) for v in final)), dead=int(dead.sum()) if os.path.exists(DEAD) else "none yet (CellViT pending)",
         pop={"all": counts(ty, LB.POP_NAMES, allm), "keep": counts(ty, LB.POP_NAMES, in_keep)},
         pop_frac={"all": frac(ty, LB.POP_NAMES, allm), "keep": frac(ty, LB.POP_NAMES, in_keep)},
         state_frac={"all": frac(st, LB.STATE_NAMES, allm), "keep": frac(st, LB.STATE_NAMES, in_keep)},
         source={"keep": {F.SRC_N[s]: counts(ty, LB.POP_NAMES, in_keep & (src == s)) for s in range(len(F.SRC_N))}},
         fill_by_lin_class={"keep": {F.LIN_L[c]: int((in_keep & (src >= F.SRC_UNKNOWN) & (k == c)).sum())
                                     for c in range(len(F.LIN_L))}},
         fill_low_classes_keep=int((in_keep & (src >= F.SRC_UNKNOWN) & np.isin(k, F.FILL_LOW)).sum()),
         cells_by_lin_class_keep={F.LIN_L[c]: counts(ty, LB.POP_NAMES, in_keep & (k == c)) for c in range(len(F.LIN_L))})
json.dump(R, open(f"{N}/out/labels.json", "w"), indent=1)
print(json.dumps({kk: R[kk] for kk in ("n_cells", "n_keep_tiles", "cells_in_keep_tiles", "pop_frac", "state_frac", "celesta_final")}))
