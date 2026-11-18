"""celltype_audit_20261118: do the generator's training channels carry the hfv labels? (envsmc; descriptive)
For the 500 sampled tiles (raw/tiles_sample.txt), read the training rasters orion-crc33/exp_channels/cell_type_* and
cell_state_* and, at every hfv CellViT centroid in the tile, record which channel is on. Agreement with
cell_assignments.csv cell_type / cell_state = the training conditioning is the hfv labeling (no later relabel).
Also the per-channel area fraction (training input composition) for ink vs ink-free tiles.
  chancheck.py W INK OUT_JSON
"""
import io
import json
import os
import sys
import tarfile

import numpy as np
import pandas as pd
from PIL import Image

W, INK, OUT = sys.argv[1:4]
TAR = tarfile.open(f"{W}/raw/o33_rasters_sample.tar")  # fetch.slurm: sampled members of orion-crc33/exp_channels
MEM = {}
for ti in TAR.getmembers():
    parts = ti.name.split("/")  # orion-crc33/exp_channels/<channel>/<tile>.<ext>
    if len(parts) == 4 and parts[1] == "exp_channels":
        MEM[(parts[2], os.path.splitext(parts[3])[0])] = ti
TY = ["healthy", "cancer", "immune"]
ST = ["prolif", "nonprolif", "dead"]


def load(ch, tid):
    ti = MEM.get((ch, tid))
    if ti is None:
        return None
    b = io.BytesIO(TAR.extractfile(ti).read())
    a = np.load(b) if ti.name.endswith(".npy") else np.array(Image.open(b))
    return a if a.ndim == 2 else a[..., 0]


tiles = open(f"{W}/raw/tiles_sample.txt").read().split()
h = pd.read_csv(f"{W}/raw/processed_crc33/cell_assignments.csv",
                usecols=["patch_id", "cell_type", "cell_state", "centroid_x_local", "centroid_y_local"])
h = h[h.patch_id.isin(tiles)]
h["state"] = h.cell_state.replace({"proliferative": "prolif", "nonproliferative": "nonprolif"})
ink = json.load(open(INK))
rows, comp = [], []
for tid in tiles:
    T = {c: load(f"cell_type_{c}", tid) for c in TY}
    S = {c: load(f"cell_state_{c}", tid) for c in ST}
    M = load("cell_masks", tid)
    if any(v is None for v in list(T.values()) + list(S.values())):
        continue
    comp.append(dict(tile=tid, ink=ink.get(tid, 0.0) > 0, mask=float((M > 0).mean()) if M is not None else np.nan,
                     **{f"t_{c}": float((T[c] > 0).mean()) for c in TY}, **{f"s_{c}": float((S[c] > 0).mean()) for c in ST}))
    for r in h[h.patch_id == tid].itertuples():
        x = int(np.clip(round(r.centroid_x_local), 0, 255)); y = int(np.clip(round(r.centroid_y_local), 0, 255))
        on_t = [c for c in TY if T[c][y, x] > 0]; on_s = [c for c in ST if S[c][y, x] > 0]
        rows.append(dict(t=r.cell_type, s=r.state, t_ch=on_t[0] if len(on_t) == 1 else ("none" if not on_t else "multi"),
                         s_ch=on_s[0] if len(on_s) == 1 else ("none" if not on_s else "multi")))
d = pd.DataFrame(rows); c = pd.DataFrame(comp)
res = dict(n_tiles=len(c), n_cells=len(d), exp_channels=sorted({k[0] for k in MEM}),
           type_agree=round(float((d.t == d.t_ch).mean()), 4), state_agree=round(float((d.s == d.s_ch).mean()), 4),
           type_confusion=pd.crosstab(d.t, d.t_ch).to_dict(), state_confusion=pd.crosstab(d.s, d.s_ch).to_dict(),
           channel_area_frac={k: {col: round(float(g[col].mean()), 4) for col in c.columns if col not in ("tile", "ink")}
                              for k, g in c.groupby(c.ink.map({True: "ink_tiles", False: "ink_free_tiles"}))})
json.dump(res, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "exp_channels"}, indent=1))
