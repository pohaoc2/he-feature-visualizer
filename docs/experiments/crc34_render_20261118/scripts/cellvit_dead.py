"""crc34_render_20261118: CellViT dead objects -> dead Lin cells (the CRC33_01 rule, hfv stages/lin_typing/labels.py
dead_from_cellvit: a Lin cell is dead when the centroid of a CellViT dead object, type_cellvit 4, falls on its label).

  python cellvit_dead.py N CV_DIR
CV_DIR = per-tile CellViT JSONs (he-feature-visualizer stages/run_cellvit_local.py e88f450, CellViT-256, Colab run of
the CRC34 H&E tiles; file <x0>_<y0>.json, centroid [cx, cy] in tile pixels). Global centroid = (x0 + cx, y0 + cy) on the
Lin cell mask (same pixel grid as the registered H&E; tiles.py asserts equal sizes). Writes N/out/dead_lin_ids.npy and
N/out/cellvit_dead.json (object / tile counts, dead on background).
"""
import glob
import json
import os
import sys

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(sys.argv[1], "hfv"))
from stages.lin_typing.labels import CELLVIT_DEAD  # noqa: E402

N, CV = sys.argv[1], sys.argv[2]
tiles = [t.strip() for t in open(f"{N}/out/tiles_keep.txt") if t.strip()]
xs, ys, ty = [], [], []
missing = 0
for t in tiles:
    f = f"{CV}/{t}.json"
    if not os.path.exists(f):
        missing += 1
        continue
    x0, y0 = map(int, t.split("_"))
    for c in json.load(open(f))["cells"]:
        xs.append(x0 + c["centroid"][0]); ys.append(y0 + c["centroid"][1]); ty.append(int(c["type_cellvit"]))
assert missing == 0, f"{missing} kept tiles without a CellViT JSON"
xs, ys, ty = np.array(xs), np.array(ys), np.array(ty)
with tifffile.TiffFile(f"{N}/lin/data__CRC34__P37_S77-CRC34-mask.ome.tif") as tf:
    lab = tf.series[0].asarray()
h, w = lab.shape
xi, yi = np.clip(xs.astype(np.int64), 0, w - 1), np.clip(ys.astype(np.int64), 0, h - 1)
under = lab[yi, xi].astype(np.int64)
is_dead = ty == CELLVIT_DEAD
hit = np.unique(under[is_dead & (under > 0)])
np.save(f"{N}/out/dead_lin_ids.npy", hit)
R = dict(rule="Lin label under the centroid of a CellViT type-4 (dead) object", kept_tiles=len(tiles),
         json_files=len(glob.glob(f"{CV}/*.json")), cellvit_objects=int(len(ty)),
         per_type={int(k): int(v) for k, v in zip(*np.unique(ty, return_counts=True))},
         cellvit_dead=int(is_dead.sum()), cellvit_dead_on_background=int((is_dead & (under == 0)).sum()),
         lin_cells_dead=int(len(hit)), objects_on_background=int((under == 0).sum()))
json.dump(R, open(f"{N}/out/cellvit_dead.json", "w"), indent=1)
print(json.dumps(R))
