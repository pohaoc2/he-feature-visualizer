"""hfv_retrain_20261118: CRC34 CellViT reuse gate (coordinator criterion, 2026-11-18, fixed before the run).

  python cvgate34.py REF_DIR NEW_DIR OUT_JSON
REF_DIR = the old CRC34 CellViT JSONs of the 50 gate tiles (seed-42 sample of the CRC34 evaluation tiles), NEW_DIR = the
same tiles rerun on Colab with TF32 off. Per tile, PASS needs all of:
  cell count equal; every cell matched with centroid distance 0; classes (type_cellvit) 100 % identical;
  contour point coordinates identical (the dead-cell call maps CellViT objects to Lin cells, so contours count).
Matching is exact: each tile's cells are compared as a sorted multiset of (centroid, contour, type_cellvit).
Byte-identical JSON files are counted separately (reported, not part of PASS). Gate PASS = 50/50 tiles pass.
"""
import json
import os
import sys

REF, NEW, OUT = sys.argv[1:4]


def key(c):
    return (tuple(float(v) for v in c["centroid"]), tuple(tuple(float(v) for v in p) for p in c["contour"]),
            int(c["type_cellvit"]))


tiles = sorted(f for f in os.listdir(REF) if f.endswith(".json"))
R = dict(n_tiles=len(tiles), tiles_pass=0, byte_identical=0, cells_ref=0, cells_new=0, fail=[])
for f in tiles:
    a = open(f"{REF}/{f}", "rb").read()
    p = f"{NEW}/{f}"
    if not os.path.exists(p):
        R["fail"].append(dict(tile=f, why="missing in new"))
        continue
    b = open(p, "rb").read()
    R["byte_identical"] += a == b
    ca, cb = json.loads(a)["cells"], json.loads(b)["cells"]
    R["cells_ref"] += len(ca)
    R["cells_new"] += len(cb)
    ka, kb = sorted(map(key, ca)), sorted(map(key, cb))
    why = []
    if len(ca) != len(cb):
        why.append(f"count {len(ca)} vs {len(cb)}")
    else:
        if [k[0] for k in ka] != [k[0] for k in kb]:
            why.append("centroids differ")
        elif [k[2] for k in ka] != [k[2] for k in kb]:
            why.append("classes differ")
        elif [k[1] for k in ka] != [k[1] for k in kb]:
            why.append("contours differ")
    if why:
        R["fail"].append(dict(tile=f, why="; ".join(why)))
    else:
        R["tiles_pass"] += 1
R["PASS"] = R["tiles_pass"] == R["n_tiles"] == 50
R["rule"] = ("per tile: count equal, all centroid distances 0, classes identical, contour points identical; "
             "50/50 tiles; byte-identical reported separately")
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps(R))
