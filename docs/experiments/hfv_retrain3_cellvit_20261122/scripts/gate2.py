"""hfv_retrain3_cellvit_20261122 PREREG §3.6 gate 2: this run's official CellViT on real H&E vs cellvit_official_20261122.

  PYTHONPATH=<PixCell cb62fa7> python gate2.py R
For crc36ho (R_crc36ho, 300 tiles; this run R/cv/c36ho_*) and crc33sec (R_crc33sec, 100 tiles; R/cv/c33_*): both JSONs
rasterised with PixCell _cellvit_json_to_instance_mask; a tile is identical when the two instance maps are the same
partition (same pixels per instance, labels may differ). Also per tile: instance counts, PQ (PixCell _compute_pq) of
this run vs the stored one. Pass = identical tiles >= 298 / 300 and 100 / 100 (the agreement that run reported for the
same tiles between its own repeats). -> R/gate2.json; exit 1 if it fails.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq

R = Path(sys.argv[1])
O = Path("/gscratch/cheme/chiu/cellvit_official_20261122/full/cv")
SETS = {"crc36ho": ("R_crc36ho", "c36ho_*", 298), "crc33sec": ("R_crc33sec", "c33_*", 100)}


def same_partition(a, b):
    if not np.array_equal(a > 0, b > 0):
        return False
    fg = a > 0
    pairs = np.unique(np.stack([a[fg], b[fg]], 1), axis=0)
    return len(pairs) == len(np.unique(a[fg])) == len(np.unique(b[fg]))


rep, ok = {}, True
for s, (ref, pat, need) in SETS.items():
    mine = {Path(p).stem: p for p in glob.glob(f"{R}/cv/{pat}/*.json")}
    rows = []
    for p in sorted(glob.glob(f"{O}/{ref}/*.json")):
        t = Path(p).stem
        if t not in mine:
            print("STOP", s, t, "missing in this run", flush=True)
            sys.exit(1)
        a = _cellvit_json_to_instance_mask(mine[t], image_hw=(256, 256))
        b = _cellvit_json_to_instance_mask(p, image_hw=(256, 256))
        rows.append(dict(tile=t, identical=bool(same_partition(a, b)), n_this=int(len(np.unique(a)) - 1),
                         n_stored=int(len(np.unique(b)) - 1), pq=float(_compute_pq(b, a)[2])))
    n_id = sum(r["identical"] for r in rows)
    rep[s] = dict(tiles=len(rows), identical=n_id, need=need, pq_min=min(r["pq"] for r in rows),
                  pq_mean=float(np.mean([r["pq"] for r in rows])),
                  count_diff_max=max(abs(r["n_this"] - r["n_stored"]) for r in rows),
                  not_identical=[r for r in rows if not r["identical"]])
    ok &= n_id >= need
    print(s, {k: v for k, v in rep[s].items() if k != "not_identical"}, flush=True)
rep["pass"] = bool(ok)
(R / "gate2.json").write_text(json.dumps(rep, indent=1) + "\n")
print("PASS" if ok else "STOP gate 2", flush=True)
sys.exit(0 if ok else 1)
