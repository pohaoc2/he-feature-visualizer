"""hfv_retrain3_cellvit_20261122 stage report: pick and extract the contact-sheet tiles, Hyak CPU.

  python sheet_pick.py R OUT
3 training tiles per slide, numpy default_rng(42).choice without replacement among the tiles whose official-CellViT
nucleus count (tiles_<slide>_OC.csv n_inst) is above that slide's median. Extracts per tile the real H&E PNG (R/he
chunk zips) and the 7 cell-channel PNGs (R/cells/cells_<slide>_OC.tar) into OUT/<slide>__<t>/, and OUT/picks.json.
"""
import csv
import glob
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np

R, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
SL = {"crc33": ("c33_*", "orion-crc33/exp_channels", ""), "crc36": ("c36_[0-9]*", "c36/exp_channels", "c36_")}
picks = {}
for s, (zp, pre, idp) in SL.items():
    rows = list(csv.DictReader(open(R / f"cells/tiles_{s}_OC.csv")))
    train = set(t.strip() for t in open(R / f"cells/train_{s}.txt"))
    rows = [r for r in rows if r["tile"] in train]
    n = np.array([int(r["n_inst"]) for r in rows])
    cand = sorted(r["tile"] for r, k in zip(rows, n) if k > np.median(n))
    pick = list(np.random.default_rng(42).choice(cand, 3, replace=False))
    picks[s] = dict(median=float(np.median(n)), candidates=len(cand), tiles=pick)
    for t in pick:
        d = OUT / f"{s}__{t}"
        d.mkdir(exist_ok=True)
        for z in glob.glob(str(R / "he" / zp.replace("_*", "_*.zip").replace("[0-9]*", "[0-9]*.zip"))):
            with zipfile.ZipFile(z) as zf:
                if f"he/{t}.png" in zf.namelist():
                    (d / "he.png").write_bytes(zf.read(f"he/{t}.png"))
        with tarfile.open(R / f"cells/cells_{s}_OC.tar") as tf:
            for m in tf:
                if m.name.endswith(f"/{idp}{t}.png"):
                    (d / f"{m.name.split('/')[-2]}.png").write_bytes(tf.extractfile(m).read())
        if not (d / "he.png").is_file() or len(list(d.glob("*.png"))) != 8:
            sys.exit(f"STOP {s} {t}: files missing in {d}")
(OUT / "picks.json").write_text(json.dumps(picks, indent=1))
print(json.dumps(picks))
