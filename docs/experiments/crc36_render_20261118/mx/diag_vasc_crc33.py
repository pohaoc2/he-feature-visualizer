"""Describe every CRC33_01 tile whose reproduced vasculature mask differs from the training mask (step 1b follow-up).

  HFV=<hfv code> diag_vasc_crc33.py MX_TIFF MARKERS_CSV REPRO_TAR TRAIN_TAR OUT_JSON
Per differing tile: differing pixels, vessel fraction (train / repro), the Otsu threshold on the uint8-scaled
percentile-normalised CD31 crop, the histogram count at the threshold bins, and whether the processed_crc33 stage-1
multiplex npy would be needed to explain it.
"""
import io
import json
import os
import sys
import tarfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mx_lib as L  # noqa: E402
from utils.normalize import percentile_norm  # noqa: E402

TIFF, MARKERS, REPRO, TRAIN, OUT = sys.argv[1:6]


def vasc(path):
    tf = tarfile.open(path)
    return tf, {m.name.split("/")[-1][:-4]: m for m in tf if m.isfile() and "/vasculature/" in m.name}


tr, dt = vasc(TRAIN)
rp, dr = vasc(REPRO)
diff = []
for t in sorted(dt):
    b1 = tr.extractfile(dt[t]).read(); b2 = rp.extractfile(dr[t]).read()
    if b1 != b2:
        diff.append((t, np.load(io.BytesIO(b1)), np.load(io.BytesIO(b2))))
print("differing tiles", len(diff), flush=True)
ci, _ = L.channel_index(MARKERS)
cd31 = L.read_plane(TIFF, ci["CD31"])
rows = []
for t, a, b in diff:
    x0, y0 = map(int, t.split("_"))
    raw = L.crop(cd31, x0, y0)
    norm = percentile_norm(raw.astype(np.float32))
    u8 = (norm * 255).clip(0, 255).astype(np.uint8)
    thr, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hist = np.bincount(u8.ravel(), minlength=256)
    lo, hi = float(np.percentile(raw.astype(np.float32), 1)), float(np.percentile(raw.astype(np.float32), 99))
    # value bands that differ: which uint8 levels are vessel in train but not in repro (or vice versa)
    lv_tr = sorted(set(u8[a & ~b].tolist())); lv_rp = sorted(set(u8[b & ~a].tolist()))
    rows.append(dict(tile=t, n_diff_px=int((a != b).sum()), frac_train=float(a.mean()), frac_repro=float(b.mean()),
                     otsu_thr_repro=float(thr), p1=lo, p99=hi, levels_train_only=lv_tr[:10], levels_repro_only=lv_rp[:10],
                     hist_at_thr=[int(hist[int(thr) + k]) for k in (-1, 0, 1, 2) if 0 <= int(thr) + k < 256],
                     touches_edge=bool(x0 + 256 > cd31.shape[1] or y0 + 256 > cd31.shape[0])))
    print(rows[-1], flush=True)
json.dump(dict(n_diff=len(diff), tiles=rows), open(OUT, "w"), indent=1)
