"""Compare a reproduced CRC33_01 channel tar with the generator training channels (step 1b), on ALL tiles.

  compare_crc33.py REPRO_TAR TRAIN_TAR OUT_JSON
Per channel: byte-equal fraction (npy bytes), array-equal fraction, max / mean / p99.9 abs diff, pooled Pearson r,
per-tile Pearson r (median, p1); vasculature also pixel agreement and IoU; the worst tiles are listed.
"""
import io
import json
import sys
import tarfile

import numpy as np

REPRO, TRAIN, OUT = sys.argv[1:4]


def index(path, prefix):
    tf = tarfile.open(path)
    d = {}
    for m in tf:
        if m.isfile():
            _, _, c, f = m.name.split("/")
            d[(c, f[:-4])] = m
    return tf, d


tr, dt = index(TRAIN, "orion-crc33")
rp, dr = index(REPRO, "orion-crc33")
R = {"repro": REPRO, "train": TRAIN}
for c in ("vasculature", "oxygen", "glucose"):
    tiles = sorted(t for (cc, t) in dt if cc == c)
    missing = [t for t in tiles if (c, t) not in dr]
    n = beq = aeq = 0
    mx, s_abs, n_px = 0.0, 0.0, 0
    sx = sy = sxx = syy = sxy = 0.0
    per_r, worst, inter, union, agree = [], [], 0, 0, 0
    p999 = []
    for t in tiles:
        if (c, t) not in dr:
            continue
        b1 = tr.extractfile(dt[(c, t)]).read()
        b2 = rp.extractfile(dr[(c, t)]).read()
        a1 = np.load(io.BytesIO(b1)); a2 = np.load(io.BytesIO(b2))
        n += 1; beq += b1 == b2; aeq += a1.dtype == a2.dtype and np.array_equal(a1, a2)
        x = a1.astype(np.float64).ravel(); y = a2.astype(np.float64).ravel()
        d = np.abs(x - y); m = float(d.max())
        mx = max(mx, m); s_abs += d.sum(); n_px += d.size
        p999.append(float(np.percentile(d, 99.9)))
        sx += x.sum(); sy += y.sum(); sxx += (x * x).sum(); syy += (y * y).sum(); sxy += (x * y).sum()
        if x.std() > 0 and y.std() > 0:
            per_r.append(float(np.corrcoef(x, y)[0, 1]))
        worst.append((m, t))
        if c == "vasculature":
            inter += int(np.sum(a1 & a2)); union += int(np.sum(a1 | a2)); agree += int(np.sum(a1 == a2))
    cov = sxy / n_px - (sx / n_px) * (sy / n_px)
    vx = sxx / n_px - (sx / n_px) ** 2; vy = syy / n_px - (sy / n_px) ** 2
    r = dict(n_train=len(tiles), n_compared=n, missing_in_repro=len(missing), byte_equal_frac=beq / n,
             array_equal_frac=aeq / n, max_abs=mx, mean_abs=s_abs / n_px,
             tile_p999_abs_max=max(p999), pearson_pooled=float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else None,
             pearson_tile_median=float(np.median(per_r)) if per_r else None,
             pearson_tile_p1=float(np.percentile(per_r, 1)) if per_r else None, n_tiles_with_r=len(per_r),
             worst_tiles=sorted(worst, reverse=True)[:5])
    if c == "vasculature":
        r.update(pixel_agreement=agree / n_px, iou=inter / union if union else None)
    R[c] = r
    print(c, json.dumps({k: v for k, v in r.items() if k != "worst_tiles"}), flush=True)
json.dump(R, open(OUT, "w"), indent=1)
