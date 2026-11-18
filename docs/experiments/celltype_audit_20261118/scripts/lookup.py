"""celltype_audit_20261118 step 2 (cellvit env: numpy, cv2, tifffile; no pandas).

Puts the he-feature-visualizer (hfv) CellViT objects of CRC33 and Lin's CRC33_01 cells in one frame and records, per
object, everything the audit needs that lives in images:
  G-frame  : hfv H&E tiles (darkdiag s3/he, same he.zip the training used) equal the crops of Lin's registered CRC33_01
             H&E at (y0, x0) -> CRC33_01 is the training slide and hfv global px = Lin px (no transform).
  G-px     : OME PhysicalSizeX of the H&E and of the mask (um/px) -> class-map grid = 112 um / px size.
  G-id     : mask label == CSV CellID (label centroid vs CSV centroid, 200 random labels, median distance).
  match    : Lin mask label under each hfv centroid (0 = background) -> many-to-one / unmatched counts.
  merge    : number of Lin CSV centroids inside each hfv contour (CellViT under-segmentation).
  ink      : min H&E grayscale (cv2 weights) in the 3x3 px around each hfv centroid and each Lin centroid
             (ink rule = style_pool.PIGMENT_LEVEL 60 from sim-exp-validation; applied downstream).
  lin2hfv  : per Lin cell, number of hfv centroids inside its mask label.
  lookup.py W   (W = /gscratch/cheme/chiu/he_celltype_audit_20261118)
"""
import csv
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import tifffile

W = sys.argv[1]
P = f"{W}/raw/processed_crc33"
L = f"{W}/lin"
HE_TILES = "/gscratch/cheme/chiu/crc16-20261024/darkdiag_20261031/s3/he"  # read only
OUT = f"{W}/out"
rng = np.random.default_rng(20261118)


def px_size(path):
    with tifffile.TiffFile(path) as t:
        m = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', t.ome_metadata or "")
        return float(m.group(1)) if m else None


def level0(path):
    with tifffile.TiffFile(path) as t:
        return t.series[0].levels[0].asarray()


def gray(rgb):
    return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2GRAY)


def main():
    res = {}
    he_path = f"{L}/CRC33_01__19510_C33_US_SCAN_OR_001__160715-2-registered.ome.tif"
    mk_path = f"{L}/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif"
    res["px_um"] = dict(he=px_size(he_path), mask=px_size(mk_path))
    print("px", res["px_um"], flush=True)

    he = level0(he_path)
    if he.shape[0] == 3:
        he = np.moveaxis(he, 0, -1)
    res["he_shape"] = list(he.shape)
    # G-frame: 50 random hfv training tiles vs crops of Lin's registered H&E
    tiles = sorted(glob.glob(f"{HE_TILES}/*.png"))
    pick = rng.choice(len(tiles), size=min(50, len(tiles)), replace=False)
    diffs = []
    for i in pick:
        name = os.path.basename(tiles[i])[:-4]
        x0, y0 = map(int, name.split("_"))
        t = cv2.cvtColor(cv2.imread(tiles[i], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        c = he[y0:y0 + t.shape[0], x0:x0 + t.shape[1]]
        diffs.append(float(np.abs(c.astype(int) - t.astype(int)).mean()) if c.shape == t.shape else float("nan"))
    res["G_frame"] = dict(n=len(diffs), mean_abs_diff=diffs, max=float(np.nanmax(diffs)),
                          ok=bool(np.nanmax(diffs) < 1.0 and not np.isnan(diffs).any()))
    print("G-frame", res["G_frame"]["max"], res["G_frame"]["ok"], flush=True)
    if not res["G_frame"]["ok"]:
        json.dump(res, open(f"{OUT}/lookup.json", "w"), indent=1)
        raise SystemExit("STOP G-frame: hfv tiles are not crops of CRC33_01 registered H&E at (y0, x0)")
    g = gray(he)
    del he

    # hfv objects
    hx, hy, hpid = [], [], []
    with open(f"{P}/cell_assignments.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            hx.append(float(row["centroid_x_global"])); hy.append(float(row["centroid_y_global"])); hpid.append(row["patch_id"])
    hx, hy = np.array(hx), np.array(hy)
    # Lin cells
    lid, lx, ly = [], [], []
    with open(f"{L}/CRC33_01__P37_S76_01-CRC33_01.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            lid.append(int(float(row["CellID"]))); lx.append(float(row["X_centroid"])); ly.append(float(row["Y_centroid"]))
    lid, lx, ly = np.array(lid), np.array(lx), np.array(ly)
    print("hfv", len(hx), "lin", len(lid), flush=True)

    def ink(x, y):
        xi = np.clip(np.round(x).astype(int), 1, g.shape[1] - 2); yi = np.clip(np.round(y).astype(int), 1, g.shape[0] - 2)
        m = np.full(len(xi), 255, np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                m = np.minimum(m, g[yi + dy, xi + dx])
        return m
    hfv_gray, lin_gray = ink(hx, hy), ink(lx, ly)
    del g

    mask = level0(mk_path)
    res["mask_shape"] = list(mask.shape)
    # G-id
    sel = rng.choice(len(lid), size=200, replace=False)
    d = []
    for i in sel:
        x, y = int(round(lx[i])), int(round(ly[i]))
        y0, x0 = max(0, y - 60), max(0, x - 60)
        win = mask[y0:y + 61, x0:x + 61]
        yy, xx = np.nonzero(win == lid[i])
        d.append(float(np.hypot(xx.mean() + x0 - lx[i], yy.mean() + y0 - ly[i])) if len(yy) else float("nan"))
    res["G_id"] = dict(n=200, median_px=float(np.nanmedian(d)), missing=int(np.isnan(d).sum()),
                       ok=bool(np.nanmedian(d) <= 1.0))
    print("G-id", res["G_id"], flush=True)
    if not res["G_id"]["ok"]:
        json.dump(res, open(f"{OUT}/lookup.json", "w"), indent=1)
        raise SystemExit("STOP G-id")
    xi = np.clip(np.round(hx).astype(int), 0, mask.shape[1] - 1); yi = np.clip(np.round(hy).astype(int), 0, mask.shape[0] - 1)
    hfv_label = mask[yi, xi].astype(np.int64)
    # merge: Lin centroids inside each hfv contour (per tile, local coords)
    order = np.argsort(lx)
    sx = lx[order]
    merge = np.full(len(hx), -1, np.int32)
    k = 0
    for fn in sorted(glob.glob(f"{P}/cellvit/*.json")):
        dct = json.load(open(fn))
        x0, y0 = map(int, dct["patch"].split("_"))
        a, b = np.searchsorted(sx, x0 - 1), np.searchsorted(sx, x0 + 257)
        cand = order[a:b]
        cand = cand[(ly[cand] >= y0 - 1) & (ly[cand] < y0 + 257)]
        pts = np.stack([lx[cand] - x0, ly[cand] - y0], 1)
        for c in dct["cells"]:
            cnt = np.array(c["contour"], np.float32).reshape(-1, 1, 2)
            if len(cnt) < 3:
                continue
            (ya, xa), (yb, xb) = c["bbox"]
            q = pts[(pts[:, 0] >= xa - 1) & (pts[:, 0] <= xb + 1) & (pts[:, 1] >= ya - 1) & (pts[:, 1] <= yb + 1)]
            n = sum(cv2.pointPolygonTest(cnt, (float(p[0]), float(p[1])), False) >= 0 for p in q)
            gx, gy = x0 + c["centroid"][0], y0 + c["centroid"][1]
            merge_key = (round(gx, 2), round(gy, 2))
            MERGE[merge_key] = n
            k += 1
    for i in range(len(hx)):
        merge[i] = MERGE.get((round(hx[i], 2), round(hy[i], 2)), -1)
    print("merge done", k, "objects; unresolved", int((merge < 0).sum()), flush=True)
    # lin2hfv
    lab_counts = np.bincount(hfv_label[hfv_label > 0], minlength=int(lid.max()) + 1)
    lin2hfv = lab_counts[lid]
    np.savez_compressed(f"{OUT}/lookup.npz", hfv_x=hx, hfv_y=hy, hfv_label=hfv_label, hfv_gray=hfv_gray, hfv_merge=merge,
                        lin_id=lid, lin_gray=lin_gray, lin2hfv=lin2hfv)
    res["hfv_on_background"] = float((hfv_label == 0).mean())
    res["lin_without_hfv"] = float((lin2hfv == 0).mean())
    json.dump(res, open(f"{OUT}/lookup.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "G_frame"}), flush=True)


MERGE = {}
if __name__ == "__main__":
    main()
