"""PQ / DICE of old (paper) and new CellViT instances against old and new GT cell_masks, same 100 CRC33 tiles."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
from tools.compute_ablation_metrics import (_compute_pq, _compute_binary_segmentation_metrics,
    _cellvit_json_to_instance_mask, _label_binary_instances, _load_spatial_array, _resize_nearest, _array_max)
R = Path(sys.argv[1]); ids = [l.strip() for l in open(R / "ids100.txt") if l.strip()]
GT = {"oldGT": R / "old33zip/orion-crc33/exp_channels/cell_masks", "newGT": R / "new33pack/orion-crc33sec/exp_channels/cell_masks"}
IM = {"old": R / "oldtiles", "new": R / "new33/fig2_crc33sec/tiles"}
V = ["a2_off_shelf", "production", "a1_concat", "a2_bypass_full_tme", "a1_per_channel"]
def gt(d, t, shape):
    arr = _load_spatial_array(d / f"{t}.png")
    if arr.shape != shape: arr = _resize_nearest(arr.astype(np.uint8), shape)
    return _label_binary_instances(arr > (0.5 if _array_max(arr) <= 1.0 else 0))
res = {}; stats = {}
for im, idir in IM.items():
    for v in V:
        for g, gdir in GT.items():
            pq, dc = [], []
            for t in ids:
                img = Image.open(idir / v / f"{t}.png")
                pred = _cellvit_json_to_instance_mask(idir / v / f"{t}_cellvit_instances.json", image_hw=(img.height, img.width))
                gi = gt(gdir, t, pred.shape)
                pq.append(_compute_pq(gi, pred)[2]); dc.append(_compute_binary_segmentation_metrics(gi, pred)[0])
            res[f"{im}img/{v}/{g}"] = dict(pq=float(np.mean(pq)), dice=float(np.mean(dc)), pq_tiles=pq)
            print(f"{im:3s} images {v:20s} vs {g}: PQ {np.mean(pq):.4f}  DICE {np.mean(dc):.4f}", flush=True)
        # prediction stats
        n, area = [], []
        for t in ids:
            img = Image.open(idir / v / f"{t}.png")
            p = _cellvit_json_to_instance_mask(idir / v / f"{t}_cellvit_instances.json", image_hw=(img.height, img.width))
            ids_ = np.unique(p[p > 0]); n.append(len(ids_)); area += [int((p == i).sum()) for i in ids_]
        stats[f"{im}img/{v}"] = dict(n_per_tile=float(np.mean(n)), median_area=float(np.median(area)) if area else 0, fg=None)
for g, gdir in GT.items():
    n, area, fg, vals = [], [], [], set()
    for t in ids:
        arr = _load_spatial_array(gdir / f"{t}.png"); vals |= set(np.unique(arr).tolist()[:5])
        gi = gt(gdir, t, (256, 256)); u = np.unique(gi[gi > 0]); n.append(len(u)); area += [int((gi == i).sum()) for i in u]
        fg.append(float((gi > 0).mean()))
    stats[g] = dict(n_per_tile=float(np.mean(n)), median_area=float(np.median(area)), fg=float(np.mean(fg)),
                    shape=list(_load_spatial_array(gdir / f"{ids[0]}.png").shape), values=sorted(vals)[:6])
for k, s in stats.items(): print(k, s)
json.dump(dict(res=res, stats=stats), open(R / "pqmatrix.json", "w"), indent=1)
