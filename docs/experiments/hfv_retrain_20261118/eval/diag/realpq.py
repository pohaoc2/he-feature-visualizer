"""PQ / DICE of CellViT on the REAL H&E (processed_crc33/cellvit, paper stage 1) against old and new GT masks."""
import sys, json
from pathlib import Path
import numpy as np
from tools.compute_ablation_metrics import (_compute_pq, _compute_binary_segmentation_metrics, _cellvit_json_to_instance_mask,
    _label_binary_instances, _load_spatial_array)
R = Path(sys.argv[1]); ids = [l.strip() for l in open(R / "ids100.txt") if l.strip()]
GT = {"oldGT": R / "old33zip/orion-crc33/exp_channels/cell_masks", "newGT": R / "new33pack/orion-crc33sec/exp_channels/cell_masks"}
out = {}
for g, d in GT.items():
    pq, dc = [], []
    for t in ids:
        pred = _cellvit_json_to_instance_mask(R / f"p33/processed_crc33/cellvit/{t}.json", image_hw=(256, 256))
        a = _load_spatial_array(d / f"{t}.png"); gi = _label_binary_instances(a > 0)
        pq.append(_compute_pq(gi, pred)[2]); dc.append(_compute_binary_segmentation_metrics(gi, pred)[0])
    out[g] = dict(pq=float(np.mean(pq)), dice=float(np.mean(dc)))
    print(f"real H&E CellViT vs {g}: PQ {np.mean(pq):.4f} DICE {np.mean(dc):.4f}")
# old GT vs new GT directly (as instance maps)
pq, dc = [], []
for t in ids:
    a = _label_binary_instances(_load_spatial_array(GT["oldGT"] / f"{t}.png") > 0)
    b = _label_binary_instances(_load_spatial_array(GT["newGT"] / f"{t}.png") > 0)
    pq.append(_compute_pq(a, b)[2]); dc.append(_compute_binary_segmentation_metrics(a, b)[0])
print(f"old GT vs new GT: PQ {np.mean(pq):.4f} DICE {np.mean(dc):.4f}")
