import sys
from pathlib import Path
import numpy as np
from tools.compute_ablation_metrics import (_compute_pq, _compute_binary_segmentation_metrics, _cellvit_json_to_instance_mask,
    _label_binary_instances, _load_spatial_array)
R = Path(sys.argv[1]); ids = [l.strip() for l in open(R / "crc36ho_p300_ids.txt") if l.strip()]
pq, dc, n1, n2 = [], [], [], []
for t in ids:
    pred = _cellvit_json_to_instance_mask(R / f"cv36/{t}.json", image_hw=(256, 256))
    gi = _label_binary_instances(_load_spatial_array(R / f"orion-crc36ho/exp_channels/cell_masks/{t}.png") > 0)
    pq.append(_compute_pq(gi, pred)[2]); dc.append(_compute_binary_segmentation_metrics(gi, pred)[0])
    n1.append(len(np.unique(gi)) - 1); n2.append(len(np.unique(pred)) - 1)
pq, dc = np.array(pq), np.array(dc)
rng = np.random.default_rng(42); bs = [pq[rng.integers(0, len(pq), len(pq))].mean() for _ in range(10000)]
print(f"CRC36 hold-out 300: real H&E CellViT vs new cell_masks PQ {pq.mean():.4f} [{np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}] DICE {dc.mean():.4f}; GT objects/tile {np.mean(n1):.1f}, CellViT objects/tile {np.mean(n2):.1f}")
np.save(R / "real_pq36.npy", pq)
