"""Alignment of the new cell_masks to CellViT on the real H&E: DICE over integer shifts and the 8 flips/transposes."""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask
R = Path(sys.argv[1])
sets = {"crc33": ([l.strip() for l in open(R / "ids100.txt") if l.strip()],
                  lambda t: R / f"p33/processed_crc33/cellvit/{t}.json",
                  lambda t: R / f"new33pack/orion-crc33sec/exp_channels/cell_masks/{t}.png"),
        "crc36": ([l.strip() for l in open(R / "c36/crc36ho_p300_ids.txt") if l.strip()],
                  lambda t: R / f"c36/cv36/{t}.json",
                  lambda t: R / f"c36/orion-crc36ho/exp_channels/cell_masks/{t}.png")}
M = 24
for name, (ids, cvp, mp) in sets.items():
    A = np.stack([_cellvit_json_to_instance_mask(cvp(t), image_hw=(256, 256)) > 0 for t in ids])
    B = np.stack([np.asarray(Image.open(mp(t))) > 0 for t in ids])
    def dice(a, b):
        i = (a & b).sum(); return 2 * i / (a.sum() + b.sum())
    best = []
    for dy in range(-M, M + 1):
        for dx in range(-M, M + 1):
            a = A[:, M:256 - M, M:256 - M]; b = B[:, M + dy:256 - M + dy, M + dx:256 - M + dx]
            best.append((dice(a, b), dy, dx))
    best.sort(reverse=True)
    a0 = A[:, M:256 - M, M:256 - M]; b0 = B[:, M:256 - M, M:256 - M]
    print(f"{name}: DICE at shift 0 = {dice(a0, b0):.4f}; best 3 shifts (dice, dy, dx): {[(round(d,4), y, x) for d, y, x in best[:3]]}")
    for k, f in {"flipud": lambda x: x[:, ::-1, :], "fliplr": lambda x: x[:, :, ::-1], "transpose": lambda x: x.transpose(0, 2, 1),
                 "rot180": lambda x: x[:, ::-1, ::-1]}.items():
        print(f"   {k:9s} DICE {dice(A, f(B)):.4f}")
    # per-tile best shift spread (coarse)
    sh = []
    for i in range(len(ids)):
        bb = max(((dice(A[i, M:256-M, M:256-M], B[i, M+dy:256-M+dy, M+dx:256-M+dx]), dy, dx) for dy in range(-M, M+1, 2) for dx in range(-M, M+1, 2)))
        sh.append(bb[1:])
    sh = np.array(sh); print(f"   per-tile best shift: median dy {np.median(sh[:,0])}, dx {np.median(sh[:,1])}; IQR dy {np.percentile(sh[:,0],25)}..{np.percentile(sh[:,0],75)}, dx {np.percentile(sh[:,1],25)}..{np.percentile(sh[:,1],75)}")
