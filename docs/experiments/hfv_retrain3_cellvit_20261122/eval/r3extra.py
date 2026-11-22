"""hfv_retrain3_cellvit_20261122 (PREREG §5.2 items 1-2): per-tile quantities evalscore3 / fig2tiles do not write, Hyak CPU.

  PYTHONPATH=<PixCell cb62fa7> HFV=<hfv copy> python r3extra.py GROUP_DIR PACK_ROOT CV_GLOB OUT_CSV
GROUP_DIR  scored Fig 2 group (cache.json, tiles/<variant>/..., fig2_per_tile.csv)
PACK_ROOT  orion-<set> root of evalpack3 (he/<t>.png, exp_channels/cell_masks/<t>.png = the official-CellViT input mask)
CV_GLOB    glob of this run's official CellViT JSON folders of the real H&E (e.g. R/cv/c36ho_*)
Per tile of fig2_per_tile.csv:
  real-H&E reference line: PQ (_compute_pq) and DICE (_compute_binary_segmentation_metrics) of the real H&E's own
    official CellViT (_cellvit_json_to_instance_mask) against the input mask read by _load_gt_instance_mask
  hematoxylin readouts (sim-exp-validation mask_truenuc_20261122 truenuc.py pix(), hematoxylin =
    stages.lin_typing.compare.hematoxylin, t = 0.0224) of the input mask on the real H&E and on each of the 5 images:
    a = mask pixels with H >= t / mask pixels; b = H >= t pixels under the mask / H >= t pixels;
    c = mean H inside - mean H outside; f1 = 2ab / (a + b)
-> OUT_CSV rows (tile, model, a, b, c, f1, frac[, pq_real, dice_real for model = real])
"""
import csv
import glob
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from tools.ablation_a1_a2.metrics_io import _compute_binary_segmentation_metrics, _image_path
from tools.compute_ablation_metrics import _cellvit_json_to_instance_mask, _compute_pq, _load_gt_instance_mask

sys.path.insert(0, os.environ["HFV"])
from stages.lin_typing.compare import hematoxylin  # noqa: E402

T = 0.0224
V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
G, PACK, CVG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])


def pix(hod, fg, t):
    """truenuc.py pix(), copied verbatim"""
    pos = hod >= t
    nf, npos = int(fg.sum()), int(pos.sum())
    both = int((fg & pos).sum())
    a = both / nf if nf else np.nan
    b = both / npos if npos else np.nan
    c = float(hod[fg].mean() - hod[~fg].mean()) if 0 < nf < fg.size else np.nan
    f1 = 2 * a * b / (a + b) if (nf and npos and (a + b) > 0) else np.nan
    return dict(a=a, b=b, c=c, f1=f1, frac=nf / fg.size)


js = {Path(p).stem: p for p in glob.glob(f"{CVG}/*.json")}
tiles = sorted({r["tile"] for r in csv.DictReader(open(G / "fig2_per_tile.csv"))})
miss = [t for t in tiles if t not in js]
if miss:
    sys.exit(f"STOP {len(miss)} tiles without a real-H&E CellViT JSON, e.g. {miss[:3]}")
rows = []
for i, t in enumerate(tiles):
    fg = np.asarray(Image.open(PACK / "exp_channels/cell_masks" / f"{t}.png")) > 0
    gt = _load_gt_instance_mask(PACK, t, shape=(256, 256))
    pr = _cellvit_json_to_instance_mask(js[t], image_hw=(256, 256))
    dice = _compute_binary_segmentation_metrics(gt, pr)[0]
    real = np.asarray(Image.open(PACK / "he" / f"{t}.png").convert("RGB"))
    rows.append(dict(tile=t, model="real", **pix(hematoxylin(real), fg, T), pq_real=float(_compute_pq(gt, pr)[2]),
                     dice_real=float(dice)))
    for v in V5:
        img = np.asarray(Image.open(_image_path(G / "tiles", v, t)).convert("RGB"))
        rows.append(dict(tile=t, model=v, **pix(hematoxylin(img), fg, T), pq_real="", dice_real=""))
    if (i + 1) % 50 == 0:
        print(f"{i + 1}/{len(tiles)}", flush=True)
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("wrote", OUT, len(rows))
