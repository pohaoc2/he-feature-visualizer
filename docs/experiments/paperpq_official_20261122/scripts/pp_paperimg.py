"""paperpq_official_20261122 PREREG §9 (descriptive, not a gate): regenerated images on the paper input (the runner-consistent
root P_run) vs the paper's own Fig 2 images (S3 share_space/04_inference/si_a1_a2/tiles/<variant>). Per tile: pixel-identical
or not, mean absolute pixel difference, LPIPS (PixCell metrics_io._load_lpips / _lpips_pair, AlexNet, CPU).
production is listed for description only (its paper images were overwritten, cellvit_official PREREG §5.2).
  python pp_paperimg.py D PAPERDIR      PAPERDIR/<variant>/<tile>.png ; writes D/paperimg/paperimg_per_tile.csv
"""
import csv
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from tools.ablation_a1_a2.metrics_io import _load_lpips, _lpips_pair

D, PAPER = Path(sys.argv[1]), Path(sys.argv[2])
ids = (D / "prep/ids100.txt").read_text().split()
VS = ["a1_concat", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf", "production"]
loss, dev = _load_lpips("cpu")
tmp = D / "paperimg/tmp"
tmp.mkdir(parents=True, exist_ok=True)
rows = []
with tarfile.open(D / "gen/P_run.tar") as tf:
    for v in VS:
        for t in ids:
            p = PAPER / v / f"{t}.png"
            if not p.exists():
                rows.append(dict(variant=v, tile=t, paper_present=0, identical="", mad="", lpips=""))
                continue
            g = tmp / f"{v}__{t}.png"
            g.write_bytes(tf.extractfile(f"out/tiles/{v}/{t}.png").read())
            a = np.asarray(Image.open(p).convert("RGB"), dtype=np.int16)
            b = np.asarray(Image.open(g).convert("RGB"), dtype=np.int16)
            same = int(a.shape == b.shape and np.array_equal(a, b))
            mad = float(np.abs(a - b).mean()) if a.shape == b.shape else float("nan")
            rows.append(dict(variant=v, tile=t, paper_present=1, identical=same, mad=mad,
                             lpips=0.0 if same else _lpips_pair(loss, dev, p, g)))
            g.unlink()
        sub = [r for r in rows if r["variant"] == v and r["paper_present"] == 1]
        print(v, "paper images", len(sub), "identical", sum(r["identical"] for r in sub),
              "mean LPIPS", np.mean([r["lpips"] for r in sub]) if sub else None, flush=True)
with open(D / "paperimg/paperimg_per_tile.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
