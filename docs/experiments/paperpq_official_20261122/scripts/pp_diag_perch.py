"""paperpq_official_20261122 gate diagnosis (Per-channel): paper-style PQ / DICE of the paper's OWN Fig 2 images, scored from
the paper's own CellViT sidecars (S3 si_a1_a2/tiles/<variant>/<tile>_cellvit_instances.json) against the paper cell_masks
(P_run root), with the same PixCell functions as pp_score.py. Also the same for our regenerated P_run images (runner CellViT).
  python pp_diag_perch.py D SIDEDIR    SIDEDIR/<variant>/<tile>_cellvit_instances.json ; prints per-variant means
"""
import json
import sys
from pathlib import Path

import numpy as np

from tools.compute_ablation_metrics import (_cellvit_json_to_instance_mask, _compute_binary_segmentation_metrics,
                                            _compute_pq, _load_gt_instance_mask)

D, SIDE = Path(sys.argv[1]), Path(sys.argv[2])
ids = (D / "prep/ids100.txt").read_text().split()
root = D / "roots/P_run"
out = {}
for v in ["a1_per_channel", "a1_concat", "a2_bypass_full_tme", "a2_off_shelf"]:
    res = {"paper_sidecar": [], "ours_runner": []}
    for t in ids:
        gt = _load_gt_instance_mask(root, t)
        for k, js in (("paper_sidecar", SIDE / v / f"{t}_cellvit_instances.json"),
                      ("ours_runner", D / "cv/runner" / f"P_run__{v}" / f"{t}.json")):
            if not js.exists():
                continue
            pred = _cellvit_json_to_instance_mask(js, image_hw=(256, 256))
            pq = _compute_pq(gt, pred)[2]
            dice = _compute_binary_segmentation_metrics(gt, pred)[0]
            n = len(np.unique(pred)) - 1
            res[k].append((pq, dice, n))
    out[v] = {k: dict(n_tiles=len(x), pq=float(np.mean([a[0] for a in x])) if x else None,
                      dice=float(np.mean([a[1] for a in x])) if x else None,
                      nuclei=float(np.mean([a[2] for a in x])) if x else None) for k, x in res.items()}
    print(v, json.dumps(out[v]), flush=True)
json.dump(out, open(D / "paperimg/diag_perch.json", "w"), indent=1)
