"""hfv_retrain_20261118: which UNI-2h preprocessing made the stored orion-crc33 features/<t>_uni.npy?

  python unicheck.py PIXCELL DATA_ROOT OUT_JSON [N=50]
checks.py found the PixCell HEAD extractor (pipeline/extract_features.py UNI2hExtractor; since commit 4bf85d7 the model is
built offline as timm vit_giant_patch14_224, pretrained=False, so resolve_data_config reads timm's built-in config of that
architecture) does not reproduce the stored UNI features while the SD3.5 VAE latents match. Same weights and images
(cv2 read -> RGB, as extract_features.py), only the transform varies:
  A  extractor as shipped: resolve_data_config(model.pretrained_cfg)
  B  pretrained_cfg of pretrained_models/uni-2h/config.json (the UNI2-h hub config: ImageNet mean/std, crop_pct 1, bilinear)
  C  as B with bicubic interpolation
B is the preprocessing sim-exp-validation already uses (forward.py:load_uni_extractor, commit ea5d5bd9, 2026-09-30);
the L4 run (TAG l4u) found B reproduces the stored features to float precision, A and C do not.
Per variant: resolved config, max abs difference, median per-tile max abs difference, min / median cosine.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch
from timm.data import create_transform, resolve_data_config
from PIL import Image

PIX, DATA, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 50
os.chdir(PIX)
sys.path.insert(0, PIX)
from pipeline.extract_features import UNI2hExtractor  # noqa: E402

ids = sorted(f[:-8] for f in os.listdir(f"{DATA}/features") if f.endswith("_uni.npy"))
ids = sorted(np.random.default_rng(20261118).choice(ids, N, replace=False).tolist())
imgs = [Image.fromarray(cv2.cvtColor(cv2.imread(f"{DATA}/he/{t}.png"), cv2.COLOR_BGR2RGB)) for t in ids]
ref = np.stack([np.load(f"{DATA}/features/{t}_uni.npy").astype(np.float64).ravel() for t in ids])
ex = UNI2hExtractor("./pretrained_models/uni-2h", device="cuda")
hub = json.load(open("./pretrained_models/uni-2h/config.json"))["pretrained_cfg"]
cfgs = {"A_shipped": resolve_data_config(ex.model.pretrained_cfg, model=ex.model),
        "B_hub_cfg": resolve_data_config(hub, model=ex.model),
        "C_hub_bicubic": resolve_data_config({**hub, "interpolation": "bicubic"}, model=ex.model)}
R = dict(gpu=torch.cuda.get_device_name(0), n=N, tiles=ids)
for k, cfg in cfgs.items():
    tf = create_transform(**cfg)
    out = []
    with torch.no_grad():
        for i in range(0, N, 8):
            x = torch.stack([tf(im) for im in imgs[i:i + 8]]).cuda()
            out.append(ex.model(x).float().cpu().numpy())
    a = np.concatenate(out).astype(np.float64)
    mx = np.abs(a - ref).max(1)
    cos = (a * ref).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(ref, axis=1))
    R[k] = dict(cfg={kk: (list(v) if isinstance(v, tuple) else v) for kk, v in cfg.items()}, transform=str(tf),
                max_abs=float(mx.max()), median_tile_max_abs=float(np.median(mx)),
                min_cosine=float(cos.min()), median_cosine=float(np.median(cos)))
    print(k, {kk: R[k][kk] for kk in ("cfg", "max_abs", "median_tile_max_abs", "min_cosine", "median_cosine")}, flush=True)
json.dump(R, open(OUT, "w"), indent=1)
