"""hfv_retrain_20261118 card check (Colab): does this GPU + environment reproduce the stored orion-crc33 features?

  python checks.py PIXCELL DATA_ROOT OLD_MASK_DIR OUT_JSON [N=50]
Re-encodes N tiles (sorted tile ids, numpy default_rng(20261118) sample) with PixCell stage1_extract_features.py
(unmodified, default batch size) and compares with the files shipped in orion-crc33.zip:
  UNI-2h   he/<t>.png              -> features/<t>_uni.npy
  SD3.5 VAE he/<t>.png             -> vae_features/<t>_sd3_vae.npy
  SD3.5 VAE old cell_masks/<t>.png -> vae_features/<t>_mask_sd3_vae.npy   (the masks the stored latents were made from)
Reported per pass: max abs difference, median per-tile max abs difference, min cosine similarity. Descriptive only; the
numbers go to the coordinator with the probe (memory rule: UNI encode check on every card).
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

PIX, DATA, OLDM, OUT = sys.argv[1:5]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 50
ids = sorted(f[:-8] for f in os.listdir(f"{DATA}/features") if f.endswith("_uni.npy"))
ids = sorted(np.random.default_rng(20261118).choice(ids, N, replace=False).tolist())
T = tempfile.mkdtemp()
for d, src in (("he", f"{DATA}/he"), ("mask", OLDM)):
    os.makedirs(f"{T}/{d}")
    for t in ids:
        shutil.copy(f"{src}/{t}.png", f"{T}/{d}/{t}.png")


def run(*a):
    subprocess.run([sys.executable, "stage1_extract_features.py", *a], cwd=PIX, check=True)


run("--image-dir", f"{T}/he", "--output-dir", f"{T}/uni", "--skip-vae")
run("--image-dir", f"{T}/he", "--output-dir", f"{T}/vae", "--skip-uni")
run("--image-dir", f"{T}/mask", "--output-dir", f"{T}/mvae", "--skip-uni", "--vae-prefix", "mask_sd3_vae")


def cmp(new_dir, old_dir, suffix):
    mx, cos = [], []
    for t in ids:
        a = np.load(f"{new_dir}/{t}_{suffix}.npy").astype(np.float64).ravel()
        b = np.load(f"{old_dir}/{t}_{suffix}.npy").astype(np.float64).ravel()
        mx.append(float(np.abs(a - b).max()))
        cos.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return dict(n=len(ids), max_abs=max(mx), median_tile_max_abs=float(np.median(mx)), min_cosine=min(cos))


import torch  # noqa: E402

R = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, tiles=ids,
         uni=cmp(f"{T}/uni", f"{DATA}/features", "uni"),
         vae_he=cmp(f"{T}/vae", f"{DATA}/vae_features", "sd3_vae"),
         vae_mask_old=cmp(f"{T}/mvae", f"{DATA}/vae_features", "mask_sd3_vae"))
json.dump(R, open(OUT, "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "tiles"}, indent=1))
