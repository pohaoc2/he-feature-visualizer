"""hfv_retrain_20261118 追加 3, decision 4(b) and the CPU timing for the placement rule.

  PYTHONPATH=<PixCell cb62fa7> python a3tok.py BUNDLE_DIR HE_DIR UNI_DIR VIRCHOW_WEIGHTS CTRANSPATH_PTH OUT_JSON

BUNDLE_DIR = decodability_cpu_bundle/ (paper UNI patch tokens uni_tokens/<id>_uni_tokens.npy, fp16, 250 CRC33 tiles).
Re-extracts UNI-2h patch tokens for those tiles with PixCell's own UNI2hExtractor.extract_patch_tokens_batch (images read
as pipeline/extract_features.py reads them: cv2.imread -> RGB), once with its built-in transform (timm
vit_giant_patch14_224 config) and once with the transform of uni-2h/config.json (the one the stored CLS features match;
same construction as eval/pxscore.py). Reports, per transform, the per-tile mean cosine over the 256 patches against the
paper tokens (median / min / max over tiles) and the relative L2 error. No threshold: the transform with the higher median
cosine is the one used (README 追加 3, decision 4b). Also times the four encoders on CPU (s / image) through
pipeline.extract_features._build_patch_extractor, the code path of `extract_features.py --encoder`.
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
from timm.data import create_transform  # noqa: E402

from pipeline.extract_features import UNI2hExtractor, _build_patch_extractor  # noqa: E402

bundle, he_dir, uni_dir, vir_w, ctp_w, out_json = sys.argv[1:7]
bundle, he_dir = Path(bundle), Path(he_dir)
ids = sorted(p.name[: -len("_uni_tokens.npy")] for p in (bundle / "uni_tokens").glob("*_uni_tokens.npy"))
print(f"{len(ids)} paper token files", flush=True)


def load(t):
    im = cv2.imread(str(he_dir / f"{t}.png"))
    if im is None:
        sys.exit(f"STOP cannot read {he_dir / f'{t}.png'}")
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


ext = UNI2hExtractor(uni_dir, device="cpu")
pre = json.load(open(Path(uni_dir) / "config.json"))["pretrained_cfg"]
transforms = {"timm_builtin": ext.transform,
              "uni2h_config_json": create_transform(**{k: pre[k] for k in ("input_size", "interpolation", "mean", "std",
                                                                            "crop_pct")}, is_training=False)}
res = {"n_tiles": len(ids), "transforms": {k: str(v) for k, v in transforms.items()}, "uni": {}, "timing_s_per_image": {}}
B = 16
for name, tf in transforms.items():
    ext.transform = tf
    cos, rel = [], []
    t0 = time.time()
    for i in range(0, len(ids), B):
        chunk = ids[i:i + B]
        got = ext.extract_patch_tokens_batch([load(t) for t in chunk]).astype(np.float32)
        for t, g in zip(chunk, got):
            ref = np.load(bundle / "uni_tokens" / f"{t}_uni_tokens.npy").astype(np.float32)
            if ref.shape != g.shape:
                sys.exit(f"STOP shape {ref.shape} vs {g.shape} for {t}")
            c = (ref * g).sum(1) / (np.linalg.norm(ref, axis=1) * np.linalg.norm(g, axis=1) + 1e-12)
            cos.append(float(c.mean()))
            rel.append(float(np.linalg.norm(ref - g) / (np.linalg.norm(ref) + 1e-12)))
    dt = (time.time() - t0) / len(ids)
    cos, rel = np.array(cos), np.array(rel)
    res["uni"][name] = {"cos_median": float(np.median(cos)), "cos_min": float(cos.min()), "cos_max": float(cos.max()),
                        "rel_l2_median": float(np.median(rel)), "rel_l2_max": float(rel.max())}
    res["timing_s_per_image"][f"uni_tokens_{name}"] = dt
    print(name, json.dumps(res["uni"][name]), f"{dt:.3f} s/img", flush=True)
res["uni_transform_used"] = max(res["uni"], key=lambda k: res["uni"][k]["cos_median"])
del ext

imgs = [load(t) for t in ids[:32]]
for enc, w in (("virchow2", vir_w), ("ctranspath", ctp_w), ("resnet50", None)):
    f = _build_patch_extractor(encoder=enc, uni_model_path=uni_dir, encoder_model_path=w, device="cpu")
    f(imgs[:2])
    t0 = time.time()
    out = f(imgs)
    res["timing_s_per_image"][enc] = (time.time() - t0) / len(imgs)
    res.setdefault("patch_shape", {})[enc] = list(out.shape[1:])
    print(enc, out.shape, f"{res['timing_s_per_image'][enc]:.3f} s/img", flush=True)
json.dump(res, open(out_json, "w"), indent=1)
print(json.dumps(res, indent=1))
