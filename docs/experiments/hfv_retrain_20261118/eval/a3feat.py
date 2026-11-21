"""hfv_retrain_20261118 追加 3, step C: patch features on CPU with PixCell's own code path (cb62fa7 == d1fc8a5 for these files).

  PYTHONPATH=<PixCell cb62fa7> python a3feat.py ENCODER IMAGE_DIR OUT_DIR UNI_DIR [ENCODER_MODEL] [A3TOK_JSON]

Calls pipeline.extract_features.cache_patch_features (the body of `extract_features.py --encoder ENCODER`), which reads
images with cv2 and uses patch_extractors.extract_{uni,virchow,ctranspath,resnet50}_patches. UNI files are named
<id>_uni_tokens.npy (as the paper's --save-uni-tokens cache), the others <id>_patches.npy (as features_patches/).
For UNI the transform is the one decision 4(b) picked from A3TOK_JSON ("uni_transform_used"): timm_builtin = PixCell as is;
uni2h_config_json = UNI2hExtractor.transform replaced after __init__ by the uni-2h/config.json transform (same
construction as eval/pxscore.py). PixCell files are not changed.
"""
import json
import os
import sys
from pathlib import Path

import torch

torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
from pipeline import extract_features as ef  # noqa: E402

enc, img_dir, out_dir, uni_dir = sys.argv[1:5]
model = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != "-" else None
tok = sys.argv[6] if len(sys.argv) > 6 else None
if Path(out_dir).exists():
    sys.exit(f"STOP {out_dir} exists (new dirs only)")
if enc == "uni":
    choice = json.load(open(tok))["uni_transform_used"]
    print(f"[a3feat] UNI transform: {choice}", flush=True)
    if choice == "uni2h_config_json":
        from timm.data import create_transform
        init = ef.UNI2hExtractor.__init__

        def patched(self, model_path, *a, **k):
            init(self, model_path, *a, **k)
            pre = json.load(open(os.path.join(str(model_path), "config.json")))["pretrained_cfg"]
            self.transform = create_transform(**{key: pre[key] for key in ("input_size", "interpolation", "mean", "std",
                                                                           "crop_pct")}, is_training=False)
            print(f"[a3feat] transform from config.json: {self.transform}", flush=True)

        ef.UNI2hExtractor.__init__ = patched
    elif choice != "timm_builtin":
        sys.exit(f"STOP unknown transform choice {choice}")
ef.cache_patch_features(img_dir, out_dir, encoder=enc, uni_model_path=uni_dir, encoder_model_path=model, batch_size=16,
                        device="cpu", patches_prefix="uni_tokens" if enc == "uni" else "patches")
n_in = len(list(Path(img_dir).glob("*.png")))
n_out = len(list(Path(out_dir).glob("*.npy")))
print(f"[a3feat] {enc}: {n_out} feature files for {n_in} images")
if n_in != n_out:
    sys.exit("STOP feature count != image count")
