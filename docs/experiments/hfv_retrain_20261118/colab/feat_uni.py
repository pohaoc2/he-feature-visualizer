"""hfv_retrain_20261118: PixCell stage1 UNI-2h extraction (unmodified pipeline.extract_features.main) with the image
transform taken from pretrained_models/uni-2h/config.json instead of timm's built-in vit_giant_patch14_224 config.

  python feat_uni.py PIXCELL VARIANT stage1-args...      VARIANT = hub | hub_bicubic  (unicheck.py variants B / C)
Used only if unicheck.py shows that this variant, not the shipped one, reproduces the stored orion-crc33 UNI features, so
the CRC36 features match the CRC33 features the generator trains with. Output files and format are stage1's.
"""
import json
import os
import sys

PIX, VAR = sys.argv[1], sys.argv[2]
os.chdir(PIX)
sys.path.insert(0, PIX)
from timm.data import create_transform, resolve_data_config  # noqa: E402
import pipeline.extract_features as X  # noqa: E402

_init = X.UNI2hExtractor.__init__


def init(self, *a, **k):
    _init(self, *a, **k)
    assert VAR in ("hub", "hub_bicubic"), VAR
    pre = json.load(open("./pretrained_models/uni-2h/config.json"))["pretrained_cfg"]
    if VAR == "hub":
        # the same construction as sim-exp-validation src/sim_exp_validation/forward.py:load_uni_extractor
        # (commit ea5d5bd9, 2026-09-30, "Fix UNI preprocessing"), so both repos share one UNI-2h preprocessing
        cfg = {key: pre[key] for key in ("input_size", "interpolation", "mean", "std", "crop_pct")}
        self.transform = create_transform(**cfg, is_training=False)
    else:
        self.transform = create_transform(**resolve_data_config({**pre, "interpolation": "bicubic"}, model=self.model))
    print(f"[feat_uni] {VAR} transform: {self.transform}", flush=True)


X.UNI2hExtractor.__init__ = init
sys.argv = ["stage1_extract_features.py"] + sys.argv[3:]
X.main()
