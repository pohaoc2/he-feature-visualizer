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
    cfg = json.load(open("./pretrained_models/uni-2h/config.json"))["pretrained_cfg"]
    if VAR == "hub_bicubic":
        cfg = {**cfg, "interpolation": "bicubic"}
    assert VAR in ("hub", "hub_bicubic"), VAR
    self.transform = create_transform(**resolve_data_config(cfg, model=self.model))
    print(f"[feat_uni] {VAR} transform: {self.transform}", flush=True)


X.UNI2hExtractor.__init__ = init
sys.argv = ["stage1_extract_features.py"] + sys.argv[3:]
X.main()
