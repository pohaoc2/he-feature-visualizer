"""Run a PixCell cb62fa7 scoring script unchanged, with the README UNI-2h preprocessing (Hyak A100 scoring).

  python pxscore.py SCRIPT.py [ARGS ...]        (cwd = the PixCell root or its shadow)
  python pxscore.py -m MODULE [ARGS ...]
README "UNI-2h 前處理": every recomputed UNI feature, including the UNI of generated images at evaluation, uses the
transform of pretrained_models/uni-2h/config.json (bilinear, ImageNet mean/std, crop_pct 1), the construction of
colab/feat_uni.py "hub" = sim-exp-validation forward.py:load_uni_extractor (ea5d5bd9). PixCell's
pipeline.extract_features.UNI2hExtractor builds its transform from timm's built-in vit_giant_patch14_224 config, which does
not reproduce the stored real-tile features (README: min cosine 0.588), so FUD / UNI cosine would compare real features
and generated features made with two different transforms. This wrapper replaces only that transform, through an import
hook, so it also holds in worker processes (fork) and wherever the class is imported from. Nothing else is touched.
Also checks NVIDIA_TF32_OVERRIDE=0 (README rule 5: scoring with TF32 off) and prints the GPU, so every log records it.
"""
import importlib.abc
import importlib.util
import json
import os
import runpy
import sys

if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
    sys.exit("STOP pxscore: NVIDIA_TF32_OVERRIDE must be 0 (README rule 5)")
sys.path.insert(0, os.getcwd())
TARGET = "pipeline.extract_features"


def _patch(mod):
    from timm.data import create_transform

    init = mod.UNI2hExtractor.__init__

    def patched(self, model_path, *a, **k):
        init(self, model_path, *a, **k)
        pre = json.load(open(os.path.join(str(model_path), "config.json")))["pretrained_cfg"]
        cfg = {key: pre[key] for key in ("input_size", "interpolation", "mean", "std", "crop_pct")}
        self.transform = create_transform(**cfg, is_training=False)
        print(f"[pxscore] UNI-2h transform from {model_path}/config.json: {self.transform}", flush=True)

    mod.UNI2hExtractor.__init__ = patched


class _Hook(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path, target=None):
        if name != TARGET:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        self._orig = spec.loader
        spec.loader = self
        return spec

    def create_module(self, spec):
        return self._orig.create_module(spec)

    def exec_module(self, module):
        self._orig.exec_module(module)
        _patch(module)


if TARGET in sys.modules:
    _patch(sys.modules[TARGET])
else:
    sys.meta_path.insert(0, _Hook())

try:
    import torch

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA"
    print(f"[pxscore] GPU {gpu}; NVIDIA_TF32_OVERRIDE=0; torch {torch.__version__} "
          f"matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} cudnn.allow_tf32={torch.backends.cudnn.allow_tf32}",
          flush=True)
except ImportError:
    pass

if sys.argv[1] == "-m":
    mod = sys.argv[2]
    sys.argv = [mod] + sys.argv[3:]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
else:
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
