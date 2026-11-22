"""Run a PixCell script unchanged, with one addition: torch.manual_seed(42) before every Vanilla ControlNet tile.

  python pxrun.py SCRIPT [ARGS ...]        (cwd = the PixCell root or its shadow)
tools/baselines/pixcell_offshelf_inference.py OffShelfPixCellInference.run_on_tile draws its noise with torch.randn_like
and sets no seed, so the paper's Vanilla images depend on the RNG state left by earlier tiles. The wrapper seeds each
tile (README deviation); nothing else in the script is touched.
"""
import functools
import os
import runpy
import sys

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402
from tools.baselines import pixcell_offshelf_inference as off  # noqa: E402

_orig = off.OffShelfPixCellInference.run_on_tile


@functools.wraps(_orig)
def _seeded(self, *args, **kwargs):
    torch.manual_seed(42)
    return _orig(self, *args, **kwargs)


off.OffShelfPixCellInference.run_on_tile = _seeded
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
