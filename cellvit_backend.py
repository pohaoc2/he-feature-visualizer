from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def run_cellvit_noop(*, he_dir: str, out_dir: str) -> None:
    """
    No-op backend: creates transparent RGBA masks per patch.
    This lets the rest of the pipeline run without CellViT installed.
    """
    he_path = Path(he_dir)
    out = Path(out_dir)
    mask_dir = out / "cellvit_mask"
    mask_dir.mkdir(parents=True, exist_ok=True)

    for p in sorted(he_path.glob("*.png")):
        im = Image.open(p)
        w, h = im.size
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(mask_dir / p.name)


def run_cellvit_cli(
    *,
    he_dir: str,
    out_dir: str,
    command: Sequence[str],
) -> None:
    """
    CLI backend runner (contract only).

    `command` should be a full argv list. The command is expected to read patches from `he_dir`
    and write outputs under `out_dir`. Exact behavior depends on the user's CellViT setup.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(list(command), check=True)

