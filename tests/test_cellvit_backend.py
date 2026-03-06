from pathlib import Path

import numpy as np
from PIL import Image


def test_cellvit_noop_writes_transparent_masks(tmp_path):
    """
    Contract:
    - `run_cellvit_noop(he_dir, out_dir)` creates `out_dir/cellvit_mask/*.png`
    - Each mask is RGBA, fully transparent by default.
    """
    he_dir = tmp_path / "he"
    he_dir.mkdir()
    out_dir = tmp_path / "cellvit_out"

    # Create two fake H&E patch PNGs
    for pid in ["0_0", "0_1"]:
        arr = np.zeros((16, 16, 3), dtype=np.uint8)
        Image.fromarray(arr).save(he_dir / f"{pid}.png")

    from cellvit_backend import run_cellvit_noop  # noqa: WPS433

    run_cellvit_noop(he_dir=str(he_dir), out_dir=str(out_dir))

    m0 = out_dir / "cellvit_mask" / "0_0.png"
    m1 = out_dir / "cellvit_mask" / "0_1.png"
    assert m0.exists() and m1.exists()

    im = Image.open(m0)
    assert im.mode == "RGBA"
    rgba = np.array(im)
    assert rgba.shape == (16, 16, 4)
    assert int(rgba[..., 3].max()) == 0
