import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


def test_patchify_cli_writes_nonempty_index(tmp_path):
    """
    Regression:
    - patchify.py should iterate all patch rows (not a hard-coded debug row).
    - For a simple 512x512 OME-TIFF and a single cell, it should keep at least one patch.
    """
    # 512x512 RGB tissue-ish image (CYX)
    h = w = 512
    he = np.full((3, h, w), 255, dtype=np.uint8)
    he[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "CRC02-HE.ome.tif"
    tifffile.imwrite(str(he_path), he, ome=True, metadata={"axes": "CYX"})

    # Minimal features CSV with one cell in top-left patch
    features_path = tmp_path / "CRC02.csv"
    features_path.write_text("Xt,Yt\n50,60\n")

    out_dir = tmp_path / "processed"

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "patchify.py"),
        "--image",
        str(he_path),
        "--features-csv",
        str(features_path),
        "--out",
        str(out_dir),
        "--stride",
        "256",
        "--features",
        "cell_mask",
        "--tissue-min",
        "0.0",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    index_path = out_dir / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert len(data["patches"]) >= 1


def test_patchify_filters_dirty_gray_background_with_hsv_method(tmp_path):
    """
    Regression for CLAM-like tissue filtering:
    - A uniform gray patch (low saturation) should be rejected as non-tissue even if it's dark.
    - Only the truly tissue-colored patch should remain when tissue_min is high.
    """
    h = w = 512
    # Dark-ish gray background (would be misclassified as tissue by grayscale thresholding)
    he = np.full((3, h, w), 200, dtype=np.uint8)
    # Only top-left patch has real "tissue color"
    he[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "CRC02-HE.ome.tif"
    tifffile.imwrite(str(he_path), he, ome=True, metadata={"axes": "CYX"})

    # One cell per patch so the cell-count filter doesn't decide the result
    features_path = tmp_path / "CRC02.csv"
    features_path.write_text(
        "Xt,Yt\n"
        "128,128\n"
        "384,128\n"
        "128,384\n"
        "384,384\n"
    )

    out_dir = tmp_path / "processed"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "patchify.py"),
        "--image",
        str(he_path),
        "--features-csv",
        str(features_path),
        "--out",
        str(out_dir),
        "--stride",
        "256",
        "--features",
        "cell_mask",
        "--tissue-min",
        "0.5",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    data = json.loads((out_dir / "index.json").read_text())
    kept = {(p["i"], p["j"]) for p in data["patches"]}
    assert kept == {(0, 0)}

