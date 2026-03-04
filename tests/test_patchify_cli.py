import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile


def _write_he(path, h, w):
    """CYX uint8 H&E: tissue-colored top-left 256x256, white background elsewhere."""
    arr = np.full((3, h, w), 255, dtype=np.uint8)
    arr[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8)[:, None, None]
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": "CYX"})


def _write_multiplex(path, h, w, n_ch=4):
    arr = np.zeros((n_ch, h, w), dtype=np.uint16)
    tifffile.imwrite(str(path), arr, ome=True, metadata={"axes": "CYX"})


def _write_meta_csv(path, names):
    lines = ["Channel ID,Target Name\n"] + [f"Channel:0:{i},{n}\n" for i, n in enumerate(names)]
    path.write_text("".join(lines))


def _run_patchify(extra_args, tmp_path):
    he = tmp_path / "he.ome.tif"
    mx = tmp_path / "mx.ome.tif"
    csv = tmp_path / "meta.csv"
    out = tmp_path / "processed"
    _write_he(he, 512, 512)
    _write_multiplex(mx, 512, 512, n_ch=4)
    _write_meta_csv(csv, ["CD31", "Ki67", "CD45", "PCNA"])
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "patchify.py"),
        "--he-image", str(he),
        "--multiplex-image", str(mx),
        "--metadata-csv", str(csv),
        "--out", str(out),
        "--stride", "256",
        "--channels", "CD31", "Ki67", "CD45", "PCNA",
    ] + extra_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result, out


def test_patchify_cli_writes_nonempty_index(tmp_path):
    """patchify CLI writes index.json with at least one tissue patch."""
    result, out = _run_patchify(["--tissue-min", "0.0"], tmp_path)
    assert result.returncode == 0, result.stderr

    index_path = out / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert len(data["patches"]) >= 1


def test_patchify_filters_dirty_gray_background_with_hsv_method(tmp_path):
    """CLAM tissue filter: gray patches are rejected, tissue-colored patch is kept."""
    he = tmp_path / "he.ome.tif"
    mx = tmp_path / "mx.ome.tif"
    csv = tmp_path / "meta.csv"
    out = tmp_path / "processed"

    # Gray background (low saturation), only top-left has tissue color
    h = w = 512
    arr = np.full((3, h, w), 200, dtype=np.uint8)
    arr[:, 0:256, 0:256] = np.array([180, 60, 120], dtype=np.uint8)[:, None, None]
    tifffile.imwrite(str(he), arr, ome=True, metadata={"axes": "CYX"})
    _write_multiplex(mx, 512, 512)
    _write_meta_csv(csv, ["CD31", "Ki67", "CD45", "PCNA"])

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "patchify.py"),
        "--he-image", str(he),
        "--multiplex-image", str(mx),
        "--metadata-csv", str(csv),
        "--out", str(out),
        "--stride", "256",
        "--channels", "CD31", "Ki67", "CD45", "PCNA",
        "--tissue-min", "0.5",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    data = json.loads((out / "index.json").read_text())
    kept = {(p["i"], p["j"]) for p in data["patches"]}
    assert kept == {(0, 0)}
