import json

import numpy as np
import tifffile
from PIL import Image


def _write_ome(path, arr, axes: str):
    tifffile.imwrite(path, arr, ome=True, metadata={"axes": axes})


def test_pipeline_generates_he_and_mux_patches(tmp_path):
    """
    Contract:
    - `generate_processed(...)` writes a processed tree with index.json + he/ + mux_rgb/
    - Patch selection uses tissue filtering, so at least one patch is kept for a synthetic tissue blob.
    """
    # Build a small H&E OME-TIFF (CYX)
    he_h, he_w = 64, 64
    he = np.full((3, he_h, he_w), 255, dtype=np.uint8)
    he[:, 16:48, 16:48] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "CRC02-HE.ome.tif"
    _write_ome(str(he_path), he, axes="CYX")

    # Build a small 36-channel multiplex OME-TIFF (CYX)
    mux = np.zeros((36, he_h, he_w), dtype=np.uint16)
    mux[0, 16:48, 16:48] = 100
    mux[10, 16:48, 16:48] = 200
    mux[35, 16:48, 16:48] = 300
    mux_path = tmp_path / "CRC02.ome.tif"
    _write_ome(str(mux_path), mux, axes="CYX")

    out_dir = tmp_path / "processed"

    from pipeline_generate import generate_processed  # noqa: WPS433

    generate_processed(
        he_path=str(he_path),
        mux_path=str(mux_path),
        out_dir=str(out_dir),
        patch_size=16,
        stride=16,
        mux_rgb_channels=(0, 10, 35),
        tissue_min=0.05,
    )

    index_path = out_dir / "index.json"
    assert index_path.exists()
    data = json.loads(index_path.read_text())
    assert "patches" in data
    assert len(data["patches"]) >= 1

    # Confirm at least one patch file exists for the first index entry
    p0 = data["patches"][0]
    patch_id = f"{p0['i']}_{p0['j']}"
    assert (out_dir / "he" / f"{patch_id}.png").exists()
    assert (out_dir / "mux_rgb" / f"{patch_id}.png").exists()


def test_pipeline_generates_vasculature_and_proxies(tmp_path):
    """
    Contract:
    - If `cd31_channel` is provided, pipeline writes:
      - vasculature/{patch_id}.png (RGBA, red-ish where CD31 is high)
      - oxygen/{patch_id}.png and glucose/{patch_id}.png (grayscale/RGBA)
    - Oxygen proxy should be higher closer to vessels (distance-to-vessel).
    """
    he_h, he_w = 32, 32
    he = np.full((3, he_h, he_w), 255, dtype=np.uint8)
    he[:, :, :] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)  # all tissue
    he_path = tmp_path / "CRC02-HE.ome.tif"
    _write_ome(str(he_path), he, axes="CYX")

    mux = np.zeros((36, he_h, he_w), dtype=np.uint16)
    # CD31 stripe at x=8..10
    mux[31, :, 8:11] = 1000
    mux_path = tmp_path / "CRC02.ome.tif"
    _write_ome(str(mux_path), mux, axes="CYX")

    out_dir = tmp_path / "processed"
    from pipeline_generate import generate_processed  # noqa: WPS433

    generate_processed(
        he_path=str(he_path),
        mux_path=str(mux_path),
        out_dir=str(out_dir),
        patch_size=16,
        stride=16,
        mux_rgb_channels=(0, 10, 35),
        tissue_min=0.01,
        cd31_channel=31,
        proxy_lambda=3.0,
    )

    idx = json.loads((out_dir / "index.json").read_text())
    p0 = idx["patches"][0]
    patch_id = f"{p0['i']}_{p0['j']}"

    vasc_path = out_dir / "vasculature" / f"{patch_id}.png"
    oxy_path = out_dir / "oxygen" / f"{patch_id}.png"
    glu_path = out_dir / "glucose" / f"{patch_id}.png"
    assert vasc_path.exists()
    assert oxy_path.exists()
    assert glu_path.exists()

    # Oxygen closer to vessel stripe should be higher than far edge
    oxy = np.array(Image.open(oxy_path).convert("L"))
    # left edge far from x=8..10 within patch
    far = int(oxy[:, 0].mean())
    near = int(oxy[:, 9].mean())
    assert near > far


def test_pipeline_renders_cell_state_from_features_csv(tmp_path):
    """
    Contract:
    - If `features_csv` is provided, pipeline writes `cell_state/{patch_id}.png`.
    - The output should have non-zero alpha at cell locations.
    """
    he_h, he_w = 32, 32
    he = np.full((3, he_h, he_w), 255, dtype=np.uint8)
    he[:, :, :] = np.array([180, 60, 120], dtype=np.uint8).reshape(3, 1, 1)
    he_path = tmp_path / "CRC02-HE.ome.tif"
    _write_ome(str(he_path), he, axes="CYX")

    mux = np.zeros((36, he_h, he_w), dtype=np.uint16)
    mux_path = tmp_path / "CRC02.ome.tif"
    _write_ome(str(mux_path), mux, axes="CYX")

    # Two cells in patch 0_0 (x,y in pixel coords)
    features_csv = tmp_path / "CRC02.csv"
    features_csv.write_text(
        "Xt,Yt,Ki67,PCNA,Vimentin,Ecadherin\n"
        "5,6,100,1,1,10\n"
        "10,11,1,1,100,1\n"
    )

    out_dir = tmp_path / "processed"
    from pipeline_generate import generate_processed  # noqa: WPS433

    generate_processed(
        he_path=str(he_path),
        mux_path=str(mux_path),
        out_dir=str(out_dir),
        patch_size=16,
        stride=16,
        tissue_min=0.01,
        features_csv=str(features_csv),
        dot_radius=0,
    )

    idx = json.loads((out_dir / "index.json").read_text())
    patch_id = f"{idx['patches'][0]['i']}_{idx['patches'][0]['j']}"
    p = out_dir / "cell_state" / f"{patch_id}.png"
    assert p.exists()
    rgba = np.array(Image.open(p).convert("RGBA"))
    assert int(rgba[..., 3].max()) > 0

