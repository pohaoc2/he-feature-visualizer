"""Tests for the opt-in official CellViT post-processing in stages/run_cellvit_local.py.

The CellViT repository is not needed: a fake model stands in for calculate_instance_map.
"""
import json
import zipfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
pytest.importorskip("torchvision")

from stages import run_cellvit_local as RC  # noqa: E402


def test_default_postprocess_is_runner():
    a = RC.parse_args(["--zip", "z.zip", "--out", "o", "--checkpoint", "c.pth", "--cellvit-repo", "r"])
    assert a.postprocess == "runner"
    assert a.magnification == 40
    assert a.save_instances is None


def test_checkpoint_normalisation_flat_nested_and_default():
    flat = {"transformations.normalize.mean": [0.5, 0.5, 0.5], "transformations.normalize.std": [0.5, 0.5, 0.5]}
    assert RC.checkpoint_normalisation(flat) == ([0.5] * 3, [0.5] * 3)
    nested = {"transformations": {"normalize": {"mean": [0.1, 0.2, 0.3], "std": [0.4, 0.5, 0.6]}}}
    assert RC.checkpoint_normalisation(nested) == ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    assert RC.checkpoint_normalisation({}) == ([0.5] * 3, [0.5] * 3)


def test_official_cell_dict_format():
    cell = {"bbox": np.array([[2, 3], [10, 12]]), "centroid": np.array([7.123, 6.456]),
            "contour": np.array([[3, 2], [11, 2], [11, 9], [3, 9]], dtype=np.int32), "type_prob": 0.8333, "type": 1}
    d = RC.official_cell_dict(cell)
    assert d["centroid"] == [7.12, 6.46]
    assert d["contour"] == [[3, 2], [11, 2], [11, 9], [3, 9]]
    assert d["bbox"] == [[2, 3], [10, 12]]
    assert d["type_cellvit"] == 1 and d["type_name"] == "Neoplastic" and d["type_prob"] == 0.8333
    json.dumps(d)  # serialisable


class FakeModel(torch.nn.Module):
    """Two-class binary map, 6 type classes, records what calculate_instance_map receives."""

    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x):
        b, _, h, w = x.shape
        return {"nuclei_binary_map": torch.full((b, 2, h, w), -10.0), "nuclei_type_map": torch.zeros(b, 6, h, w),
                "hv_map": torch.zeros(b, 2, h, w), "seen_mean": x.mean()}

    def calculate_instance_map(self, pred, magnification=40):
        self.seen = (pred, magnification)
        b, _, h, w = pred["nuclei_binary_map"].shape
        inst = torch.zeros(b, h, w)
        inst[:, 2:6, 2:6] = 1
        cells = [{1: {"bbox": np.array([[2, 2], [6, 6]]), "centroid": np.array([3.5, 3.5]),
                      "contour": np.array([[2, 2], [5, 2], [5, 5], [2, 5]]), "type_prob": 1.0, "type": 5}}
                 for _ in range(b)]
        return inst, cells


def test_official_postprocess_softmax_and_types():
    m = FakeModel()
    out = m(torch.zeros(2, 3, 16, 16))
    inst, cells = RC.official_postprocess(m, out, magnification=20)
    pred, mag = m.seen
    assert mag == 20
    # softmax of zeros = uniform probabilities that sum to 1 over the channel axis
    assert torch.allclose(pred["nuclei_binary_map"].sum(1), torch.ones(2, 16, 16))
    assert torch.allclose(pred["nuclei_type_map"], torch.full((2, 6, 16, 16), 1 / 6))
    assert inst.dtype == np.int32 and inst.shape == (2, 16, 16) and inst[0, 3, 3] == 1
    assert len(cells) == 2 and cells[0][0]["type_cellvit"] == 5 and cells[0][0]["type_name"] == "Epithelial"


def _png_zip(tmp_path, n=3):
    from PIL import Image
    z = tmp_path / "he.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in range(n):
            p = tmp_path / f"t{i}.png"
            Image.fromarray(np.full((16, 16, 3), 200, np.uint8)).save(p)
            zf.write(p, arcname=f"t{i}.png")
    return z


def _run(tmp_path, monkeypatch, extra):
    m = FakeModel()
    cfg = {"transformations.normalize.mean": [0.5] * 3, "transformations.normalize.std": [0.5] * 3}
    calls = {}

    def fake_build(ckpt, repo, device, return_config=False):
        calls["return_config"] = return_config
        return (m, 6, cfg) if return_config else (m, 6)

    monkeypatch.setattr(RC, "build_model", fake_build)
    out = tmp_path / "cv"
    RC.main(["--zip", str(_png_zip(tmp_path)), "--out", str(out), "--checkpoint", "x", "--cellvit-repo", "y",
             "--num-workers", "0", "--batch-size", "2"] + extra)
    return out, m


def test_main_official_writes_runner_format_and_instances(tmp_path, monkeypatch):
    npz = tmp_path / "inst.npz"
    out, m = _run(tmp_path, monkeypatch, ["--postprocess", "official", "--save-instances", str(npz)])
    js = sorted(out.glob("*.json"))
    assert [j.stem for j in js] == ["t0", "t1", "t2"]
    d = json.loads(js[0].read_text())
    assert d["patch"] == "t0" and d["cells"][0]["type_name"] == "Epithelial"
    z = np.load(npz)
    assert z["stems"].tolist() == ["t0", "t1", "t2"] and z["inst"].shape == (3, 16, 16)


def test_main_runner_default_does_not_call_official(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(RC, "official_postprocess", lambda *a, **k: called.append(1))
    out, m = _run(tmp_path, monkeypatch, [])
    assert not called and m.seen is None
    assert json.loads((out / "t0.json").read_text())["cells"] == []  # sigmoid(-10) < 0.5 -> no nuclei


def test_runner_normalisation_unchanged():
    assert RC.IMAGENET_MEAN == [0.485, 0.456, 0.406] and RC.IMAGENET_STD == [0.229, 0.224, 0.225]


def test_save_instances_requires_official(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run(tmp_path, monkeypatch, ["--save-instances", str(tmp_path / "x.npz")])
