import hashlib
import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from stages.lin_typing import compare, labels, nuclei, render
from stages.lin_typing import celltype_celesta_fig3d as F

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "stages/lin_typing/celltype_celesta_fig3d.py"
# sha256 of sim-exp-validation scripts/analysis/tcmvp/celltype_celesta_fig3d.py at a3800e69
SEV_SHA256 = "55774345e86b77d24dc6c8869d1024c8149d09d4e7040afc4419e8f00a759837"
SEV_ROOT = Path(os.environ.get("SEV_ROOT", ROOT.parent / "sim-exp-validation"))
SEV_FIG3D = SEV_ROOT / "scripts/analysis/tcmvp/celltype_celesta_fig3d.py"


def test_vendored_typing_is_the_sim_side_file():
    assert hashlib.sha256(VENDORED.read_bytes()).hexdigest() == SEV_SHA256


@pytest.mark.skipif(not SEV_FIG3D.exists(), reason="sim-exp-validation checkout not found (set SEV_ROOT)")
def test_typing_matches_sim_side_function_now():
    spec = importlib.util.spec_from_file_location("sev_fig3d", SEV_FIG3D)
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    rng = np.random.default_rng(0)
    calls = list(F.KNOWN) + [None, ""]
    final = [calls[i] for i in rng.integers(len(calls), size=5000)]
    cls = rng.integers(len(F.LIN_L), size=5000)
    a, b = F.classify_celesta_fig3d(final, cls), S.classify_celesta_fig3d(final, cls)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_lin_class_at_uses_the_112um_grid():
    cmap = np.arange(6).reshape(2, 3)
    g = labels.LIN_GRID_UM / labels.MPP
    assert labels.lin_class_at([0, g + 1, 10 * g], [0, g + 1, -5], cmap).tolist() == [0, 4, 2]


def test_dead_from_cellvit_and_state_order():
    dead, n = labels.dead_from_cellvit([5, 6, 7], hfv_label=[5, 0, 7, 6], hfv_type_cellvit=[4, 4, 1, 1])
    assert dead.tolist() == [True, False, False]
    assert n == dict(cellvit_dead=2, cellvit_dead_on_background=1, lin_cells_dead=1)
    st = labels.cell_state([400, 400, 100, 331.5], 331.5, [True, False, False, False])
    assert st.tolist() == [labels.DEAD, labels.PROLIF, labels.NONPROLIF, labels.NONPROLIF]


def test_assign_combines_typing_and_state():
    ty, src, st = labels.assign(["Epithelial", "Unknown"], [3, 8], [500, 0], 331.5, [False, True])
    assert ty.tolist() == [F.CANCER, F.IMMUNE] and src.tolist() == [F.SRC_EPI, F.SRC_UNKNOWN]
    assert st.tolist() == [labels.PROLIF, labels.DEAD]


def _disk_at(shape, cy, cx, r):
    y, x = np.mgrid[:shape[0], :shape[1]]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def test_strict_erosion_recovers_an_isolated_dilated_nucleus():
    import cv2
    nuc = _disk_at((64, 64), 32, 30, 8)                  # convex: dilation then erosion gives it back exactly
    cell = cv2.dilate(nuc.astype(np.uint8), nuclei.disk(5)) > 0
    lab = cell.astype(np.int32) * 7
    strict, loose = nuclei.erode_labels(lab)
    assert np.array_equal(strict == 7, nuc) and np.array_equal(loose == 7, nuc)
    assert nuclei.touching_labels(lab).size == 0
    # non-convex: the closing keeps the nucleus and fills only concave notches narrower than the disk
    nuc2 = nuc | _disk_at((64, 64), 32, 38, 6)
    strict2, _ = nuclei.erode_labels((cv2.dilate(nuc2.astype(np.uint8), nuclei.disk(5)) > 0).astype(np.int32))
    assert not (nuc2 & ~(strict2 == 1)).any() and ((strict2 == 1) & ~nuc2).sum() <= 0.02 * nuc2.sum()


def test_touching_cells_bracket_the_nucleus():
    seg = pytest.importorskip("skimage.segmentation")
    n1, n2 = _disk_at((64, 80), 32, 30, 7), _disk_at((64, 80), 32, 47, 7)   # 3 px gap < 2 x 5 px dilation
    nuc = n1.astype(np.int32) + 2 * n2.astype(np.int32)
    lab = seg.expand_labels(nuc, distance=5)
    strict, loose = nuclei.erode_labels(lab)
    for k, n in ((1, n1), (2, n2)):
        assert not ((strict == k) & ~n).any()          # strict never exceeds the nucleus
        assert not (n & ~(loose == k)).any()           # loose always contains it
        assert (strict == k).sum() < n.sum()           # and strict loses some of it at the contact
    assert set(nuclei.touching_labels(lab).tolist()) == {1, 2}


def test_gap_separated_cells_as_in_lin_mask():
    seg = pytest.importorskip("skimage.segmentation")
    n1, n2 = _disk_at((64, 80), 32, 30, 7), _disk_at((64, 80), 32, 47, 7)
    lab = seg.expand_labels(n1.astype(np.int32) + 2 * n2.astype(np.int32), distance=5)
    gap = (lab == 1) & np.roll(lab == 2, -1, axis=1)          # cut a one-pixel background line at the contact
    lab[gap] = 0
    assert nuclei.gap_lines(lab).sum() >= gap.sum()
    strict, loose = nuclei.erode_labels(lab)
    for k, n in ((1, n1), (2, n2)):
        assert not ((strict == k) & ~n).any() and (strict == k).sum() < n.sum()
        assert not (n & ~(loose == k)).any()
    assert set(nuclei.touching_labels(lab).tolist()) == {1, 2}


def test_stamp_keeps_largest_part_and_rejects_big_objects():
    m = np.zeros((60, 60), bool)
    m[10:20, 10:30] = True
    m[40, 40] = True
    s, a = nuclei.stamp(m)
    assert s.shape == (48, 48) and a == 200 and s[19:29, 14:34].all()
    assert nuclei.stamp(np.ones((50, 10), bool)) == (None, 0)
    assert nuclei.stamp(np.zeros((5, 5), bool)) == (None, 0)


def test_target_area_falls_back_to_loose_bound():
    assert render.target_area([10, 0, 0], [20, 30, 0]).tolist() == [10, 30, 1]


def test_ink_tiles():
    assert render.ink_tiles({"a": 0.0, "b": 1e-4, "c": 0}) == {"b"}


def test_dice_and_auc():
    a = np.zeros((4, 4), bool); a[:2] = True
    assert compare.dice(a, a) == 1.0 and compare.dice(a, ~a) == 0.0
    assert compare.auc([2, 3], [0, 1]) == 1.0 and compare.auc([1], [1]) == 0.5 and compare.auc([0], [1]) == 0.0


SEV_SRC = SEV_ROOT / "src"


@pytest.fixture
def sev():
    if not (SEV_SRC / "sim_exp_validation/morphology_renderer.py").exists():
        pytest.skip("sim-exp-validation src not found (set SEV_ROOT)")
    return render.renderer(str(SEV_SRC))


def _library(rasterize):
    m = np.zeros((48, 48), bool)
    m[19:29, 19:29] = True                               # 10 x 10 px, unchanged by pick's quarter turns / flips
    return rasterize.NucleusShapes(np.stack([m, m, m]), np.array([0, 1, 2]), np.array([100, 100, 100]),
                                   nuclei.TYPE_NAMES)


def test_render_tile_uses_sim_renderer_and_partitions(sev):
    rasterize, mr, channels = sev
    lib = _library(rasterize)
    table = render.CellTable([11, 12], [1000 + 100, 1000 + 255], [2000 + 50, 2000 + 128],
                             [labels.POP_NAMES.index("cancer"), labels.POP_NAMES.index("immune")],
                             [labels.PROLIF, labels.DEAD], [100, 100])
    t, inst = render.render_tile(table, 1000, 2000, lib, mr)
    names = channels.CHANNEL_NAMES[:7]
    assert t.shape == (7, 256, 256)
    ch = dict(zip(names, t))
    assert ch["cell_masks"].sum() == 100 + 60             # second cell straddles the right edge (6 of 10 columns)
    types = ch["cell_type_healthy"] + ch["cell_type_cancer"] + ch["cell_type_immune"]
    states = ch["cell_state_prolif"] + ch["cell_state_nonprolif"] + ch["cell_state_dead"]
    assert np.array_equal(types, ch["cell_masks"]) and np.array_equal(states, ch["cell_masks"])
    assert ch["cell_type_cancer"].sum() == 100 and ch["cell_state_dead"].sum() == 60
    assert [i["id"] for i in inst] == [11, 12] and inst[1]["clipped"]
    # the straddling cell keeps the same outline in the neighbouring tile
    t2, _ = render.render_tile(table, 1256, 2000, lib, mr)
    assert dict(zip(names, t2))["cell_masks"].sum() == 40


def test_tile_agents_match_agent_stamps_contract(sev):
    rasterize, mr, _ = sev
    lib = _library(rasterize)
    table = render.CellTable([3], [10], [10], [0], [1], [100])
    agents, pos = render.tile_agents(table, 0, 0, lib)
    a = agents[0]
    assert a["outline"].shape == (48, 48) and a["type"] == "healthy" and a["state"] == "nonprolif"
    assert pos[3]["x"] == 10 and pos[3]["y"] == 10
    assert render.tile_agents(table, 512, 512, lib) == ([], {})
