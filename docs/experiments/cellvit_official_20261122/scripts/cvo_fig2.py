"""cellvit_official_20261122 (PREREG.md §5): per-tile PQ of the hfv paper Fig 2 under both CellViT post-processings.
Hyak CPU; nothing generated; CellViT JSONs from cvo_infer.slurm (official: $D/full/cv, runner rerun: $D/full_runner/cv;
real H&E of the retrained sets: official $D/full/cv/R_<set>, runner = hfv_retrain2 stage3/cv/R_<set>, the same runner).

  PYTHONPATH=<PixCell cb62fa7> python cvo_fig2.py D OUT_DIR

PixCell functions only (tools/compute_ablation_metrics.py): _cellvit_json_to_instance_mask (contours filled),
_label_binary_instances (8-connected), _load_gt_instance_mask (cell_masks > 0, 8-connected), _compute_pq (IoU >= 0.5).
Per tile t, variant v, path p in {runner, official}:
  pq_main_p  = PQ(label(raster(CellViT_p(real H&E)) > 0), raster(CellViT_p(v)))   the paper's PQ: the input mask is the
               real-H&E CellViT drawn as a binary cell_masks PNG (realref.py masks), read back as 8-connected components
  pq_inst_p  = PQ(raster(CellViT_p(real H&E)), raster(CellViT_p(v)))              the same without binarising (secondary)
  pq_lin_p   = PQ(Lin cell_masks of the R0 pack, raster(CellViT_p(v)))            retrained sets only (README secondary PQ)
  ref_lin_p  = PQ(Lin cell_masks, raster(CellViT_p(real H&E)))                    retrained sets only (reference line)
  n_det_p    = generated objects; n_real_p = real-H&E objects
Groups: P = paper images (CRC33; Vanilla 1000 tiles of cache.json metric_tile_ids, Concat / Grouped+bypass / Per-channel
the 100 Fig 2 tiles); 36 = crc36ho 300, 33 = crc33sec 100 (retrained hfv_retrain_20261118 Fig 2 images, five variants).
Gates: every JSON present; retrained real H&E of score_fig2 tiles/gt == the stage-3 R_<set>.zip images (pixels).
-> OUT_DIR/tiles_<group>.csv
"""
import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from tools.compute_ablation_metrics import (_cellvit_json_to_instance_mask, _compute_pq, _label_binary_instances,
                                            _load_gt_instance_mask)

D = Path(sys.argv[1])
OUT = Path(sys.argv[2])
S3 = Path("/gscratch/cheme/chiu/hfv_retrain2_20261121/stage3")
CVDIR = {"official": D / "full/cv", "runner": D / "full_runner/cv"}
REALRET = {"official": lambda s: D / f"full/cv/R_{s}", "runner": lambda s: S3 / f"cv/R_{s}"}
SETS = {"36": "crc36ho", "33": "crc33sec"}
VAR_P = ("van", "cat", "gbp", "pch")
VAR_R = ("van", "grp", "cat", "gbp", "pch")


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


def inst(d, t):
    f = Path(d) / f"{t}.json"
    if not f.is_file():
        stop(f"missing {f}")
    return _cellvit_json_to_instance_mask(f, image_hw=(256, 256))


def names(z):
    with zipfile.ZipFile(z) as zf:
        return sorted(Path(n).stem for n in zf.namelist())


def pq(a, b):
    return float(_compute_pq(a, b)[2])


def nobj(m):
    return int(len(np.unique(m)) - 1)


OUT.mkdir(parents=True)
# gate: the retrained sets' real H&E (score_fig2 tiles/gt, uploaded to D/zips/retgt_<set>) equals the stage-3 R zips
for k, s in SETS.items():
    with zipfile.ZipFile(S3 / f"zips/R_{s}.zip") as zf:
        for n in zf.namelist():
            a = np.asarray(Image.open(io.BytesIO(zf.read(n))).convert("RGB"))
            b = np.asarray(Image.open(D / f"zips/retgt_{s}/{n}").convert("RGB"))
            if not np.array_equal(a, b):
                stop(f"real H&E {s} {n} differs between score_fig2 tiles/gt and R_{s}.zip")
    print(f"gate: real H&E {s} identical ({len(zf.namelist())})", flush=True)

rows = []
ids_van = names(D / "zips/F2_P_van.zip")
ids100 = names(D / "zips/F2_P_cat.zip")
for v in VAR_P:
    for t in (ids_van if v == "van" else ids100):
        r = dict(group="P", variant=v, tile=t, in100=int(t in ids100))
        for p, cv in CVDIR.items():
            real = inst(cv / "F2_P_gt", t)
            gen = inst(cv / f"F2_P_{v}", t)
            r |= {f"pq_main_{p}": pq(_label_binary_instances(real > 0), gen), f"pq_inst_{p}": pq(real, gen),
                  f"n_det_{p}": nobj(gen), f"n_real_{p}": nobj(real)}
        rows.append(r)
    print("P", v, flush=True)
with open(OUT / "tiles_P.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

for k, s in SETS.items():
    rows = []
    ids = names(D / f"zips/F2_{k}_cat.zip")
    for v in VAR_R:
        if names(D / f"zips/F2_{k}_{v}.zip") != ids:
            stop(f"{k} {v} tile ids differ")
        for t in ids:
            lin = _load_gt_instance_mask(S3 / f"packs/R0/orion-{s}", t, shape=(256, 256))
            r = dict(group=k, variant=v, tile=t, n_lin=nobj(lin))
            for p, cv in CVDIR.items():
                real = inst(REALRET[p](s), t)
                gen = inst(cv / f"F2_{k}_{v}", t)
                r |= {f"pq_main_{p}": pq(_label_binary_instances(real > 0), gen), f"pq_inst_{p}": pq(real, gen),
                      f"pq_lin_{p}": pq(lin, gen), f"ref_lin_{p}": pq(lin, real), f"n_det_{p}": nobj(gen),
                      f"n_real_{p}": nobj(real)}
            rows.append(r)
        print(k, v, flush=True)
    with open(OUT / f"tiles_{k}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
print("DONE", flush=True)
