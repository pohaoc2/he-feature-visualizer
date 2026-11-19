"""crc34_render_20261118: CellViT inputs for CRC34 (cellvit env).

  cvprep34.py N OLD_TAR
OLD_TAR = processed_CRC34_stage1_gpu_products.tar (the paper-era CRC34 stage-1 GPU products, copied read only from S3;
the S3 object is not changed). Its CellViT JSONs are processed_CRC34/cellvit/<x0>_<y0>.json.

1. Grid check (coordinator rule, hfv_retrain_20261118 README, before any run): the old JSONs may be reused only if they
   sit on the new tile grid with identical tile origins and size. Checked:
     - every new and old origin is a multiple of the 256 px stride (same lattice, same origin 0, 0);
     - old tile size: the old index (processed_CRC34/index.json, if present) patch size, and the old H&E tile PNGs (if
       present) are 256 x 256 and pixel-equal to the new crop of Lin's registered H&E on the 50 gate tiles;
     - every new kept tile has an old JSON (coverage).
2. Gate tiles: 50 tiles drawn from the CRC34 evaluation pool (tissue tiles minus ink tiles = N/out/tiles_keep.txt,
   the training filters) with numpy default_rng(42).choice(sorted(pool), 50, replace=False).
3. N/data/cvgate34_inputs.tar: he_tiles.zip (he/<t>.png, stored, the new H&E tile PNGs from tiles.py) + ref/<t>.json
   (the old JSONs, byte copies) + ids.txt; the layout cellvit_kit uses (tiles/he_tiles.zip, ref/) and the names
   run_cellvit_local.py writes, as colab/cvgate34.py REF_DIR NEW_DIR expects.
4. N/data/crc34_he_tiles_keep.tar: he/<t>.png of every kept tile (for cv34 / UNI / VAE on Colab).
5. If the grid check passes: the old JSONs of the kept tiles extracted to N/cvold/json (for cellvit_dead.py, only after
   the Colab gate passes).
Writes N/out/cvprep34.json.
"""
import hashlib
import io
import json
import os
import re
import sys
import tarfile
import zipfile

import cv2
import numpy as np
import tifffile

N, OLD = sys.argv[1:3]
PS = 256
T = json.load(open(f"{N}/out/tiles_crc34.json"))
tiles = list(T["tiles"])
keep = [t.strip() for t in open(f"{N}/out/tiles_keep.txt") if t.strip()]
assert set(keep) == {t for t in tiles if T["ink"][t] <= 0}, "STOP tiles_keep.txt != tissue tiles minus ink"
rng = np.random.default_rng(42)
gate = sorted(rng.choice(sorted(keep), 50, replace=False).tolist())
R = dict(new_tiles=len(tiles), new_keep=len(keep), gate_rule="numpy default_rng(42).choice(sorted(keep), 50, replace=False)",
         gate_tiles=gate, old_tar=OLD, old_tar_bytes=os.path.getsize(OLD))

rx_cv = re.compile(r"(?:^|/)cellvit/(\d+_\d+)\.json$")
rx_he = re.compile(r"(?:^|/)he/(\d+_\d+)\.png$")
old_json, old_he, old_index, kinds = {}, {}, None, {}
gs, ks = set(gate), set(keep)
with tarfile.open(OLD) as tf:
    for m in tf:
        if not m.isfile():
            continue
        top = "/".join(m.name.split("/")[:2])
        kinds[top] = kinds.get(top, 0) + 1
        a = rx_cv.search(m.name)
        if a:
            old_json[a.group(1)] = m.name
            continue
        b = rx_he.search(m.name)
        if b:
            old_he[b.group(1)] = m.name
            continue
        if m.name.endswith("/index.json") and m.name.count("/") == 1:
            old_index = json.load(tf.extractfile(m))
R["old_members_by_dir"] = kinds
R["old_json"] = len(old_json)
R["old_he_png"] = len(old_he)


def on_lattice(t):
    x, y = map(int, t.split("_"))
    return x % PS == 0 and y % PS == 0


g = dict(new_on_lattice=all(map(on_lattice, tiles)), old_on_lattice=all(map(on_lattice, old_json)),
         keep_without_old_json=len(ks - set(old_json)), gate_without_old_json=len(gs - set(old_json)),
         new_tissue_not_in_old=len(set(tiles) - set(old_json)), old_not_in_new_tissue=len(set(old_json) - set(tiles)))
if old_index is not None:
    ps = {old_index.get(k) for k in ("patch_size", "tile_size", "patch") if k in old_index}
    pp = old_index.get("patches", [])
    g["old_index_keys"] = sorted(k for k in old_index if k != "patches")
    g["old_index_patch_size"] = sorted(x for x in ps if x is not None)
    g["old_index_patches"] = len(pp)
    if pp and "x0" in pp[0]:
        oi = {f"{int(p['x0'])}_{int(p['y0'])}" for p in pp}
        g["old_index_equals_old_json_set"] = oi == set(old_json)
        g["old_index_equals_new_tissue"] = oi == set(tiles)
# old H&E tiles vs the new crop (gate tiles)
if old_he:
    with tifffile.TiffFile(f"{N}/lin/data__CRC34__19510_C34_US_SCAN_OR_001__160949-registered.ome.tif") as t_:
        s = t_.series[0]
        arr = s.levels[0].asarray()
        ax = s.axes.upper()
    if ax.startswith("C") or ax.startswith("S"):
        arr = np.moveaxis(arr, 0, -1)
    shapes, neq = set(), []
    with tarfile.open(OLD) as tf:
        for t in gate:
            if t not in old_he:
                neq.append((t, "no old png"))
                continue
            im = cv2.imdecode(np.frombuffer(tf.extractfile(old_he[t]).read(), np.uint8), cv2.IMREAD_COLOR)
            shapes.add(im.shape)
            x0, y0 = map(int, t.split("_"))
            new = cv2.cvtColor(np.ascontiguousarray(arr[y0:y0 + PS, x0:x0 + PS, :3]), cv2.COLOR_RGB2BGR)
            if im.shape != new.shape or not np.array_equal(im, new):
                neq.append((t, "pixels differ"))
    del arr
    g["old_he_shapes_gate"] = sorted(map(list, shapes))
    g["old_he_vs_new_crop_gate_not_equal"] = neq
size_ok = (g.get("old_index_patch_size") in (None, [], [PS])) and (not old_he or g["old_he_shapes_gate"] == [[PS, PS, 3]])
g["PASS"] = bool(g["new_on_lattice"] and g["old_on_lattice"] and g["keep_without_old_json"] == 0 and size_ok
                 and not g.get("old_he_vs_new_crop_gate_not_equal"))
g["size_evidence"] = ("old index patch size" if g.get("old_index_patch_size") else "") + \
                     (" + old H&E PNG shape/pixels" if old_he else "") or "none beyond the 256 px lattice"
R["grid_check"] = g
print("grid check", json.dumps(g), flush=True)

# ---- new H&E PNGs -> gate zip and kept-tile tar ----
he_tar = f"{N}/data/crc34_he_tiles.tar"
KEEP_TAR, GATE_TAR = f"{N}/data/crc34_he_tiles_keep.tar", f"{N}/data/cvgate34_inputs.tar"
for p in (KEEP_TAR, GATE_TAR):
    assert not os.path.exists(p), f"STOP {p} exists"
zbuf = io.BytesIO()
zf = zipfile.ZipFile(zbuf, "w", compression=zipfile.ZIP_STORED)
n_keep = 0
with tarfile.open(he_tar) as src, tarfile.open(KEEP_TAR + ".part", "w") as out:
    for m in src:
        if not m.isfile():
            continue
        a = rx_he.search(m.name)
        if not a or a.group(1) not in ks:
            continue
        b = src.extractfile(m).read()
        m2 = tarfile.TarInfo(f"he/{a.group(1)}.png")
        m2.size, m2.mtime, m2.mode = len(b), m.mtime, 0o644
        out.addfile(m2, io.BytesIO(b))
        n_keep += 1
        if a.group(1) in gs:
            zf.writestr(f"he/{a.group(1)}.png", b)
zf.close()
assert n_keep == len(keep), f"STOP kept tar has {n_keep} of {len(keep)}"
os.replace(KEEP_TAR + ".part", KEEP_TAR)


def add(tf, name, b):
    m = tarfile.TarInfo(name)
    m.size, m.mode = len(b), 0o644
    tf.addfile(m, io.BytesIO(b))


with tarfile.open(OLD) as src, tarfile.open(GATE_TAR + ".part", "w") as out:
    add(out, "he_tiles.zip", zbuf.getvalue())
    add(out, "ids.txt", ("\n".join(gate) + "\n").encode())
    for t in gate:
        add(out, f"ref/{t}.json", src.extractfile(old_json[t]).read())
os.replace(GATE_TAR + ".part", GATE_TAR)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""):
            h.update(c)
    return h.hexdigest()


R["outputs"] = {os.path.basename(p): dict(path=p, bytes=os.path.getsize(p), md5=md5(p)) for p in (KEEP_TAR, GATE_TAR)}
R["outputs"]["crc34_he_tiles_keep.tar"]["members"] = n_keep
# ---- old JSONs of the kept tiles (only if the grid matches) ----
if g["PASS"]:
    D = f"{N}/cvold/json"
    os.makedirs(D, exist_ok=True)
    with tarfile.open(OLD) as src:
        for t in keep:
            with open(f"{D}/{t}.json", "wb") as f:
                f.write(src.extractfile(old_json[t]).read())
    R["old_json_extracted"] = dict(dir=D, n=len(keep))
json.dump(R, open(f"{N}/out/cvprep34.json", "w"), indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "gate_tiles"}), flush=True)
