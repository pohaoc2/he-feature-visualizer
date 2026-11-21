"""hfv_retrain_20261118 追加 3: inputs for the decodability rerun (README 追加 3, decisions 2 and 4). New folders only.

  python a3prep.py zip   ORION_CRC33_ZIP IDS_HE IDS_CH OUT      # he/<id>.png for IDS_HE; every exp_channels/<ch>/<id>.* and
                                                                # features/<id>_uni.npy (CLS) for IDS_CH -> OUT/orion-crc33-old/
  python a3prep.py tar   TAR IDS OUT [STRIP]                    # members whose file stem is in IDS (he/, exp_channels/<ch>/) -> OUT
  python a3prep.py splits PAPER_CV_JSON PAPER_IDS KEEP_IDS OUT_IDS OUT_CV
        Paper CV folds minus the tiles not in KEEP_IDS: same format as PixCell save_cv_splits (version, tile_count,
        tile_ids_sha1, block_size_px, n_splits, splits[train_idx, test_idx]); every kept tile stays in its paper fold;
        indices are re-numbered over the kept list, which keeps the paper's (numeric) order. STOP if PAPER_IDS does not
        hash to the paper file's tile_ids_sha1.
  python a3prep.py ids   TILE_LIST N SEED OUT                   # PixCell's subsample: sorted numeric list, rng(SEED).choice(N)
"""
import hashlib
import json
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np


def read_ids(p):
    return [l.strip() for l in open(p) if l.strip()]


def sha(ids):
    d = hashlib.sha1()
    for t in ids:
        d.update(str(t).encode("utf-8"))
        d.update(b"\n")
    return d.hexdigest()


def numkey(t):
    m = re.match(r"^(\d+)_(\d+)$", t)
    return int(m.group(1)), int(m.group(2))


def cmd_zip(zpath, ids_he, ids_ch, out):
    he, ch = set(read_ids(ids_he)), set(read_ids(ids_ch))
    out = Path(out) / "orion-crc33-old"
    if out.exists():
        sys.exit(f"STOP {out} exists")
    n = {"he": 0, "exp": 0, "cls": 0}
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = info.filename.split("/")
            name = parts[-1]
            stem = name.rsplit(".", 1)[0]
            if len(parts) >= 2 and parts[-2] == "he" and stem in he:
                dst, k = out / "he" / name, "he"
            elif len(parts) >= 3 and parts[-3] == "exp_channels" and stem in ch:
                dst, k = out / "exp_channels" / parts[-2] / name, "exp"
            elif len(parts) >= 2 and parts[-2] == "features" and name.endswith("_uni.npy") and name[:-8] in ch:
                dst, k = out / "features_cls" / name, "cls"
            else:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dst, "wb") as f:
                f.write(src.read())
            n[k] += 1
    print(json.dumps(n))
    chans = sorted(p.name for p in (out / "exp_channels").iterdir())
    counts = {c: len(list((out / "exp_channels" / c).iterdir())) for c in chans}
    print(json.dumps(counts))
    if n["he"] != len(he):
        sys.exit(f"STOP he: {n['he']} of {len(he)}")
    bad = {c: v for c, v in counts.items() if v != len(ch)}
    if bad:
        sys.exit(f"STOP exp_channels counts != {len(ch)}: {bad}")


def cmd_tar(tpath, ids, out, strip="0"):
    ids, out, strip = set(read_ids(ids)), Path(out), int(strip)
    n = 0
    with tarfile.open(tpath) as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.split("/")[strip:]
            stem = parts[-1].rsplit(".", 1)[0]
            if stem not in ids:
                continue
            dst = out.joinpath(*parts)
            dst.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(dst, "wb") as f:
                f.write(src.read())
            n += 1
    print(f"{tpath}: {n} files for {len(ids)} ids -> {out}")


def cmd_splits(cv_json, paper_ids, keep_ids, out_ids, out_cv):
    cv = json.load(open(cv_json))
    ids = read_ids(paper_ids)
    if sha(ids) != cv["tile_ids_sha1"] or len(ids) != cv["tile_count"]:
        sys.exit("STOP paper ids do not hash to the paper cv_splits tile_ids_sha1")
    keep = set(read_ids(keep_ids))
    kept = [t for t in ids if t in keep]
    new_index = {t: i for i, t in enumerate(kept)}
    splits = []
    for s in cv["splits"]:
        tr = [new_index[ids[i]] for i in s["train_idx"] if ids[i] in new_index]
        te = [new_index[ids[i]] for i in s["test_idx"] if ids[i] in new_index]
        splits.append({"train_idx": tr, "test_idx": te})
    allte = sorted(i for s in splits for i in s["test_idx"])
    if allte != list(range(len(kept))):
        sys.exit("STOP kept test folds do not partition the kept tiles")
    Path(out_ids).write_text("\n".join(kept) + "\n")
    doc = {"version": cv["version"], "tile_count": len(kept), "tile_ids_sha1": sha(kept),
           "block_size_px": cv["block_size_px"], "n_splits": cv["n_splits"], "splits": splits,
           "derived_from": {"paper_cv": str(cv_json), "paper_tile_ids_sha1": cv["tile_ids_sha1"],
                            "dropped": len(ids) - len(kept), "rule": "paper folds minus tiles without new channels"}}
    json.dump(doc, open(out_cv, "w"))
    print(f"{len(ids)} -> {len(kept)} tiles; test sizes {[len(s['test_idx']) for s in splits]}")


def cmd_ids(tile_list, n, seed, out):
    ids = sorted(set(read_ids(tile_list)), key=numkey)
    keep = np.sort(np.random.default_rng(int(seed)).choice(len(ids), size=int(n), replace=False))
    sel = [ids[i] for i in keep]
    Path(out).write_text("\n".join(sel) + "\n")
    print(f"{len(ids)} -> {len(sel)} sha1 {sha(sel)}")


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"zip": cmd_zip, "tar": cmd_tar, "splits": cmd_splits, "ids": cmd_ids}.get(c, lambda *x: sys.exit(__doc__))(*a)
