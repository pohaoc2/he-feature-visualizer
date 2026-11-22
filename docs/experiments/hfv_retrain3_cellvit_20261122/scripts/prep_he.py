"""hfv_retrain3_cellvit_20261122 (PREREG §3.1): chunk zips of the real H&E tiles for official CellViT, Hyak CPU.

  python prep_he.py R CHUNK
Sources (read only):
  CRC33_01  paperpq_official_20261122/in/orion-crc33.zip  orion-crc33/he/<t>.png, t in hfv_lintype_20261118/out/tiles_keep.txt
  CRC36     hfv_crc36_20261118/data/crc36_he_tiles.tar     he/<t>.png, t in hfv_retrain2 ref/c36_train_ids.txt (training)
            and hfv_eval_20261118/packs/crc36ho_pool.txt (held-out region, all 1,092)
Writes R/he/<group>_<NN>.zip (members he/<t>.png, bytes copied, ZIP_STORED) and R/he/manifest.json
(group -> chunk -> tiles, sha256 of every PNG). Gates: every listed tile found exactly once; no tile in two groups.
"""
import hashlib
import json
import os
import sys
import tarfile
import zipfile

R, CHUNK = sys.argv[1], int(sys.argv[2])
C = "/gscratch/cheme/chiu"
LISTS = {
    "c33": open(f"{C}/hfv_lintype_20261118/out/tiles_keep.txt").read().split(),
    "c36": open(f"{C}/hfv_retrain2_20261121/ref/c36_train_ids.txt").read().split(),
    "c36ho": open(f"{C}/hfv_eval_20261118/packs/crc36ho_pool.txt").read().split(),
}


def stop(m):
    print("STOP", m, flush=True)
    sys.exit(1)


def main():
    out = f"{R}/he"
    os.makedirs(out, exist_ok=False)
    if set(LISTS["c36"]) & set(LISTS["c36ho"]):
        stop("CRC36 training and held-out lists overlap")
    data = {}
    with zipfile.ZipFile(f"{C}/paperpq_official_20261122/in/orion-crc33.zip") as z:
        want = set(LISTS["c33"])
        for n in z.namelist():
            if n.startswith("orion-crc33/he/") and n.endswith(".png"):
                t = n.split("/")[-1][:-4]
                if t in want:
                    if ("c33", t) in data:
                        stop(f"c33 {t} twice")
                    data[("c33", t)] = z.read(n)
    w36 = {t: "c36" for t in LISTS["c36"]} | {t: "c36ho" for t in LISTS["c36ho"]}
    with tarfile.open(f"{C}/hfv_crc36_20261118/data/crc36_he_tiles.tar") as tf:
        for m in tf:
            if m.isfile() and m.name.startswith("he/") and m.name.endswith(".png"):
                t = m.name.split("/")[-1][:-4]
                if t in w36:
                    if (w36[t], t) in data:
                        stop(f"c36 {t} twice")
                    data[(w36[t], t)] = tf.extractfile(m).read()
    man = {}
    for g, ids in LISTS.items():
        miss = [t for t in ids if (g, t) not in data]
        if miss:
            stop(f"{g}: {len(miss)} tiles without H&E, e.g. {miss[:3]}")
        ids = sorted(ids)
        man[g] = {}
        for k in range(0, len(ids), CHUNK):
            name = f"{g}_{k // CHUNK:02d}"
            with zipfile.ZipFile(f"{out}/{name}.zip.part", "w", zipfile.ZIP_STORED) as z:
                for t in ids[k:k + CHUNK]:
                    z.writestr(f"he/{t}.png", data[(g, t)])
            os.replace(f"{out}/{name}.zip.part", f"{out}/{name}.zip")
            man[g][name] = {t: hashlib.sha256(data[(g, t)]).hexdigest() for t in ids[k:k + CHUNK]}
        print(g, len(ids), "tiles", len(man[g]), "chunks", flush=True)
    json.dump(man, open(f"{out}/manifest.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
