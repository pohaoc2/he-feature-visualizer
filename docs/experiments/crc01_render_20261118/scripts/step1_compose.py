"""crc01_render_20261118 step 1: Lin H&E tissue-class composition of every Lin slide, and TUM cell counts.

Inputs (public): Lin S3 bucket listing (data/<slide>/ registered H&E names), Lin class-images (GitHub, orion-crc
HE_patch_prediction/gamma-only/orion/class-images, 112 um grid), Lin CSVs for the slides whose cell count is asked.
TUM cells = cells whose (X_centroid, Y_centroid) falls in a TUM grid cell (same lookup as stages/lin_typing/labels.py).
Usage: python step1_compose.py OUT_DIR HFV_DIR [slide ...]   (CSV counted only for listed slides)
"""
import json, os, re, subprocess, sys, urllib.parse
import numpy as np
from PIL import Image

OUT, HFV = sys.argv[1], sys.argv[2]
CSV_SLIDES = sys.argv[3:]
sys.path.insert(0, HFV)
from stages.lin_typing import labels as LB  # noqa: E402
from stages.lin_typing.celltype_celesta_fig3d import LIN_L  # noqa: E402

B = "https://lin-2023-orion-crc.s3.amazonaws.com"
G = "https://raw.githubusercontent.com/labsyspharm/orion-crc/main/HE_patch_prediction/gamma-only/orion/class-images"
os.makedirs(f"{OUT}/classmaps", exist_ok=True)


def curl(url, dst=None):
    """HTTP GET with curl (the env's Python has no CA bundle)."""
    cmd = ["curl", "-sS", "--fail", "--retry", "3", url] + (["-o", dst] if dst else [])
    return subprocess.run(cmd, check=True, capture_output=dst is None).stdout
keys, tok = [], None
while True:
    q = "list-type=2&prefix=data/" + (f"&continuation-token={urllib.parse.quote(tok)}" if tok else "")
    s = curl(f"{B}/?{q}").decode()
    keys += re.findall(r"<Key>(.*?)</Key>\s*<LastModified>.*?</LastModified>\s*<ETag>.*?</ETag>\s*<Size>(\d+)</Size>", s)
    m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", s)
    if not m:
        break
    tok = m.group(1)
files = {k: int(n) for k, n in keys}
json.dump(files, open(f"{OUT}/lin_bucket_listing.json", "w"), indent=0)
slides = {}
for k in files:
    m = re.match(r"data/(CRC\d+(?:_\d+)?)/(\d+)_(\w+?)_US_SCAN_OR_001__(\d+(?:-\d+)?)-registered\.ome\.tif$", k)
    if m:
        slides[m.group(1)] = dict(he=k, cls=f"{m.group(2)}${m.group(3)}$US$SCAN$OR$001 _{m.group(4)}-registered.tiff",
                                  csv=[x for x in files if x.startswith(f"data/{m.group(1)}/") and x.endswith(".csv")
                                       and not x.endswith("markers.csv")])
read = 0
res = {}
grid_px = LB.LIN_GRID_UM / LB.MPP
for sl in sorted(slides):
    d = slides[sl]
    dst = f"{OUT}/classmaps/{sl}_lin_he_class.tiff"
    if not os.path.exists(dst):
        try:
            curl(f"{G}/{urllib.parse.quote(d['cls'])}", dst)
        except Exception as e:  # noqa: BLE001
            res[sl] = dict(error=f"class map: {e}", **d); continue
    read += os.path.getsize(dst)
    cm = np.array(Image.open(dst))
    cnt = np.bincount(cm.ravel(), minlength=len(LIN_L))
    tissue = cnt[1:].sum()
    r = dict(classmap=d["cls"], shape=list(cm.shape), grid_counts=dict(zip(LIN_L, cnt.tolist())),
             tum_grid=int(cnt[3]), tum_frac_tissue=round(float(cnt[3] / max(tissue, 1)), 4),
             norm_frac_tissue=round(float(cnt[1] / max(tissue, 1)), 4),
             csv_bytes=sum(files[c] for c in d["csv"]), he_bytes=files[d["he"]])
    if sl in CSV_SLIDES:
        for c in d["csv"]:
            loc = f"{OUT}/{c.replace('/', '__')}"
            if not os.path.exists(loc) or os.path.getsize(loc) != files[c]:
                subprocess.run(["curl", "-sS", "--fail", "--retry", "3", "-o", loc + ".part", f"{B}/{c}"], check=True)
                assert os.path.getsize(loc + ".part") == files[c], f"STOP size mismatch {c}"
                os.replace(loc + ".part", loc)
            read += files[c]
            import pandas as pd
            df = pd.read_csv(loc, usecols=["X_centroid", "Y_centroid"])
            k = LB.lin_class_at(df.X_centroid.values, df.Y_centroid.values, cm)
            cc = np.bincount(k, minlength=len(LIN_L))
            r.setdefault("cells", {})[c] = dict(n=int(len(df)), by_class=dict(zip(LIN_L, cc.tolist())))
        r["tum_cells"] = int(sum(v["by_class"]["TUM"] for v in r["cells"].values()))
    res[sl] = r
    print(sl, r.get("tum_frac_tissue"), r.get("tum_grid"), r.get("tum_cells", ""), flush=True)
json.dump(dict(slides=res, bytes_read=read, grid_px=grid_px), open(f"{OUT}/step1_composition.json", "w"), indent=1)
print("bytes_read", read)
