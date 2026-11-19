"""hfv_retrain_20261118 Colab side: move files with one-day presigned S3 links only (no AWS credentials in Colab).

  python s3io.py get  LINKS.json NAME DEST            download links["get"][NAME] to DEST: files > 200 MB as parallel
                                                      ranged GETs (S3_RANGES, default 16), resumable per range; checks
                                                      the size and the S3 ETag when LINKS["meta"][NAME] carries it
  python s3io.py put  LINKS.json SLOT FILE            upload FILE to links["put"][SLOT] (single PUT or multipart)
  python s3io.py list LINKS.json                      show slot names and expiry

A multipart slot is single-use: its upload id is consumed by the first completed upload, so a second upload through the
same slot gets 404 (NoSuchUpload). Slots that carry both a single and a multipart link use the single PUT (reusable
until expiry) for files under 5 GB and the multipart link only for larger files.
LINKS.json is written on Hyak by presign.py and fetched once from a presigned GET link pasted into the notebook; it is
never committed. A "put" slot is {"single": url} (file < 5 GB) or {"multipart": {"part_bytes": n, "parts": [url, ...],
"complete": url}}; multipart parts are sent in order and completed with the returned ETags.
PUTs send an empty Content-Type: urllib otherwise adds application/x-www-form-urlencoded, which breaks a link whose
signature covers Content-Type (SigV2 links of the first L4 boot; presign.py now signs SigV4).
"""
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 1 << 24


def _req(url, method="GET", data=None, headers=None, tries=5):
    for k in range(tries):
        try:
            r = urllib.request.Request(url, data=data, method=method, headers=headers or {})
            return urllib.request.urlopen(r, timeout=600)
        except Exception as e:  # network blips on Colab; 4xx (expired link) are re-raised at the end
            if k == tries - 1:
                raise
            print(f"retry {k + 1} {method}: {e}", flush=True)
            time.sleep(5 * (k + 1))


def _size(url):
    return int(_req(url, headers={"Range": "bytes=0-0"}).headers["Content-Range"].split("/")[-1])


def _fetch_range(url, part, lo, hi):
    """download bytes lo..hi (inclusive) into the file part; resumes from the bytes already in part"""
    want = hi - lo + 1
    for k in range(6):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        if have == want:
            return
        if have > want:
            os.remove(part)
            continue
        try:
            r = _req(url, headers={"Range": f"bytes={lo + have}-{hi}"})
            with open(part, "ab") as f:
                while True:
                    b = r.read(CHUNK)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:   # a dropped stream mid-range: resume from what is on disk
            print(f"range {lo}-{hi} retry {k + 1}: {e}", flush=True)
            time.sleep(3 * (k + 1))
    assert os.path.getsize(part) == want, f"range {lo}-{hi} incomplete"


def _etag_ok(path, etag, size):
    """S3 ETag check: plain md5 for a single-part object; for a multipart object ("<md5 of part md5s>-N") the part
    size is not stored, so try the MiB part sizes that give N parts (common upload sizes first)"""
    etag = etag.strip('"')
    if "-" not in etag:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(CHUNK), b""):
                h.update(b)
        return h.hexdigest() == etag
    n = int(etag.split("-")[1])
    mib = 1 << 20
    fits = [m for m in range(5, 5121) if -(-size // (m * mib)) == n]
    common = [m for m in (8, 16, 32, 64, 100, 128, 256, 512, 1024) if m in fits]
    for m in (common + [x for x in fits if x not in common])[:4]:
        ps, ds = m * mib, []
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(ps), b""):
                ds.append(hashlib.md5(b).digest())
        if hashlib.md5(b"".join(ds)).hexdigest() + f"-{n}" == etag:
            return True
    return False


def get(links, name, dest):
    """parallel ranged GETs (S3_RANGES ranges, default 16, for files > 200 MB), each range resumable, then
    concatenated in order; checks the size and, when links["meta"][name] has it, the S3 ETag"""
    url = links["get"][name]
    meta = links.get("meta", {}).get(name, {})
    size = _size(url)
    if "size" in meta:
        assert size == int(meta["size"]), f"{name}: link size {size} != S3 size {meta['size']}"
    ok = dest + ".ok"
    if os.path.exists(dest) and os.path.getsize(dest) == size and os.path.exists(ok):
        print(f"{dest} complete ({size} bytes)")
        return
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    t0 = time.time()
    n = max(1, min(int(os.environ.get("S3_RANGES", "16")), size // (64 << 20))) if size > 200 * 10**6 else 1
    step = -(-size // n)
    rng = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(n) if i * step < size]
    with ThreadPoolExecutor(len(rng)) as ex:
        for fu in [ex.submit(_fetch_range, url, f"{dest}.r{i:02d}", lo, hi) for i, lo, hi in rng]:
            fu.result()
    t1 = time.time()
    with open(dest, "wb") as out:
        for i, _, _ in rng:
            with open(f"{dest}.r{i:02d}", "rb") as f:
                shutil.copyfileobj(f, out, CHUNK)
            os.remove(f"{dest}.r{i:02d}")
    assert os.path.getsize(dest) == size, "size mismatch"
    if "etag" in meta:
        assert _etag_ok(dest, meta["etag"], size), f"{name}: ETag mismatch"
    open(ok, "w").write(f"{size} {meta.get('etag', '')}\n")
    print(f"{dest} {size} bytes {len(rng)} ranges {size / max(t1 - t0, 1e-6) / 1e6:.0f} MB/s "
          f"(check {'etag' if 'etag' in meta else 'size'} ok)", flush=True)


def put(links, slot, path):
    s = links["put"][slot]
    n = os.path.getsize(path)
    t0 = time.time()
    if "single" in s and n < 5 * 10**9:   # a single-PUT link can be reused until it expires
        with open(path, "rb") as f:
            _req(s["single"], "PUT", data=f.read(), headers={"Content-Length": str(n), "Content-Type": ""})
    else:
        m = s["multipart"]
        ps = int(m["part_bytes"])
        need = -(-n // ps)
        assert need <= len(m["parts"]), f"file needs {need} parts, slot has {len(m['parts'])}"
        etags = []
        with open(path, "rb") as f:
            for i in range(need):
                b = f.read(ps)
                r = _req(m["parts"][i], "PUT", data=b, headers={"Content-Length": str(len(b)), "Content-Type": ""})
                etags.append(r.headers["ETag"])
                print(f"part {i + 1}/{need}", flush=True)
        body = "<CompleteMultipartUpload>" + "".join(
            f"<Part><PartNumber>{i + 1}</PartNumber><ETag>{e}</ETag></Part>" for i, e in enumerate(etags)
        ) + "</CompleteMultipartUpload>"
        r = _req(m["complete"], "POST", data=body.encode(), headers={"Content-Type": "application/xml"})
        out = r.read().decode()
        assert "<Error>" not in out, out
    print(f"uploaded {path} -> {slot} ({n} bytes, {n / max(time.time() - t0, 1e-6) / 1e6:.0f} MB/s)", flush=True)


if __name__ == "__main__":
    cmd, lp = sys.argv[1], sys.argv[2]
    L = json.load(open(lp))
    if cmd == "get":
        get(L, sys.argv[3], sys.argv[4])
    elif cmd == "put":
        put(L, sys.argv[3], sys.argv[4])
    else:
        print("expires", L.get("expires"))
        print("get:", sorted(L["get"]))
        print("put:", sorted(L["put"]))
