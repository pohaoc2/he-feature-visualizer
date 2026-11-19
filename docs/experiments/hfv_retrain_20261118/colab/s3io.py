"""hfv_retrain_20261118 Colab side: move files with one-day presigned S3 links only (no AWS credentials in Colab).

  python s3io.py get  LINKS.json NAME DEST            download links["get"][NAME] to DEST (resumable, size-checked)
  python s3io.py put  LINKS.json SLOT FILE            upload FILE to links["put"][SLOT] (single PUT or multipart)
  python s3io.py list LINKS.json                      show slot names and expiry

LINKS.json is written on Hyak by presign.py and fetched once from a presigned GET link pasted into the notebook; it is
never committed. A "put" slot is {"single": url} (file < 5 GB) or {"multipart": {"part_bytes": n, "parts": [url, ...],
"complete": url}}; multipart parts are sent in order and completed with the returned ETags.
"""
import json
import os
import sys
import time
import urllib.request

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


def get(links, name, dest):
    url = links["get"][name]
    size = int(_req(url, headers={"Range": "bytes=0-0"}).headers["Content-Range"].split("/")[-1])
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    if have == size:
        print(f"{dest} complete ({size} bytes)")
        return
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    t0 = time.time()
    with open(dest, "ab") as f:
        while have < size:
            r = _req(url, headers={"Range": f"bytes={have}-"})
            while True:
                b = r.read(CHUNK)
                if not b:
                    break
                f.write(b)
                have += len(b)
    assert os.path.getsize(dest) == size, "size mismatch"
    print(f"{dest} {size} bytes {size / max(time.time() - t0, 1e-6) / 1e6:.0f} MB/s", flush=True)


def put(links, slot, path):
    s = links["put"][slot]
    n = os.path.getsize(path)
    t0 = time.time()
    if "single" in s:
        assert n < 5 * 10**9, "single PUT limited to 5 GB"
        with open(path, "rb") as f:
            _req(s["single"], "PUT", data=f.read(), headers={"Content-Length": str(n)})
    else:
        m = s["multipart"]
        ps = int(m["part_bytes"])
        need = -(-n // ps)
        assert need <= len(m["parts"]), f"file needs {need} parts, slot has {len(m['parts'])}"
        etags = []
        with open(path, "rb") as f:
            for i in range(need):
                b = f.read(ps)
                r = _req(m["parts"][i], "PUT", data=b, headers={"Content-Length": str(len(b))})
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
