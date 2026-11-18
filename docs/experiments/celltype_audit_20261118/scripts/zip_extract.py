"""Extract selected members of a large zip on S3 with ranged GETs (no full download).

Reads the central directory (zip64 aware), groups the wanted members by prefix, downloads one contiguous byte span per
prefix and inflates each member from its local header. Bytes fetched are printed per prefix (volume report).

  zip_extract.py s3://BUCKET/KEY OUTDIR PREFIX [PREFIX ...]
A PREFIX selects every member whose name starts with it (e.g. processed_crc33/cell_assignments.csv,
processed_crc33/cellvit/). Needs the `aws` CLI on PATH (Hyak: module load coenv/aws/2.17.32).
"""
import io
import json
import os
import tarfile
import struct
import subprocess
import sys
import tempfile
import zlib


def s3(url):
    b, k = url[5:].split("/", 1)
    return b, k


def rng(b, k, a, z, out):
    subprocess.run(["aws", "s3api", "get-object", "--bucket", b, "--key", k, "--range", f"bytes={a}-{z}", out],
                   check=True, capture_output=True)
    with open(out, "rb") as f:
        return f.read()


def central_dir(b, k, tmp):
    head = json.loads(subprocess.run(["aws", "s3api", "head-object", "--bucket", b, "--key", k],
                                     check=True, capture_output=True).stdout)
    size = int(head["ContentLength"])
    tail = rng(b, k, size - 70000, size - 1, os.path.join(tmp, "tail"))
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = rng(b, k, cd_off, cd_off + cd_size - 1, os.path.join(tmp, "cd"))
    p, out = 0, []
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        meth, = struct.unpack("<H", cd[p + 10:p + 12])
        csize, usize = struct.unpack("<II", cd[p + 20:p + 28])
        n, e, c = struct.unpack("<HHH", cd[p + 28:p + 34])
        off, = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + n].decode()
        ex = cd[p + 46 + n:p + 46 + n + e]
        q = 0
        while q < len(ex):
            hid, hl = struct.unpack("<HH", ex[q:q + 4])
            if hid == 1:
                v, t = ex[q + 4:q + 4 + hl], 0
                if usize == 0xFFFFFFFF:
                    usize, = struct.unpack("<Q", v[t:t + 8]); t += 8
                if csize == 0xFFFFFFFF:
                    csize, = struct.unpack("<Q", v[t:t + 8]); t += 8
                if off == 0xFFFFFFFF:
                    off, = struct.unpack("<Q", v[t:t + 8]); t += 8
            q += 4 + hl
        out.append(dict(name=name, meth=meth, csize=csize, usize=usize, off=off))
        p += 46 + n + e + c
    return size, out


def main(url, outdir, *prefixes):
    keep = set(open(os.environ["TILES"]).read().split()) if os.environ.get("TILES") else None
    skip = os.environ.get("SKIP_EXISTING") == "1"

    def wanted(name):
        base = os.path.splitext(os.path.basename(name))[0]
        return keep is None or "_" not in base or not base.replace("_", "").isdigit() or base in keep
    tar = tarfile.open(os.environ["TAR"], "a") if os.environ.get("TAR") else None
    b, k = s3(url)
    tmp = tempfile.mkdtemp(dir=outdir)
    size, members = central_dir(b, k, tmp)
    total = 0
    for pre in prefixes:
        sel = [m for m in members if m["name"].startswith(pre) and not m["name"].endswith("/")]
        if not sel:
            raise SystemExit(f"STOP: no member matches {pre}")
        sel = [m for m in sel if wanted(m["name"])]
        if skip and tar is None:
            sel = [m for m in sel if not (os.path.exists(os.path.join(outdir, m["name"]))
                                          and os.path.getsize(os.path.join(outdir, m["name"])) == m["usize"])]
        if not sel:
            print(f"{pre}: nothing to do", flush=True)
            continue
        lo = min(m["off"] for m in sel)
        hi = min(max(m["off"] + m["csize"] + 30 + 1024 + 64 for m in sel), size - 1)  # + local header slack
        span = os.path.join(tmp, "span")
        rng(b, k, lo, hi, span)
        total += hi - lo + 1
        with open(span, "rb") as f:
            buf = f.read()
        for m in sel:
            p = m["off"] - lo
            assert buf[p:p + 4] == b"PK\x03\x04", m["name"]
            n, e = struct.unpack("<HH", buf[p + 26:p + 30])
            data = buf[p + 30 + n + e:p + 30 + n + e + m["csize"]]
            if m["meth"] == 8:
                data = zlib.decompress(data, -15)
            elif m["meth"] != 0:
                raise SystemExit(f"STOP: compression method {m['meth']} for {m['name']}")
            assert len(data) == m["usize"], m["name"]
            if tar is not None:
                ti = tarfile.TarInfo(m["name"]); ti.size = len(data)
                tar.addfile(ti, io.BytesIO(data))
                continue
            dst = os.path.join(outdir, m["name"])
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "wb") as g:
                g.write(data)
        os.remove(span)
        print(f"{pre}: {len(sel)} members, {(hi - lo + 1) / 1e6:.1f} MB fetched", flush=True)
    if tar is not None:
        tar.close()
    print(f"TOTAL fetched {total / 1e9:.3f} GB from {url}", flush=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    os.rmdir(tmp)


if __name__ == "__main__":
    main(*sys.argv[1:])
