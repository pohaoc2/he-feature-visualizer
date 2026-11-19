"""Extract a zip read sequentially from stdin, so the archive is never stored next to its extracted copy (Tillicum
storage rule, 2026-11-18):   curl -sf URL | python streamunzip.py OUT [EXCLUDE_REGEX]
Members are found by their local headers (orion-crc33.zip: 166,085 members, stored or deflated, no data descriptors,
checked from its central directory). Every member's CRC-32 and size are checked; names matching EXCLUDE_REGEX are read
and checked but not written. A member with a data descriptor or another compression method stops the run.
"""
import os
import re
import struct
import sys
import zlib

OUT = sys.argv[1]
EX = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
f = sys.stdin.buffer
buf = bytearray()


def read(n):
    global buf
    while len(buf) < n:
        b = f.read(max(1 << 20, n - len(buf)))
        if not b:
            sys.exit("STOP truncated stream")
        buf += b
    out = bytes(buf[:n])
    del buf[:n]
    return out


n_w = n_x = n_b = 0
while True:
    sig = read(4)
    if sig != b"PK\x03\x04":
        if sig in (b"PK\x01\x02", b"PK\x05\x06", b"PK\x06\x06"):
            break   # central directory: all members read
        sys.exit(f"STOP unexpected signature {sig!r}")
    ver, flag, meth, _, _, crc, csz, usz, nl, xl = struct.unpack("<HHHHHIIIHH", read(26))
    name = read(nl).decode("utf-8" if flag & 0x800 else "cp437")
    extra = read(xl)
    if flag & 8:
        sys.exit(f"STOP data descriptor not handled ({name})")
    if csz == 0xFFFFFFFF or usz == 0xFFFFFFFF:   # zip64 sizes in the extra field
        i = 0
        while i + 4 <= len(extra):
            hid, hl = struct.unpack("<HH", extra[i:i + 4])
            if hid == 1:
                v = struct.unpack("<" + "Q" * (hl // 8), extra[i + 4:i + 4 + hl])
                if usz == 0xFFFFFFFF:
                    usz, v = v[0], v[1:]
                if csz == 0xFFFFFFFF:
                    csz = v[0]
            i += 4 + hl
    if ".." in name.split("/") or name.startswith("/"):
        sys.exit(f"STOP unsafe member name {name}")
    path = os.path.join(OUT, name)
    if name.endswith("/"):
        os.makedirs(path, exist_ok=True)
    write = not name.endswith("/") and not (EX and EX.search(name))
    fo = None
    if write:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fo = open(path + ".part", "wb")
    if meth not in (0, 8):
        sys.exit(f"STOP unsupported method {meth} ({name})")
    z = zlib.decompressobj(-15) if meth == 8 else None
    c = size = 0
    left = csz
    while left:
        d = read(min(left, 1 << 22))
        left -= len(d)
        u = z.decompress(d) if z else d
        c, size = zlib.crc32(u, c), size + len(u)
        if fo:
            fo.write(u)
    if z:
        u = z.flush()
        c, size = zlib.crc32(u, c), size + len(u)
        if fo:
            fo.write(u)
    if c != crc or size != usz:
        sys.exit(f"STOP CRC/size mismatch {name}: crc {c:08x} vs {crc:08x}, size {size} vs {usz}")
    if fo:
        fo.close()
        os.replace(path + ".part", path)
        n_w += 1
        n_b += size
    elif not name.endswith("/"):
        n_x += 1
while f.read(1 << 22):   # drain the central directory so curl does not fail on a closed pipe
    pass
print(f"streamunzip: wrote {n_w} files ({n_b} bytes), read and skipped {n_x} excluded", flush=True)
