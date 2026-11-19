"""md5 of every regular file in a tar read from stdin (nothing written to disk):
  curl -sf URL | python tarmd5.py NAME      -> "<md5>  ./<path below the directory NAME>" lines, sorted
Used to compare the pretrained tars Colab downloads with the copies already on Tillicum without storing the tars.
"""
import hashlib
import sys
import tarfile

NAME = sys.argv[1]
rows = []
with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as tf:
    for m in tf:
        if not m.isfile():
            continue
        parts = m.name.split("/")
        if NAME not in parts:
            sys.exit(f"STOP {m.name} not under {NAME}")
        rel = "./" + "/".join(parts[parts.index(NAME) + 1:])
        h = hashlib.md5()
        r = tf.extractfile(m)
        for b in iter(lambda: r.read(1 << 24), b""):
            h.update(b)
        rows.append(f"{h.hexdigest()}  {rel}")
print("\n".join(sorted(rows, key=lambda s: s[34:])))
