"""hfv_retrain_20261118 scoring on Colab L4, local side: one boot with one-day presigned links (README deviation
"評分改在 Colab L4", 2026-11-20).

  python presign_score.py OUT TAG "ITEM [ITEM ...]"
ITEM = KIND:SRC:GROUP:PACK[:LIST] as in score_hyak.slurm. GET: scorekit (inputs/scorekit.tar.gz = evalscore.py,
pxscore.py, fig2tiles.py, s3io.py), code (PixCell cb62fa7), req, uni-2h, cellvit_kit, evalpack_<PACK>, list_<LIST>,
src_<SRC> (outputs/<SRC>). PUT (single, reusable until expiry): log_<TAG>, score_<GROUP> -> outputs/score_<GROUP>.tar
(the key score_hyak.slurm writes and figs_hyak.slurm reads), scoreres_<GROUP> -> outputs/scoreres_<GROUP>.tar.
Writes OUT/boot_score_<TAG>.sh (mode 600, outside git), uploads it and prints the notebook line. LINKS carries meta
{name: {size, etag}} from head_object; s3io.py get checks both.
"""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colab"))
import presign as ps  # noqa: E402

out, tag, items = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split()
out.mkdir(parents=True, exist_ok=True)
P = ps.P
keys = {"scorekit": P + "inputs/scorekit.tar.gz", "code": P + "inputs/pixcell_cb62fa7.tar.gz", "req": ps.UG + "req.txt",
        "uni-2h": ps.UG + "uni-2h.tar", "cellvit_kit": P + "inputs/cellvit_kit.tar"}
put_keys = {f"log_{tag}": P + f"outputs/log_{tag}"}
for it in items:
    kind, src, group, pack, *lst = it.split(":")
    assert kind in ("fig2", "abl", "loo", "decomp", "crc34"), it
    keys[f"evalpack_{pack}"] = P + f"inputs/eval/evalpack_{pack}.tar"
    if lst:
        keys[f"list_{lst[0]}"] = P + f"inputs/eval/{lst[0]}"
    keys[f"src_{src}"] = P + f"outputs/{src}"
    put_keys[f"score_{group}"] = P + f"outputs/score_{group}.tar"
    put_keys[f"scoreres_{group}"] = P + f"outputs/scoreres_{group}.tar"
meta = {}
for n, k in keys.items():
    h = ps.s3.head_object(Bucket=ps.B, Key=k)   # raises if an input is missing
    meta[n] = {"size": h["ContentLength"], "etag": h["ETag"]}
g = {n: ps.get(k) for n, k in keys.items()}
pu = {n: {"single": ps.s3.generate_presigned_url("put_object", Params=dict(Bucket=ps.B, Key=k), ExpiresIn=ps.DAY)}
      for n, k in put_keys.items()}
links = {"expires": (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                     + datetime.timedelta(seconds=ps.DAY)).isoformat() + "Z", "get": g, "meta": meta, "put": pu}
body = (HERE / "boot_score.sh").read_text().split("\n", 1)[1]
f = out / f"boot_score_{tag}.sh"
f.write_text(f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nITEMS='{' '.join(items)}'\nTAG='{tag}'\n" + body)
os.chmod(f, 0o600)
ps.s3.upload_file(str(f), ps.B, P + f"boot/boot_score_{tag}.sh")
print(f"!curl -sf '{ps.get(P + f'boot/boot_score_{tag}.sh')}' -o /content/boot_score_{tag}.sh && bash /content/boot_score_{tag}.sh")
