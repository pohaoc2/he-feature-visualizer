"""hfv_retrain_20261118 follow-up 1: one Colab L4 boot per a4 tile shard (eval/boot_a4.sh), one-day presigned links.

  python presign_a4.py OUT TAG SHARD NSHARD
GET: a4kit (inputs/a4kit.tar.gz = a4run.py, s3io.py), code (PixCell cb62fa7), req and the four pretrained tars (the
2026-11-11 Colab set, as presign_eval.py), cellvit_kit, evalpack_crc33a4, a4_inputs (inputs/eval/), infer_tr_concat
(outputs/infer_tr_concat_e20.tar). PUT (single, reusable until expiry): log_<TAG>, a4gen_<SHARD+1> ->
outputs/a4gen_crc33a4_shard<SHARD+1>.tar. Writes OUT/boot_a4_<TAG>.sh (mode 600, outside git), uploads it and prints the
notebook line. LINKS carries meta {name: {size, etag}} from head_object; s3io.py get checks both.
"""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colab"))
import presign as ps  # noqa: E402

out, tag, shard, nshard = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
P = ps.P
keys = {"a4kit": P + "inputs/a4kit.tar.gz", "code": P + "inputs/pixcell_cb62fa7.tar.gz", "req": ps.UG + "req.txt",
        "cellvit_kit": P + "inputs/cellvit_kit.tar", "evalpack_crc33a4": P + "inputs/eval/evalpack_crc33a4.tar",
        "a4_inputs": P + "inputs/eval/a4_inputs.tar", "infer_tr_concat": P + "outputs/infer_tr_concat_e20.tar"}
for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae", "uni-2h"):
    keys[d] = ps.UG + d + ".tar"
k = shard + 1
put_keys = {f"log_{tag}": P + f"outputs/log_{tag}", f"a4gen_{k}": P + f"outputs/a4gen_crc33a4_shard{k}.tar"}
meta = {}
for n, key in keys.items():
    h = ps.s3.head_object(Bucket=ps.B, Key=key)   # raises if an input is missing
    meta[n] = {"size": h["ContentLength"], "etag": h["ETag"]}
g = {n: ps.get(key) for n, key in keys.items()}
pu = {n: {"single": ps.s3.generate_presigned_url("put_object", Params=dict(Bucket=ps.B, Key=key), ExpiresIn=ps.DAY)}
      for n, key in put_keys.items()}
links = {"expires": (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                     + datetime.timedelta(seconds=ps.DAY)).isoformat() + "Z", "get": g, "meta": meta, "put": pu}
body = (HERE / "boot_a4.sh").read_text().split("\n", 1)[1]
f = out / f"boot_a4_{tag}.sh"
f.write_text(f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nSHARD='{shard}'\nNSHARD='{nshard}'\nTAG='{tag}'\n" + body)
os.chmod(f, 0o600)
ps.s3.upload_file(str(f), ps.B, P + f"boot/boot_a4_{tag}.sh")
print(f"!curl -sf '{ps.get(P + f'boot/boot_a4_{tag}.sh')}' -o /content/boot_a4_{tag}.sh && bash /content/boot_a4_{tag}.sh")
