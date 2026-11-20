"""hfv_retrain_20261118: presign the real-H&E CellViT boot (boot_realcv.sh; README deviation "PQ 參考遮罩", 2026-11-20).

  python presign_realcv.py OUT TAG "PACK [PACK ...]"
GET: scorekit, code, req, uni-2h, cellvit_kit (as presign_score.py, so the setup block matches boot_score.sh) and
realhe_<PACK> (inputs/eval/realhe_<PACK>.zip). PUT (single): log_<TAG>, realcv_<PACK> -> outputs/realcv_<PACK>.tar.
Writes OUT/boot_realcv_<TAG>.sh (mode 600, outside git), uploads it to boot/ and prints the notebook line.
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
    keys[f"realhe_{it}"] = P + f"inputs/eval/realhe_{it}.zip"
    k = P + f"outputs/realcv_{it}.tar"
    try:
        ps.s3.head_object(Bucket=ps.B, Key=k)
        sys.exit(f"STOP {k} exists (new keys only)")
    except ps.s3.exceptions.ClientError:
        pass
    put_keys[f"realcv_{it}"] = k
meta = {}
for n, k in keys.items():
    h = ps.s3.head_object(Bucket=ps.B, Key=k)
    meta[n] = {"size": h["ContentLength"], "etag": h["ETag"]}
g = {n: ps.get(k) for n, k in keys.items()}
pu = {n: {"single": ps.s3.generate_presigned_url("put_object", Params=dict(Bucket=ps.B, Key=k), ExpiresIn=ps.DAY)}
      for n, k in put_keys.items()}
links = {"expires": (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                     + datetime.timedelta(seconds=ps.DAY)).isoformat() + "Z", "get": g, "meta": meta, "put": pu}
body = (HERE / "boot_realcv.sh").read_text().split("\n", 1)[1]
f = out / f"boot_realcv_{tag}.sh"
f.write_text(f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nITEMS='{' '.join(items)}'\nTAG='{tag}'\n" + body)
os.chmod(f, 0o600)
ps.s3.upload_file(str(f), ps.B, P + f"boot/boot_realcv_{tag}.sh")
print(f"!curl -sf '{ps.get(P + f'boot/boot_realcv_{tag}.sh')}' -o /content/boot_realcv_{tag}.sh && bash /content/boot_realcv_{tag}.sh")
