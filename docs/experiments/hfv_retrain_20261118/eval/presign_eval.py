"""hfv_retrain_20261118 evaluation, local side: one Colab generation boot with one-day presigned links.

  python presign_eval.py OUT TAG "TASK [TASK ...]"
TASKS as in evalgen.py (e.g. "speed:orion-crc36ho:crc36ho_p300_ids.txt fig2:orion-crc36ho:crc36ho_p300_ids.txt").
GET: evalkit (inputs/evalkit.tar.gz), code, req, the four pretrained tars, every evalpack_* and list file under
inputs/eval/ (list_<file name>), infer_<run> (outputs/infer_<run>_e20.tar). PUT: log_<TAG>, gen_<TAG>_<k> for each task
(multipart, 16 x 512 MB). Writes OUT/boot_eval_<TAG>.sh (mode 600, outside git), uploads it and prints the notebook line.
Uses presign.py's signing (SigV4, host header only).
"""
import datetime
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "colab"))
import presign as ps  # noqa: E402

out, tag, tasks = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
P = ps.P
g = {"evalkit": ps.get(P + "inputs/evalkit.tar.gz"), "code": ps.get(P + "inputs/pixcell_cb62fa7.tar.gz"),
     "req": ps.get(ps.UG + "req.txt")}
for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae", "uni-2h"):
    g[d] = ps.get(ps.UG + d + ".tar")
for o in ps.s3.list_objects_v2(Bucket=ps.B, Prefix=P + "inputs/eval/").get("Contents", []):
    name = o["Key"].rsplit("/", 1)[1]
    if name.startswith("evalpack_") and name.endswith(".tar"):
        g[name[:-4]] = ps.get(o["Key"])
    elif "/paper_lists/" not in o["Key"] and name.endswith((".txt", ".json")) and not name.endswith(".tar.json"):
        g["list_" + name] = ps.get(o["Key"])
for r in ("tr_concat", "tr_grouped", "tr_bypass", "tr_perch"):
    g["infer_" + r] = ps.get(P + f"outputs/infer_{r}_e20.tar")
n = len(tasks.split())
pu = dict(ps.put_slot(s) for s in [f"log_{tag}"] + [f"gen_{tag}_{k}:multipart:16:512" for k in range(1, n + 1)])
for k in range(1, n + 1):   # reusable single PUT for files < 5 GB (a rerun of the same boot must be able to upload again)
    pu[f"gen_{tag}_{k}"]["single"] = ps.s3.generate_presigned_url(
        "put_object", Params=dict(Bucket=ps.B, Key=P + f"outputs/gen_{tag}_{k}"), ExpiresIn=ps.DAY)
links = {"expires": (datetime.datetime.utcnow() + datetime.timedelta(seconds=ps.DAY)).isoformat() + "Z", "get": g, "put": pu}
body = (HERE / "boot_eval.sh").read_text().split("\n", 1)[1]
f = out / f"boot_eval_{tag}.sh"
f.write_text(f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nTASKS='{tasks}'\nTAG='{tag}'\n" + body)
os.chmod(f, 0o600)
ps.s3.upload_file(str(f), ps.B, P + f"boot/boot_eval_{tag}.sh")
print(f"!curl -sf '{ps.get(P + f'boot/boot_eval_{tag}.sh')}' -o /content/boot_eval_{tag}.sh && bash /content/boot_eval_{tag}.sh")
