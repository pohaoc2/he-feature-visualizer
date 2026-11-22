"""paperpq_official_20261122, local side: one Colab L4 boot (boot_pp.sh) with one-day presigned links.

  python presign_pp.py OUT TAG "ROOT:FAMILY [ROOT:FAMILY ...]"
GET: kit (inputs/kit.tar.gz: pp_gen.py, pp_appear.py, pxrun.py, pxscore.py, s3io.py, ids100.txt), roots (inputs/roots.tar
from pp_prep.slurm), PixCell cb62fa7, req.txt and the four pretrained tars (the hfv_retrain Colab inputs), the paper
checkpoints (share_space/03_checkpoints) and the retrain-1 inference tars. PUT: log_<TAG>, gen_<ROOT>. Signing as hfv
colab/presign.py (SigV4, host header only); LINKS.meta = {name: {size, etag}} for s3io.py's checks. Writes
OUT/boot_pp_<TAG>.sh (mode 600, outside git), uploads it and prints the notebook line. Links are never committed.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

HERE = Path(__file__).resolve().parent
B, DAY = "bagherilab-working", 86400
P2 = "pohao/paperpq_official_20261122/"
HF = "pohao/hfv_retrain_20261118/"
UG = "pohao/sim-exp-validation/colab/uni-gen-test-20261111/"
CK = "pohao/share_space/03_checkpoints/"
PAPER = {"production": "concat_retrain_post_fix/full_seed_42/checkpoint", "a1_concat": "concat/full_seed_42/checkpoint",
         "a1_per_channel": "tillicum_95470/per_channel_95470_3/checkpoints", "a2_bypass_full_tme": "bypass/full_seed_42/checkpoint"}
s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version="s3v4"))
out, tag, tasks = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
fams = {t.split(":")[1] for t in tasks.split()}
keys = {"kit": P2 + "inputs/kit.tar.gz", "roots": P2 + "inputs/roots.tar", "code": HF + "inputs/pixcell_cb62fa7.tar.gz",
        "req": UG + "req.txt"}
for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae", "uni-2h"):
    keys[d] = UG + d + ".tar"
if "P" in fams:
    for v, p in PAPER.items():
        keys[f"ck_{v}_cn"] = CK + p + "/step_0002600/controlnet_epoch_20_step_2600.pth"
        keys[f"ck_{v}_tme"] = CK + p + "/step_0002600/tme_module.pth"
    keys["ck_perch_cfg"] = CK + "tillicum_95470/per_channel_95470_3/config.py"
if "R" in fams:
    for r in ("tr_concat", "tr_grouped", "tr_bypass", "tr_perch"):
        keys["infer_" + r] = HF + f"outputs/infer_{r}_e20.tar"
g, meta = {}, {}
for n, k in keys.items():
    h = s3.head_object(Bucket=B, Key=k)
    meta[n] = {"size": h["ContentLength"], "etag": h["ETag"]}
    g[n] = s3.generate_presigned_url("get_object", Params=dict(Bucket=B, Key=k), ExpiresIn=DAY)
pu = {}
for slot in [f"log_{tag}"] + [f"gen_{t.split(':')[0]}" for t in tasks.split()]:
    pu[slot] = {"single": s3.generate_presigned_url("put_object", Params=dict(Bucket=B, Key=P2 + "outputs/" + slot),
                                                    ExpiresIn=DAY)}
exp = (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) + datetime.timedelta(seconds=DAY)).isoformat() + "Z"
links = {"expires": exp, "get": g, "meta": meta, "put": pu}
body = (HERE / "boot_pp.sh").read_text().split("\n", 1)[1]
f = out / f"boot_pp_{tag}.sh"
f.write_text(f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nTASKS='{tasks}'\nTAG='{tag}'\n" + body)
os.chmod(f, 0o600)
s3.upload_file(str(f), B, P2 + f"boot/boot_pp_{tag}.sh")
u = s3.generate_presigned_url("get_object", Params=dict(Bucket=B, Key=P2 + f"boot/boot_pp_{tag}.sh"), ExpiresIn=DAY)
print(f"!curl -sf '{u}' -o /content/boot_pp_{tag}.sh && bash /content/boot_pp_{tag}.sh")
