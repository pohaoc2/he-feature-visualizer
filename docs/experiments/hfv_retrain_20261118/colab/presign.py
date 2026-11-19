"""hfv_retrain_20261118 local side: one-day presigned links for a Colab boot (needs boto3 + this machine's AWS login).

  python presign.py OUT TAG TASKS [--put NAME[:multipart:N_PARTS:PART_MB] ...]
Writes OUT/boot_<TAG>.sh = LINKS (JSON) + TASKS + TAG + boot.sh, uploads it to the run prefix and prints the one line the
notebook runs. OUT stays outside git (mode 600); links expire after one day and are never committed.
GET names: code, kit, req, the four pretrained tars, orion-crc33, newch33, cellvit_kit, he36, crc36ch.
PUT slots: log_<TAG>, result_<TAG>, masklat_crc33_01, cellvit_crc36, feat_crc36, masklat_crc36 by default; --put adds more (multipart slots for files > 5 GB).
An extra argument train:RUN adds, for every 2nd epoch and epoch 20, the checkpoint slots ckpt_RUN_eNN_cn (multipart,
24 x 512 MB), _tme and _meta, and GET links of the same keys (train_wrap.py --fetch-resume); GET names c36train
(inputs/c36train.tar) and masklat_crc33_01 are always included.
"""
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

HERE = Path(__file__).resolve().parent
B = "bagherilab-working"
P = "pohao/hfv_retrain_20261118/"
UG = "pohao/sim-exp-validation/colab/uni-gen-test-20261111/"   # pretrained tars + req.txt of the 2026-11-11 Colab runs
DAY = 86400
# SigV4: only the host header is signed, so a client's Content-Type cannot break a PUT / POST link (the default SigV2
# links of the first L4 boot signed Content-Type and urllib's default form type made every PUT fail with 403)
s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version="s3v4"))


def get(key):
    return s3.generate_presigned_url("get_object", Params=dict(Bucket=B, Key=key), ExpiresIn=DAY)


def put_slot(spec):
    name, *mp = spec.split(":")
    key = P + "outputs/" + name
    if not mp:
        return name, {"single": s3.generate_presigned_url("put_object", Params=dict(Bucket=B, Key=key), ExpiresIn=DAY)}
    n_parts, part_mb = int(mp[1]), int(mp[2])
    up = s3.create_multipart_upload(Bucket=B, Key=key)["UploadId"]
    parts = [s3.generate_presigned_url("upload_part", Params=dict(Bucket=B, Key=key, UploadId=up, PartNumber=i + 1),
                                       ExpiresIn=DAY) for i in range(n_parts)]
    comp = s3.generate_presigned_url("complete_multipart_upload", Params=dict(Bucket=B, Key=key, UploadId=up),
                                     ExpiresIn=DAY, HttpMethod="POST")
    return name, {"multipart": {"upload_id": up, "part_bytes": part_mb << 20, "parts": parts, "complete": comp}}


def main():
    out, tag, tasks = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    extra_put = [a for a in sys.argv[4:] if not a.startswith("--")]
    out.mkdir(parents=True, exist_ok=True)
    g = {"code": get(P + "inputs/pixcell_cb62fa7.tar.gz"), "kit": get(P + "inputs/kit.tar.gz"), "req": get(UG + "req.txt"),
         "orion-crc33": get("pohao/share_space/02_datasets/orion-crc33.zip"),
         "newch33": get(P + "inputs/new_exp_channels_crc33_01.tar"),
         "cellvit_kit": get(P + "inputs/cellvit_kit.tar"), "he36": get(P + "inputs/crc36_he_tiles.tar"),
         "crc36ch": get(P + "inputs/crc36_exp_channels_pass1.tar")}
    for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae", "uni-2h"):
        g[d] = get(UG + d + ".tar")
    g["c36train"] = get(P + "inputs/c36train.tar")
    g["masklat_crc33_01"] = get(P + "outputs/masklat_crc33_01")
    runs = [a.split(":", 1)[1] for a in extra_put if a.startswith("train:")]
    extra_put = [a for a in extra_put if not a.startswith("train:")]
    for run in runs:
        for e in sorted(set(range(2, 21, 2)) | {20}):
            k = f"ckpt_{run}_e{e:02d}"
            extra_put += [f"{k}_cn:multipart:24:512", f"{k}_tme", f"{k}_meta"]
            for x in ("cn", "tme", "meta"):
                g[f"{k}_{x}"] = get(P + f"outputs/{k}_{x}")
    pu = dict(put_slot(s) for s in [f"log_{tag}", f"result_{tag}", "masklat_crc33_01", "cellvit_crc36", "feat_crc36", "masklat_crc36"] + extra_put)
    import datetime
    links = {"expires": (datetime.datetime.utcnow() + datetime.timedelta(seconds=DAY)).isoformat() + "Z", "get": g, "put": pu}
    body = (HERE / "boot.sh").read_text().split("\n", 1)[1]
    head = f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nTASKS='{tasks}'\nTAG='{tag}'\n"
    f = out / f"boot_{tag}.sh"
    f.write_text(head + body)
    os.chmod(f, 0o600)
    s3.upload_file(str(f), B, P + f"boot/boot_{tag}.sh")
    (out / f"links_{tag}.json").write_text(json.dumps(links))
    os.chmod(out / f"links_{tag}.json", 0o600)
    print(f"!curl -sf '{get(P + f'boot/boot_{tag}.sh')}' -o /content/boot_{tag}.sh && bash /content/boot_{tag}.sh")


if __name__ == "__main__":
    main()
