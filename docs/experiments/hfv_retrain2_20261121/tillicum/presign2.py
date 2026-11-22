"""hfv_retrain2_20261121 local side: one-day presigned links for the Tillicum setup + Concat retrain 2 (needs boto3 and
this machine's AWS login; nothing else). Same scheme as hfv_retrain_20261118/colab/presign.py (SigV4, one day, file mode
600, never committed, no AWS credentials on Tillicum).

  python presign2.py OUT RUN        (RUN = tr_concat2) -> OUT/links_RUN.json
GET: code, kit, req, the four pretrained tars, orion-crc33 (share_space zip) from the first retrain; newch33_v2,
masklat33_v2, c36train_v2 from this run's inputs/; ckpt_RUN_eNN_* (resume). PUT: log_RUN_h200, ckpt_RUN_eNN_* under
this run's outputs/.
"""
import datetime
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.config import Config

B = "bagherilab-working"
P1 = "pohao/hfv_retrain_20261118/"
P2 = "pohao/hfv_retrain2_20261121/"
UG = "pohao/sim-exp-validation/colab/uni-gen-test-20261111/"
DAY = 86400
s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version="s3v4"))
get = lambda key: s3.generate_presigned_url("get_object", Params=dict(Bucket=B, Key=key), ExpiresIn=DAY)


def put_slot(name, parts=None, mb=None):
    key = P2 + "outputs/" + name
    if parts is None:
        return {"single": s3.generate_presigned_url("put_object", Params=dict(Bucket=B, Key=key), ExpiresIn=DAY)}
    up = s3.create_multipart_upload(Bucket=B, Key=key)["UploadId"]
    return {"multipart": {"upload_id": up, "part_bytes": mb << 20,
                          "parts": [s3.generate_presigned_url("upload_part", Params=dict(Bucket=B, Key=key, UploadId=up, PartNumber=i + 1),
                                                              ExpiresIn=DAY) for i in range(parts)],
                          "complete": s3.generate_presigned_url("complete_multipart_upload", Params=dict(Bucket=B, Key=key, UploadId=up),
                                                                ExpiresIn=DAY, HttpMethod="POST")}}


def main():
    out, run = Path(sys.argv[1]), sys.argv[2]
    out.mkdir(parents=True, exist_ok=True)
    g = {"code": get(P1 + "inputs/pixcell_cb62fa7.tar.gz"), "kit": get(P1 + "inputs/kit.tar.gz"), "req": get(UG + "req.txt"),
         "orion-crc33": get("pohao/share_space/02_datasets/orion-crc33.zip")}
    for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae", "uni-2h"):
        g[d] = get(UG + d + ".tar")
    for n in ("newch33_v2", "masklat33_v2", "c36train_v2"):
        g[n] = get(P2 + f"inputs/{n}.tar")
    pu = {f"log_{run}_h200": put_slot(f"log_{run}_h200")}
    for e in range(2, 21, 2):
        k = f"ckpt_{run}_e{e:02d}"
        pu[f"{k}_cn"], pu[f"{k}_tme"], pu[f"{k}_meta"] = put_slot(f"{k}_cn", 24, 512), put_slot(f"{k}_tme"), put_slot(f"{k}_meta")
        for x in ("cn", "tme", "meta"):
            g[f"{k}_{x}"] = get(P2 + f"outputs/{k}_{x}")
    links = {"expires": (datetime.datetime.utcnow() + datetime.timedelta(seconds=DAY)).isoformat() + "Z", "get": g, "put": pu}
    f = out / f"links_{run}.json"
    f.write_text(json.dumps(links))
    os.chmod(f, 0o600)
    print("wrote", f, "(copy to Tillicum W/links.json and W/links_RUN.json; never commit)")


if __name__ == "__main__":
    main()
