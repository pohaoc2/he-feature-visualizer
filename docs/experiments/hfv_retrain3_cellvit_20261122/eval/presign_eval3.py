"""hfv_retrain3_cellvit_20261122, local side: one Colab L4 generation boot with one-day presigned links (fallback for a
generation job that sits PENDING on Hyak; user rule 2026-11-22).

  python presign_eval3.py OUT TAG "TASK [TASK ...]"
Copy of hfv_retrain_20261118 eval/presign_eval.py; the Colab body is that run's eval/boot_eval.sh unchanged (evalkit with
evalgen.py / pxrun.py / s3io.py, the same code the Hyak gen3.slurm reads). Changed: the link names the boot asks for
point at this run's objects:
  evalpack_<set>  -> this run's inputs/eval/evalpack3_<set>.tar (official-CellViT cell channels, method C, new latents)
  infer_tr_<v>    -> this run's outputs/infer_tr3_<v>_e20.tar (folder names tr_<v>; slimck3.slurm)
  list_<file>     -> the first retrain's inputs/eval/<file> (same tile lists)
and the PUT slots (log_<TAG>, gen_<TAG>_<k>) are under this run's outputs/. Only the checkpoints that exist are
signed (abl needs Concat only). Writes OUT/boot_eval_<TAG>.sh (mode 600, outside git), uploads it under this run's
boot/ and prints the notebook line (contains a one-day link; never committed).
"""
import datetime
import json
import os
import sys
from pathlib import Path

R1 = Path(__file__).resolve().parents[2] / "hfv_retrain_20261118"
sys.path.insert(0, str(R1 / "colab"))
import presign as ps  # noqa: E402

out, tag, tasks = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
out.mkdir(parents=True, exist_ok=True)
P1, P3 = ps.P, "pohao/hfv_retrain3_cellvit_20261122/"
keys = {"evalkit": P1 + "inputs/evalkit.tar.gz", "code": P1 + "inputs/pixcell_cb62fa7.tar.gz", "req": ps.UG + "req.txt"}
for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae", "uni-2h"):
    keys[d] = ps.UG + d + ".tar"
for o in ps.s3.list_objects_v2(Bucket=ps.B, Prefix=P1 + "inputs/eval/").get("Contents", []):
    name = o["Key"].rsplit("/", 1)[1]
    if "/paper_lists/" not in o["Key"] and name.endswith((".txt", ".json")) and not name.endswith(".tar.json"):
        keys["list_" + name] = o["Key"]
for s in ("crc36ho", "crc33sec"):
    keys["evalpack_" + s] = P3 + f"inputs/eval/evalpack3_{s}.tar"
have = {o["Key"] for o in ps.s3.list_objects_v2(Bucket=ps.B, Prefix=P3 + "outputs/infer_").get("Contents", [])}
for v in ("concat", "grouped", "bypass", "perch"):
    k = P3 + f"outputs/infer_tr3_{v}_e20.tar"
    if k in have:
        keys["infer_tr_" + v] = k
need = {"infer_tr_concat"} | ({f"infer_tr_{v}" for v in ("grouped", "bypass", "perch")} if "fig2:" in tasks else set())
miss = sorted(need - set(keys))
if miss:
    sys.exit(f"STOP checkpoints not on S3 yet: {miss}")
g = {n: ps.get(k) for n, k in keys.items()}
meta = {}
for n, k in keys.items():
    h = ps.s3.head_object(Bucket=ps.B, Key=k)
    meta[n] = {"size": h["ContentLength"], "etag": h["ETag"]}
ps.P = P3   # PUT slots under this run's outputs/
n = len(tasks.split())
pu = dict(ps.put_slot(s) for s in [f"log_{tag}"] + [f"gen_{tag}_{k}:multipart:16:512" for k in range(1, n + 1)])
for k in range(1, n + 1):
    pu[f"gen_{tag}_{k}"]["single"] = ps.s3.generate_presigned_url(
        "put_object", Params=dict(Bucket=ps.B, Key=P3 + f"outputs/gen_{tag}_{k}"), ExpiresIn=ps.DAY)
links = {"expires": (datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                     + datetime.timedelta(seconds=ps.DAY)).isoformat() + "Z", "get": g, "meta": meta, "put": pu}
body = (R1 / "eval" / "boot_eval.sh").read_text().split("\n", 1)[1]
f = out / f"boot_eval_{tag}.sh"
f.write_text(f"#!/bin/bash\nLINKS='{json.dumps(links)}'\nTASKS='{tasks}'\nTAG='{tag}'\n" + body)
os.chmod(f, 0o600)
ps.s3.upload_file(str(f), ps.B, P3 + f"boot/boot_eval_{tag}.sh")
line = f"!curl -sf '{ps.get(P3 + f'boot/boot_eval_{tag}.sh')}' -o /content/boot_eval_{tag}.sh && bash /content/boot_eval_{tag}.sh"
(out / f"notebook_line_{tag}.txt").write_text(line + "\n")
os.chmod(out / f"notebook_line_{tag}.txt", 0o600)
print(f"wrote {f} and {out / f'notebook_line_{tag}.txt'} (links expire {links['expires']})")
