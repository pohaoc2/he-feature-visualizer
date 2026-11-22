"""hfv_retrain2 stage 3 (PREREG §9.1, §9.3) Colab L4: one-day presigned links for the model-N jobs that are not app 10
(c6A / r2B: the ch6 ablation halves; r2B also arm ND, 384 images, and Fig 2 Concat on crc36ho 300 + crc33sec 100). Each session normally runs right after an
app-10 session in the same runtime (r2_boot.sh reuses /content/hfvk2). Needs boto3 and this machine's AWS login; links
are never committed or printed (written mode 600 into OUT, outside git). Link helpers = sim-exp-validation
scripts/colab/app7_presign.py (bucket, 2026-11-11 bundle prefix BP).

  boot OUT TAG CKMD5   r2_boot.sh with URL_* GET links (req.txt and three pretrained tars from BP; PixCell cb62fa7 from
                       hfv_retrain_20261118, read only; k2.tar = the app-10 prefix infer_tr_concat2_e20.tar = model N,
                       read only; kit.tar.gz = r2_gen.py, evalgen_r2.py, pxrun.py, uploaded here; the job inputs from
                       hfv_retrain2_20261121/stage3/inputs/), PUT links for the log and one jobs_out_<out>.tar per job
                       -> OUT/boot_TAG.sh, OUT/TAG_curl.txt (the notebook line)
  get OUT              OUT/get.txt "<name> <GET link>" for the three jobs_out_*.tar (Hyak)
  fetch OUT            colab logs and jobs_out_*.tar into OUT
Every signed GET object must exist (head_object) before the boot is written.
"""
import io
import os
import re
import sys
import tarfile
from pathlib import Path

RP = "pohao/hfv_retrain2_20261121/stage3/colab/"
os.environ["RUN_PREFIX"] = RP
HERE = Path(__file__).resolve().parent
SEV = Path(os.environ.get("SEV_REPO", HERE.parents[4] / "sim-exp-validation"))
sys.path.insert(0, str(SEV / "scripts/colab"))
import app7_presign as A7  # noqa: E402

B, BP, s3, link, private = A7.B, A7.BP, A7.s3, A7.link, A7.private
assert A7.RP == RP
HP = "pohao/hfv_retrain_20261118/"
IP = "pohao/hfv_retrain2_20261121/stage3/inputs/"
K2 = "pohao/sim-exp-validation/colab/sim2real-app10-20261121/infer_tr_concat2_e20.tar"
# 2026-11-21 ch6 ablation first (sim-exp-validation prereg_ch6_ablation.md): each session runs one half of it; session B
# then runs the retrain-2 stage-3 jobs (independent of the ablation). App 10 waits for the ablation's rule.
SESS = {"c6A": ["r2gen:ch6a"], "r2B": ["r2gen:ch6b", "r2gen:r2ND", "fig2c:crc36ho", "fig2c:crc33sec"]}
KINP = {"r2ND": IP, "ch6a": "pohao/sim-exp-validation/colab/sim2real-ch6-20261121/inputs/",
        "ch6b": "pohao/sim-exp-validation/colab/sim2real-ch6-20261121/inputs/"}
LISTS = {"crc36ho": "crc36ho_p300_ids.txt", "crc33sec": "crc33sec_fig2_ids.txt"}


def outname(job):
    kind, name = job.split(":")
    return name if kind == "r2gen" else f"fig2c_{name}"


OUTS = [f"jobs_out_{outname(j)}.tar" for js in SESS.values() for j in js]


def var(n):
    return "URL_" + n.replace(".", "_").replace("-", "_")


def main():
    mode, out = sys.argv[1], Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    if mode == "boot":
        tag, ckmd5 = sys.argv[3], sys.argv[4]
        if tag not in SESS or not re.fullmatch(r"[0-9a-f]{32}", ckmd5):
            raise SystemExit(f"tag {tag} / md5 {ckmd5}")
        kit = io.BytesIO()
        with tarfile.open(fileobj=kit, mode="w:gz") as tf:
            for f in ("r2_gen.py", "evalgen_r2.py", "pxrun.py"):
                tf.add(HERE / f, arcname=f)
        s3.put_object(Bucket=B, Key=RP + f"kit_{tag}.tar.gz", Body=kit.getvalue())
        keys = {"req.txt": BP + "req.txt", "pixcell.tar.gz": HP + "inputs/pixcell_cb62fa7.tar.gz", "k2.tar": K2,
                "kit.tar.gz": RP + f"kit_{tag}.tar.gz"}
        for d in ("pixcell-256", "pixcell-256-controlnet", "sd-3.5-vae"):
            keys[d + ".tar"] = BP + d + ".tar"
        for j in SESS[tag]:
            kind, name = j.split(":")
            if kind == "r2gen":
                keys[f"kin_{name}.tar"] = KINP[name] + f"kin_{name}.tar"
            else:
                keys[f"evalpackN_{name}.tar"] = IP + f"evalpackN_{name}.tar"
                keys[f"list_{name}.txt"] = IP + LISTS[name]
        for k in keys.values():
            s3.head_object(Bucket=B, Key=k)
        head = "#!/bin/bash\n" + "".join(f"{var(n)}='{link('get_object', k)}'\n" for n, k in keys.items())
        head += f"LOGURL='{link('put_object', RP + f'colab_{tag}.log')}'\n"
        for j in SESS[tag]:
            o = outname(j)
            head += f"OUTURL_{o.replace('-', '_')}='{link('put_object', RP + f'jobs_out_{o}.tar')}'\n"
        head += f"TAG='{tag}'\nJOBS='{' '.join(SESS[tag])}'\nCKMD5='{ckmd5}'\n"
        body = (HERE / "r2_boot.sh").read_text().split("\n", 1)[1]
        private(out / f"boot_{tag}.sh", head + body)
        s3.upload_file(str(out / f"boot_{tag}.sh"), B, RP + f"boot_{tag}.sh")
        line = (f"!curl -sf '{link('get_object', RP + f'boot_{tag}.sh')}' -o /content/boot_{tag}.sh "
                f"&& bash /content/boot_{tag}.sh\n")
        private(out / f"{tag}_curl.txt", line)
        print(f"{tag}: {out / f'{tag}_curl.txt'}; log s3://{B}/{RP}colab_{tag}.log; jobs {SESS[tag]}")
    elif mode == "get":
        private(out / "get.txt", "".join(f"{n} {link('get_object', RP + n)}\n" for n in OUTS))
        print(out / "get.txt")
    elif mode == "fetch":
        for n in [f"colab_{t}.log" for t in SESS] + OUTS:
            try:
                s3.download_file(B, RP + n, str(out / n))
                print("fetched", n)
            except Exception as e:
                print("missing", n, type(e).__name__)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
