"""hfv_retrain_20261118 speed probe (Colab): ~50 optimizer steps at effective batch 80 per variant config.

  python probe_sweep.py PIXCELL DATA_ROOT OUT_DIR TAG CONFIG [CONFIG ...]
For each config, the micro-batch is the largest divisor of 80 that fits (tried 80, 40, 20, 16, 10, 8, 5, 4, 2 in turn; a
CUDA out-of-memory run moves to the next one), with gradient accumulation making up 80 (train_wrap.py). Writes
OUT_DIR/probe_<TAG>.json: per config the GPU, micro-batch, accumulation, samples/s over steps 11-50, peak memory, and the
micro-batches that ran out of memory.
First, outputs of earlier tasks in this runtime that did not reach S3 (no <file>.sent marker; the first L4 boot's PUTs
failed on a link-signature mismatch, fixed in s3io.py, which this task re-fetches) are sent again.
"""
import json
import os
import subprocess
import sys

PIX, DATA, OUT, TAG = sys.argv[1:5]
CONFIGS = sys.argv[5:]
HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = 50
sys.path.insert(0, HERE)
import s3io  # noqa: E402
W = "/content/hfv"
if os.path.exists(f"{W}/links.json"):
    L = json.load(open(f"{W}/links.json"))
    for slot in ("masklat_crc33_01", "cellvit_crc36", "feat_crc36"):
        f = f"{W}/{slot}.tar"
        if os.path.exists(f) and not os.path.exists(f + ".sent") and slot in L["put"]:
            try:
                s3io.put(L, slot, f)
                open(f + ".sent", "w").close()
            except Exception as e:
                print(f"resend {slot} failed: {e}", flush=True)
res = {}
for cfg in CONFIGS:
    name = os.path.basename(cfg).replace(".py", "")
    res[name] = dict(oom=[])
    for mb in (80, 40, 20, 16, 10, 8, 5, 4, 2):
        wd = f"{OUT}/probe_{TAG}/{name}_mb{mb}"
        pj = f"{wd}/probe.json"
        log = f"{wd}.log"
        os.makedirs(wd, exist_ok=True)
        with open(log, "w") as f:
            rc = subprocess.run([sys.executable, "-u", f"{HERE}/train_wrap.py", "--pixcell", PIX, "--config", cfg,
                                 "--data-root", DATA, "--work-dir", wd, "--micro-bs", str(mb), "--max-steps", str(STEPS),
                                 "--probe-json", pj], stdout=f, stderr=subprocess.STDOUT).returncode
        txt = open(log, errors="replace").read()
        if rc == 0 and os.path.exists(pj):
            res[name].update(json.load(open(pj)))
            break
        if "out of memory" in txt.lower() or "OutOfMemoryError" in txt:
            res[name]["oom"].append(mb)
            print(f"{name} micro-bs {mb}: out of memory", flush=True)
            continue
        res[name]["error"] = dict(micro_bs=mb, rc=rc, tail=txt[-3000:])
        print(f"STOP {name} micro-bs {mb} failed rc {rc} (not OOM):\n{txt[-3000:]}", flush=True)
        break
    json.dump(res, open(f"{OUT}/probe_{TAG}.json", "w"), indent=1)
    print(name, {k: v for k, v in res[name].items() if k != "error"}, flush=True)
