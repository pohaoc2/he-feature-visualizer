"""paperpq_official_20261122 (PREREG.md §5): Fig 2 generation on one root, then appearance scoring on the same GPU.

  python pp_gen.py --pixcell P --work W --root ROOT --family P|R --ids IDS --d D
P     PixCell cb62fa7 (unmodified) with pretrained_models/{pixcell-256, pixcell-256-controlnet, sd-3.5-vae, uni-2h}
W     node-local work folder; images land in W/out/tiles/<variant>/<tile>.png, scores in W/out/
ROOT  one data root from pp_prep.py (P_run, P_off, R_run, R_off)
family P = CellViT-mask generators (paper checkpoints, D/in/ck) + PixCell off-the-shelf; R = shape-library-mask
       generators (retrain-1 epoch-20 inference checkpoints, hfv_eval_20261118/ck/infer_e20) without off-the-shelf
Generation = tools/ablation_a1_a2/build_cache.py run unchanged through pxrun.py (30 steps, guidance 1.5, seed 42; the
off-the-shelf model is seeded with torch.manual_seed(42) per tile, hfv_retrain README deviation). The paths build_cache.py
hard-codes (INFERENCE_VARIANTS checkpoint folders, the per-channel config, data/orion-crc33) are served by a shadow root of
symlinks, the hfv_retrain eval/evalgen.py construction; no PixCell file changes. TF32: generation runs with the torch
default (NVIDIA_TF32_OVERRIDE unset, as the Fig 2 runs); appearance scoring runs with NVIDIA_TF32_OVERRIDE=0 (pxscore.py).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
a = argparse.ArgumentParser()
for k in ("pixcell", "work", "root", "ids", "d"):
    a.add_argument(f"--{k}", required=True)
a.add_argument("--family", required=True, choices=["P", "R"])
a.add_argument("--ck", default=None, help="checkpoint folder (Colab): P -> CK/<variant>/step_0002600 (+ CK/a1_per_channel/config.py); R -> CK/<tr_run>/step_*")
A = a.parse_args()
P, W, ROOT, IDS, D = (Path(getattr(A, k)).resolve() for k in ("pixcell", "work", "root", "ids", "d"))
PY = sys.executable
REL = {  # build_cache.py INFERENCE_VARIANTS ckpt_dir per variant
    "a1_concat": "checkpoints/concat_95470_0/checkpoints/step_0002600",
    "production": "checkpoints/production_retrain_post_fix/full_seed_42/checkpoint/step_0002600",
    "a1_per_channel": "checkpoints/debug/per_channel_95470_3/checkpoints/step_0002600",
    "a2_bypass_full_tme": "checkpoints/a2_a3/a2_bypass/full_seed_42/checkpoint/step_0002600",
}
if A.family == "P":
    CKD = Path(A.ck) if A.ck else D / "in/ck"
    CK = {v: CKD / v / "step_0002600" for v in REL}
    PERCH_CFG = CKD / "a1_per_channel/config.py"
    VARIANTS = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]
else:
    E = Path(A.ck) if A.ck else Path("/gscratch/cheme/chiu/hfv_eval_20261118/ck/infer_e20")
    run = {"a1_concat": "tr_concat", "production": "tr_grouped", "a1_per_channel": "tr_perch", "a2_bypass_full_tme": "tr_bypass"}
    CK = {}
    for v, r in run.items():
        s = sorted((E / r).glob("step_*"))
        if len(s) != 1:
            sys.exit(f"STOP {r}: {len(s)} step folders")
        CK[v] = s[0]
    PERCH_CFG = P / "configs/config_controlnet_exp_a1_per_channel.py"
    VARIANTS = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel"]
for v, c in CK.items():
    if not list(c.glob("controlnet_epoch_*_step_*.pth")) or not (c / "tme_module.pth").is_file():
        sys.exit(f"STOP checkpoint {v}: {c} lacks controlnet_epoch_*_step_*.pth or tme_module.pth")


def link(dst, src):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


S = W / "shadow"
for e in P.iterdir():
    if e.name not in ("data", "checkpoints", "inference_output"):
        link(S / e.name, e)
link(S / "data/orion-crc33", ROOT)
for v, rel in REL.items():
    link(S / rel, CK[v])
pc = S / "checkpoints/debug/per_channel_95470_3"
for f in sorted((P / "configs").glob("*.py")):   # mmcv resolves _base_ next to config.py (evalgen.py note)
    link(pc / f.name, f)
link(pc / "config.py", PERCH_CFG)
G = W / "out"
G.mkdir(parents=True, exist_ok=True)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
(G / "gpu.txt").write_text(gpu + "\n")
print(f"[pp_gen] GPU {gpu}; family {A.family}; root {ROOT}; checkpoints {json.dumps({k: str(v) for k, v in CK.items()})}",
      flush=True)


def run(cmd, tf32_off):
    env = {k: v for k, v in os.environ.items() if k != "NVIDIA_TF32_OVERRIDE"}
    if tf32_off:
        env["NVIDIA_TF32_OVERRIDE"] = "0"
    # scoring only: + the hfv_score pylib (lpips 0.1.4, pandas; score_hyak.slurm), kept off the generation path
    extra = [os.environ["PP_SCORE_PYLIB"]] if tf32_off else []
    env["PYTHONPATH"] = os.pathsep.join([str(S)] + extra + [x for x in [os.environ.get("PYTHONPATH")] if x])
    t0 = time.time()
    print(f"[pp_gen] $ {' '.join(map(str, cmd))} (TF32 override {'0' if tf32_off else 'unset'})", flush=True)
    rc = subprocess.run([str(c) for c in cmd], cwd=S, env=env).returncode
    print(f"[pp_gen] rc {rc} {time.time() - t0:.0f} s", flush=True)
    return rc, time.time() - t0


timing = {}
rc, timing["generate_s"] = run([PY, "-u", HERE / "pxrun.py", "tools/ablation_a1_a2/build_cache.py", "--cache-dir", G,
                                "--tile-ids-file", IDS, "--tile-id-cache-key", "metric_tile_ids", "--variants", *VARIANTS,
                                "--device", "cuda"], tf32_off=False)
ids = [t.strip() for t in open(IDS) if t.strip()]
n = {v: sum((G / "tiles" / v / f"{t}.png").is_file() for t in ids) for v in VARIANTS}
print(f"[pp_gen] images per variant {n}", flush=True)
if rc != 0 or any(x != len(ids) for x in n.values()):
    json.dump(dict(rc=rc, n=n, timing=timing), open(G / "FAILED", "w"))
    sys.exit("STOP generation")
# per-variant steady rate from file times (first image and model loading excluded)
for v in VARIANTS:
    ts = sorted((G / "tiles" / v / f"{t}.png").stat().st_mtime for t in ids)
    timing[f"{v}_img_per_min"] = round((len(ts) - 1) / ((ts[-1] - ts[0]) / 60), 2)
rc2, timing["appearance_s"] = run([PY, "-u", HERE / "pxscore.py", HERE / "pp_appear.py", "--cache-dir", G, "--orion-root", ROOT,
                                   "--ids", IDS, "--variants", *VARIANTS, "--device", "cuda"], tf32_off=True)
json.dump(dict(rc_gen=rc, rc_appear=rc2, n=n, timing=timing, gpu=gpu), open(G / ("DONE" if rc2 == 0 else "FAILED"), "w"), indent=1)
sys.exit(0 if rc2 == 0 else 1)
