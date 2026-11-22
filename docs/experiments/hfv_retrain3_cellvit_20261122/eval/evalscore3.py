"""hfv_retrain3_cellvit_20261122 scoring (PREREG §5.2, §5.5): copy of hfv_retrain_20261118 eval/evalscore.py with two
changes only: (1) CellViT runs the official post-processing (--postprocess official --magnification 40, as cv3.slurm and
cellvit_official_20261122); (2) the device of the PixCell scoring commands comes from EVALSCORE_DEVICE (default cuda;
cpu on Hyak cheme, where the CellViT of the real H&E ran). Everything else unchanged.

Original header (hfv_retrain_20261118 evaluation, scoring side (Hyak A100, TF32 off; README rule 5)):

  python evalscore.py --pixcell P --work W TASK [TASK ...]
P     PixCell cb62fa7 (unmodified) with pretrained_models/uni-2h
W     work folder (shadow root, CellViT batches); each TASK scores a generated group folder in place
TASK  fig2:GROUP:DATA:IDS   si_a1_a2-style cache (build_cache.py + sensitivity_eval.py --mode render output)
      abl:GROUP:DATA:MAP    15-condition ablation (GROUP/{paired,unpaired}_ablation/ablation_results; MAP = the unpaired
                            style mapping JSON)
      loo:GROUP:DATA        sub-channel LOO (run_subchannel_loo.py output)
      decomp:GROUP:DATA     UNI x TME decomposition (GROUP/generated/<tile>/<mode>.png)
      crc34:GROUP:DATA      CRC34 frozen inference (run_evaluation.py output), run_crc34_metrics.sh steps
DATA  the evaluation pack root the group was generated from (real H&E, stored real UNI, cell_masks)
Every step is a PixCell cb62fa7 script run unchanged, in the order of its own documented pipeline:
  fig2   metrics_io.py export-cellvit -> CellViT -> metrics_io.py import-cellvit -> metrics_io.py compute (FUD, LPIPS,
         PQ, DICE, HED; tools/ablation_a1_a2/metrics_io.py) -> sensitivity_eval.py --mode score (group-drop LPIPS)
         -> eval/fig2tiles.py (the same per-tile metrics kept per tile, for the text-definition ΔLPIPS and its CI)
  abl    per mode: tools/cellvit/export_batch.py --zip -> CellViT -> tools/cellvit/import_results.py ->
         tools/compute_ablation_metrics.py --metrics lpips aji pq dice iou accuracy style_hed -> tools/compute_fid.py
         (README_stage3.md "Ablation + Metrics Workflow"; run_crc34_metrics.sh)
  loo    aggregate_subchannel_loo -> export_subchannel_loo_cellvit --zip -> CellViT -> import_results ->
         aggregate_subchannel_layout_metrics (docs/analysis/fix_vasculature_c34_c10_runbook.md)
  decomp a2_decomposition.metrics (manifests) -> export_batch --zip -> CellViT -> import_results ->
         compute_ablation_metrics --metrics lpips pq dice style_hed -> a2_decomposition.metrics --compute-fud ->
         --summarize (docs/handovers/HANDOVER.md)
  crc34  per mode: build_all_only_cache -> export_batch --zip -> CellViT -> import_results -> compute_ablation_metrics
         --metrics lpips pq dice style_hed -> compute_fid --only-all -> aggregate_all_only_metrics
CellViT = he-feature-visualizer stages/run_cellvit_local.py (CellViT-256; env CV_PY, CV_REPO, CV_CKPT, CV_RUNNER), the
runner of the paper and of the CRC34 gate. PixCell commands run through eval/pxscore.py (UNI-2h transform from
uni-2h/config.json, README "UNI-2h 前處理"). Every command runs with NVIDIA_TF32_OVERRIDE=0; the GPU name goes into
GROUP/score_gpu.txt and the log. Paths PixCell hard-codes (data/orion-crc33 in sensitivity_eval.py, the per-channel
config path of build_cache.INFERENCE_VARIANTS) come from a shadow root of symlinks (W/shadow_<group>), as in evalgen.py.
Any rc != 0 or a CellViT output count that differs from the exported PNG count stops that task (GROUP/SCORE_FAILED);
exit 1 when any task stopped.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
# helper scripts (fig2tiles.py) beside this file; jobs whose slurm copied only evalscore.py + pxscore.py into node-local
# /tmp read them from the deployed scripts folder
AUX = HERE if (HERE / "fig2tiles.py").is_file() else Path("/gscratch/cheme/chiu/hfv_score_20261119/scripts")
a = argparse.ArgumentParser()
a.add_argument("--pixcell", required=True)
a.add_argument("--work", required=True)
a.add_argument("tasks", nargs="+")
A = a.parse_args()
P, W = Path(A.pixcell).resolve(), Path(A.work).resolve()
PY = sys.executable
CV = {k: os.environ[k] for k in ("CV_PY", "CV_REPO", "CV_CKPT", "CV_RUNNER")}
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
try:
    _q = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    GPU = _q.stdout.strip().splitlines()[0] if _q.returncode == 0 else ""
except (FileNotFoundError, IndexError):
    GPU = ""
if not GPU:   # CPU dry tests only
    GPU = os.environ.get("EVALSCORE_GPU") or sys.exit("STOP no GPU found (nvidia-smi) and EVALSCORE_GPU not set")
print(f"[evalscore] GPU {GPU}; NVIDIA_TF32_OVERRIDE=0", flush=True)
DEV = os.environ.get("EVALSCORE_DEVICE", "cuda")
V4 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel"]
V5 = V4 + ["a2_off_shelf"]


class Stop(Exception):
    pass


def shadow(name, data_root):
    S = W / f"shadow_{name}"
    S.mkdir(parents=True, exist_ok=True)

    def link(dst, src):
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(src)

    for e in P.iterdir():
        if e.name not in ("data", "checkpoints", "inference_output"):
            link(S / e.name, e)
    link(S / "data" / "orion-crc33", Path(data_root).resolve())
    pc = S / "checkpoints/debug/per_channel_95470_3"   # same per-channel config links as evalgen.py shadow()
    for f in sorted((P / "configs").glob("*.py")):
        link(pc / f.name, f)
    link(pc / "config.py", P / "configs/config_controlnet_exp_a1_per_channel.py")
    return S


def run(cmd, cwd, log, env_extra=None):
    t0 = time.time()
    cmd = [str(c) for c in cmd]
    print(f"[evalscore] $ {' '.join(cmd)}", flush=True)
    env = dict(os.environ, NVIDIA_TF32_OVERRIDE="0")
    env["PYTHONPATH"] = os.pathsep.join([str(cwd)] + [x for x in [os.environ.get("PYTHONPATH")] if x])
    env.update(env_extra or {})
    with open(log, "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%S')} GPU {GPU} $ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[evalscore] rc {rc} {time.time() - t0:.0f} s", flush=True)
    if rc != 0:
        raise Stop(f"rc {rc}: {' '.join(cmd[:4])}")


def px(S, log, *args):
    run([PY, "-u", HERE / "pxscore.py", *args], S, log)


def cellvit(zip_path, out, log, n_expect):
    run([CV["CV_PY"], "-u", CV["CV_RUNNER"], "--zip", zip_path, "--out", out, "--checkpoint", CV["CV_CKPT"],
         "--cellvit-repo", CV["CV_REPO"], "--batch-size", "32", "--no-skip-existing", "--postprocess", "official",
         "--magnification", "40"], W, log,
        {"PYTHONPATH": CV["CV_REPO"]})
    n = len(list(Path(out).glob("*.json")))
    print(f"[evalscore] CellViT {n} JSON for {n_expect} PNG", flush=True)
    if n != n_expect:
        raise Stop(f"CellViT wrote {n} JSON for {n_expect} PNG ({zip_path})")


def zip_dir(d, z):
    files = sorted(Path(d).glob("*.png"))
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:
        for f in files:
            zf.write(f, f.name)
    return len(files)


def zip_count(z):
    with zipfile.ZipFile(z) as zf:
        return sum(n.endswith(".png") for n in zf.namelist())


def cv_tree(S, log, root, batch, exporter="tools/cellvit/export_batch.py"):
    """export a cache tree (zip), CellViT, import the sidecars beside the generated PNGs"""
    px(S, log, exporter, "--cache-root", root, "--output-dir", batch, "--overwrite", "--zip")
    res = Path(str(batch) + "_res")
    cellvit(Path(str(batch) + ".zip"), res, log, zip_count(Path(str(batch) + ".zip")))
    px(S, log, "tools/cellvit/import_results.py", "--manifest", Path(batch) / "manifest.json", "--results-dir", res,
       "--prefer-ext", ".json", "--sidecar-suffix", "_cellvit_instances")


bad = []
for task in A.tasks:
    kind, gdir, data, *rest = task.split(":")
    G, D = Path(gdir).resolve(), Path(data).resolve()
    log = G / "score.log"
    (G / "SCORE_FAILED").unlink(missing_ok=True)
    (G / "SCORE_DONE").unlink(missing_ok=True)
    (G / "score_gpu.txt").write_text(GPU + "\n")
    S = shadow(G.name, D)
    B = W / "cv" / G.name
    B.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        if kind == "fig2":
            ids = Path(rest[0]).resolve()
            px(S, log, "tools/ablation_a1_a2/metrics_io.py", "export-cellvit", "--cache-dir", G, "--out-dir", B / "fig2",
               "--mode", "hardlink", "--variants", *V5)
            n = zip_dir(B / "fig2" / "images", B / "fig2.zip")
            cellvit(B / "fig2.zip", B / "fig2_res", log, n)
            px(S, log, "tools/ablation_a1_a2/metrics_io.py", "import-cellvit", "--manifest", B / "fig2" / "manifest.json",
               "--results-dir", B / "fig2_res")
            px(S, log, "tools/ablation_a1_a2/metrics_io.py", "compute", "--cache-dir", G, "--orion-root", D,
               "--device", DEV, "--uni-model", P / "pretrained_models/uni-2h", "--variants", *V5)
            px(S, log, "tools/ablation_a1_a2/sensitivity_eval.py", "--mode", "score", "--cache-dir", G,
               "--tile-ids-file", ids, "--device", DEV, "--variants", *V5)
            px(S, log, AUX / "fig2tiles.py", "--cache-dir", G, "--orion-root", D, "--device", DEV)
        elif kind == "abl":
            for mode in ("paired", "unpaired"):
                R = G / f"{mode}_ablation" / "ablation_results"
                sty = ["--style-mapping-json", Path(rest[0]).resolve()] if mode == "unpaired" else []
                cv_tree(S, log, R, B / mode)
                px(S, log, "tools/compute_ablation_metrics.py", "--cache-dir", R, "--orion-root", D, "--metrics", "lpips",
                   "aji", "pq", "dice", "iou", "accuracy", "style_hed", "--device", DEV, "--jobs", "4", *sty)
                px(S, log, "tools/compute_fid.py", "--cache-dir", R, "--orion-root", D, "--device", DEV,
                   "--feature-backend", "uni", *sty)
        elif kind == "loo":
            px(S, log, "-m", "tools.stage3.aggregate_subchannel_loo", "--out-dir", G)
            cv_tree(S, log, G, B / "loo", exporter="tools/stage3/export_subchannel_loo_cellvit.py")
            px(S, log, "-m", "tools.stage3.aggregate_subchannel_layout_metrics", "--out-dir", G, "--orion-root", D)
        elif kind == "decomp":
            M = G / "decomposition_metrics"
            px(S, log, "-m", "src.a2_decomposition.metrics", "--generated-root", G / "generated", "--out-dir", M)
            cv_tree(S, log, M, B / "decomp")
            px(S, log, "tools/compute_ablation_metrics.py", "--cache-dir", M, "--orion-root", D, "--metrics", "lpips",
               "pq", "dice", "style_hed", "--device", DEV, "--jobs", "4")
            px(S, log, "-m", "src.a2_decomposition.metrics", "--compute-fud", "--metrics-root", M, "--orion-root", D,
               "--uni-model", P / "pretrained_models/uni-2h", "--device", DEV, "--fud-json", G / "fud_scores.json")
            px(S, log, "-m", "src.a2_decomposition.metrics", "--summarize", "--metrics-root", M, "--fud-json",
               G / "fud_scores.json", "--out-csv", G / "decomposition_summary.csv", "--representative-json",
               G / "representative_tile.json")
        elif kind == "crc34":
            for mode in ("paired", "unpaired"):
                C = G / f"{mode}_cache"
                px(S, log, "tools/stage3/build_all_only_cache.py", "--run-output", G, "--mode", mode, "--data-root", D,
                   "--out", C)
                sty = ["--style-mapping-json", C / "style_mapping.json"] if mode == "unpaired" else []
                cv_tree(S, log, C, B / mode)
                px(S, log, "tools/compute_ablation_metrics.py", "--cache-dir", C, "--orion-root", D, "--metrics", "lpips",
                   "pq", "dice", "style_hed", "--device", DEV, *sty)
                px(S, log, "tools/compute_fid.py", "--cache-dir", C, "--orion-root", D, "--feature-backend", "uni",
                   "--only-all", *sty, "--output", C / "fud_scores.json")
                px(S, log, "tools/stage3/aggregate_all_only_metrics.py", "--cache-dir", C, "--label", f"CRC34 {mode}")
        else:
            raise Stop(f"unknown task kind {kind}")
        (G / "SCORE_DONE").write_text(json.dumps(dict(task=task, gpu=GPU, seconds=round(time.time() - t0))) + "\n")
        print(f"[evalscore] {task}: done {time.time() - t0:.0f} s", flush=True)
    except Stop as e:
        (G / "SCORE_FAILED").write_text(json.dumps(dict(task=task, gpu=GPU, error=str(e))) + "\n")
        print(f"[evalscore] STOP {task}: {e}", flush=True)
        bad.append(task)
print(f"[evalscore] summary: {len(A.tasks) - len(bad)} done, stopped {bad}", flush=True)
sys.exit(1 if bad else 0)
