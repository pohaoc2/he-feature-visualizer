"""hfv_retrain_20261118 follow-up 1 (coordinator 2026-11-20): Fig 3 a4 UNI-probe sweep with the retrained Concat, one tile
shard per Colab L4 session (eval/boot_a4.sh), generation + CellViT in the same session.

  python a4run.py --pixcell P --ck CK --root R/orion-crc33a4 --inputs A4IN --work W --shard K --nshard N

What is run, unchanged from the paper's shared-tile run (inference_output_20260527.zip a1_concat/a4_uni_probe/
sweep_shards.sh): PixCell cb62fa7 src.a4_uni_probe.edit.run_sweep with --alphas -1 0 1, --fixed-tile-ids (the manifest,
here the 28 eligible tiles), --k-tiles 30, --num-steps 20, guidance 2.5, seed 42, --tile-shard-count N; features.npz /
labels.npz / probe_results.csv / probe_directions/ of that run (A4IN = S3 inputs/eval/a4_inputs.tar). Differences:
  * checkpoint = the retrained Concat (CK/tr_concat/step_*, epoch 20 inference copy), config =
    configs/config_controlnet_exp_a1_concat.py (as evalgen.py), data root = the crc33a4 pack (new cell channels);
  * the six swept attributes are pinned to the ones the paper swept (eccentricity_mean, nuclear_area_mean, nuclei_density,
    texture_e_contrast, texture_h_contrast, texture_h_energy = fig_combined_uni_decomp_v2.F_SWEEP_ATTRS). The zip's
    probe_results.csv was refit after that sweep and its appearance top 3 would now pick texture_e_homogeneity instead
    of texture_h_energy, so _select_sweep_attrs is replaced by the fixed list (coordinator approved); the directions are
    still read by run_sweep from probe_directions/ (cos 1.000000 with the paper's sweep/<attr>/w_targeted.npy) and the
    random directions come from its own seeded generator (equal to the paper's w_random.npy).
Then, as the paper's run_generated_cellvit_local.sh: postprocess export-cellvit-batch -> CellViT-256
(he-feature-visualizer stages/run_cellvit_local.py, env CV_*; NVIDIA_TF32_OVERRIDE=0) -> tools/cellvit/import_results.py
--prefer-ext .json --sidecar-suffix .png. refresh-metrics / specificity run on Hyak after the shards are merged.
cwd = P (PixCell reads pretrained_models/ relative to it); the sweep folder is P/a4out -> W/out/a4_crc33a4, so image
paths in metrics.csv are relative (a4out/a4_uni_probe/sweep/...), as the paper's were (inference_output/...).
A4RUN_GPU / A4RUN_DRY_STEPS: CPU plumbing tests only (see the code). The group records its GPU in gpu.txt and stops if resumed on another GPU type. Exit 1 on any failure.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ATTRS = ["eccentricity_mean", "nuclear_area_mean", "nuclei_density",
         "texture_e_contrast", "texture_h_contrast", "texture_h_energy"]
a = argparse.ArgumentParser()
for k in ("--pixcell", "--ck", "--root", "--inputs", "--work"):
    a.add_argument(k, required=True)
a.add_argument("--shard", type=int, required=True)
a.add_argument("--nshard", type=int, required=True)
A = a.parse_args()
P, CK, ROOT, IN, W = (Path(x).resolve() for x in (A.pixcell, A.ck, A.root, A.inputs, A.work))
PY = sys.executable
try:
    GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().splitlines()[0]
except (FileNotFoundError, IndexError):
    GPU = os.environ.get("A4RUN_GPU") or sys.exit("STOP no GPU found (nvidia-smi) and A4RUN_GPU not set")
DRY_STEPS = os.environ.get("A4RUN_DRY_STEPS")   # plumbing tests on a CPU node only (never set on Colab): fewer
                                                # denoising steps and no CellViT; outputs are thrown away
G = W / "out" / "a4_crc33a4"
O = G / "a4_uni_probe"
LOG = G / f"a4run_shard{A.shard + 1}of{A.nshard}.log"


def stop(msg):
    print(f"STOP {msg}", flush=True)
    sys.exit(1)


def run(cmd, env_extra=None):
    cmd = [str(c) for c in cmd]
    t0 = time.time()
    print(f"[a4run] $ {' '.join(cmd)}", flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(P)] + [x for x in [os.environ.get("PYTHONPATH")] if x])
    env.update(env_extra or {})
    with open(LOG, "a") as f:
        f.write(f"\n### {time.strftime('%Y-%m-%dT%H:%M:%S')} GPU {GPU} $ {' '.join(cmd)}\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=P, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[a4run] rc {rc} {time.time() - t0:.0f} s", flush=True)
    if rc != 0:
        stop(f"rc {rc}: {' '.join(cmd[:4])} (see {LOG})")


G.mkdir(parents=True, exist_ok=True)
g = G / "gpu.txt"
if g.is_file() and g.read_text().strip() != GPU:
    stop(f"group was started on {g.read_text().strip()}, this GPU is {GPU}")
g.write_text(GPU + "\n")
steps = sorted((CK / "tr_concat").glob("step_*"))
if len(steps) != 1 or not list(steps[0].glob("controlnet_epoch_*_step_*.pth")) or not (steps[0] / "tme_module.pth").is_file():
    stop(f"checkpoint: expected one step_* with controlnet + tme_module under {CK / 'tr_concat'}")
O.mkdir(parents=True, exist_ok=True)
for n in ("features.npz", "labels.npz", "probe_results.csv"):
    if not (O / n).is_file():
        shutil.copy2(IN / n, O / n)
if not (O / "probe_directions").is_dir():
    shutil.copytree(IN / "probe_directions", O / "probe_directions")
ids = [t.strip() for t in open(IN / "a4_ids28.txt") if t.strip()]
(G / "fixed_tiles.json").write_text(json.dumps({"tile_ids": ids, "source": "paper shared_tiles.json minus "
                                                "20992_46080, 26368_50688 (not eligible)"}, indent=1))
for t in ids:
    for c in ("cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
              "cell_state_nonprolif", "cell_state_dead", "vasculature", "oxygen", "glucose"):
        if len(list((ROOT / "exp_channels" / c).glob(f"{t}.*"))) != 1:
            stop(f"pack: {c}/{t} missing or duplicated")
link = P / "a4out"
if link.is_symlink() or link.exists():
    link.unlink()
link.symlink_to(G)

# ---- generation (run_sweep with the attribute list pinned)
drv = G / "a4_sweep_driver.py"
drv.write_text(f'''import argparse, sys
sys.path.insert(0, {str(P)!r})
import src.a4_uni_probe.edit as edit
ATTRS = {ATTRS!r}
edit._select_sweep_attrs = lambda probe_csv, top_k, attr_pool="morphology": list(ATTRS)
edit.run_sweep(argparse.Namespace(
    out_dir="a4out/a4_uni_probe", top_k_attrs=3, attr_pool="pinned",
    checkpoint_dir={str(steps[0])!r}, config_path={str(P / "configs/config_controlnet_exp_a1_concat.py")!r},
    data_root={str(ROOT)!r}, exp_channels_dir={str(ROOT / "exp_channels")!r},
    alphas=[-1.0, 0.0, 1.0], fixed_tile_ids={str(G / "fixed_tiles.json")!r}, k_tiles=30, seed=42,
    num_steps={int(DRY_STEPS) if DRY_STEPS else 20}, guidance_scale=2.5, tile_shard_index={A.shard}, tile_shard_count={A.nshard}))
print("sweep done")
''')
t_gen = time.time()
run([PY, "-u", drv])
pngs = sorted(O.glob("sweep/*/*/*/alpha_*.png"))
n_tiles = len(ids[len(ids) * A.shard // A.nshard: len(ids) * (A.shard + 1) // A.nshard])
expect = len(ATTRS) * n_tiles * 2 * 3
print(f"[a4run] generated {len(pngs)} PNG (expected {expect}) in {time.time() - t_gen:.0f} s on {GPU}", flush=True)
if len(pngs) != expect:
    stop(f"generated {len(pngs)} PNG, expected {expect}")
(G / f"speed_shard{A.shard + 1}.json").write_text(json.dumps(
    {"gpu": GPU, "images": len(pngs), "seconds_incl_model_load": round(time.time() - t_gen, 1)}))

# ---- CellViT on this shard's PNGs (paper: run_generated_cellvit_local.sh)
B = W / "cv" / f"a4_shard{A.shard + 1}"
if B.exists():
    shutil.rmtree(B)
B.parent.mkdir(parents=True, exist_ok=True)
run([PY, "-u", "-m", "src.a4_uni_probe.postprocess", "export-cellvit-batch", "--out-dir", "a4out/a4_uni_probe",
     "--batch-dir", B / "batch", "--overwrite", "--zip"])
with zipfile.ZipFile(str(B / "batch") + ".zip") as zf:
    n_zip = sum(n.endswith(".png") for n in zf.namelist())
if n_zip != expect:
    stop(f"CellViT batch has {n_zip} PNG, expected {expect}")
if DRY_STEPS:
    print("[a4run] dry test: CellViT skipped", flush=True)
    sys.exit(0)
CV = {k: os.environ[k] for k in ("CV_PY", "CV_REPO", "CV_CKPT", "CV_RUNNER")}
run([CV["CV_PY"], "-u", CV["CV_RUNNER"], "--zip", str(B / "batch") + ".zip", "--out", B / "res", "--checkpoint",
     CV["CV_CKPT"], "--cellvit-repo", CV["CV_REPO"], "--batch-size", "32", "--no-skip-existing"],
    {"PYTHONPATH": CV["CV_REPO"], "NVIDIA_TF32_OVERRIDE": "0"})
n_json = len(list((B / "res").glob("*.json")))
print(f"[a4run] CellViT {n_json} JSON for {n_zip} PNG", flush=True)
if n_json != n_zip:
    stop(f"CellViT wrote {n_json} JSON for {n_zip} PNG")
run([PY, "-u", "tools/cellvit/import_results.py", "--manifest", B / "batch" / "manifest.json", "--results-dir",
     B / "res", "--prefer-ext", ".json", "--sidecar-suffix", ".png"])
side = sorted(O.glob("sweep/*/*/*/alpha_*.png.json"))
print(f"[a4run] sidecars {len(side)} for {len(pngs)} PNG", flush=True)
if len(side) != len(pngs):
    stop(f"{len(side)} sidecars for {len(pngs)} PNG")
print("[a4run] shard done", flush=True)
