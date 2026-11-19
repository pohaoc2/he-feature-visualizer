"""hfv_retrain_20261118 evaluation, generation side (Colab L4 or Hyak A100; scoring runs later on Hyak, README rules).

  python evalgen.py --pixcell P --work W --ck CK --roots R TASK [TASK ...]
P    PixCell cb62fa7 (unmodified), with pretrained_models/{pixcell-256, pixcell-256-controlnet, sd-3.5-vae, uni-2h}
W    work folder (shadow root, views, outputs under W/out/<group>)
CK   folder with the inference checkpoints: CK/<run>/step_*/{controlnet_epoch_*_step_*.pth, tme_module.pth}
     (run = tr_concat, tr_grouped, tr_bypass, tr_perch; eval/slimck.py)
R    folder with the extracted evaluation packs (R/orion-crc36ho, R/orion-crc33sec, R/orion-crc34eval) and their lists
TASK (group = the comparison group = one output folder, generated on one GPU type):
  speed:ROOT:IDS         throwaway speed probe: Concat, first 16 tiles at 30 steps (build_cache, 1 image per tile) and
                         first 8 tiles x 15 conditions at 20 steps (ablation; its per-tile cell_mask.png is not an
                         image and is not counted). Per command: wall seconds (model loading included), seconds to the
                         first finished tile, steady images per minute from the finished-tile times (first tile and
                         model loading excluded: (tiles - 1) x images per tile / (last - first finish)), and the
                         loading overhead = first finish - one steady tile interval. Writes
                         W/out/speed_<ROOT>_<gpu>.json; the images are not used
  fig2:ROOT:IDS          Fig 2 group: Concat, Grouped, Grouped + bypass, Per-channel and Vanilla ControlNet
                         (build_cache.py, 30 steps, guidance 1.5, seed 42), then the group-drop renders of the
                         conditioned variants (sensitivity_eval.py --mode render; LPIPS scoring on Hyak)
  abl:ROOT:IDS:MAP       15-condition Concat ablation, paired then unpaired with the MAP derangement
                         (generate_ablation_subset_cache.py, 20 steps, guidance 2.5, seed 42)
  loo:ROOT:IDS           Fig 4 sub-channel drops, baseline regenerated in the same run (run_subchannel_loo.py)
  decomp:ROOT:IDS        Fig 3 UNI x TME decomposition, four modes (src.a2_decomposition.main --worker generate) on a
                         view root holding only IDS (the script takes the first sample-n tiles of the root)
  crc34:ROOT             CRC34 frozen inference (run_evaluation.py --n-tiles 1000 --seed 42 --no-ablation --resume)
IDS / MAP are file names inside R. Paths the PixCell scripts hard-code (build_cache.py INFERENCE_VARIANTS checkpoint
folders and per-channel config, data/orion-crc33 in build_cache.py / sensitivity_eval.py) are served by a shadow root
of symlinks (W/shadow): every PixCell entry, plus data/orion-crc33 -> the group's data root and the hard-coded
checkpoint folders -> CK. No PixCell file is changed.
Vanilla ControlNet: tools/baselines/pixcell_offshelf_inference.py draws its starting noise with torch.randn_like and no
seed; here torch.manual_seed(42) is set before each tile (pxrun.py) so the baseline is reproducible (README deviation).
Each group records its GPU in W/out/<group>/gpu.txt and stops when resumed on another GPU type.
Every PixCell command runs with cwd = the shadow root and PYTHONPATH = the shadow root (the scripts import tools.* and
src.* without adding the repo to sys.path). View roots link each channel file under its own extension (the packs keep
vasculature / oxygen / glucose as .npy, the cell channels as .png). Exit code 1 when any task has rc != 0 or stops.
EVALGEN_GPU (optional) names the device when nvidia-smi is absent (dry tests only).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
a = argparse.ArgumentParser()
a.add_argument("--pixcell", required=True)
a.add_argument("--work", required=True)
a.add_argument("--ck", required=True)
a.add_argument("--roots", required=True)
a.add_argument("tasks", nargs="+")
A = a.parse_args()
P, W, CK, R = (Path(x).resolve() for x in (A.pixcell, A.work, A.ck, A.roots))
PY = sys.executable
CH = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
      "cell_state_nonprolif", "cell_state_dead", "vasculature", "oxygen", "glucose"]
try:
    GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.strip().splitlines()[0]
except (FileNotFoundError, IndexError):
    GPU = os.environ.get("EVALGEN_GPU") or sys.exit("STOP no GPU found (nvidia-smi) and EVALGEN_GPU not set")
print(f"[evalgen] GPU {GPU}", flush=True)


def step_dir(run):
    s = sorted((CK / run).glob("step_*"))
    if len(s) != 1 or not list(s[0].glob("controlnet_epoch_*_step_*.pth")) or not (s[0] / "tme_module.pth").is_file():
        sys.exit(f"STOP checkpoint {run}: expected one step_* folder with controlnet + tme_module under {CK / run}")
    return s[0]


HARD = {  # build_cache.py INFERENCE_VARIANTS ckpt_dir -> run
    "checkpoints/concat_95470_0/checkpoints/step_0002600": "tr_concat",
    "checkpoints/production_retrain_post_fix/full_seed_42/checkpoint/step_0002600": "tr_grouped",
    "checkpoints/debug/per_channel_95470_3/checkpoints/step_0002600": "tr_perch",
    "checkpoints/a2_a3/a2_bypass/full_seed_42/checkpoint/step_0002600": "tr_bypass",
}


def link(dst, src):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def shadow(data_root, runs):
    S = W / "shadow"
    S.mkdir(parents=True, exist_ok=True)
    for e in P.iterdir():
        if e.name not in ("data", "checkpoints", "inference_output"):
            if not (S / e.name).is_symlink():
                link(S / e.name, e)
    link(S / "data" / "orion-crc33", data_root)
    for rel, run in HARD.items():
        if run in runs:
            link(S / rel, step_dir(run))
    link(S / "checkpoints/debug/per_channel_95470_3/config.py", P / "configs/config_controlnet_exp_a1_per_channel.py")
    return S


def ids_of(name):
    return [t.strip() for t in open(R / name) if t.strip()]


def view(root, name):
    """data root holding only the tiles of R/name (symlinks to the pack)"""
    src = R / root
    V = W / "views" / f"{root}__{Path(name).stem}"
    if (V / "VIEW_OK_v2").is_file():
        return V
    if V.exists():
        shutil.rmtree(V)
    for t in ids_of(name):
        for c in CH:
            f = sorted((src / "exp_channels" / c).glob(f"{t}.*"))
            if len(f) != 1:
                sys.exit(f"STOP view {root}: {len(f)} files for {c}/{t}")
            link(V / "exp_channels" / c / f[0].name, f[0])
        files = [src / "features" / f"{t}_uni.npy", src / "he" / f"{t}.png"] + \
                [src / "vae_features" / f"{t}_{s}.npy" for s in ("sd3_vae", "mask_sd3_vae")]
        for f in files:
            if not f.is_file():
                sys.exit(f"STOP view {root}: missing {f}")
            link(V / f.relative_to(src), f)
    bad = [p for p in V.rglob("*") if p.is_symlink() and not p.exists()]
    if bad:
        sys.exit(f"STOP view {root}: {len(bad)} dangling links, e.g. {bad[0]}")
    (V / "VIEW_OK_v2").write_text("")
    return V


def group_dir(group):
    G = W / "out" / group
    G.mkdir(parents=True, exist_ok=True)
    g = G / "gpu.txt"
    if g.is_file() and g.read_text().strip() != GPU:
        print(f"STOP group {group} was started on {g.read_text().strip()}, this GPU is {GPU}", flush=True)
        return None
    g.write_text(GPU + "\n")
    return G


def run(cmd, cwd, log):
    t0 = time.time()
    print(f"[evalgen] $ {' '.join(map(str, cmd))}", flush=True)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(cwd)] + [x for x in [os.environ.get("PYTHONPATH")] if x]))
    with open(log, "a") as f:
        rc = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"[evalgen] rc {rc} {time.time() - t0:.0f} s (log {log})", flush=True)
    return rc


def pngs(d):
    return sorted(Path(d).rglob("*.png"), key=lambda p: p.stat().st_mtime)


def speed_stats(tiles, t_start, t_end):
    """tiles: {tile: [image files]}; finish time of a tile = its newest image"""
    fin = sorted(max(f.stat().st_mtime for f in fs) for fs in tiles.values() if fs)
    n_img = sum(len(fs) for fs in tiles.values())
    out = dict(tiles=len(fin), images=n_img, wall_s=round(t_end - t_start, 1))
    if len(fin) >= 2:
        per_tile = n_img / len(fin)
        step = (fin[-1] - fin[0]) / (len(fin) - 1)
        out.update(first_tile_s=round(fin[0] - t_start, 1), steady_s_per_tile=round(step, 2),
                   steady_images_per_min=round(per_tile * 60 / step, 2), load_overhead_s=round(fin[0] - t_start - step, 1))
    return out


def rate(files):
    if len(files) < 2:
        return None
    t = [f.stat().st_mtime for f in files]
    return round((len(files) - 1) / ((t[-1] - t[0]) / 60), 2)


ALL4 = ["tr_concat", "tr_grouped", "tr_bypass", "tr_perch"]
CONCAT = "configs/config_controlnet_exp_a1_concat.py"
summary = []
for task in A.tasks:
    kind, *args = task.split(":")
    root = args[0]
    group = f"{kind}_{root.replace('orion-', '')}"
    if kind == "speed":
        group += "_" + GPU.replace(" ", "_")
    G = group_dir(group)
    if G is None:
        summary.append((task, "STOP gpu"))
        continue
    log = G / "gen.log"
    t0 = time.time()
    rc = 0
    for stale in ("DONE", "FAILED"):
        (G / stale).unlink(missing_ok=True)
    if kind == "speed":   # throwaway: start clean
        for d in ("s30", "s20"):
            shutil.rmtree(G / d, ignore_errors=True)
        ids = ids_of(args[1])
        (G / "ids16.txt").write_text("".join(f"{t}\n" for t in ids[:16]))
        (G / "ids8.txt").write_text("".join(f"{t}\n" for t in ids[:8]))
        V = view(root, args[1])
        S = shadow(V, ["tr_concat"])
        a = time.time()
        rc |= run([PY, "-u", "tools/ablation_a1_a2/build_cache.py", "--cache-dir", G / "s30", "--tile-ids-file", G / "ids16.txt",
                   "--tile-id-cache-key", "metric_tile_ids", "--variants", "a1_concat", "--device", "cuda"], S, log)
        b = time.time()
        rc |= run([PY, "-u", "tools/stage3/generate_ablation_subset_cache.py", "--config", CONCAT, "--checkpoint-dir",
                   step_dir("tr_concat"), "--data-root", V, "--tile-list", G / "ids8.txt", "--output-dir", G / "s20"], S, log)
        c = time.time()
        t30 = {f.stem: [f] for f in (G / "s30/tiles/a1_concat").glob("*.png")}
        t20 = {d.name: [f for f in d.rglob("*.png") if f.name != "cell_mask.png"] for d in (G / "s20").iterdir() if d.is_dir()}
        r = dict(gpu=GPU, rc=rc, steps30=speed_stats(t30, a, b), steps20=speed_stats(t20, b, c))
        json.dump(r, open(W / "out" / f"speed_{root}_{GPU.replace(' ', '_')}.json", "w"), indent=1)
        print(f"[evalgen] speed {json.dumps(r)}", flush=True)
    elif kind == "fig2":
        V = view(root, args[1])
        S = shadow(V, ALL4)
        rc |= run([PY, "-u", HERE / "pxrun.py", "tools/ablation_a1_a2/build_cache.py", "--cache-dir", G, "--tile-ids-file",
                   R / args[1], "--tile-id-cache-key", "metric_tile_ids", "--variants", "a1_concat", "production",
                   "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf", "--device", "cuda"], S, log)
        rc |= run([PY, "-u", "tools/ablation_a1_a2/sensitivity_eval.py", "--mode", "render", "--cache-dir", G,
                   "--tile-ids-file", R / args[1], "--variants", "a1_concat", "production", "a2_bypass_full_tme",
                   "a1_per_channel", "--device", "cuda"], S, log)
    elif kind == "abl":
        S = shadow(R / root, ["tr_concat"])   # full pack root: unpaired style tiles need their UNI
        for mode, extra in (("paired", []), ("unpaired", ["--style-mapping-json", R / args[2]])):
            rc |= run([PY, "-u", "tools/stage3/generate_ablation_subset_cache.py", "--config", CONCAT, "--checkpoint-dir",
                       step_dir("tr_concat"), "--data-root", R / root, "--tile-list", R / args[1], "--output-dir",
                       G / f"{mode}_ablation" / "ablation_results", "--skip-existing"] + extra, S, log)
    elif kind == "loo":
        ids = ids_of(args[1])
        V = view(root, args[1])
        S = shadow(V, ["tr_concat"])
        rc |= run([PY, "-u", "tools/stage3/run_subchannel_loo.py", "--config-path", P / CONCAT, "--checkpoint-dir",
                   step_dir("tr_concat"), "--data-root", V, "--baseline-root", G / "no_cached_baseline", "--tile-list",
                   R / args[1], "--out-dir", G, "--n-tiles", len(ids), "--regenerate-baseline"], S, log)
    elif kind == "decomp":
        ids = ids_of(args[1])
        V = view(root, args[1])
        S = shadow(V, ["tr_concat"])
        base = [PY, "-u", "-m", "src.a2_decomposition.main", "--config-path", P / CONCAT, "--checkpoint-dir",
                step_dir("tr_concat"), "--data-root", V, "--out-dir", G, "--sample-n", len(ids)]
        rc |= run(base, S, log)
        rc |= run(base + ["--worker", "generate"], S, log)
    elif kind == "crc34":
        S = shadow(R / root, ["tr_concat"])
        rc |= run([PY, "-u", "tools/stage3/run_evaluation.py", "--config", CONCAT, "--checkpoint-dir", step_dir("tr_concat"),
                   "--data-root", R / root, "--output-dir", G, "--n-tiles", "1000", "--seed", "42", "--no-ablation",
                   "--resume"], S, log)
    else:
        sys.exit(f"STOP unknown task {task}")
    n = len(pngs(G))
    (G / "DONE" if rc == 0 else G / "FAILED").write_text(json.dumps(dict(task=task, gpu=GPU, rc=rc, pngs=n,
                                                                          seconds=round(time.time() - t0))) + "\n")
    summary.append((task, f"rc {rc} pngs {n} {time.time() - t0:.0f} s"))
    print(f"[evalgen] {task}: rc {rc}, {n} png, {time.time() - t0:.0f} s", flush=True)
print("[evalgen] summary", json.dumps(summary), flush=True)
sys.exit(0 if all(r.startswith("rc 0 ") for _, r in summary) else 1)
