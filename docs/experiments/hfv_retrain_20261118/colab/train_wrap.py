"""hfv_retrain_20261118: run PixCell stage-2 training (repo unmodified, cb62fa7) with three wrapper-level additions.

  python train_wrap.py --pixcell P --config configs/config_controlnet_exp_a1_concat.py --data-root R --work-dir W
                       [--micro-bs 20] [--max-steps 50] [--probe-json out.json] [--resume auto] [--save-every-epoch]
                       [--seed 42] [--num-epochs 20] [--num-workers 8]

1. Effective batch stays at the paper's 80: train_batch_size = micro, gradient_accumulation_steps = 80 / micro
   (accelerate divides the loss by the accumulation count, so the gradient equals one bs-80 step; the TME encoders use
   GroupNorm / LayerNorm only, no BatchNorm, so a smaller micro-batch does not change the forward pass).
   The repo calls ema_update once per micro-batch; with accumulation that would apply the 0.9999 decay `accum` times per
   optimizer step, so here EMA is applied only on the micro-batch that closes an optimizer step (as at bs 80).
   (Inference reads `state_dict`, not the EMA weights - train_scripts/mapping_weights_helper.py - so this only keeps the
   saved EMA comparable.)
2. --max-steps N stops after N optimizer steps (speed probe); --probe-json writes samples/s over the steps after the
   first 10 (data-loader warm-up excluded) and the peak CUDA memory.
3. Resume: --save-every-epoch saves a checkpoint at every epoch end (repo save_model_epochs = 1), so a dropped Colab
   session resumes at an epoch boundary with skip_step 0. --resume auto picks the newest
   <work_dir>/checkpoints/step_*/controlnet_epoch_*_step_*.pth and its tme_module.pth (optimizer + LR schedule kept,
   reset_tme_optimizer = False on resume only). The repo reads the epoch from the file name minus one
   (initialize_models._resume_from_checkpoint: start_epoch = E - 1), which is right for a mid-epoch save but would
   repeat a whole epoch from an epoch-end save; so the resume path is an alias named epoch E + 1 (same step) in
   <work_dir>/resume/, and training continues with epoch E + 1. (The data order of the epochs after a resume differs
   from an uninterrupted run.)
4. --upload-every K (with --links, --run-name): after each epoch-end save only the newest checkpoint stays on the Colab
   disk; every K epochs and at the last epoch it is uploaded in a background thread (training goes on) to the presigned
   slots ckpt_<run>_e<NN>_cn (multipart), _tme, then _meta (a small JSON with the file names, sent last, so a meta
   object means a complete checkpoint). The next save waits for that upload before deleting the old directory.
   --fetch-resume: with no local checkpoint, the newest uploaded one (GET links of the same names) is downloaded first,
   so a new Colab runtime resumes where the last one stopped.
The total step count is the repo's: len(dataloader) x num_epochs optimizer steps (len counts micro-batches, so it is
divided by the accumulation count here to keep the same number of optimizer steps as bs 80).
"""
import argparse
import glob
import json
import os
import re
import sys
import time

EFFECTIVE_BS = 80
os.environ["MPLBACKEND"] = "Agg"   # Colab exports the notebook inline backend, which matplotlib rejects outside a kernel

ap = argparse.ArgumentParser()
ap.add_argument("--pixcell", required=True)
ap.add_argument("--config", required=True)
ap.add_argument("--data-root", required=True)
ap.add_argument("--work-dir", required=True)
ap.add_argument("--micro-bs", type=int, default=EFFECTIVE_BS)
ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--probe-json", default="")
ap.add_argument("--resume", default="")
ap.add_argument("--save-every-epoch", action="store_true")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--num-epochs", type=int, default=0)
ap.add_argument("--num-workers", type=int, default=0)
ap.add_argument("--upload-every", type=int, default=0)
ap.add_argument("--links", default="")
ap.add_argument("--run-name", default="")
ap.add_argument("--fetch-resume", action="store_true")
a = ap.parse_args()
assert EFFECTIVE_BS % a.micro_bs == 0, "micro-batch must divide 80"
accum = EFFECTIVE_BS // a.micro_bs

os.chdir(a.pixcell)
sys.path.insert(0, a.pixcell)
os.makedirs(a.work_dir, exist_ok=True)

resume_ckpt = None
HERE = os.path.dirname(os.path.abspath(__file__))
if a.fetch_resume and not glob.glob(f"{a.work_dir}/checkpoints/step_*/controlnet_epoch_*_step_*.pth"):
    sys.path.insert(0, HERE)
    import s3io  # noqa: E402
    L = json.load(open(a.links))
    for e in range(40, 0, -1):
        k = f"ckpt_{a.run_name}_e{e:02d}"
        if f"{k}_meta" not in L["get"]:
            continue
        try:
            s3io._req(L["get"][f"{k}_meta"], headers={"Range": "bytes=0-0"}, tries=1)
        except Exception:
            continue
        mp = f"{a.work_dir}/fetch_meta.json"
        s3io.get(L, f"{k}_meta", mp)
        meta = json.load(open(mp))
        d = f"{a.work_dir}/checkpoints/{meta['step_dir']}"
        os.makedirs(d, exist_ok=True)
        s3io.get(L, f"{k}_cn", f"{d}/{meta['cn']}.part")
        s3io.get(L, f"{k}_tme", f"{d}/tme_module.pth")
        os.rename(f"{d}/{meta['cn']}.part", f"{d}/{meta['cn']}")
        print(f"[train_wrap] fetched {k} -> {d}/{meta['cn']}", flush=True)
        break
if a.resume == "auto":
    c = sorted(glob.glob(f"{a.work_dir}/checkpoints/step_*/controlnet_epoch_*_step_*.pth"),
               key=lambda p: int(re.search(r"_step_(\d+)\.pth$", p).group(1)))
    resume_ckpt = c[-1] if c else None
elif a.resume:
    resume_ckpt = a.resume
resume_tme_dir = os.path.dirname(resume_ckpt) if resume_ckpt else None
if resume_ckpt and a.save_every_epoch:
    # epoch-end checkpoint: alias named epoch E + 1 so the repo starts at epoch E + 1 (see 3. above)
    E, S = map(int, re.search(r"controlnet_epoch_(\d+)_step_(\d+)\.pth$", resume_ckpt).groups())
    os.makedirs(f"{a.work_dir}/resume", exist_ok=True)
    alias = f"{a.work_dir}/resume/controlnet_epoch_{E + 1}_step_{S}.pth"
    if os.path.lexists(alias):
        os.remove(alias)
    os.symlink(os.path.abspath(resume_ckpt), alias)
    print(f"[train_wrap] resume from epoch {E} end (step {S}) via {alias}", flush=True)
    resume_ckpt = alias

# child config: the variant config + data root / batch / accumulation / checkpoint cadence (written next to the run)
child = os.path.join(a.work_dir, "config_wrap.py")
base = os.path.abspath(os.path.join(a.pixcell, a.config))
lines = [f"_base_ = [{base!r}]",
         f"exp_data_root = {os.path.abspath(a.data_root)!r}",
         f"train_batch_size = {a.micro_bs}",
         f"gradient_accumulation_steps = {accum}",
         f"root = {a.pixcell!r}",
         f"base_model_path = {a.pixcell + '/pretrained_models/pixcell-256/transformer'!r}",
         f"load_from = {a.pixcell + '/pretrained_models/pixcell-256/transformer'!r}",
         f"controlnet_load_from = {a.pixcell + '/pretrained_models/pixcell-256-controlnet/controlnet/diffusion_pytorch_model.safetensors'!r}",
         f"vae_pretrained = {a.pixcell + '/pretrained_models/sd-3.5-vae/vae'!r}"]
if a.num_epochs:
    lines.append(f"num_epochs = {a.num_epochs}")
if a.num_workers:
    lines.append(f"num_workers = {a.num_workers}")
if a.save_every_epoch:
    lines += ["save_model_epochs = 1", "save_model_steps = 10**9"]   # epoch boundaries only (resume with skip_step 0)
if a.max_steps:
    lines += ["save_final_checkpoint = False", "save_model_steps = 10**9", "log_interval = 10"]
if resume_ckpt:
    lines += [f"resume_tme_checkpoint = {resume_tme_dir!r}", "reset_tme_optimizer = False"]
open(child, "w").write("\n".join(lines) + "\n")

import torch  # noqa: E402
import train_scripts.train_controlnet_exp as T  # noqa: E402

_ema = T.ema_update
_calls = {"n": 0}
_t = []


def ema_sync_only(model_ema, model, rate):
    _calls["n"] += 1
    if _calls["n"] % accum == 0:
        _ema(model_ema, model, rate)
        _t.append(time.time())


T.ema_update = ema_sync_only
_setup = T.setup_training_state


def setup_wrapped(config, accelerator, logger, args, train_dataloader, *rest):
    out = _setup(config, accelerator, logger, args, train_dataloader, *rest)
    out["total_steps"] = -(-len(train_dataloader) // accum) * config.num_epochs
    if a.max_steps:
        out["total_steps"] = min(out["total_steps"], a.max_steps)
    logger.info(f"[train_wrap] micro_bs={a.micro_bs} accum={accum} total_steps={out['total_steps']} "
                f"resume={resume_ckpt}")
    return out


T.setup_training_state = setup_wrapped

_up = {"thread": None, "failed": []}
if a.upload_every:
    # see 4. above
    import shutil
    import threading
    sys.path.insert(0, HERE)
    import s3io  # noqa: E402
    _save = T.save_checkpoint_with_tme
    _links = json.load(open(a.links))

    def _upload(k, d, cn):
        try:
            s3io.put(_links, f"{k}_cn", cn)
            s3io.put(_links, f"{k}_tme", f"{d}/tme_module.pth")
            mp = f"{d}/meta.json"
            json.dump(dict(cn=os.path.basename(cn), step_dir=os.path.basename(d)), open(mp, "w"))
            s3io.put(_links, f"{k}_meta", mp)
            print(f"[train_wrap] uploaded {cn} as {k}", flush=True)
        except Exception as e:
            _up["failed"].append(k)
            print(f"[train_wrap] UPLOAD_FAILED {k}: {e}", flush=True)

    def _join():
        if _up["thread"] is not None:
            _up["thread"].join()
            _up["thread"] = None

    def save_and_sync(accelerator, controlnet, tme_module, model_ema, optimizer, optimizer_tme, lr_scheduler,
                      lr_scheduler_tme, step, epoch, config, logger):
        _save(accelerator, controlnet, tme_module, model_ema, optimizer, optimizer_tme, lr_scheduler,
              lr_scheduler_tme, step, epoch, config, logger)
        d = os.path.join(config.work_dir, "checkpoints", f"step_{step:07d}")
        _join()   # the previous upload reads the previous directory
        for old in glob.glob(os.path.join(config.work_dir, "checkpoints", "step_*")):
            if os.path.abspath(old) != os.path.abspath(d):
                shutil.rmtree(old)
        if epoch % a.upload_every == 0 or epoch == config.num_epochs:
            cn = glob.glob(f"{d}/controlnet_epoch_*_step_*.pth")[0]
            k = f"ckpt_{a.run_name}_e{epoch:02d}"
            _up["thread"] = threading.Thread(target=_upload, args=(k, d, cn), daemon=False)
            _up["thread"].start()
            logger.info(f"[train_wrap] uploading {cn} as {k} in the background")

    T.save_checkpoint_with_tme = save_and_sync
argv = ["stage2_train.py", child, "--work-dir", a.work_dir, "--seed", str(a.seed)]
if resume_ckpt:
    argv += ["--resume-from", resume_ckpt]
sys.argv = argv
t0 = time.time()
T.main()
if _up["thread"] is not None:
    _up["thread"].join()
if _up["failed"]:
    print(f"STOP uploads failed: {_up['failed']}", flush=True)
    sys.exit(1)
if a.probe_json:
    w = 10
    st = _t[w:]
    sps = (len(st) - 1) * EFFECTIVE_BS / (st[-1] - st[0]) if len(st) > 1 else None
    p = torch.cuda.get_device_properties(0)
    json.dump(dict(gpu=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 1), config=a.config, micro_bs=a.micro_bs,
                   accum=accum, optimizer_steps=len(_t), timed_steps=max(len(st) - 1, 0),
                   samples_per_s=None if sps is None else round(sps, 2),
                   sec_per_step=None if sps is None else round(EFFECTIVE_BS / sps, 3),
                   peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                   peak_reserved_gb=round(torch.cuda.max_memory_reserved() / 1e9, 2),
                   wall_s=round(time.time() - t0, 1), torch=torch.__version__, cuda=torch.version.cuda),
              open(a.probe_json, "w"), indent=1)
    print(open(a.probe_json).read())
