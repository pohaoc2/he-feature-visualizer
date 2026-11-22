"""hfv_retrain2 stage 3 (PREREG §9.3, arm ND): copy of sim-exp-validation scripts/analysis/sim2real_apps/app8_gen_k.py
(app 8 / 9 / 10 generation path, unchanged) with two edits only: the row count comes from the design file (app8_gen_k.py
requires exactly 1152 rows) and each row names its reference tensor (ref_key) instead of the app-8 root -> __k / __e
rule. Design file: KIN/design.json. Everything else (models, scheduler, guidance 2.5, 20 steps, seeds, mask latent from
ctrl, gates, outputs) is app8_gen_k.py's.

  python r2_gen.py PIXCELL CK KIN OUT
PIXCELL  PixCell cb62fa7 checkout with pretrained_models/{pixcell-256, pixcell-256-controlnet, sd-3.5-vae}
CK       folder holding tr_concat/step_*/{controlnet_epoch_*_step_*.pth, tme_module.pth} (infer_tr_concat_e20.tar)
KIN      app8_stage.py kin (roots/, ref.npz, uni.npz, design_k.json)
OUT      new folder: generated/<id>.png, rows.json (design row + image + image_sha256), status.json

The path is the one hfv's evaluation uses (run_evaluation.run_one -> tile_pipeline.generate_tile): read_config +
load_all_models(config, config_path, step_dir, "cuda"), make_inference_scheduler(num_steps=20), per row
ctrl = load_exp_channels(tile, config.data.active_channels, config.image_size, <root>/exp_channels) and
generate_from_ctrl(ctrl, uni_embeds = the style vector as (1, 1, 1, 1536) float32, guidance_scale 2.5, seed); the mask
latent is computed inside it from ctrl (encode_ctrl_mask_latent), as in every hfv Concat generation. Only the "all
channels" condition is made (no ablation groups), seeds 42 and 43.
Gates (STOP before any image): one step_* with both files; config = Concat (all_channels, 10 active channels in the
CELL + MX order); for every tile and root, torch.equal(load_exp_channels(...), ref.npz <tile>__k / __e); 1152 design rows,
ids unique. After: 1152 PNGs.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

PIX, CK, KIN, OUT = (Path(p) for p in sys.argv[1:5])
CFG = "configs/config_controlnet_exp_a1_concat.py"
ORDER = ["cell_masks", "cell_type_healthy", "cell_type_cancer", "cell_type_immune", "cell_state_prolif",
         "cell_state_nonprolif", "cell_state_dead", "vasculature", "oxygen", "glucose"]
T0 = time.time()


def stop(m):
    print(f"STOP {m}", flush=True)
    sys.exit(1)


if OUT.exists():
    stop(f"{OUT} exists")
steps = sorted((CK / "tr_concat").glob("step_*"))
if len(steps) != 1 or len(list(steps[0].glob("controlnet_epoch_*_step_*.pth"))) != 1 \
        or not (steps[0] / "tme_module.pth").is_file():
    stop(f"checkpoint layout under {CK}")
os.chdir(PIX)
sys.path.insert(0, str(PIX))
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from diffusion.utils.misc import read_config  # noqa: E402
from tools.stage3.common import make_inference_scheduler  # noqa: E402
from tools.stage3.tile_pipeline import generate_from_ctrl, load_all_models, load_exp_channels  # noqa: E402

config = read_config(CFG)
config._filename = CFG
if getattr(config, "tme_input_mode", None) != "all_channels" or list(config.data.active_channels) != ORDER:
    stop(f"config {getattr(config, 'tme_input_mode', None)} {list(config.data.active_channels)}")
design = json.loads((KIN / "design.json").read_text())
rows = design["rows"]
NR = int(design["n_rows"])
if len(rows) != NR or len({r["id"] for r in rows}) != NR:
    stop("design rows")
ref = np.load(KIN / "ref.npz")
uni = np.load(KIN / "uni.npz")
ctrl = {}
for r in rows:
    key = (r["root"], r["tile"])
    if key in ctrl:
        continue
    c = load_exp_channels(r["tile"], config.data.active_channels, config.image_size,
                          KIN / "roots" / r["root"] / "exp_channels")
    want = torch.from_numpy(ref[r["ref_key"]])
    if c.dtype != want.dtype or not torch.equal(c, want):
        stop(f"loaded channels != ref for {key}: max diff {float((c - want).abs().max())}")
    ctrl[key] = c
print(f"gate: {len(ctrl)} (root, tile) controls load to ref exactly", flush=True)
gpu = torch.cuda.get_device_name()
models = load_all_models(config, CFG, steps[0], "cuda")
sched = make_inference_scheduler(num_steps=20, device="cuda")
(OUT / "generated").mkdir(parents=True)
out_rows = []
for i, r in enumerate(rows):
    u = torch.from_numpy(np.asarray(uni[r["reference"]], np.float32)).view(1, 1, 1, 1536)
    img, _ = generate_from_ctrl(ctrl[(r["root"], r["tile"])], models=models, config=config, scheduler=sched,
                                uni_embeds=u, device="cuda", guidance_scale=2.5, seed=int(r["seed"]))
    f = OUT / "generated" / f"{r['id']}.png"
    Image.fromarray(img).save(f)
    out_rows.append(r | dict(image=f"generated/{r['id']}.png",
                             image_sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    if (i + 1) % 96 == 0:
        print(f"[{time.time() - T0:6.0f}s] {i + 1}/{len(rows)}", flush=True)
(OUT / "rows.json").write_text(json.dumps(out_rows, indent=1) + "\n")
n = len(list((OUT / "generated").glob("*.png")))
st = dict(state="complete" if n == NR else "incomplete", images=n, gpu=gpu, seconds=time.time() - T0,
          checkpoint=str(steps[0]), config=CFG, torch=torch.__version__)
(OUT / "status.json").write_text(json.dumps(st, indent=1) + "\n")
print(st, flush=True)
if n != NR:
    stop(f"{n} images")
