"""hfv_retrain_20261118 evaluation: inference copy of one final checkpoint (no change to any weight).

  python slimck.py META_JSON CN_PTH TME_PTH OUT_DIR RUN
The training checkpoint (train_wrap.py slot ckpt_<run>_e20_*) holds state_dict, state_dict_ema, optimizer, scheduler
and epoch (about 10 GB). Inference (tools/stage3/tile_pipeline.py load_all_models -> load_controlnet_weights_flexible ->
_extract_state_dict) reads only "state_dict", and tme_module.pth["model_state"]. Writes
OUT_DIR/RUN/<step_dir>/<cn name> = {"state_dict", "epoch"} and tme_module.pth copied byte for byte; reloads the copy and
STOPs unless every tensor is identical (torch.equal) to the original. The full checkpoint stays on S3.
"""
import json
import os
import shutil
import sys

import torch

meta_p, cn_p, tme_p, out, run = sys.argv[1:6]
meta = json.load(open(meta_p))
d = os.path.join(out, run, meta["step_dir"])
os.makedirs(d, exist_ok=True)
c = torch.load(cn_p, map_location="cpu", mmap=True, weights_only=False)
sd = c["state_dict"]
dst = os.path.join(d, meta["cn"])
torch.save({"state_dict": sd, "epoch": c.get("epoch")}, dst + ".part")
os.replace(dst + ".part", dst)
shutil.copyfile(tme_p, os.path.join(d, "tme_module.pth"))
r = torch.load(dst, map_location="cpu", weights_only=False)["state_dict"]
bad = [k for k in sd if k not in r or not torch.equal(sd[k], r[k])] + [k for k in r if k not in sd]
if bad:
    print(f"STOP {run}: {len(bad)} tensors differ, e.g. {bad[:3]}")
    sys.exit(1)
print(json.dumps(dict(run=run, step_dir=meta["step_dir"], cn=meta["cn"], epoch=c.get("epoch"), tensors=len(sd),
                      params=int(sum(v.numel() for v in sd.values())), bytes=os.path.getsize(dst), identical=True)))
