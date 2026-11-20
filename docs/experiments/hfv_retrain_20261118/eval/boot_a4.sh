#!/bin/bash
# hfv_retrain_20261118 follow-up 1 (coordinator 2026-11-20): Fig 3 a4 UNI-probe sweep with the retrained Concat on Colab L4,
# one tile shard per session (2 sessions, hfv_eval_L4_a / hfv_eval_L4_b), generation + CellViT in the session
# (eval/a4run.py). Not run directly: presign_a4.py prepends LINKS (one-day presigned links, never committed), SHARD,
# NSHARD and TAG, uploads boot_a4_<TAG>.sh, and the notebook runs it with one curl line. No AWS credentials in Colab.
#   environment: the one of boot_eval.sh (uv venv Python 3.12, torch 2.10.0+cu128, torchvision 0.25.0, xformers 0.0.35,
#   req.txt) plus the CellViT extras of boot_score.sh (scikit-image 0.26.0, numba 0.68.0, llvmlite 0.50.0,
#   numpy 2.0.2), installed only when the runtime does not already have them; PixCell cb62fa7, the four pretrained tars
#   and the CellViT kit (runner md5 7e67da08, CellViT-256.pth md5 b3634bd5) only when missing. Inputs: evalpack_crc33a4,
#   a4_inputs.tar, infer_tr_concat (outputs/infer_tr_concat_e20.tar). Output: the group folder of this shard ->
#   outputs/a4gen_crc33a4_shard<K>.tar; log every 2 min -> outputs/log_<TAG>.
set -u
export MPLBACKEND=Agg
W=/content/hfve; mkdir -p $W; LOG=$W/colab_$TAG.log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s)
printf '%s' "$LINKS" > $W/links.json; chmod 600 $W/links.json
PY=$W/env/bin/python
st() { echo "### $(( $(date +%s) - T0 )) s  $*"; python3 $W/a4kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1 || true; }
getf() { python3 - "$1" "$2" <<'PYG'
import json, sys, urllib.request
u = json.load(open("/content/hfve/links.json"))["get"][sys.argv[1]]
with urllib.request.urlopen(u) as r, open(sys.argv[2], "wb") as f:
    while True:
        b = r.read(1 << 24)
        if not b: break
        f.write(b)
PYG
}
get() { python3 $W/a4kit/s3io.py get $W/links.json "$1" "$2"; }
put() { python3 $W/a4kit/s3io.py put $W/links.json "$1" "$2"; }
mkdir -p $W/a4kit; getf a4kit $W/a4kit.tar.gz && tar -xzf $W/a4kit.tar.gz -C $W/a4kit && ls $W/a4kit || { echo "STOP a4kit"; exit 1; }
( while sleep 120; do echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)" >> $LOG
    python3 $W/a4kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1; done ) & LOGPID=$!
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h /content | tail -1; free -g | sed -n 2p; nproc
st "start shard $SHARD of $NSHARD"
mkdir -p $W/dl $W/pt $W/roots $W/ck
export S3_RANGES=8
DLP=""
dl() { ( get $1 $2 || echo "DOWNLOAD_FAILED $1" ) & DLP="$DLP $!"; }
ENVOK=0; [ -x $PY ] && $PY -c "import torch,torchvision,mmcv,diffusers,xformers,timm,skimage,numba" >/dev/null 2>&1 && ENVOK=1
[ $ENVOK = 1 ] || { getf req $W/req.txt || { st "STOP req"; kill $LOGPID; exit 1; }; }
[ -d $W/PixCell/train_scripts ] || dl code $W/dl/code.bin
for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do [ -e $W/PixCell/pretrained_models/$d ] || dl $d $W/dl/$d.bin; done
[ -f $W/cellvit/ckpt/CellViT-256.pth ] || dl cellvit_kit $W/dl/cv.tar
[ -f $W/roots/orion-crc33a4.ok ] || dl evalpack_crc33a4 $W/dl/pack_crc33a4.tar
[ -d $W/ck/tr_concat ] || dl infer_tr_concat $W/dl/ck_tr_concat.tar
rm -rf $W/a4in; dl a4_inputs $W/dl/a4_inputs.tar
if [ $ENVOK = 0 ]; then
  pip -q install uv >/dev/null 2>&1
  [ -x $PY ] || uv venv -q --python 3.12 $W/env; $PY --version
  uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  echo "setuptools==75.2.0" > $W/bc.txt
  uv pip install -q --python $PY --build-constraints $W/bc.txt xformers==0.0.35 -r $W/req.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || { st "STOP: pip failed"; kill $LOGPID; exit 1; }
  uv pip install -q --python $PY scikit-image==0.26.0 numba==0.68.0 llvmlite==0.50.0 numpy==2.0.2 || { st "STOP: cellvit deps"; kill $LOGPID; exit 1; }
  st pip
fi
$PY -c "import torch,torchvision,numpy,scipy,skimage,numba,mmcv,diffusers,xformers,timm;print('env',torch.__version__,torchvision.__version__,numpy.__version__,scipy.__version__,skimage.__version__,numba.__version__,mmcv.__version__,diffusers.__version__,xformers.__version__,timm.__version__,torch.cuda.get_device_name())" \
  || { st "STOP: env import"; kill $LOGPID; exit 1; }
[ -n "$DLP" ] && wait $DLP; ls -la $W/dl
grep -q DOWNLOAD_FAILED $LOG && { st "STOP: a download failed"; kill $LOGPID; exit 1; }
st downloads
if [ -f $W/dl/code.bin ]; then mkdir -p $W/PixCell/pretrained_models && tar -xzf $W/dl/code.bin -C $W/PixCell && rm $W/dl/code.bin || { st "STOP PixCell"; kill $LOGPID; exit 1; }; fi
for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do
  if [ -f $W/dl/$d.bin ]; then tar -xf $W/dl/$d.bin -C $W/pt && rm $W/dl/$d.bin || { st "STOP $d"; kill $LOGPID; exit 1; }
    p=$(find $W/pt -maxdepth 5 -type d -name $d | head -1); ln -sfn $p $W/PixCell/pretrained_models/$d; fi
  [ -e $W/PixCell/pretrained_models/$d ] || { st "STOP pretrained $d missing"; kill $LOGPID; exit 1; }
done
K=$W/cellvit
if [ -f $W/dl/cv.tar ]; then mkdir -p $K && tar -xf $W/dl/cv.tar -C $K && rm $W/dl/cv.tar || { st "STOP cellvit kit"; kill $LOGPID; exit 1; }; fi
md5sum $K/hfv/stages/run_cellvit_local.py $K/ckpt/CellViT-256.pth
md5sum $K/hfv/stages/run_cellvit_local.py | grep -q '^7e67da08' && md5sum $K/ckpt/CellViT-256.pth | grep -q '^b3634bd5' \
  || { st "STOP CellViT runner / checkpoint md5"; kill $LOGPID; exit 1; }
if [ ! -f $W/roots/orion-crc33a4.ok ]; then tar -xf $W/dl/pack_crc33a4.tar -C $W/roots && rm $W/dl/pack_crc33a4.tar* \
    && touch $W/roots/orion-crc33a4.ok || { st "STOP pack"; kill $LOGPID; exit 1; }; fi
[ -d $W/ck/tr_concat ] || { tar -xf $W/dl/ck_tr_concat.tar -C $W/ck && rm $W/dl/ck_tr_concat.tar*; } || { st "STOP checkpoint"; kill $LOGPID; exit 1; }
mkdir -p $W/a4in && tar -xf $W/dl/a4_inputs.tar -C $W/a4in && rm $W/dl/a4_inputs.tar || { st "STOP a4 inputs"; kill $LOGPID; exit 1; }
ls $W/roots $W/ck/tr_concat $W/a4in/a4_inputs; st setup
export CV_PY=$PY CV_REPO=$K/CellViT CV_CKPT=$K/ckpt/CellViT-256.pth CV_RUNNER=$K/hfv/stages/run_cellvit_local.py
export HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache OMP_NUM_THREADS=$(nproc)
rm -rf $W/out/a4_crc33a4
$PY -u $W/a4kit/a4run.py --pixcell $W/PixCell --ck $W/ck --root $W/roots/orion-crc33a4 --inputs $W/a4in/a4_inputs \
  --work $W --shard $SHARD --nshard $NSHARD; rc=$?
k=$((SHARD + 1)); up=1
tar -cf $W/a4gen_$k.tar -C $W/out a4_crc33a4 && put a4gen_$k $W/a4gen_$k.tar && { up=0; rm -f $W/a4gen_$k.tar; }
st "DONE shard $k rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"; kill $LOGPID
