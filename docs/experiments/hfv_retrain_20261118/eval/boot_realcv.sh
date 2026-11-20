#!/bin/bash
# hfv_retrain_20261118: CellViT-256 on the REAL H&E of the evaluation sets (coordinator 2026-11-20, README deviation
# "PQ 參考遮罩"), the paper-comparable PQ reference. Same environment and CellViT as eval/boot_score.sh (runner md5
# 7e67da08, CellViT 05097e1, CellViT-256.pth md5 b3634bd5, NVIDIA_TF32_OVERRIDE=0, --batch-size 32 as evalscore.py
# cellvit()); setup block copied from boot_score.sh so later scoring boots in the same runtime find the same setup.
# Not run directly: presign_realcv.py prepends LINKS, ITEMS (pack names) and TAG.
#   per ITEM (= pack name): realhe_<pack>.zip (inputs/eval/, the pack's he/<id>.png, ids unchanged) -> runner -> one
#   JSON per tile -> realcv_<pack> (S3 outputs/realcv_<pack>.tar, folder realcv_<pack>/<id>.json); STOP-style failure
#   count when the JSON count differs from the PNG count.
set -u
export MPLBACKEND=Agg NVIDIA_TF32_OVERRIDE=0
W=/content/hfve; mkdir -p $W; LOG=$W/colab_$TAG.log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s)
printf '%s' "$LINKS" > $W/links.json; chmod 600 $W/links.json
PY=$W/env/bin/python
st() { echo "### $(( $(date +%s) - T0 )) s  $*"; python3 $W/skit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1 || true; }
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
get() { python3 $W/skit/s3io.py get $W/links.json "$1" "$2"; }
put() { python3 $W/skit/s3io.py put $W/links.json "$1" "$2"; }
mkdir -p $W/skit; getf scorekit $W/scorekit.tar.gz && tar -xzf $W/scorekit.tar.gz -C $W/skit && ls $W/skit || { echo "STOP scorekit"; exit 1; }
( while sleep 120; do echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)" >> $LOG
    python3 $W/skit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1; done ) & LOGPID=$!
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h /content | tail -1; free -g | sed -n 2p; nproc
st start
mkdir -p $W/dl $W/pt $W/roots $W/sgen $W/score
export S3_RANGES=8
DLP=""; SEEN=" "
dl() { case "$SEEN" in *" $1 "*) return ;; esac; SEEN="$SEEN$1 "; ( get $1 $2 || echo "DOWNLOAD_FAILED $1" ) & DLP="$DLP $!"; }
SETUP=0; [ -f /content/HFV_EVAL_SETUP_OK ] || SETUP=1
SCSET=0; [ -f /content/HFV_SCORE_SETUP_OK ] || SCSET=1
[ $SETUP = 1 ] && { getf req $W/req.txt || { st "STOP req"; kill $LOGPID; exit 1; }; dl code $W/dl/code.bin; dl uni-2h $W/dl/uni-2h.bin; }
[ $SCSET = 1 ] && dl cellvit_kit $W/dl/cv.tar
for it in $ITEMS; do dl realhe_$it $W/dl/realhe_$it.zip; done
if [ $SETUP = 1 ]; then
  pip -q install uv >/dev/null 2>&1
  [ -x $PY ] || uv venv -q --python 3.12 $W/env; $PY --version
  uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  echo "setuptools==75.2.0" > $W/bc.txt
  uv pip install -q --python $PY --build-constraints $W/bc.txt xformers==0.0.35 -r $W/req.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || { st "STOP: pip failed"; kill $LOGPID; exit 1; }
  st pip
fi
if [ $SCSET = 1 ]; then
  command -v uv >/dev/null || pip -q install uv >/dev/null 2>&1
  uv pip install -q --python $PY scikit-image==0.26.0 numba==0.68.0 llvmlite==0.50.0 numpy==2.0.2 || { st "STOP: cellvit deps"; kill $LOGPID; exit 1; }
  uv pip install -q --python $PY --no-deps lpips==0.1.4 pandas==2.2.3 python-dateutil==2.9.0.post0 pytz==2024.2 tzdata==2024.2 six==1.16.0 \
    || { st "STOP: scoring packages"; kill $LOGPID; exit 1; }
fi
$PY -c "import torch,torchvision,numpy,scipy,skimage,numba,lpips,pandas,timm;print('env',torch.__version__,torchvision.__version__,numpy.__version__,scipy.__version__,skimage.__version__,numba.__version__,pandas.__version__,timm.__version__,torch.cuda.get_device_name())" \
  || { st "STOP: env import"; kill $LOGPID; exit 1; }
[ -n "$DLP" ] && wait $DLP; ls -la $W/dl
grep -q DOWNLOAD_FAILED $LOG && { st "STOP: a download failed"; kill $LOGPID; exit 1; }
st downloads
if [ $SETUP = 1 ]; then
  mkdir -p $W/PixCell/pretrained_models; tar -xzf $W/dl/code.bin -C $W/PixCell && tar -xf $W/dl/uni-2h.bin -C $W/pt && rm $W/dl/code.bin $W/dl/uni-2h.bin \
    || { st "STOP PixCell / uni-2h"; kill $LOGPID; exit 1; }
  p=$(find $W/pt -maxdepth 5 -type d -name uni-2h | head -1); ln -sfn $p $W/PixCell/pretrained_models/uni-2h
fi
[ -f $W/PixCell/pretrained_models/uni-2h/config.json ] || { st "STOP uni-2h config.json missing"; kill $LOGPID; exit 1; }
K=$W/cellvit
if [ $SCSET = 1 ]; then
  mkdir -p $K && tar -xf $W/dl/cv.tar -C $K && rm $W/dl/cv.tar || { st "STOP cellvit kit"; kill $LOGPID; exit 1; }
  mkdir -p $W/torch_home
  TORCH_HOME=$W/torch_home $PY -c "import torchvision; torchvision.models.alexnet(weights='IMAGENET1K_V1')" >/dev/null || { st "STOP alexnet"; kill $LOGPID; exit 1; }
  a=$(find $W/torch_home -name 'alexnet-owt-*.pth' | head -1); sha256sum $a
  sha256sum $a | grep -q '^7be5be79' || { st "STOP alexnet sha256"; kill $LOGPID; exit 1; }
  touch /content/HFV_SCORE_SETUP_OK
fi
md5sum $K/hfv/stages/run_cellvit_local.py $K/ckpt/CellViT-256.pth
md5sum $K/hfv/stages/run_cellvit_local.py | grep -q '^7e67da08' && md5sum $K/ckpt/CellViT-256.pth | grep -q '^b3634bd5' \
  || { st "STOP CellViT runner / checkpoint md5"; kill $LOGPID; exit 1; }
st setup
export TORCH_HOME=$W/torch_home HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache OMP_NUM_THREADS=$(nproc)
mkdir -p $W/realcv; cd $W/realcv
for it in $ITEMS; do
  n=$(python3 -c "import zipfile,sys; print(sum(n.endswith('.png') for n in zipfile.ZipFile(sys.argv[1]).namelist()))" $W/dl/realhe_$it.zip)
  rm -rf $W/realcv/realcv_$it
  PYTHONPATH=$K/CellViT $PY -u $K/hfv/stages/run_cellvit_local.py --zip $W/dl/realhe_$it.zip --out $W/realcv/realcv_$it \
    --checkpoint $K/ckpt/CellViT-256.pth --cellvit-repo $K/CellViT --batch-size 32 --no-skip-existing > $W/realcv/realcv_$it.log 2>&1; rc=$?
  m=$(ls $W/realcv/realcv_$it | grep -c '\.json$'); echo "$it: png $n json $m rc $rc"; tail -3 $W/realcv/realcv_$it.log
  cp $W/realcv/realcv_$it.log $W/realcv/realcv_$it/run.log; nvidia-smi --query-gpu=name --format=csv,noheader > $W/realcv/realcv_$it/gpu.txt
  up=1; tar -cf $W/realcv_$it.tar -C $W/realcv realcv_$it && put realcv_$it $W/realcv_$it.tar && { up=0; rm -f $W/realcv_$it.tar; }
  [ $rc -eq 0 ] && [ "$n" = "$m" ] && [ $up -eq 0 ] || NFAIL=$((${NFAIL:-0} + 1))
  st "task $it png $n json $m rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
st "DONE failed tasks ${NFAIL:-0}"; kill $LOGPID
