#!/bin/bash
# hfv_retrain_20261118 scoring, Colab L4 side (README deviation "評分改在 Colab L4", 2026-11-20). Same steps as
# eval/score_hyak.slurm: eval/evalscore.py runs PixCell cb62fa7's own scoring scripts unchanged (UNI-2h transform via
# eval/pxscore.py) and CellViT-256 (he-feature-visualizer stages/run_cellvit_local.py, md5 7e67da08, CellViT 05097e1,
# CellViT-256.pth md5 b3634bd5 = the Hyak copies) with NVIDIA_TF32_OVERRIDE=0. Not run directly: presign_score.py
# prepends LINKS (one-day presigned links, never committed), ITEMS and TAG, uploads boot_score_<TAG>.sh, and the notebook
# runs it with one curl line. No AWS credentials in Colab.
#   ITEM = KIND:SRC:GROUP:PACK[:LIST] as in score_hyak.slurm (SRC = generation object under S3 outputs/).
#   setup (once per runtime): the environment of boot_eval.sh (uv venv Python 3.12, torch 2.10.0+cu128, torchvision
#   0.25.0, xformers 0.0.35, req.txt) when /content/HFV_EVAL_SETUP_OK is absent; then (once, /content/HFV_SCORE_SETUP_OK)
#   the Hyak CellViT env's extra packages (scikit-image 0.26.0, numba 0.68.0, llvmlite 0.50.0, numpy 2.0.2) and the
#   Hyak scoring pylib versions (lpips 0.1.4, pandas 2.2.3, python-dateutil 2.9.0.post0, pytz 2024.2, tzdata 2024.2,
#   six 1.16.0; --no-deps), the CellViT kit, and the LPIPS AlexNet weights (sha256 must start with 7be5be79).
#   After each item: GROUP folder -> score_<GROUP> (= S3 outputs/score_<GROUP>.tar, the key figs_hyak.slurm reads),
#   small result files (json / csv / logs, the score_hyak.slurm filter) -> scoreres_<GROUP>; log every 2 min.
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
for it in $ITEMS; do IFS=: read -r kind src group pack list <<< "$it"
  [ -f $W/roots/orion-$pack.ok ] || dl evalpack_$pack $W/dl/pack_$pack.tar
  [ -z "$list" ] || [ -f $W/roots/$list ] || dl list_$list $W/roots/$list
  dl src_$src $W/dl/src_$src.tar
done
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
TASKS=""
for it in $ITEMS; do IFS=: read -r kind src group pack list <<< "$it"
  if [ ! -f $W/roots/orion-$pack.ok ]; then tar -xf $W/dl/pack_$pack.tar -C $W/roots && rm $W/dl/pack_$pack.tar* \
      && touch $W/roots/orion-$pack.ok || { st "STOP pack $pack"; kill $LOGPID; exit 1; }; fi
  rm -rf $W/sgen/$group; tar -xf $W/dl/src_$src.tar -C $W/sgen $group && rm $W/dl/src_$src.tar* || { st "STOP gen $src"; kill $LOGPID; exit 1; }
  echo "$group: $(find $W/sgen/$group -name '*.png' | wc -l) png"
  TASKS="$TASKS $kind:$W/sgen/$group:$W/roots/orion-$pack${list:+:$W/roots/$list}"
done
st inputs
export CV_PY=$PY CV_REPO=$K/CellViT CV_CKPT=$K/ckpt/CellViT-256.pth CV_RUNNER=$K/hfv/stages/run_cellvit_local.py
export TORCH_HOME=$W/torch_home HF_HOME=$W/cache/hf XDG_CACHE_HOME=$W/cache OMP_NUM_THREADS=$(nproc)
cd $W/score
for t in $TASKS; do
  $PY -u $W/skit/evalscore.py --pixcell $W/PixCell --work $W/score $t; rc=$?
  g=$(basename $(echo $t | cut -d: -f2))
  up=1
  (cd $W/sgen && find $g \( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.txt' -o -name 'SCORE_*' -o -name DONE \) \
     -not -name '*_cellvit_instances.json' -not -name 'manifest.json' -not -name 'subchannel_loo_diff_stats.json' -print0 \
     | tar -cf $W/scoreres_$g.tar --null -T - ) && put scoreres_$g $W/scoreres_$g.tar && \
  tar -cf $W/score_$g.tar -C $W/sgen $g && put score_$g $W/score_$g.tar && { up=0; rm -f $W/score_$g.tar; }
  [ $rc -eq 0 ] && [ $up -eq 0 ] || NFAIL=$((${NFAIL:-0} + 1))
  st "task $t rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
st "DONE failed tasks ${NFAIL:-0}"; kill $LOGPID
