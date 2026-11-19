#!/bin/bash
# hfv_retrain_20261118 evaluation, Colab side (generation only; scoring on Hyak). Not run directly: presign_eval.py
# prepends LINKS (one-day presigned links, never committed), TASKS and TAG, uploads boot_eval_<TAG>.sh, and the notebook
# runs it with one curl line. No AWS credentials in Colab: every input comes through a GET link, every output goes back
# through a PUT link.
#   setup (once per runtime, /content/HFV_EVAL_SETUP_OK): the training environment of colab/boot.sh (uv venv Python 3.12,
#         torch 2.10.0+cu128, torchvision 0.25.0, xformers 0.0.35, req.txt), PixCell cb62fa7, the four pretrained tars
#   and the evaluation packs and inference checkpoints the TASKS need (all downloads at once, large files as
#   parallel ranged GETs with size + ETag check), then eval/evalgen.py TASKS (one GPU type per
#   comparison group). After each task its group folder goes back as gen_<TAG>_<k>.tar (k = task number, 1-based);
#   the log is pushed every 2 min to log_<TAG>.
set -u
export MPLBACKEND=Agg
W=/content/hfve; mkdir -p $W; LOG=$W/colab_$TAG.log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s)
printf '%s' "$LINKS" > $W/links.json; chmod 600 $W/links.json
PY=$W/env/bin/python
st() { echo "### $(( $(date +%s) - T0 )) s  $*"; python3 $W/kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1 || true; }
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
get() { python3 $W/kit/s3io.py get $W/links.json "$1" "$2"; }
mkdir -p $W/kit; getf evalkit $W/evalkit.tar.gz && tar -xzf $W/evalkit.tar.gz -C $W/kit && ls $W/kit
( while sleep 120; do echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)" >> $LOG
    python3 $W/kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1; done ) & LOGPID=$!
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /content | tail -1; free -g | sed -n 2p; nproc
st start
# every download of this boot starts here at once (s3io.py get: files > 200 MB as parallel ranged GETs, each checked
# against the S3 size and ETag); pip runs while they download
mkdir -p $W/dl $W/pt $W/roots $W/ck
export S3_RANGES=8   # ranges per large file; about a dozen files download at once
DLP=""; SEEN=" "
dl() { case "$SEEN" in *" $1 "*) return ;; esac; SEEN="$SEEN$1 "; ( get $1 $2 || echo "DOWNLOAD_FAILED $1" ) & DLP="$DLP $!"; }
SETUP=0; [ -f /content/HFV_EVAL_SETUP_OK ] || SETUP=1
[ $SETUP = 1 ] && { getf req $W/req.txt || { st "STOP req"; kill $LOGPID; exit 1; }; }
[ $SETUP = 1 ] && for n in code pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do dl $n $W/dl/$n.bin; done
for t in $TASKS; do IFS=: read -r kind root rest <<< "$t"
  [ -f $W/roots/$root.ok ] || dl evalpack_${root#orion-} $W/dl/pack_${root#orion-}.tar
  for f in ${rest//:/ }; do [ -f $W/roots/$f ] || dl list_$f $W/roots/$f; done
  runs=tr_concat; case $kind in fig2) runs="tr_concat tr_grouped tr_bypass tr_perch" ;; esac
  for r in $runs; do [ -d $W/ck/$r ] || dl infer_$r $W/dl/ck_$r.tar; done
done
if [ $SETUP = 1 ]; then
  pip -q install uv >/dev/null 2>&1
  [ -x $PY ] || uv venv -q --python 3.12 $W/env; $PY --version
  uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  echo "setuptools==75.2.0" > $W/bc.txt
  uv pip install -q --python $PY --build-constraints $W/bc.txt xformers==0.0.35 -r $W/req.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || { echo "STOP: pip failed"; st STOP; kill $LOGPID; exit 1; }
  $PY -c "import torch,mmcv,diffusers,timm,xformers;print('env',torch.__version__,torch.cuda.get_device_name(),mmcv.__version__,diffusers.__version__,xformers.__version__)"
  st pip
fi
[ -n "$DLP" ] && wait $DLP; ls -la $W/dl   # never a bare wait: it would also wait for the log loop
grep -q DOWNLOAD_FAILED $LOG && { echo "STOP: a download failed"; st STOP; kill $LOGPID; exit 1; }
st downloads
if [ $SETUP = 1 ]; then
  mkdir -p $W/PixCell/pretrained_models; tar -xzf $W/dl/code.bin -C $W/PixCell
  for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do tar -xf $W/dl/$d.bin -C $W/pt; rm $W/dl/$d.bin
    p=$(find $W/pt -maxdepth 5 -type d -name $d | head -1); ln -sfn $p $W/PixCell/pretrained_models/$d; done
  touch /content/HFV_EVAL_SETUP_OK; st setup
fi
for t in $TASKS; do IFS=: read -r kind root rest <<< "$t"
  if [ ! -f $W/roots/$root.ok ]; then tar -xf $W/dl/pack_${root#orion-}.tar -C $W/roots && rm $W/dl/pack_${root#orion-}.tar* \
      && touch $W/roots/$root.ok || { st "STOP pack $root"; kill $LOGPID; exit 1; }; fi
  runs=tr_concat; case $kind in fig2) runs="tr_concat tr_grouped tr_bypass tr_perch" ;; esac
  for r in $runs; do [ -d $W/ck/$r ] || { tar -xf $W/dl/ck_$r.tar -C $W/ck && rm $W/dl/ck_$r.tar*; } \
      || { st "STOP checkpoint $r"; kill $LOGPID; exit 1; }; done
done
ls $W/roots $W/ck/*; st inputs
k=0
for t in $TASKS; do k=$((k + 1))
  $PY -u $W/kit/evalgen.py --pixcell $W/PixCell --work $W --ck $W/ck --roots $W/roots "$t"; rc=$?
  IFS=: read -r kind root rest <<< "$t"; g=${kind}_${root#orion-}; [ $kind = speed ] && g=$(ls -d $W/out/speed_${root#orion-}_*/ | tail -1 | xargs basename)   # directory only, not the .json
  up=1; tar -cf $W/gen_${TAG}_$k.tar -C $W/out $g $(cd $W/out && ls speed_*.json 2>/dev/null) && \
    python3 $W/kit/s3io.py put $W/links.json gen_${TAG}_$k $W/gen_${TAG}_$k.tar && { up=0; rm $W/gen_${TAG}_$k.tar; }
  [ $rc -eq 0 ] && [ $up -eq 0 ] || NFAIL=$((${NFAIL:-0} + 1))
  st "task $k $t rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
st "DONE failed tasks ${NFAIL:-0}"; kill $LOGPID
