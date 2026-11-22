#!/bin/bash
# paperpq_official_20261122 (PREREG.md §5), Colab L4 side: generation + appearance scoring of the roots in TASKS.
# Not run directly: presign_pp.py prepends LINKS (one-day presigned links, never committed), TASKS ("ROOT:FAMILY ...")
# and TAG, uploads boot_pp_<TAG>.sh, and the notebook cell runs it with one curl line. No AWS credentials in Colab.
# Setup = hfv_retrain eval/boot_eval.sh (uv venv Python 3.12, torch 2.10.0+cu128, torchvision 0.25.0, xformers 0.0.35,
# req.txt; PixCell cb62fa7; the four pretrained tars) + lpips 0.1.4 for scoring. Checkpoints: family P = the paper
# checkpoints (share_space/03_checkpoints, as build_cache.py INFERENCE_VARIANTS names them, + the per-channel config.py);
# family R = the retrain-1 epoch-20 inference tars. Each root's out/ folder goes back as gen_<ROOT>; log every 2 min.
set -u
export MPLBACKEND=Agg
W=/content/pp; mkdir -p $W; LOG=$W/colab_$TAG.log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s)
printf '%s' "$LINKS" > $W/links.json; chmod 600 $W/links.json
PY=$W/env/bin/python
getf() { python3 - "$1" "$2" <<'PYG'
import json, sys, urllib.request
u = json.load(open("/content/pp/links.json"))["get"][sys.argv[1]]
with urllib.request.urlopen(u) as r, open(sys.argv[2], "wb") as f:
    while True:
        b = r.read(1 << 24)
        if not b: break
        f.write(b)
PYG
}
mkdir -p $W/kit; getf kit $W/kit.tar.gz && tar -xzf $W/kit.tar.gz -C $W/kit && ls $W/kit
st() { echo "### $(( $(date +%s) - T0 )) s  $*"; python3 $W/kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1 || true; }
get() { python3 $W/kit/s3io.py get $W/links.json "$1" "$2"; }
( while sleep 120; do echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)" >> $LOG
    python3 $W/kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1; done ) & LOGPID=$!
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; df -h /content | tail -1; free -g | sed -n 2p; nproc
st start
mkdir -p $W/dl $W/pt $W/ck $W/empty
export S3_RANGES=8
DLP=""
dl() { ( get $1 $2 || echo "DOWNLOAD_FAILED $1" ) & DLP="$DLP $!"; }
getf req $W/req.txt || { st "STOP req"; kill $LOGPID; exit 1; }
for n in code roots pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do dl $n $W/dl/$n.bin; done
FAMS=$(for t in $TASKS; do echo ${t#*:}; done | sort -u)
for f in $FAMS; do
  if [ $f = P ]; then
    for v in production a1_concat a1_per_channel a2_bypass_full_tme; do mkdir -p $W/ck/$v/step_0002600
      dl ck_${v}_cn $W/ck/$v/step_0002600/controlnet_epoch_20_step_2600.pth; dl ck_${v}_tme $W/ck/$v/step_0002600/tme_module.pth; done
    dl ck_perch_cfg $W/ck/a1_per_channel/config.py
  else
    for r in tr_concat tr_grouped tr_bypass tr_perch; do dl infer_$r $W/dl/ck_$r.tar; done
  fi
done
pip -q install uv >/dev/null 2>&1
[ -x $PY ] || uv venv -q --python 3.12 $W/env; $PY --version
uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
echo "setuptools==75.2.0" > $W/bc.txt
uv pip install -q --python $PY --build-constraints $W/bc.txt xformers==0.0.35 -r $W/req.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || { st "STOP pip"; kill $LOGPID; exit 1; }
uv pip install -q --python $PY --no-deps lpips==0.1.4 || { st "STOP pip lpips"; kill $LOGPID; exit 1; }
$PY -c "import torch,mmcv,diffusers,timm,xformers,lpips,pandas;print('env',torch.__version__,torch.cuda.get_device_name(),mmcv.__version__,diffusers.__version__,xformers.__version__)"
st pip
[ -n "$DLP" ] && wait $DLP; ls -la $W/dl; du -sh $W/ck/* 2>/dev/null
grep -q DOWNLOAD_FAILED $LOG && { st "STOP a download failed"; kill $LOGPID; exit 1; }
st downloads
mkdir -p $W/PixCell/pretrained_models; tar -xzf $W/dl/code.bin -C $W/PixCell
for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do tar -xf $W/dl/$d.bin -C $W/pt; rm $W/dl/$d.bin
  p=$(find $W/pt -maxdepth 5 -type d -name $d | head -1); ln -sfn $p $W/PixCell/pretrained_models/$d; done
tar -xf $W/dl/roots.bin -C $W && rm $W/dl/roots.bin
for f in $FAMS; do [ $f = R ] && for r in tr_concat tr_grouped tr_bypass tr_perch; do tar -xf $W/dl/ck_$r.tar -C $W/ck && rm $W/dl/ck_$r.tar; done; done
ls $W/roots $W/ck; md5sum $W/ck/*/step_*/tme_module.pth $W/ck/a1_per_channel/config.py 2>/dev/null
st inputs
for t in $TASKS; do ROOT=${t%%:*}; FAM=${t#*:}
  rm -rf $W/work_$ROOT; mkdir -p $W/work_$ROOT
  PP_SCORE_PYLIB=$W/empty $PY -u $W/kit/pp_gen.py --pixcell $W/PixCell --work $W/work_$ROOT --root $W/roots/$ROOT \
    --family $FAM --ids $W/kit/ids100.txt --d $W --ck $W/ck; rc=$?
  up=1; tar -cf $W/gen_$ROOT.tar -C $W/work_$ROOT out && python3 $W/kit/s3io.py put $W/links.json gen_$ROOT $W/gen_$ROOT.tar && up=0
  [ $rc -eq 0 ] && [ $up -eq 0 ] || NFAIL=$((${NFAIL:-0} + 1))
  st "task $ROOT rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
st "DONE failed tasks ${NFAIL:-0}"; kill $LOGPID
