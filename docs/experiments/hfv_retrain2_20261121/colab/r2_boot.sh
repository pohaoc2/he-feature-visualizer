#!/bin/bash
# hfv_retrain2 stage 3 (PREREG §9) + sim2real app 10: model-N generation on one Colab L4 session.
# Setup = sim-exp-validation scripts/colab/app10_boot_k.sh line for line (folder /content/hfvk2, marker SETUP_K2_OK, uv
# venv Python 3.12, torch 2.10.0+cu128, torchvision 0.25.0, xformers 0.0.35, req.txt; PixCell cb62fa7; pixcell-256,
# pixcell-256-controlnet, sd-3.5-vae; model N = the app-10 prefix infer_tr_concat2_e20.tar, top dir tr_concat, md5 gated
# against CKMD5), so a runtime that already ran app10A / app10B reuses its environment and the same checkpoint bytes.
# Not run directly: r2_presign.py prepends URL_* (one-day presigned GET links), LOGURL, OUTURL_<job> (PUT), JOBS, TAG
# and CKMD5.
# No AWS credentials or HF token in Colab. JOBS (run in order, each output uploaded right after it finishes):
#   r2gen:NAME   r2_gen.py on kin_NAME.tar (row count + ref_key from design.json) -> jobs_out_NAME.tar
#   fig2c:SET    evalgen_r2.py fig2c:orion-SET:LIST on evalpackN_SET.tar          -> jobs_out_fig2c_SET.tar
W=/content/w_${TAG}; mkdir -p $W; LOG=$W/colab.log; exec > $LOG 2>&1
T0=$(date +%s); st() { echo "### $(( $(date +%s) - T0 )) s  $*"; curl -s -o /dev/null -T $LOG "$LOGURL"; }
( i=0; while sleep 60; do i=$((i+1)); [ $((i % 5)) = 1 ] && echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)"
    curl -s -o /dev/null -T $LOG "$LOGURL"; done ) & LOGPID=$!
fail() { st "STOP: $*"; kill $LOGPID; exit 1; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv; python3 --version; df -h /content | tail -1; free -g | sed -n 2p
st "start jobs: $JOBS"
H=/content/hfvk2; PY=$H/env/bin/python; PIX=$H/PixCell; D=$W/dl; mkdir -p $H $D $W/roots $W/kins $W/ret
dl() { local n=$1 u=$2 sz ch i a b; sz=$(curl -s -r 0-0 -D - -o /dev/null "$u" | tr -d '\r' | awk -F/ 'tolower($0) ~ /^content-range/ {print $2}')
  ch=$(( (sz + 7) / 8 )); for i in 0 1 2 3 4 5 6 7; do a=$(( i * ch )); b=$(( a + ch - 1 )); [ $a -ge $sz ] && break
    ( until curl -sf --speed-limit 100000 --speed-time 30 -r $a-$b -o $n.p$i "$u"; do echo "retry $n part $i"; sleep 2; done ) & done; wait
  cat $n.p? > $n; rm $n.p?; [ "$(stat -c %s $n)" = "$sz" ] && echo "got $n $sz" || echo "SIZE_MISMATCH $n"; }
var() { echo "URL_$(echo $1 | tr '.-' '__')"; }
DL=""; get() { local n; for n in "$@"; do ( eval u=\$$(var $n); dl $D/$n "$u" ) & DL="$DL $!"; done; }
cd $D
need="kit.tar.gz"
for j in $JOBS; do IFS=: read -r kind name <<< "$j"
  case $kind in r2gen) need="$need kin_$name.tar" ;; fig2c) need="$need evalpackN_$name.tar list_$name.txt" ;; *) fail "job $j" ;; esac
done
if [ ! -f /content/SETUP_K2_OK ]; then
  get req.txt pixcell.tar.gz pixcell-256.tar pixcell-256-controlnet.tar sd-3.5-vae.tar k2.tar $need
  pip -q install uv >/dev/null 2>&1
  [ -x $PY ] || uv venv -q --python 3.12 $H/env; $PY --version
  uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128; st torch
  wait $DL; st downloads; ls -la $D; grep -q SIZE_MISMATCH $LOG && fail "download size"
  m=$(md5sum $D/k2.tar | cut -d' ' -f1); [ "$m" = "$CKMD5" ] || fail "N checkpoint md5 $m != $CKMD5"
  echo "setuptools==75.2.0" > $W/bc.txt
  uv pip install -q --python $PY --build-constraints $W/bc.txt xformers==0.0.35 -r $D/req.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || fail "pip"
  mkdir -p $PIX/pretrained_models $H/pt $H/ck
  tar -xzf $D/pixcell.tar.gz -C $PIX && rm $D/pixcell.tar.gz || fail "PixCell"
  for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae; do
    tar -xf $D/$d.tar -C $H/pt && rm $D/$d.tar || fail "$d"
    p=$(find $H/pt -maxdepth 5 -type d -name $d | head -1); ln -sfn $p $PIX/pretrained_models/$d
    [ -e $PIX/pretrained_models/$d ] || fail "pretrained $d missing"
  done
  tar -xf $D/k2.tar -C $H/ck && rm $D/k2.tar || fail "checkpoint"
  [ "$(ls $H/ck)" = tr_concat ] || fail "checkpoint layout: $(ls $H/ck)"
  echo "$CKMD5" > $H/ck/K2_MD5; touch /content/SETUP_K2_OK
else
  get $need; wait $DL; grep -q SIZE_MISMATCH $LOG && fail "download size"; st "reused setup; downloads"
  [ "$(cat $H/ck/K2_MD5)" = "$CKMD5" ] || fail "reused checkpoint md5 $(cat $H/ck/K2_MD5) != $CKMD5"
fi
$PY -c "import torch,torchvision,numpy,mmcv,diffusers,xformers,timm,cv2;print('env',torch.__version__,torchvision.__version__,numpy.__version__,mmcv.__version__,diffusers.__version__,xformers.__version__,timm.__version__,torch.cuda.get_device_name())" \
  || fail "env import"
rm -rf $W/kit; mkdir -p $W/kit; tar -xzf $D/kit.tar.gz -C $W/kit || fail kit
ls $H/ck/tr_concat/*; sha256sum $W/kit/* $H/ck/tr_concat/step_*/*.pth; st setup
export HF_HOME=$H/cache/hf XDG_CACHE_HOME=$H/cache OMP_NUM_THREADS=4
NFAIL=0
for j in $JOBS; do IFS=: read -r kind name <<< "$j"
  case $kind in
    r2gen)
      mkdir -p $W/kins/$name && tar -xf $D/kin_$name.tar -C $W/kins/$name || fail "kin $name"
      st "job $j start"; $PY -u $W/kit/r2_gen.py $PIX $H/ck $W/kins/$name $W/ret/$name; rc=$?; out=$name ;;
    fig2c)
      [ -d $W/roots/orion-$name ] || tar -xf $D/evalpackN_$name.tar -C $W/roots || fail "pack $name"
      cp $D/list_$name.txt $W/roots/list_$name.txt
      st "job $j start"; $PY -u $W/kit/evalgen_r2.py --pixcell $PIX --work $W/eg --ck $H/ck --roots $W/roots "fig2c:orion-$name:list_$name.txt"
      rc=$?; out=fig2c_$name; rm -rf $W/ret/$out; cp -r $W/eg/out/fig2c_$name $W/ret/$out ;;
  esac
  cp $LOG $W/ret/$out/colab_run.log
  tar -cf $W/jobs_out_$out.tar -C $W/ret $out || fail "tar $out"
  eval u=\$OUTURL_$(echo $out | tr '.-' '__')
  code=$(curl -s -o /dev/null -w "%{http_code}" -T $W/jobs_out_$out.tar "$u")
  [ $rc = 0 ] && [ "$code" = 200 ] || NFAIL=$((NFAIL + 1))
  st "job $j rc $rc upload http $code"
done
st "DONE failed jobs $NFAIL"; kill $LOGPID
