#!/bin/bash
# hfv_retrain_20261118 Colab side. Not run directly: presign.py prepends LINKS (a JSON of one-day presigned links, never
# committed), TASKS and TAG, uploads the result as boot_<TAG>.sh, and the notebook runs it with one curl | bash line.
# No AWS credentials in Colab: every input comes through a GET link, every output goes back through a PUT link.
#   setup   (once per runtime; /content/HFV_SETUP_OK): uv venv Python 3.12 + torch 2.10.0+cu128 / torchvision 0.25.0 /
#           xformers 0.0.35 + the pinned PixCell environment of the Hyak env (req.txt, same as the 2026-11-11 Colab runs);
#           PixCell cb62fa7 (git archive), pretrained weights (pixcell-256, pixcell-256-controlnet, sd-3.5-vae, uni-2h);
#           orion-crc33.zip, then the new CRC33_01 cell channels (lintype_render_20261118, loose nuclei, ink tiles
#           excluded: 9,419 tiles) replace the seven old cell channels; vasculature / oxygen / glucose stay as shipped.
#   checks  : checks.py - stored UNI / VAE / old-mask-VAE features re-encoded on this card (50 tiles)
#   masklat : SD3.5 VAE mask latents of the new cell_masks (stage1_extract_features.py --vae-prefix mask_sd3_vae
#             --skip-uni, unmodified), returned as masklat_crc33_01.tar, then installed into vae_features/
#   probe   : probe_sweep.py on the four paper variant configs (Concat, Grouped, Grouped + bypass, Per-channel)
# Results (logs, json) return as result_<TAG>.tar.gz; the log is pushed every 2 min to log_<TAG>.
set -u
W=/content/hfv; mkdir -p $W; LOG=$W/colab_$TAG.log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s)
printf '%s' "$LINKS" > $W/links.json; chmod 600 $W/links.json
PY=$W/env/bin/python
put() { python3 $W/kit/s3io.py put $W/links.json "$1" "$2"; }
st() { echo "### $(( $(date +%s) - T0 )) s  $*"; python3 $W/kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1 || true; }
( while sleep 120; do echo "gpu $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>&1)" >> $LOG
    python3 $W/kit/s3io.py put $W/links.json log_$TAG $LOG >/dev/null 2>&1; done ) & LOGPID=$!
getf() { python3 - "$1" "$2" <<'PYG'
import json, sys, urllib.request
u = json.load(open("/content/hfv/links.json"))["get"][sys.argv[1]]
with urllib.request.urlopen(u) as r, open(sys.argv[2], "wb") as f:
    while True:
        b = r.read(1 << 24)
        if not b: break
        f.write(b)
PYG
}
# kit first (s3io.py), with plain urllib
mkdir -p $W/kit; getf kit $W/kit.tar.gz && tar -xzf $W/kit.tar.gz -C $W/kit && ls $W/kit
nvidia-smi --query-gpu=name,memory.total --format=csv; df -h /content | tail -1; free -g | sed -n 2p; nproc
st start
if [ ! -f /content/HFV_SETUP_OK ]; then
  mkdir -p $W/dl $W/pt $W/data
  # parallel downloads (s3io.py get resumes and checks the size)
  for n in code req pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h orion-crc33 newch33; do
    ( python3 $W/kit/s3io.py get $W/links.json $n $W/dl/$n.bin || echo "DOWNLOAD_FAILED $n" ) & done
  pip -q install uv >/dev/null 2>&1
  uv venv -q --python 3.12 $W/env; $PY --version
  uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  st torch
  wait; st downloads; ls -la $W/dl
  grep -q DOWNLOAD_FAILED $LOG && { echo "STOP: a download failed"; st STOP; kill $LOGPID; exit 1; }
  echo "setuptools==75.2.0" > $W/bc.txt
  uv pip install -q --python $PY --build-constraints $W/bc.txt xformers==0.0.35 -r $W/dl/req.bin \
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || { echo "STOP: pip failed"; st STOP; exit 1; }
  $PY -c "import torch,mmcv,diffusers,timm,xformers,accelerate;print('env',torch.__version__,torch.cuda.get_device_name(),mmcv.__version__,diffusers.__version__,accelerate.__version__,xformers.__version__)"
  st pip
  mkdir -p $W/PixCell; tar -xzf $W/dl/code.bin -C $W/PixCell
  for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do tar -xf $W/dl/$d.bin -C $W/pt; rm $W/dl/$d.bin; done
  mkdir -p $W/PixCell/pretrained_models
  for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do
    p=$(find $W/pt -maxdepth 5 -type d -name $d | head -1); ln -sfn $p $W/PixCell/pretrained_models/$d; done
  ls -la $W/PixCell/pretrained_models/*/
  cd $W/data; unzip -q $W/dl/orion-crc33.bin; rm $W/dl/orion-crc33.bin
  D=$W/data/orion-crc33; mkdir -p $W/data/old33
  for c in cell_masks cell_type_healthy cell_type_cancer cell_type_immune cell_state_prolif cell_state_nonprolif cell_state_dead; do
    mv $D/exp_channels/$c $W/data/old33/$c; done
  mv $D/metadata/exp_index.hdf5 $W/data/old33/exp_index.hdf5
  tar -xf $W/dl/newch33.bin -C $W/data; rm $W/dl/newch33.bin
  for c in $D/exp_channels/*; do echo "$(ls $c | wc -l) $c"; done
  $PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print({k: len(h[k]) for k in h})"
  touch /content/HFV_SETUP_OK; st setup
fi
D=$W/data/orion-crc33; P=$W/PixCell; R=$W/res_$TAG; mkdir -p $R
for t in $TASKS; do
  case $t in
    checks) (cd $P && $PY -u $W/kit/checks.py $P $D $W/data/old33/cell_masks $R/checks.json); st "checks rc $?" ;;
    masklat)
      if [ ! -f $W/MASKLAT_OK ]; then
        (cd $P && $PY -u stage1_extract_features.py --image-dir $D/exp_channels/cell_masks --output-dir $W/masklat \
           --vae-prefix mask_sd3_vae --skip-uni); rc=$?; st "masklat rc $rc"
        n=$(ls $W/masklat | grep -c _mask_sd3_vae.npy); echo "mask latents $n"
        if [ $rc = 0 ] && [ $n = $(ls $D/exp_channels/cell_masks | wc -l) ]; then
          tar -cf $W/masklat_crc33_01.tar -C $W masklat && put masklat_crc33_01 $W/masklat_crc33_01.tar
          mkdir -p $W/data/old33/vae_mask; mv $D/vae_features/*_mask_sd3_vae.npy $W/data/old33/vae_mask/
          cp $W/masklat/*_mask_sd3_vae.npy $D/vae_features/ && touch $W/MASKLAT_OK; st "masklat installed"
        else echo "STOP: masklat incomplete"; fi
      fi ;;
    probe)
      (cd $P && $PY -u $W/kit/probe_sweep.py $P $D $R $TAG configs/config_controlnet_exp_a1_concat.py \
         configs/config_controlnet_exp.py configs/config_controlnet_exp_a2_bypass.py configs/config_controlnet_exp_a1_per_channel.py)
      st "probe rc $?" ;;
    *) echo "unknown task $t" ;;
  esac
done
cp $LOG $R/; tar --exclude='*.pth' -czf $W/result_$TAG.tar.gz -C $W res_$TAG && put result_$TAG $W/result_$TAG.tar.gz
st DONE; kill $LOGPID
