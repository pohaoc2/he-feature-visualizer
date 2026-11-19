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
#   cvgate  : CellViT-256 card gate (50 CRC34 tiles vs the reference JSONs)   cv36 : CellViT on the CRC36 H&E tiles
#   feat36  : UNI-2h + SD3.5 VAE features of the CRC36 H&E tiles   masklat36 : mask latents of the CRC36 cell_masks
#   unicheck: which UNI-2h preprocessing reproduces the stored orion-crc33 UNI features (unicheck.py)
# Results (logs, json) return as result_<TAG>.tar.gz; the log is pushed every 2 min to log_<TAG>.
set -u
export MPLBACKEND=Agg   # Colab exports the notebook inline backend; scripts here run outside the kernel
W=/content/hfv; mkdir -p $W; LOG=$W/colab_$TAG.log
exec > >(tee -a $LOG) 2>&1
T0=$(date +%s)
printf '%s' "$LINKS" > $W/links.json; chmod 600 $W/links.json
PY=$W/env/bin/python
put() { python3 $W/kit/s3io.py put $W/links.json "$1" "$2" && touch "$2.sent"; }
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
    ( python3 $W/kit/s3io.py get $W/links.json $n $W/dl/$n.bin || echo "DOWNLOAD_FAILED $n" ) & DLP="${DLP:-} $!"; done
  pip -q install uv >/dev/null 2>&1
  [ -x $PY ] || uv venv -q --python 3.12 $W/env; $PY --version   # a rerun after an interrupted setup reuses the venv
  uv pip install -q --python $PY torch==2.10.0+cu128 torchvision==0.25.0+cu128 --index-url https://download.pytorch.org/whl/cu128
  st torch
  wait $DLP; st downloads; ls -la $W/dl   # only the download PIDs (a bare wait would also wait on the log loop)
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
      getf kit $W/kit.tar.gz && tar -xzf $W/kit.tar.gz -C $W/kit   # re-fetch: picks up wrapper fixes made after boot
      (cd $P && $PY -u $W/kit/probe_sweep.py $P $D $R $TAG configs/config_controlnet_exp_a1_concat.py \
         configs/config_controlnet_exp.py configs/config_controlnet_exp_a2_bypass.py configs/config_controlnet_exp_a1_per_channel.py)
      st "probe rc $?" ;;
    cvgate|cv36)
      # CellViT-256 (he-feature-visualizer stages/run_cellvit_local.py e88f450, unmodified) in the same environment
      # (the Hyak CellViT env freeze equals req.txt + torch / torchvision / xformers). cvgate = the 50-tile CRC34 gate of
      # crc16-20261024/cellvit (cellvit_gate.py compare: counts equal, centroids within 2 px); cv36 = all CRC36 H&E tiles.
      # CellViT also imports scikit-image (watershed) and numba (tools.py), which req.txt lacks: the Hyak CellViT env's versions
      # (scikit-image 0.26.0, numba 0.68.0, llvmlite 0.50.0; numpy stays 2.0.2). TF32 off for CellViT, the CRC16 rule
      # (prereg_crc16.md revision 1: with TF32 the CRC34 gate had one extra background contour, without it 50/50 matched)
      $PY -c "import skimage, numba, sys; sys.exit(not (skimage.__version__ == '0.26.0' and numba.__version__ == '0.68.0'))" 2>/dev/null || \
        uv pip install -q --python $PY scikit-image==0.26.0 numba==0.68.0 llvmlite==0.50.0 numpy==2.0.2
      $PY -c "import skimage, numba, numpy; print('cellvit deps', skimage.__version__, numba.__version__, numpy.__version__)"
      K=$W/cellvit; if [ ! -d $K/CellViT ]; then mkdir -p $K; python3 $W/kit/s3io.py get $W/links.json cellvit_kit $W/dl/cv.tar && tar -xf $W/dl/cv.tar -C $K && rm $W/dl/cv.tar; fi
      CV="env NVIDIA_TF32_OVERRIDE=0 $PY -u $K/hfv/stages/run_cellvit_local.py --checkpoint $K/ckpt/CellViT-256.pth --cellvit-repo $K/CellViT --no-skip-existing"
      if [ $t = cvgate ]; then
        PYTHONPATH=$K/CellViT $CV --zip $K/tiles/he_tiles.zip --out $R/cvgate; rc=$?
        $PY $K/cellvit_gate.py compare --ref $K/ref/ref --new $R/cvgate --out $R/cvgate.json; cat $R/cvgate.json; st "cvgate rc $rc"
      else
        if [ ! -f $W/he36/zip.ok ]; then mkdir -p $W/he36; python3 $W/kit/s3io.py get $W/links.json he36 $W/dl/he36.tar
          tar -xf $W/dl/he36.tar -C $W/he36 && rm $W/dl/he36.tar && (cd $W/he36 && zip -q -0 -r he36.zip he) && touch $W/he36/zip.ok; fi
        PYTHONPATH=$K/CellViT $CV --zip $W/he36/he36.zip --out $W/cv36 --batch-size 32; rc=$?; st "cv36 rc $rc"
        echo "cellvit json $(find $W/cv36 -name '*.json' | wc -l) of $(ls $W/he36/he | wc -l) tiles"
        [ $rc = 0 ] && tar -cf $W/cellvit_crc36.tar -C $W cv36 && put cellvit_crc36 $W/cellvit_crc36.tar
      fi ;;
    feat36)
      # UNI-2h (pooled 1536) and SD3.5 VAE latents of the CRC36 H&E tiles, PixCell stage1_extract_features.py unmodified,
      # two passes into features/ and vae_features/ as the training loader expects
      if [ ! -f $W/he36/zip.ok ]; then mkdir -p $W/he36; python3 $W/kit/s3io.py get $W/links.json he36 $W/dl/he36.tar
        tar -xf $W/dl/he36.tar -C $W/he36 && rm $W/dl/he36.tar && (cd $W/he36 && zip -q -0 -r he36.zip he) && touch $W/he36/zip.ok; fi
      (cd $P && $PY -u stage1_extract_features.py --image-dir $W/he36/he --output-dir $W/f36/features --skip-vae) ; r1=$?
      (cd $P && $PY -u stage1_extract_features.py --image-dir $W/he36/he --output-dir $W/f36/vae_features --skip-uni) ; r2=$?
      st "feat36 rc $r1 $r2"; echo "uni $(ls $W/f36/features | wc -l) vae $(ls $W/f36/vae_features | wc -l)"
      [ $r1 = 0 ] && [ $r2 = 0 ] && tar -cf $W/feat_crc36.tar -C $W f36 && put feat_crc36 $W/feat_crc36.tar ;;
    masklat36)
      # SD3.5 VAE mask latents of the CRC36 cell_masks channel (first-pass render; cell_masks does not depend on the
      # CellViT dead labels, which change only the state channels - checked by md5 after the dead re-render on Hyak)
      mkdir -p $W/c36; python3 $W/kit/s3io.py get $W/links.json crc36ch $W/dl/c36.tar && tar -xf $W/dl/c36.tar -C $W/c36 && rm $W/dl/c36.tar
      (cd $P && $PY -u stage1_extract_features.py --image-dir $W/c36/orion-crc36/exp_channels/cell_masks --output-dir $W/masklat36 \
         --vae-prefix mask_sd3_vae --skip-uni); rc=$?; st "masklat36 rc $rc"
      echo "mask latents $(ls $W/masklat36 | wc -l) of $(ls $W/c36/orion-crc36/exp_channels/cell_masks | wc -l)"
      [ $rc = 0 ] && tar -cf $W/masklat_crc36.tar -C $W masklat36 && put masklat_crc36 $W/masklat_crc36.tar ;;
    uni36hub|uni36hubbic)
      # CRC36 UNI-2h with the uni-2h config.json transform (feat_uni.py), when unicheck shows it reproduces the stored CRC33
      V=hub; [ $t = uni36hubbic ] && V=hub_bicubic
      if [ ! -f $W/he36/zip.ok ]; then mkdir -p $W/he36; python3 $W/kit/s3io.py get $W/links.json he36 $W/dl/he36.tar
        tar -xf $W/dl/he36.tar -C $W/he36 && rm $W/dl/he36.tar && touch $W/he36/zip.ok; fi
      $PY -u $W/kit/feat_uni.py $P $V --image-dir $W/he36/he --output-dir $W/u36/features --skip-vae; rc=$?; st "$t rc $rc"
      echo "uni $(ls $W/u36/features | wc -l)"
      [ $rc = 0 ] && tar -cf $W/uni_crc36_$V.tar -C $W u36 && put uni_crc36_$V $W/uni_crc36_$V.tar ;;
    unifix)
      # unicheck, then the CRC36 UNI features again with the variant closest to the stored CRC33 features (highest median
      # cosine) when that variant is not the shipped one (feat36 already used the shipped one)
      (cd $P && $PY -u $W/kit/unicheck.py $P $D $R/unicheck.json); st "unicheck rc $?"
      B=$($PY -c "import json; r=json.load(open('$R/unicheck.json')); print(max(('A_shipped','B_hub_cfg','C_hub_bicubic'), key=lambda k: r[k]['median_cosine']))")
      echo "unicheck best $B"
      V=""; [ "$B" = B_hub_cfg ] && V=hub; [ "$B" = C_hub_bicubic ] && V=hub_bicubic
      if [ -n "$V" ]; then
        if [ ! -f $W/he36/zip.ok ]; then mkdir -p $W/he36; python3 $W/kit/s3io.py get $W/links.json he36 $W/dl/he36.tar
          tar -xf $W/dl/he36.tar -C $W/he36 && rm $W/dl/he36.tar && touch $W/he36/zip.ok; fi
        $PY -u $W/kit/feat_uni.py $P $V --image-dir $W/he36/he --output-dir $W/u36/features --skip-vae; rc=$?; st "uni36 $V rc $rc"
        echo "uni $(ls $W/u36/features | wc -l)"
        [ $rc = 0 ] && tar -cf $W/uni_crc36_$V.tar -C $W u36 && put uni_crc36_$V $W/uni_crc36_$V.tar
      fi ;;
    unicheck) (cd $P && $PY -u $W/kit/unicheck.py $P $D $R/unicheck.json); st "unicheck rc $?" ;;
    train:*)
      # train:RUN:CONFIG:MICRO_BS - one paper variant on CRC33_01 + CRC36 (train_wrap.py: effective batch 80, 20 epochs,
      # epoch-end checkpoints, every 2nd epoch and the last uploaded in the background, resume from S3 on a new runtime)
      IFS=: read -r _ RUN CFG MBS ALLOW <<< "$t"
      # the card first: Colab can fall back to a 40 GB A100 or an L4 without notice. MICRO_BS auto = from the probes
      # (effective batch stays 80): >= 70 GB measured micro 80 for all four (A100-SXM4-80GB, peak 39-80 GB);
      # 35-70 GB not measured, from the measured peaks (peak grows about linearly with micro): Concat 40, others 20;
      # an L4 / T4 stops here unless the fifth field is allowl4 (L4 measured: Concat 16, others 5)
      G=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1); GM=${G##*, }
      echo "train GPU: $G MiB"
      case "$G" in *L4*|*T4*|*V100*|*P100*)
        echo "WARNING: $RUN is on ${G%%,*}, not an A100. Training does not start (about 3.6-3.9x slower than A100;"
        echo "WARNING: rerun with the fifth TASKS field allowl4 only after the coordinator checked)"
        [ "${ALLOW:-}" = allowl4 ] || { st "STOP $RUN not on A100: ${G%%,*}"; continue; } ;; esac
      if [ "$MBS" = auto ]; then
        if [ "$GM" -ge 70000 ]; then MBS=80
        elif [ "$GM" -ge 35000 ]; then MBS=20; [ "$CFG" = config_controlnet_exp_a1_concat ] && MBS=40
        else MBS=5; [ "$CFG" = config_controlnet_exp_a1_concat ] && MBS=16; fi
      fi
      echo "train $RUN: config $CFG micro-batch $MBS x accumulation $((80 / MBS)) = 80"; st "train $RUN start mbs $MBS"
      getf kit $W/kit.tar.gz && tar -xzf $W/kit.tar.gz -C $W/kit
      if [ ! -f $W/MASKLAT_OK ]; then   # new CRC33_01 mask latents (masklat task of an earlier runtime, S3)
        python3 $W/kit/s3io.py get $W/links.json masklat_crc33_01 $W/dl/ml33.tar && tar -xf $W/dl/ml33.tar -C $W && rm $W/dl/ml33.tar
        n=$(ls $W/masklat | grep -c _mask_sd3_vae.npy); echo "mask latents $n"
        [ $n = $(ls $D/exp_channels/cell_masks | wc -l) ] || { echo "STOP masklat_crc33_01 incomplete"; st STOP; break; }
        mkdir -p $W/data/old33/vae_mask; mv $D/vae_features/*_mask_sd3_vae.npy $W/data/old33/vae_mask/
        cp $W/masklat/*_mask_sd3_vae.npy $D/vae_features/ && touch $W/MASKLAT_OK; st "masklat installed"
      fi
      if [ ! -f $W/C36_OK ]; then   # CRC36 training tiles (crc36_render_20261118 assemble36.py) + combined index
        python3 $W/kit/s3io.py get $W/links.json c36train $W/dl/c36train.tar && tar -xf $W/dl/c36train.tar -C $W/data && rm $W/dl/c36train.tar
        $PY -u $W/kit/merge36.py $D $W/data/c36 && touch $W/C36_OK || { echo "STOP merge36"; st STOP; break; }
        st "c36 merged"
      fi
      $PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print('index', {k: len(h[k]) for k in h})"
      $PY -u $W/kit/train_wrap.py --pixcell $P --config configs/$CFG.py --data-root $D --work-dir $W/runs/$RUN \
        --micro-bs $MBS --save-every-epoch --upload-every 2 --links $W/links.json --run-name $RUN --resume auto --fetch-resume
      st "train $RUN rc $?" ;;
    *) echo "unknown task $t" ;;
  esac
done
cp $LOG $R/; tar --exclude='*.pth' -czf $W/result_$TAG.tar.gz -C $W res_$TAG && put result_$TAG $W/result_$TAG.tar.gz
st DONE; kill $LOGPID
