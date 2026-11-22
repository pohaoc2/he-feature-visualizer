#!/bin/bash
# hfv_retrain3_cellvit_20261122 on Tillicum (copy of hfv_retrain2 setup2.sh; only the run folder, tar names _v3 and their md5 changed; v2 comments kept): the Colab boot.sh setup + train-task data steps, run on the LOGIN node before the
# GPU job (no downloads or pip in GPU time; coordinator rule 2026-11-18). Storage rule (2026-11-18): every new file under
# /gpfs/scrubbed/pohaoc2/hfv_retrain3_cellvit_20261122 (scrubbed: removed automatically after a period without access;
# checkpoints also go to S3); archives streamed (curl | tar x / streamunzip.py), never stored next to their contents;
# pretrained weights and the Python env read from /gpfs/projects/bagherilab/pohaoc2 (not copied), after checking them
# against what Colab uses. Inputs come through one-day presigned GET links (links.json from colab/presign.py, mode 600,
# never committed); no AWS credentials on Tillicum.
#   bash setup.sh W        W = /gpfs/scrubbed/pohaoc2/hfv_retrain3_cellvit_20261122, holding links.json and tools/ (this dir)
# Result state = orion-crc33 with the seven old cell channels, the three old environment channels, the old index and the
# old mask latents left out; newch33_v3; masklat33_v3; c36train_v3 merged by the unchanged kit/merge36.py.
set -euo pipefail
W=$1; L=$W/links.json; TL=$W/tools
PROJ=/gpfs/projects/bagherilab/pohaoc2; E=$PROJ/envs/pixcell; PY=$E/bin/python; T=$PROJ/models/pretrained_models
st() { echo "### $(date -Is) $*"; }
url() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['get'][sys.argv[2]])" $L $1; }
get() { curl -sSf --retry 5 "$(url $1)"; }
# v2: the three new tars are checked against the md5 written by pack.py (pack.json) while they stream in
MD5_newch33_v3=TBD; MD5_masklat33_v3=TBD; MD5_c36train_v3=TBD
getv() { mkdir -p $W/md5; get $1 | python3 -c "
import hashlib, sys
h = hashlib.md5(); i, o = sys.stdin.buffer, sys.stdout.buffer
for b in iter(lambda: i.read(1 << 22), b''):
    h.update(b); o.write(b)
o.flush(); open(sys.argv[1], 'w').write(h.hexdigest())" $W/md5/$1.md5; }
chk() { local want=MD5_$1; [ "$(cat $W/md5/$1.md5)" = "${!want}" ] || { echo "STOP md5 $1 $(cat $W/md5/$1.md5) != ${!want}"; exit 1; }; echo "md5 ok $1"; }
cd $W; st start
df -h /gpfs/home /gpfs/projects/bagherilab /gpfs/scrubbed 2>/dev/null || true
# ---- kit and PixCell cb62fa7 (the Colab tars) ----
# phases with a marker (ok_*) are skipped on a rerun
if [ ! -f ok_code ]; then
mkdir -p kit PixCell; get kit | tar -xz -C kit; get code | tar -xz -C PixCell; ls kit; st code; touch ok_code; fi
# ---- pretrained weights: Colab tars (streamed, md5 only) vs the Tillicum copies ----
if [ ! -f ok_pretrained ]; then
mkdir -p PixCell/pretrained_models md5
for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do
  get $d | python3 $TL/tarmd5.py $d > md5/colab_$d.txt
  (cd $T/$d && find . -type f | LC_ALL=C sort | xargs md5sum) | python3 -c "import sys; r=[l.rstrip('\n') for l in sys.stdin]; print('\n'.join(sorted(r, key=lambda s: s[34:])))" > md5/tillicum_$d.txt
  # every file of the Colab tar must be on Tillicum with the same md5; extra Tillicum files (HF cache metadata, README,
  # model_index.json, ...) are listed, not loaded by train_wrap's paths (transformer/, controlnet/, vae/)
  miss=$(LC_ALL=C comm -23 <(LC_ALL=C sort md5/colab_$d.txt) <(LC_ALL=C sort md5/tillicum_$d.txt))
  [ -s md5/colab_$d.txt ] && [ -z "$miss" ] || { echo "STOP pretrained $d: Colab files missing or different on Tillicum:"; echo "$miss" | head; exit 1; }
  echo "pretrained $d: all $(wc -l < md5/colab_$d.txt) Colab files identical; Tillicum extra files: $(LC_ALL=C comm -13 <(LC_ALL=C sort md5/colab_$d.txt) <(LC_ALL=C sort md5/tillicum_$d.txt) | awk '{print $2}' | paste -sd' ' -)"
  # the model directory PixCell sees holds exactly the Colab file set: one symlink per Colab file to the (identical)
  # Tillicum file, no copies; the extra Tillicum files stay invisible to the run
  rm -rf PixCell/pretrained_models/$d
  while read -r _ f; do f=${f#./}; mkdir -p "$(dirname PixCell/pretrained_models/$d/$f)"; ln -s "$T/$d/$f" "PixCell/pretrained_models/$d/$f"; done < md5/colab_$d.txt
done
st pretrained; touch ok_pretrained; fi
# ---- env: the Tillicum env vs the Colab pins (req.txt + torch / torchvision / xformers) ----
get req > req.txt
$PY -m pip freeze 2>/dev/null > env_freeze.txt
python3 - req.txt env_freeze.txt <<'EOF'
import sys
n = lambda s: s.lower().replace("-", "_")
req = {n(l.split("==")[0]): l.split("==")[1].strip() for l in open(sys.argv[1]) if "==" in l and not l.startswith("#")}
req.update(torch="2.10.0+cu128", torchvision="0.25.0+cu128", xformers="0.0.35")
fr = {n(l.split("==")[0]): l.split("==")[1].strip() for l in open(sys.argv[2]) if "==" in l}
print("env missing vs Colab:", [k for k in req if k not in fr])
print("env version differs vs Colab:", {k: (req[k], fr[k]) for k in req if k in fr and fr[k] != req[k]})
EOF
(cd PixCell && $PY -c "import torch,mmcv,diffusers,timm,xformers,accelerate,transformers,numpy,h5py; import train_scripts.train_controlnet_exp; print('env import ok', torch.__version__, mmcv.__version__, diffusers.__version__, timm.__version__, xformers.__version__, accelerate.__version__, transformers.__version__, numpy.__version__)")
st env
# ---- data ----
mkdir -p data
# v2: the three old environment channels are left out too (replaced by method C in newch33_v3)
X='^orion-crc33/(exp_channels/(cell_masks|cell_type_healthy|cell_type_cancer|cell_type_immune|cell_state_prolif|cell_state_nonprolif|cell_state_dead|vasculature|oxygen|glucose)/|metadata/exp_index\.hdf5$|vae_features/[^/]*_mask_sd3_vae\.npy$)'
[ -f ok_orion ] || { get orion-crc33 | python3 $TL/streamunzip.py data "$X"; st orion-crc33; touch ok_orion; }
# v2: PRE_ONLY=1 stops here (the v2 tars are still being packed); a later run skips the finished phases by their markers
[ "${PRE_ONLY:-}" = 1 ] && { st "pre-only phases done"; exit 0; }
D=$W/data/orion-crc33
getv newch33_v3 | tar -x -C data; chk newch33_v3   # v2: 7 L1 cell channels + method C ch7-9 + index
for c in $D/exp_channels/*; do echo "$(ls $c | wc -l) $c"; done
$PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print({k: len(h[k]) for k in h})"
st newch33
getv masklat33_v3 | tar -x -C $W; chk masklat33_v3   # v2: mask latents of the L1 cell_masks
n=$(ls $W/masklat | grep -c _mask_sd3_vae.npy); echo "mask latents $n"
[ $n = $(ls $D/exp_channels/cell_masks | wc -l) ] || { echo "STOP masklat33_v3 incomplete"; exit 1; }
[ $(ls $D/vae_features | grep -c _mask_sd3_vae.npy) = 0 ] || { echo "STOP old mask latents present"; exit 1; }
mv $W/masklat/*_mask_sd3_vae.npy $D/vae_features/ && rmdir $W/masklat && touch $W/MASKLAT_OK; st "masklat installed"
getv c36train_v3 | tar -x -C data; chk c36train_v3   # v2: L1 cell channels, method C ch7-9, new mask latents; UNI + H&E latents byte-copied
$PY -u $W/kit/merge36.py $D $W/data/c36 && touch $W/C36_OK
$PY -c "import h5py; h=h5py.File('$D/metadata/exp_index.hdf5'); print('index', {k: len(h[k]) for k in h})"
du -sh $W/data; df -h /gpfs/home /gpfs/projects/bagherilab 2>/dev/null || true
touch $W/SETUP_OK; st "setup done"
