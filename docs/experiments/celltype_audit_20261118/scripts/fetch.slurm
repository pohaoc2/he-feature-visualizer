#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/he_celltype_audit_20261118/logs/%x-%j.out
# celltype_audit_20261118 step 1 (inventory fetch, read-only on every source).
#  - he-feature-visualizer stage-1 output for CRC33 (own S3, ranged reads; only the per-cell tables, CellViT JSONs and
#    type/state rasters, not the 26 GB multiplex tiles);
#  - the generator's training channels (orion-crc33.zip exp_channels type/state/mask rasters + metadata);
#  - Lin 2023 public data for CRC33_01 and CRC33_02 (CSV, cell mask, registered H&E) + Lin H&E class maps (GitHub);
#  - every file size checked against the source listing; any mismatch -> exit 1 (STOP, no retry).
set -euo pipefail
W=/gscratch/cheme/chiu/he_celltype_audit_20261118
PY=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python
set +u; module load coenv/aws/2.17.32; set -u
echo "== $(date -Is) $(hostname)"
S=s3://bagherilab-working/pohao/share_space
cd $W/raw
# per-tile rasters only for a fixed random sample of tiles (the raster checks are descriptive; 93k small files stalled
# job 43359509 on n3305). Sample: 500 tile ids from index.json, numpy default_rng(20261118), written once.
export SKIP_EXISTING=1  # only for the per-file members (tables, CellViT JSONs)
[ -s $W/raw/tiles_sample.txt ] || $PY -c "
import json, numpy as np
ids = sorted(p['id'] if isinstance(p, dict) and 'id' in p else f\"{p['x0']}_{p['y0']}\" for p in json.load(open('$W/raw/processed_crc33/index.json'))['patches'])
r = np.random.default_rng(20261118); open('$W/raw/tiles_sample.txt', 'w').write('\\n'.join(sorted(r.choice(ids, 500, replace=False))) + '\\n')"
if [ ! -f p33.done ]; then
  rm -f $W/raw/p33_rasters_sample.tar
  $PY -u $W/scripts/zip_extract.py $S/01_stage1_features/processed_crc33.zip $W/raw \
    processed_crc33/cell_assignments.csv processed_crc33/cellvit_mx_features.csv processed_crc33/cell_shape_features.csv \
    processed_crc33/cell_summary.json processed_crc33/index.json processed_crc33/registration/ processed_crc33/analysis/ \
    processed_crc33/cellvit/
  TILES=$W/raw/tiles_sample.txt TAR=$W/raw/p33_rasters_sample.tar $PY -u $W/scripts/zip_extract.py $S/01_stage1_features/processed_crc33.zip $W/raw \
    processed_crc33/cell_types/ processed_crc33/cell_states/ processed_crc33/cell_masks/
  touch p33.done
fi
if [ ! -f o33.done ]; then
  rm -f $W/raw/o33_rasters_sample.tar
  TILES=$W/raw/tiles_sample.txt TAR=$W/raw/o33_rasters_sample.tar $PY -u $W/scripts/zip_extract.py $S/02_datasets/orion-crc33.zip $W/raw \
    orion-crc33/metadata/ orion-crc33/exp_channels/cell_masks/ orion-crc33/exp_channels/cell_type_cancer/ \
    orion-crc33/exp_channels/cell_type_healthy/ orion-crc33/exp_channels/cell_type_immune/ \
    orion-crc33/exp_channels/cell_state_prolif/ orion-crc33/exp_channels/cell_state_nonprolif/ \
    orion-crc33/exp_channels/cell_state_dead/
  touch o33.done
fi
B=https://lin-2023-orion-crc.s3.amazonaws.com/data
cd $W/lin
for fs in CRC33_01/P37_S76_01-CRC33_01.csv:222836315 CRC33_01/P37_S76_01-CRC33_01-mask.ome.tif:84491354 \
          CRC33_01/19510_C33_US_SCAN_OR_001__160715-2-registered.ome.tif:390336926 CRC33_01/markers.csv:128 \
          CRC33_02/P37_S76_02-CRC33_02.csv:293352057; do
  f=${fs%%:*}; n=${fs##*:}; o=${f/\//__}
  if [ "$(stat -c %s $o 2>/dev/null || echo 0)" != $n ]; then
    curl -sS --fail --retry 3 -o $o.part "$B/$f"
    test "$(stat -c %s $o.part)" = $n || { echo "STOP size mismatch $f"; exit 1; }
    mv $o.part $o
  fi
  echo "$o $n bytes"
done
G=https://raw.githubusercontent.com/labsyspharm/orion-crc/main/HE_patch_prediction/gamma-only/orion/class-images
for t in "19510\$C33\$US\$SCAN\$OR\$001 _160715-2-registered.tiff:CRC33_01_lin_he_class.tiff" \
         "19510\$C33\$US\$SCAN\$OR\$001 _160715-registered.tiff:CRC33_02_lin_he_class.tiff" \
         "19510\$C16\$US\$SCAN\$OR\$001 _152020-registered.tiff:CRC16_lin_he_class.tiff"; do
  src=${t%%:*}; dst=${t##*:}
  enc=$($PY -c "import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))" "$src")
  [ -s $dst ] || curl -sS --fail --retry 3 -o $dst "$G/$enc"
  ls -l $dst
done
# the CRC16 map must equal the one the main repo uses (same source, same bytes)
cmp CRC16_lin_he_class.tiff /gscratch/cheme/chiu/crc16-20261024/tc_lincell_20261118/CRC16_lin_he_class.tiff && echo "CRC16 class map identical to main repo copy"
du -sh $W/raw $W/lin
echo "== $(date -Is) done"
