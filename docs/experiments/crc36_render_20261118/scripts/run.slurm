#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc36_20261118/logs/%x-%A_%a.out
# crc36_render_20261118: the lintype_render_20261118 pipeline on Lin CRC36 (user decision 2026-11-18: CRC36 replaces CRC01
# as the added tumour-rich training slide; C36 hypermutant, low grade; unused by sim-exp-validation evaluations). MODE:
#   mxfetch: the CRC36 19-channel multiplex OME-TIFF (105,398,296,221 bytes, size-checked) for the vasculature / O2 / glucose
#            channels
#   fetch  : Lin CRC36 cell mask, registered H&E, markers.csv (size-checked); byte copies of the CELESTA / gate scripts
#            (md5 against their sources)
#   gates  : C36 column of Lin's sumAllsample (lin_sumall_col.py, value-matched) + lin_gate.py on the CRC36 CSV
#   cprep / ctile : CELESTA prep (k-d depth 4, margin 1000 px) / one core tile (array 1-16), scripts and envs of celltype_pub
#   tiles  : tissue tile grid (hfv patchify rule, checked on CRC33_01 against its 10,379 training tiles), H&E tile PNGs,
#            ink tiles (style_pool.pigment_fraction > 0, checked on CRC33_01 against darkdiag ink.json)
#   labels : per-cell population / state (CELESTA + Fig. 3d, Ki-67 Lin gate)
#   test|full : nuclei, Lin-nucleus library, renderer channels, H&E comparison
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N=/gscratch/cheme/chiu/hfv_crc36_20261118
A=/gscratch/cheme/chiu/he_celltype_audit_20261118                 # read only
E=/gscratch/cheme/chiu/crc16-20261024/celltype_pub_20261118        # environments + script originals, read only
LR=/gscratch/cheme/chiu/crc16-20261024/lin_reinit_20261106         # sumAllsample, read only
PYC=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python     # tifffile, cv2, h5py, skimage
PYG=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python # pandas, sklearn
CSV=$N/lin/data__CRC36__P37_S79-CRC36.csv
echo "== $(date -Is) $(hostname) MODE=$MODE task=${SLURM_ARRAY_TASK_ID:-}"
case $MODE in
  mxfetch)
    cd $N/lin; f=P37_S79_Full_A24_C59qX_E15_20220302_235254_496641-zlib.ome.tiff; n=105398296221; o=data__CRC36__$f
    if [ "$(stat -c %s $o 2>/dev/null || echo 0)" != $n ]; then
      curl -sS --fail --retry 3 -C - -o $o.part "https://lin-2023-orion-crc.s3.amazonaws.com/data/CRC36/$f"
      test "$(stat -c %s $o.part)" = $n || { echo "STOP size mismatch $f"; exit 1; }
      mv $o.part $o
    fi
    echo "$o $n bytes"
    ;;
  fetch)
    B=https://lin-2023-orion-crc.s3.amazonaws.com/data/CRC36
    cd $N/lin
    for fs in P37_S79-CRC36-mask.ome.tif:254462842 19510_C36_US_SCAN_OR_001__161442-registered.ome.tif:727189714 markers.csv:128; do
      f=${fs%%:*}; n=${fs##*:}; o=data__CRC36__$f
      if [ "$(stat -c %s $o 2>/dev/null || echo 0)" != $n ]; then
        curl -sS --fail --retry 3 -o $o.part "$B/$f"
        test "$(stat -c %s $o.part)" = $n || { echo "STOP size mismatch $f"; exit 1; }
        mv $o.part $o
      fi
      echo "$o $n bytes"
    done
    mkdir -p $N/pub $N/gates
    for f in celltype_pub_celesta.R celesta_prior.csv; do cp $E/run/$f $N/pub/; done
    for f in lin_gate.py findcutoff.py lin_sumall_col.py; do cp $A/scripts/$f $N/gates/; done
    (cd $E/run && md5sum celltype_pub_celesta.R celesta_prior.csv) | (cd $N/pub && md5sum -c -)
    (cd $A/scripts && md5sum lin_gate.py findcutoff.py lin_sumall_col.py) | (cd $N/gates && md5sum -c -)
    (cd $N/pub && md5sum *) > $N/pub/MD5; (cd $N/gates && md5sum *.py) > $N/gates/MD5
    ;;
  gates)
    O=$N/out; mkdir -p $O
    $PYC -u $N/gates/lin_sumall_col.py $LR/matlab-sumAllsample-20220710.mat $LR/lin_sumall.json $O/lin_sumall_crc36.json C36 crc36
    LIN_SLIDE=crc36 $PYG -u $N/gates/lin_gate.py $CSV $O/lin_sumall_crc36.json $O/lin_gates_crc36.json > $O/lin_gate.log
    tail -n 5 $O/lin_gate.log
    ;;
  cprep) $E/env_r/bin/Rscript $N/pub/celltype_pub_celesta.R prep $CSV $N/pub/celesta_prior.csv $N/celesta 4 1000 ;;
  ctile) $E/env_r/bin/Rscript $N/pub/celltype_pub_celesta.R tile $N/pub/celesta_prior.csv $N/celesta $SLURM_ARRAY_TASK_ID ;;
  tiles)  PYTHONPATH=$A/pylib $PYC -u $N/scripts/tiles.py $N $N/hfv $N/sev/src ${SLURM_CPUS_PER_TASK:-1} ;;
  labels) $PYG -u $N/scripts/labels_crc36.py $N $N/hfv ;;
  test|full) PYTHONPATH=$A/pylib $PYC -u $N/scripts/nuclei_render_crc36.py $N $N/hfv $N/sev/src $MODE ${SLURM_CPUS_PER_TASK:-1} ;;
  *) echo "STOP unknown MODE"; exit 1 ;;
esac
echo "== done $(date -Is)"
