#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc34_20261118/logs/%x-%A_%a.out
# crc34_render_20261118: the crc36_render_20261118 pipeline (same rules, thresholds, envs and code paths) on Lin CRC34
# (patient 2, held out, never used for training). Coordinator decision 2026-11-18, before any evaluation result: the
# CRC34 evaluation inputs are rebuilt with the CRC36 pipeline; the old orion-CRC34.tar is no longer used and stays on S3.
# MODE:
#   mxfetch: the CRC34 19-channel multiplex OME-TIFF (137,941,102,041 bytes, size-checked) for vasculature / O2 / glucose
#   fetch  : Lin CRC34 cell mask, registered H&E, CSV, markers.csv (size-checked; sizes = Lin bucket listing); the Lin
#            H&E class map from orion-crc GitHub class-images (the source of CRC36_lin_he_class.tiff, crc01 step1_compose.py),
#            md5 against the copy step1_compose.py fetched (hfv_crc01_20261118/lin/classmaps); byte copies of the CELESTA /
#            gate scripts (md5 against their sources)
#   gates  : C34 column of Lin's sumAllsample (lin_sumall_col.py, value-matched) + lin_gate.py on the CRC34 CSV
#   cprep / ctile : CELESTA prep (k-d depth 4, margin 1000 px) / one core tile (array 1-16), scripts and envs of celltype_pub
#   tiles  : tissue tile grid, H&E tile PNGs, ink tiles (tiles.py, CRC33_01 checks included, unchanged)
#   labels : per-cell population / state (CELESTA + Fig. 3d, Ki-67 Lin gate)
#   test|full : nuclei, Lin-nucleus library, renderer channels, H&E comparison
#   cvold  : the old CRC34 CellViT JSONs (processed_CRC34_stage1_gpu_products.tar, read from S3, kept there unchanged)
#   cvprep : grid check old JSONs vs the new tile grid, the 50 seed-42 gate tiles, gate tar, kept-tile H&E tar
#   cvdead : CellViT JSONs (CV_SRC = s3: Colab cv34 output; or a directory of reused old JSONs) -> dead Lin ids; first-pass
#            outputs move to out/pass1_nodead and data/pass1_nodead, then MODE=labels and MODE=full run again
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
N=/gscratch/cheme/chiu/hfv_crc34_20261118
A=/gscratch/cheme/chiu/he_celltype_audit_20261118                 # read only
E=/gscratch/cheme/chiu/crc16-20261024/celltype_pub_20261118        # environments + script originals, read only
LR=/gscratch/cheme/chiu/crc16-20261024/lin_reinit_20261106         # sumAllsample, read only
PYC=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python     # tifffile, cv2, h5py, skimage
PYG=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python # pandas, sklearn
CSV=$N/lin/data__CRC34__P37_S77-CRC34.csv
S3=s3://bagherilab-working/pohao
echo "== $(date -Is) $(hostname) MODE=$MODE task=${SLURM_ARRAY_TASK_ID:-}"
case $MODE in
  cvold)
    set +u; module load coenv/aws/2.17.32; set -u
    mkdir -p $N/cvold && cd $N/cvold
    aws s3 cp --quiet $S3/share_space/01_stage1_features/processed_CRC34_stage1_gpu_products.tar old.tar
    test "$(stat -c %s old.tar)" = 3798118400 || { echo "STOP size mismatch old tar"; exit 1; }
    md5sum old.tar > old.tar.md5; cat old.tar.md5
    tar -tvf old.tar > old_members.txt; wc -l old_members.txt; awk '{print $6}' old_members.txt | cut -d/ -f1-2 | sort | uniq -c | sort -rn | head -20
    ;;
  cvprep) $PYC -u $N/scripts/cvprep34.py $N ${CVOLD_DIR} ;;
  cvdead)
    mkdir -p $N/cv34
    if [ "${CV_SRC:-s3}" = s3 ]; then
      set +u; module load coenv/aws/2.17.32; set -u
      aws s3 cp --quiet $S3/hfv_retrain_20261118/outputs/cellvit_crc34 $N/cv34/cellvit_crc34.tar
      tar -xf $N/cv34/cellvit_crc34.tar -C $N/cv34 && rm $N/cv34/cellvit_crc34.tar
      CVD=$N/cv34/cv34
    else
      CVD=$CV_SRC
    fi
    $PYC -u $N/scripts/cellvit_dead.py $N $CVD
    mkdir -p $N/out/pass1_nodead $N/data/pass1_nodead
    for f in cells_labels.npz labels.json cells_nuclei.npz compare_tiles.npz nuclei_render.json render_tiles.npz nucleus_shapes_lin_crc34.npz; do
      if [ -e $N/out/$f ]; then mv $N/out/$f $N/out/pass1_nodead/; fi; done
    if [ -e $N/data/crc34_exp_channels.tar ]; then mv $N/data/crc34_exp_channels.tar $N/data/pass1_nodead/; fi
    ;;
  mxfetch)
    cd $N/lin; f=P37_S77_Full_A24_C59qX_E15_20220302_235222_359806-zlib.ome.tiff; n=137941102041; o=data__CRC34__$f
    if [ "$(stat -c %s $o 2>/dev/null || echo 0)" != $n ]; then
      curl -sS --fail --retry 3 -C - -o $o.part "https://lin-2023-orion-crc.s3.amazonaws.com/data/CRC34/$f"
      test "$(stat -c %s $o.part)" = $n || { echo "STOP size mismatch $f"; exit 1; }
      mv $o.part $o
    fi
    echo "$o $n bytes"
    ;;
  fetch)
    B=https://lin-2023-orion-crc.s3.amazonaws.com/data/CRC34
    cd $N/lin
    for fs in P37_S77-CRC34-mask.ome.tif:468371320 19510_C34_US_SCAN_OR_001__160949-registered.ome.tif:1290571158 \
              P37_S77-CRC34.csv:1493691021 markers.csv:128; do
      f=${fs%%:*}; n=${fs##*:}; o=data__CRC34__$f
      if [ "$(stat -c %s $o 2>/dev/null || echo 0)" != $n ]; then
        curl -sS --fail --retry 3 -o $o.part "$B/$f"
        test "$(stat -c %s $o.part)" = $n || { echo "STOP size mismatch $f"; exit 1; }
        mv $o.part $o
      fi
      echo "$o $n bytes"
    done
    mkdir -p classmaps
    G=https://raw.githubusercontent.com/labsyspharm/orion-crc/main/HE_patch_prediction/gamma-only/orion/class-images
    curl -sS --fail --retry 3 -o classmaps/CRC34_lin_he_class.tiff "$G/19510%24C34%24US%24SCAN%24OR%24001%20_160949-registered.tiff"
    echo "83b40b84b6110edf4f66190ede0e1842  classmaps/CRC34_lin_he_class.tiff" | md5sum -c -
    mkdir -p $N/pub $N/gates
    for f in celltype_pub_celesta.R celesta_prior.csv; do cp $E/run/$f $N/pub/; done
    for f in lin_gate.py findcutoff.py lin_sumall_col.py; do cp $A/scripts/$f $N/gates/; done
    (cd $E/run && md5sum celltype_pub_celesta.R celesta_prior.csv) | (cd $N/pub && md5sum -c -)
    (cd $A/scripts && md5sum lin_gate.py findcutoff.py lin_sumall_col.py) | (cd $N/gates && md5sum -c -)
    (cd $N/pub && md5sum *) > $N/pub/MD5; (cd $N/gates && md5sum *.py) > $N/gates/MD5
    ;;
  gates)
    O=$N/out; mkdir -p $O
    $PYC -u $N/gates/lin_sumall_col.py $LR/matlab-sumAllsample-20220710.mat $LR/lin_sumall.json $O/lin_sumall_crc34.json C34 crc34
    LIN_SLIDE=crc34 $PYG -u $N/gates/lin_gate.py $CSV $O/lin_sumall_crc34.json $O/lin_gates_crc34.json > $O/lin_gate.log
    tail -n 5 $O/lin_gate.log
    ;;
  cprep) $E/env_r/bin/Rscript $N/pub/celltype_pub_celesta.R prep $CSV $N/pub/celesta_prior.csv $N/celesta 4 1000 ;;
  ctile) $E/env_r/bin/Rscript $N/pub/celltype_pub_celesta.R tile $N/pub/celesta_prior.csv $N/celesta $SLURM_ARRAY_TASK_ID ;;
  tiles)  PYTHONPATH=$A/pylib $PYC -u $N/scripts/tiles.py $N $N/hfv $N/sev/src ${SLURM_CPUS_PER_TASK:-1} ;;
  labels) $PYG -u $N/scripts/labels_crc34.py $N $N/hfv ;;
  test|full) PYTHONPATH=$A/pylib $PYC -u $N/scripts/nuclei_render_crc34.py $N $N/hfv $N/sev/src $MODE ${SLURM_CPUS_PER_TASK:-1} ;;
  *) echo "STOP unknown MODE"; exit 1 ;;
esac
echo "== done $(date -Is)"
