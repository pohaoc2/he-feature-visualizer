#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 stage-1 diagnosis (PREREG.md §2) for one evaluation set. Inputs staged to node-local /tmp.
#   sbatch --job-name=diag-SET diag.slurm SET
# Any failure is a STOP: no retry, no requeue.
set -uo pipefail
SET=$1
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
RB=/gscratch/cheme/chiu/hfv_robust_20261120/res2/$SET
case $SET in
  crc33sec) TIF=/gscratch/cheme/chiu/he_celltype_audit_20261118/lin/CRC33_01__P37_S76_01-CRC33_01-mask.ome.tif
            LIN=/gscratch/cheme/chiu/hfv_lintype_20261118/out ;;
  crc36ho)  TIF=/gscratch/cheme/chiu/hfv_crc36_20261118/lin/data__CRC36__P37_S79-CRC36-mask.ome.tif
            LIN=/gscratch/cheme/chiu/hfv_crc36_20261118/out ;;
  *) echo "STOP set"; exit 1 ;;
esac
O=$R/res/$SET; W=/tmp/diag_${SET}_$SLURM_JOB_ID
stop() { echo "STOP $*"; exit 1; }
[ -e $O ] && stop "$O exists (new folders only)"
mkdir -p $R/res $W/real $W/pack; trap 'rm -rf $W' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $SET"
module load coenv/aws/2.17.32
PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
aws s3 cp --only-show-errors $S/inputs/eval/realhe_$SET.zip $W/r.zip && $PY -m zipfile -e $W/r.zip $W/real || stop realhe
aws s3 cp --only-show-errors $S/outputs/realcv_$SET.tar - | tar -x -C $W || stop realcv
tar -xf /gscratch/cheme/chiu/hfv_eval_20261118/packs/evalpack_$SET.tar -C $W/pack --wildcards "*/exp_channels/cell_masks/*" || stop pack
PK=$(dirname $(dirname $(find $W/pack -type d -name cell_masks | head -1)))
RH=$(dirname $(find $W/real -name '*.png' | head -1))
echo "### $(date -Is) real $(ls $RH | wc -l) png, realcv $(ls $W/realcv_$SET | grep -c json) json, masks $(ls $PK/exp_channels/cell_masks | wc -l)"
RPX=$(python3 -c "import json;print(json.load(open('$RB/info.json'))['match_radius_px'])")
export HFV=/gscratch/cheme/chiu/hfv_lintype_20261118/hfv OMP_NUM_THREADS=1
PYTHONPATH=/gscratch/cheme/chiu/hfv_figs_20261118/PixCell $PY -u $R/scripts/diag.py tiles $SET $TIF $LIN $RH \
  $W/realcv_$SET $RB $PK $RPX $O $SLURM_CPUS_PER_TASK || stop diag
echo "### $(date -Is) DONE"
