#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121 PREREG §8: L1b cross-PQ on the stage-1 eval tiles of one set (same inputs as diag.slurm).
#   sbatch --job-name=l1b-SET l1b.slurm SET        Any failure is a STOP: no retry, no requeue.
set -uo pipefail
SET=$1
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
RB=/gscratch/cheme/chiu/hfv_robust_20261120/res2/$SET
case $SET in
  crc33sec) LIN=/gscratch/cheme/chiu/hfv_lintype_20261118/out; LIB=$LIN/nucleus_shapes_lin_crc33_01.npz ;;
  crc36ho)  LIN=/gscratch/cheme/chiu/hfv_crc36_20261118/out; LIB=$LIN/nucleus_shapes_lin_crc36.npz ;;
  *) echo "STOP set"; exit 1 ;;
esac
O=$R/l1b2/$SET; W=/tmp/l1b_${SET}_$SLURM_JOB_ID
stop() { echo "STOP $*"; exit 1; }
[ -e $O ] && stop "$O exists (new folders only)"
mkdir -p $R/l1b2 $W/pack; trap 'rm -rf $W' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $SET sev $(cat $R/sev3/COMMIT)"
module load coenv/aws/2.17.32
PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
aws s3 cp --only-show-errors $S/outputs/realcv_$SET.tar - | tar -x -C $W || stop realcv
tar -xf /gscratch/cheme/chiu/hfv_eval_20261118/packs/evalpack_$SET.tar -C $W/pack --wildcards "*/exp_channels/cell_masks/*" || stop pack
PK=$(dirname $(dirname $(find $W/pack -type d -name cell_masks | head -1)))
export HFV=/gscratch/cheme/chiu/hfv_lintype_20261118/hfv SEV_SRC=$R/sev3/src OMP_NUM_THREADS=1
PYTHONPATH=/gscratch/cheme/chiu/hfv_figs_20261118/PixCell $PY -u $R/scripts/l1b.py $SET $LIN $LIB $W/realcv_$SET $RB $PK \
  $R/res/$SET/tiles.csv $O $SLURM_CPUS_PER_TASK || stop l1b
echo "### $(date -Is) DONE"
