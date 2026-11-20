#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_robust_20261120/logs/%x-%j.out
# hfv_retrain_20261118 PQ robustness readouts (eval/robust.py) for one evaluation set, all on node-local /tmp (gscratch
# small-file creation is slow): stage the scored groups (fig2 / abl / loo / decomp, whichever exist), the real H&E and
# the real-H&E CellViT JSONs from S3; StarDist 2D_versatile_he on every image (P processes x T threads); per-image metrics
# and the bootstrap summary. Results (csv / json, StarDist label npz, image list, logs) -> R/res/SET (new folder).
#   sbatch --job-name=rob-SET eval/robust_set.slurm SET LIN_DIR
# Any failure is a STOP: no retry, no requeue.
set -uo pipefail
SET=$1; LIN=$2
R=/gscratch/cheme/chiu/hfv_robust_20261120; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
O=$R/res/$SET; W=/tmp/rob_${SET}_$SLURM_JOB_ID
stop() { echo "STOP $*"; exit 1; }
[ -e $O ] && stop "$O exists (new folders only)"
mkdir -p $O $W/score $W/real $W/realcv $W/sd $W/PixCell; trap 'rm -rf $W' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) $SET lin $LIN"
source $R/scripts/robust_envsetup.sh
module load coenv/aws/2.17.32
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $W/PixCell || stop PixCell
for k in fig2 abl loo decomp; do
  key=$S/outputs/score_${k}_$SET.tar
  aws s3 ls $key >/dev/null 2>&1 || { echo "no $k for $SET"; continue; }
  mkdir $W/score/${k}_$SET
  aws s3 cp --only-show-errors $key $W/t.tar || stop "download $key"
  tar -xf $W/t.tar -C $W/score/${k}_$SET --exclude='*/features/*' --exclude='*/sensitivity_tiles/*' \
    --exclude='*.npy' --exclude='*.pt' || stop "untar $key"
  rm -f $W/t.tar; echo "### $(date -Is) $k: $(find $W/score/${k}_$SET -name '*.png' | wc -l) png"
done
mkdir $W/real/$SET
aws s3 cp --only-show-errors $S/inputs/eval/realhe_$SET.zip $W/r.zip && $PY -m zipfile -e $W/r.zip $W/real/$SET || stop "realhe"
aws s3 cp --only-show-errors $S/outputs/realcv_$SET.tar - | tar -x -C $W/realcv || stop "realcv"
echo "### $(date -Is) real $(ls $W/real/$SET | wc -l) png, realcv $(ls $W/realcv/realcv_$SET | grep -c json) json"
cd $W/PixCell || stop cd
PYTHONPATH=$W/PixCell $PY -u $R/scripts/robust.py sdlist $SET $W/score $W/real $W/list.tsv || stop sdlist
cp $W/list.tsv $O/
P=8; export SD_THREADS=$((SLURM_CPUS_PER_TASK / P)) OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / P))
pids=()
for k in $(seq 0 $((P - 1))); do
  $PY -u $R/scripts/robust.py stardist $W/list.tsv $W/sd $k $P > $O/sd_$k.log 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait $p || stop "stardist process (see $O/sd_*.log)"; done
cp $W/sd/*.npz $W/sd/sd_model.json $O/ || stop "copy sd"
echo "### $(date -Is) stardist done: $(cat $W/sd/sd_model.json | tr -d '\n ')"
export OMP_NUM_THREADS=1
PYTHONPATH=$W/PixCell $PY -u $R/scripts/robust.py metrics $SET $W/score $W/real $W/realcv/realcv_$SET $W/sd $LIN $W/out \
  $SLURM_CPUS_PER_TASK || stop metrics
PYTHONPATH=$W/PixCell $PY -u $R/scripts/robust.py stats $W/out || stop stats
cp $W/out/* $O/ || stop "copy out"
echo "### $(date -Is) DONE"
