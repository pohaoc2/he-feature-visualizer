#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122 scoring (PREREG §5.2, §5.5) on Hyak cheme CPU: copy of hfv_retrain_20261118
# eval/score_hyak.slurm with evalscore3.py (official CellViT post-processing; PixCell scoring on CPU via
# EVALSCORE_DEVICE=cpu). Same env, pylib (lpips, pandas) and AlexNet weights of hfv_score_20261119 (read only). Packs =
# this run's inputs/eval/evalpack3_<PACK>.tar; generation tars from this run's outputs/; lists from the first retrain's
# inputs/eval/. Results -> R/res/<GROUP>/ and outputs/score_<GROUP>.tar.
#   sbatch --job-name=r3score-NAME score3.slurm "KIND:SRC:GROUP:PACK[:LIST] ..."
set -uo pipefail
ITEMS=$1
D=/gscratch/cheme/chiu/hfv_score_20261119; R3=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; Wd=/tmp/hfv_score3_$SLURM_JOB_ID; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; P=$Wd/PixCell; S=s3://bagherilab-working/pohao/hfv_retrain_20261118; S3=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122
export PY=$C/env/bin/python NVIDIA_TF32_OVERRIDE=0 MPLBACKEND=Agg HF_HOME=$Wd/cache/hf XDG_CACHE_HOME=$Wd/cache TORCH_HOME=$D/torch_home
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$D/pylib EVALSCORE_DEVICE=cpu EVALSCORE_GPU=cpu-$(hostname)
export CV_PY=$C/env/bin/python CV_REPO=$C/CellViT CV_CKPT=$C/ckpt/CellViT-256.pth CV_RUNNER=$R3/hfv/stages/run_cellvit_local.py   # runner 467a54f (md5 96ecbbb3), the copy cv3.slurm used for the real H&E
module load coenv/aws/2.17.32
mkdir -p $Wd/roots $Wd/gen $R3/res
trap 'rm -rf $Wd' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) items $ITEMS"
lscpu | grep 'Model name'
(cd $C/hfv && git log --oneline -1 2>/dev/null); md5sum $CV_CKPT
mkdir -p $P/pretrained_models
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $P || { echo "STOP PixCell"; exit 1; }
ln -sfn $A/uni-2h $P/pretrained_models/uni-2h
cp $R3/eval/evalscore3.py $Wd/evalscore.py; cp $D/scripts/pxscore.py $D/scripts/fig2tiles.py $Wd/
TASKS=""
for it in $ITEMS; do IFS=: read -r kind src group pack list <<< "$it"
  [ -f $Wd/roots/$pack.ok ] || { aws s3 cp --only-show-errors $S3/inputs/eval/evalpack3_$pack.tar - | tar -x -C $Wd/roots && touch $Wd/roots/$pack.ok; } || { echo "STOP pack $pack"; exit 1; }
  [ -z "$list" ] || [ -f $Wd/roots/$list ] || aws s3 cp --only-show-errors $S/inputs/eval/$list $Wd/roots/$list || { echo "STOP list $list"; exit 1; }
  aws s3 cp --only-show-errors $S3/outputs/$src - | tar -x -C $Wd/gen $group || { echo "STOP gen $src"; exit 1; }
  echo "$group: $(find $Wd/gen/$group -name '*.png' | wc -l) png"
  TASKS="$TASKS $kind:$Wd/gen/$group:$Wd/roots/orion-$pack${list:+:$Wd/roots/$list}"
done
echo "### $(date -Is) inputs"
cd $Wd
rc_all=0
for t in $TASKS; do
  $PY -u $Wd/evalscore.py --pixcell $P --work $Wd $t; rc=$?
  g=$(basename $(echo $t | cut -d: -f2))
  if [ $rc -eq 0 ] && [ "${t%%:*}" = fig2 ]; then   # PREREG §5.2 items 1-2: real-H&E reference line + hematoxylin readouts
    pk=$(echo $t | cut -d: -f3); case $pk in *crc36ho) cvg="$R3/cv/c36ho_*";; *crc33sec) cvg="$R3/cv/c33_*";; esac
    HFV=$R3/hfv PYTHONPATH=$P:$D/pylib $PY -u $R3/eval/r3extra.py $Wd/gen/$g $pk "$cvg" $Wd/gen/$g/r3extra.csv || rc=1
  fi
  mkdir -p $R3/res/$g
  (cd $Wd/gen/$g && find . \( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.txt' -o -name 'SCORE_*' -o -name DONE \) \
     -not -name '*_cellvit_instances.json' -not -name 'manifest.json' -not -name 'subchannel_loo_diff_stats.json' -print0 \
     | tar -cf - --null -T - ) | tar -xf - -C $R3/res/$g
  up=1; tar -cf $Wd/score_$g.tar -C $Wd/gen $g && aws s3 cp --only-show-errors $Wd/score_$g.tar $S3/outputs/score_$g.tar && { up=0; rm $Wd/score_$g.tar; }
  [ $rc -eq 0 ] && [ $up -eq 0 ] || rc_all=1
  echo "### $(date -Is) task $t rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
echo "### $(date -Is) done rc $rc_all"
exit $rc_all
