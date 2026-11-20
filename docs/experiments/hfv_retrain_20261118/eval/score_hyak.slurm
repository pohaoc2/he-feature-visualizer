#!/bin/bash
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_score_20261119/logs/%x-%j.out
# hfv_retrain_20261118 evaluation, scoring on a Hyak checkpoint A100 with TF32 off (README rule 5). eval/evalscore.py
# runs PixCell cb62fa7's own scoring scripts unchanged (UNI-2h transform via eval/pxscore.py) and CellViT-256 (he-feature-
# visualizer stages/run_cellvit_local.py, the CRC34 gate runner and env). No requeue: a preempted or failed job is a STOP
# (coordinator rule), never retried automatically.
# Each ITEM = KIND:SRC:GROUP:PACK[:LIST]  KIND fig2|abl|loo|decomp|crc34; SRC = the generation tar under S3 outputs/;
# GROUP = the group folder inside it; PACK = crc36ho|crc33sec|crc34eval (inputs/eval/evalpack_<PACK>.tar); LIST = the
# fig2 tile list / ablation style mapping (inputs/eval/). Work folder = node-local /tmp (gscratch file creation is slow on
# GPU nodes). After each item: GROUP folder (images + sidecars + scores) -> S3 outputs/score_<GROUP>.tar, small result
# files (json / csv / logs) -> D/res/<GROUP>/.
# Env: crc16-20261024/cellvit/env (the generation and CellViT env) + D/pylib (lpips 0.1.4, pandas 2.2.3, added with
# pip --no-deps --target; not in that env) ; AlexNet weights for LPIPS pre-fetched in D/torch_home (sha256 7be5be79...).
#   sbatch --job-name=ev-score-NAME --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/score_hyak.slurm "ITEM [ITEM ...]"
set -uo pipefail
ITEMS=$1
D=/gscratch/cheme/chiu/hfv_score_20261119; Wd=/tmp/hfv_score_$SLURM_JOB_ID; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; P=$Wd/PixCell; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
export PY=$C/env/bin/python NVIDIA_TF32_OVERRIDE=0 MPLBACKEND=Agg HF_HOME=$Wd/cache/hf XDG_CACHE_HOME=$Wd/cache TORCH_HOME=$D/torch_home
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$D/pylib
export CV_PY=$C/env/bin/python CV_REPO=$C/CellViT CV_CKPT=$C/ckpt/CellViT-256.pth CV_RUNNER=$C/hfv/stages/run_cellvit_local.py
module load coenv/aws/2.17.32
mkdir -p $Wd/roots $Wd/gen $D/res
trap 'rm -rf $Wd' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) items $ITEMS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
(cd $C/hfv && git log --oneline -1 2>/dev/null); md5sum $CV_CKPT
mkdir -p $P/pretrained_models
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $P || { echo "STOP PixCell"; exit 1; }
ln -sfn $A/uni-2h $P/pretrained_models/uni-2h
cp $D/scripts/evalscore.py $D/scripts/pxscore.py $D/scripts/fig2tiles.py $Wd/
TASKS=""
for it in $ITEMS; do IFS=: read -r kind src group pack list <<< "$it"
  [ -f $Wd/roots/$pack.ok ] || { aws s3 cp --only-show-errors $S/inputs/eval/evalpack_$pack.tar - | tar -x -C $Wd/roots && touch $Wd/roots/$pack.ok; } || { echo "STOP pack $pack"; exit 1; }
  [ -z "$list" ] || [ -f $Wd/roots/$list ] || aws s3 cp --only-show-errors $S/inputs/eval/$list $Wd/roots/$list || { echo "STOP list $list"; exit 1; }
  aws s3 cp --only-show-errors $S/outputs/$src - | tar -x -C $Wd/gen $group || { echo "STOP gen $src"; exit 1; }
  echo "$group: $(find $Wd/gen/$group -name '*.png' | wc -l) png"
  TASKS="$TASKS $kind:$Wd/gen/$group:$Wd/roots/orion-$pack${list:+:$Wd/roots/$list}"
done
echo "### $(date -Is) inputs"
cd $Wd
rc_all=0
for t in $TASKS; do
  $PY -u $Wd/evalscore.py --pixcell $P --work $Wd $t; rc=$?
  g=$(basename $(echo $t | cut -d: -f2))
  mkdir -p $D/res/$g
  (cd $Wd/gen/$g && find . \( -name '*.json' -o -name '*.csv' -o -name '*.log' -o -name '*.txt' -o -name 'SCORE_*' -o -name DONE \) \
     -not -name '*_cellvit_instances.json' -not -name 'manifest.json' -not -name 'subchannel_loo_diff_stats.json' -print0 \
     | tar -cf - --null -T - ) | tar -xf - -C $D/res/$g
  up=1; tar -cf $Wd/score_$g.tar -C $Wd/gen $g && aws s3 cp --only-show-errors $Wd/score_$g.tar $S/outputs/score_$g.tar && { up=0; rm $Wd/score_$g.tar; }
  [ $rc -eq 0 ] && [ $up -eq 0 ] || rc_all=1
  echo "### $(date -Is) task $t rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
echo "### $(date -Is) done rc $rc_all"
exit $rc_all
