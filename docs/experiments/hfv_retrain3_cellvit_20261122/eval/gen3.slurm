#!/bin/bash
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122 generation (PREREG §5.1, §5.5): copy of hfv_retrain_20261118 eval/gen_hyak.slurm,
# evalgen.py / pxrun.py unchanged (read from hfv_eval_20261118/scripts). Changed: run folder; evaluation packs =
# this run's inputs/eval/evalpack3_<set>.tar (official-CellViT cell channels, method C ch7-9, new mask latents); tile
# lists from the first retrain's inputs/eval/ (same files); checkpoints = this run's outputs/infer_tr3_<v>_e20.tar
# (folder names tr_<v>, slimck3.slurm); outputs to this run's outputs/gen_hyak_<group>.tar.
#   sbatch --job-name=r3gen-NAME --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) gen3.slurm NAME "TASK [TASK ...]"
set -uo pipefail
NAME=$1; TASKS=$2
E=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; K=$E/gen/$NAME; E1=/gscratch/cheme/chiu/hfv_eval_20261118; Wd=/tmp/hfv_eval_$SLURM_JOB_ID; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; P=$Wd/PixCell; S=s3://bagherilab-working/pohao/hfv_retrain_20261118; S3=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122
export MPLBACKEND=Agg HF_HOME=$Wd/cache/hf XDG_CACHE_HOME=$Wd/cache OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load coenv/aws/2.17.32
mkdir -p $K $Wd/roots $Wd/ck $Wd/dl
trap 'rm -rf $Wd' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} $(hostname) tasks $TASKS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if [ ! -f $P/ok ]; then rm -rf $P; mkdir -p $P/pretrained_models
  aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $P || { echo "STOP PixCell"; exit 1; }
  for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do ln -sfn $A/$d $P/pretrained_models/$d; done; touch $P/ok; fi
for t in $TASKS; do IFS=: read -r kind root rest <<< "$t"
  [ -f $Wd/roots/$root.ok ] || { aws s3 cp --only-show-errors $S3/inputs/eval/evalpack3_${root#orion-}.tar - | tar -x -C $Wd/roots && touch $Wd/roots/$root.ok; } || { echo "STOP pack $root"; exit 1; }
  for f in ${rest//:/ }; do [ -f $Wd/roots/$f ] || aws s3 cp --only-show-errors $S/inputs/eval/$f $Wd/roots/$f || { echo "STOP list $f"; exit 1; }; done
  runs=tr_concat; [ $kind = fig2 ] && runs="tr_concat tr_grouped tr_bypass tr_perch"
  for r in $runs; do [ -d $Wd/ck/$r ] || aws s3 cp --only-show-errors $S3/outputs/infer_${r/tr_/tr3_}_e${EP:-20}.tar - | tar -x -C $Wd/ck || { echo "STOP checkpoint $r"; exit 1; }; done
done
cp $E1/scripts/evalgen.py $E1/scripts/pxrun.py $Wd/
for t in $TASKS; do IFS=: read -r kind root rest <<< "$t"; g=${kind}_${root#orion-}
  if [ -f $K/sent_$g ]; then echo "skip $t (done and sent)"; continue; fi
  $PY -u $Wd/evalgen.py --pixcell $P --work $Wd --ck $Wd/ck --roots $Wd/roots "$t"; rc=$?
  [ $kind = speed ] && g=$(cd $Wd/out && ls -d speed_${root#orion-}_*/ | tail -1); g=${g%/}   # directory only, not the .json
  up=1; tar -cf $Wd/gen_hyak_$g.tar -C $Wd/out $g $(cd $Wd/out && ls speed_*.json 2>/dev/null) && \
    aws s3 cp --only-show-errors $Wd/gen_hyak_$g.tar $S3/outputs/gen_hyak_$g.tar && { up=0; rm $Wd/gen_hyak_$g.tar; }
  [ $rc -eq 0 ] && [ $up -eq 0 ] && touch $K/sent_$g   # only a finished and uploaded group is skipped on requeue
  cp $Wd/out/$g/DONE $Wd/out/$g/FAILED $Wd/out/$g/gen.log $Wd/out/speed_*.json $K/ 2>/dev/null
  echo "### $(date -Is) task $t rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
echo "### $(date -Is) done"
