#!/bin/bash
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_eval_20261118/logs/%x-%j.out
# hfv_retrain_20261118 evaluation, generation on a Hyak checkpoint A100 (README rules: one GPU type per comparison group;
# scoring separately). Same eval/evalgen.py as the Colab boot; environment = the Hyak CellViT env of gpu34.slurm (the
# Colab req.txt + torch / torchvision / xformers), PixCell cb62fa7 (S3 inputs) + the Hyak pretrained files the Colab
# tars were made from (sev-conditioning-recovery-20261006/assets; all files md5-identical to the Colab tars).
# Inputs from S3 (inputs/eval/ packs and lists, outputs/infer_<run>_e20.tar); after each task its group folder goes to
# S3 outputs/gen_hyak_<group>.tar (and E/gen/NAME/). Work folder = node-local /tmp: creating one file on gscratch took
# about 1 s on the GPU nodes (2026-11-18), so packs, PixCell and images stay on the node. A requeued job starts the
# unfinished groups again from scratch in its own /tmp (finished groups are marked in E/gen/NAME/sent_<group>).
#   sbatch --job-name=ev-gen-NAME --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) eval/gen_hyak.slurm NAME "TASK [TASK ...]"
set -uo pipefail
NAME=$1; TASKS=$2
E=/gscratch/cheme/chiu/hfv_eval_20261118; K=$E/gen/$NAME; Wd=/tmp/hfv_eval_$SLURM_JOB_ID; PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python
A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; P=$Wd/PixCell; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
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
  [ -f $Wd/roots/$root.ok ] || { aws s3 cp --only-show-errors $S/inputs/eval/evalpack_${root#orion-}.tar - | tar -x -C $Wd/roots && touch $Wd/roots/$root.ok; } || { echo "STOP pack $root"; exit 1; }
  for f in ${rest//:/ }; do [ -f $Wd/roots/$f ] || aws s3 cp --only-show-errors $S/inputs/eval/$f $Wd/roots/$f || { echo "STOP list $f"; exit 1; }; done
  runs=tr_concat; [ $kind = fig2 ] && runs="tr_concat tr_grouped tr_bypass tr_perch"
  for r in $runs; do [ -d $Wd/ck/$r ] || aws s3 cp --only-show-errors $S/outputs/infer_${r}_e${EP:-20}.tar - | tar -x -C $Wd/ck || { echo "STOP checkpoint $r"; exit 1; }; done
done
cp $E/scripts/evalgen.py $E/scripts/pxrun.py $Wd/
for t in $TASKS; do IFS=: read -r kind root rest <<< "$t"; g=${kind}_${root#orion-}
  if [ -f $K/sent_$g ]; then echo "skip $t (done and sent)"; continue; fi
  $PY -u $Wd/evalgen.py --pixcell $P --work $Wd --ck $Wd/ck --roots $Wd/roots "$t"; rc=$?
  [ $kind = speed ] && g=$(cd $Wd/out && ls -d speed_${root#orion-}_*/ | tail -1); g=${g%/}   # directory only, not the .json
  up=1; tar -cf $Wd/gen_hyak_$g.tar -C $Wd/out $g $(cd $Wd/out && ls speed_*.json 2>/dev/null) && \
    aws s3 cp --only-show-errors $Wd/gen_hyak_$g.tar $S/outputs/gen_hyak_$g.tar && { up=0; rm $Wd/gen_hyak_$g.tar; }
  [ $rc -eq 0 ] && [ $up -eq 0 ] && touch $K/sent_$g   # only a finished and uploaded group is skipped on requeue
  cp $Wd/out/$g/DONE $Wd/out/$g/FAILED $Wd/out/$g/gen.log $Wd/out/speed_*.json $K/ 2>/dev/null
  echo "### $(date -Is) task $t rc $rc upload $([ $up -eq 0 ] && echo ok || echo FAILED)"
done
echo "### $(date -Is) done"
