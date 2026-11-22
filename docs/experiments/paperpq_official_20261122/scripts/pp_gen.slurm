#!/bin/bash
#SBATCH --account=ckpt-cheme
#SBATCH --partition=ckpt
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/paperpq_official_20261122/logs/%x-%j.out
# paperpq_official_20261122 (PREREG.md §5): generation + appearance scoring of one root on a Hyak checkpoint A100.
# Environment = the hfv_retrain gen_hyak.slurm / score_hyak.slurm one: crc16-20261024/cellvit/env, PixCell cb62fa7 (S3
# inputs tar), pretrained files of sev-conditioning-recovery-20261006/assets, hfv_score_20261119 pylib + torch_home (LPIPS
# AlexNet). Work folder = node-local /tmp (gscratch file creation is slow on GPU nodes). No requeue: a preempted or failed
# job is a STOP, never resubmitted unchanged.
#   sbatch --job-name=pp_gen_ROOT --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) pp_gen.slurm ROOT FAMILY
set -uo pipefail
ROOT=$1; FAM=$2
CC=/gscratch/cheme/chiu; D=$CC/paperpq_official_20261122; Wd=/tmp/pp_gen_$SLURM_JOB_ID; PY=$CC/crc16-20261024/cellvit/env/bin/python
A=$CC/sev-conditioning-recovery-20261006/assets; P=$Wd/PixCell; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
export MPLBACKEND=Agg HF_HOME=$Wd/cache/hf XDG_CACHE_HOME=$Wd/cache OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCH_HOME=$CC/hfv_score_20261119/torch_home PP_SCORE_PYLIB=$CC/hfv_score_20261119/pylib
unset NVIDIA_TF32_OVERRIDE
module load coenv/aws/2.17.32
test ! -e $D/gen/$ROOT.tar || { echo "STOP $D/gen/$ROOT.tar exists"; exit 1; }
mkdir -p $D/gen $Wd/code
trap 'rm -rf $Wd' EXIT
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) root $ROOT family $FAM"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
mkdir -p $P/pretrained_models
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $P || { echo "STOP PixCell"; exit 1; }
for d in pixcell-256 pixcell-256-controlnet sd-3.5-vae uni-2h; do ln -sfn $A/$d $P/pretrained_models/$d; done
tar -xf $D/roots.tar -C $Wd roots/$ROOT || { echo "STOP roots"; exit 1; }
cp $D/scripts/pp_gen.py $D/scripts/pp_appear.py $D/scripts/pxrun.py $D/scripts/pxscore.py $Wd/code/
echo "### $(date -Is) inputs ready"
$PY -u $Wd/code/pp_gen.py --pixcell $P --work $Wd --root $Wd/roots/$ROOT --family $FAM --ids $D/prep/ids100.txt --d $D; rc=$?
tar -cf $D/gen/$ROOT.tar -C $Wd out || rc=1
echo "### $(date -Is) done rc $rc"
exit $rc
