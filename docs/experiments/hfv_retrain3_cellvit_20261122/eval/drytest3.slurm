#!/bin/bash
#SBATCH --job-name=r3-scoredry
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --no-requeue
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122/logs/%x-%j.out
# hfv_retrain3_cellvit_20261122: throwaway CPU dry test of score3's code path (evalscore3.py with official CellViT
# post-processing on CPU, fig2tiles.py, r3extra.py) on 3 tiles of the first retrain's CRC33 Fig 2 and ablation images
# (read only) against this run's evalpack3_crc33sec masks. Measures CPU seconds per image; nothing goes into results.
set -uo pipefail
D=/gscratch/cheme/chiu/hfv_score_20261119; R3=/gscratch/cheme/chiu/hfv_retrain3_cellvit_20261122; Wd=/tmp/hfv_dry3_$SLURM_JOB_ID
C=/gscratch/cheme/chiu/crc16-20261024/cellvit; A=/gscratch/cheme/chiu/sev-conditioning-recovery-20261006/assets; P=$Wd/PixCell
S=s3://bagherilab-working/pohao/hfv_retrain_20261118; S3=s3://bagherilab-working/pohao/hfv_retrain3_cellvit_20261122
export PY=$C/env/bin/python NVIDIA_TF32_OVERRIDE=0 MPLBACKEND=Agg HF_HOME=$Wd/cache/hf XDG_CACHE_HOME=$Wd/cache TORCH_HOME=$D/torch_home
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONPATH=$D/pylib EVALSCORE_DEVICE=cpu EVALSCORE_GPU=cpu-dry CUDA_VISIBLE_DEVICES=
export CV_PY=$C/env/bin/python CV_REPO=$C/CellViT CV_CKPT=$C/ckpt/CellViT-256.pth CV_RUNNER=$R3/hfv/stages/run_cellvit_local.py
module load coenv/aws/2.17.32
mkdir -p $Wd/roots $Wd/full $Wd/gen; trap 'rm -rf $Wd' EXIT
echo "### $(date -Is) $(hostname)"; lscpu | grep 'Model name'
mkdir -p $P/pretrained_models
aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $P
ln -sfn $A/uni-2h $P/pretrained_models/uni-2h
cp $R3/eval/evalscore3.py $Wd/evalscore.py; cp $D/scripts/pxscore.py $D/scripts/fig2tiles.py $Wd/
aws s3 cp --only-show-errors $S3/inputs/eval/evalpack3_crc33sec.tar - | tar -x -C $Wd/roots
aws s3 cp --only-show-errors $S/inputs/eval/crc33sec_p300_unpaired.json $Wd/roots/
for f in gen_c33fig2b_1:fig2_crc33sec gen_hyak_abl_crc33sec.tar:abl_crc33sec; do
  aws s3 cp --only-show-errors $S/outputs/${f%%:*} - | tar -x -C $Wd/full ${f##*:}; done
echo "### $(date -Is) inputs"
G=$Wd/gen; F=$Wd/full
mkdir -p $G/fig2_crc33sec/tiles $G/fig2_crc33sec/sensitivity_tiles
T3=$(ls $F/fig2_crc33sec/tiles/a1_concat | head -3 | sed 's/.png//'); printf "%s\n" $T3 > $Wd/roots/ids3.txt
for v in $(ls $F/fig2_crc33sec/tiles); do mkdir -p $G/fig2_crc33sec/tiles/$v; for t in $T3; do cp $F/fig2_crc33sec/tiles/$v/$t.png $G/fig2_crc33sec/tiles/$v/; done; done
for v in $(ls $F/fig2_crc33sec/sensitivity_tiles); do [ -d $F/fig2_crc33sec/sensitivity_tiles/$v ] || continue; for g in $(ls $F/fig2_crc33sec/sensitivity_tiles/$v); do [ -d $F/fig2_crc33sec/sensitivity_tiles/$v/$g ] || continue; mkdir -p $G/fig2_crc33sec/sensitivity_tiles/$v/$g; for t in $T3; do cp $F/fig2_crc33sec/sensitivity_tiles/$v/$g/$t.png $G/fig2_crc33sec/sensitivity_tiles/$v/$g/; done; done; done
$PY -c "import json,sys; c=json.load(open('$F/fig2_crc33sec/cache.json')); c['metric_tile_ids']=open('$Wd/roots/ids3.txt').read().split(); json.dump(c,open('$G/fig2_crc33sec/cache.json','w'))"
for m in paired unpaired; do mkdir -p $G/abl_crc33sec/${m}_ablation/ablation_results; for t in $(ls $F/abl_crc33sec/${m}_ablation/ablation_results | head -3); do cp -r $F/abl_crc33sec/${m}_ablation/ablation_results/$t $G/abl_crc33sec/${m}_ablation/ablation_results/; done; done
R=$Wd/roots/orion-crc33sec
cd $Wd
t0=$(date +%s)
$PY -u evalscore.py --pixcell $P --work $Wd fig2:$G/fig2_crc33sec:$R:$Wd/roots/ids3.txt; echo "### fig2 rc $? $(( $(date +%s) - t0 )) s"
t0=$(date +%s)
$PY -u evalscore.py --pixcell $P --work $Wd abl:$G/abl_crc33sec:$R:$Wd/roots/crc33sec_p300_unpaired.json; echo "### abl rc $? $(( $(date +%s) - t0 )) s"
HFV=$R3/hfv PYTHONPATH=$P:$D/pylib $PY -u $R3/eval/r3extra.py $G/fig2_crc33sec $R "$R3/cv/c33_*" $G/fig2_crc33sec/r3extra.csv; echo "### r3extra rc $?"
for g in fig2_crc33sec abl_crc33sec; do echo "== $g"; cat $G/$g/SCORE_* ; grep -iE "error|Traceback|STOP|postprocess|official" $G/$g/score.log | head -20; tail -8 $G/$g/score.log; done
cat $G/fig2_crc33sec/fig2_per_tile.csv; cat $G/fig2_crc33sec/r3extra.csv
find $G -name '*_cellvit_instances.json' | head -2; find $G -name '*.png' | wc -l
mkdir -p $R3/dry && cp -r $G $R3/dry/ && echo "kept $R3/dry"; echo "### $(date -Is) done"
