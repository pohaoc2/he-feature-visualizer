#!/bin/bash
#SBATCH --job-name=ev-figs
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_figs_20261118/logs/%x-%j.out
# hfv_retrain_20261118 paper-figure re-render (README "論文 figure 用新模型重繪"). PixCell cb62fa7 figure scripts run unchanged
# in the re-render checkout F/PixCell (from inputs/pixcell_cb62fa7.tar.gz) through eval/figrender.py (guard before every
# script; Vanilla bands for Fig 2d / si_performance_ranking). Inputs: the CRC33 secondary group scored on the A100
# (S3 outputs/score_<group>.tar) + evalpack_crc33sec, placed under F/inputs and linked into the checkout where PixCell
# hard-codes them; panels marked "unchanged" / "not redone" read the original inputs from S3 share_space/05_results.
# Fig 2b training curves: the four Tillicum H200 training logs (outputs/log_tr_<run>_h200, one run each, steps 50-6500)
# at the train_log.jsonl paths of log_utils.LOG_SOURCES; PixCell's _read_log parses plain-text logs (its own production
# fallback train_log.log is one). Each figure script is one step; a failing step is recorded (FIG_FAILED) and reported,
# never retried. Outputs stay in F/PixCell; at the end figures/pngs_updated + results/paper_figures go to S3
# hfv_retrain_20261118/figures_rerender/.
#   sbatch --exclude=$(bash /gscratch/cheme/chiu/slow_nodes/slow_nodes.sh list) figs_hyak.slurm
set -uo pipefail
F=/gscratch/cheme/chiu/hfv_figs_20261118; R=$F/PixCell; I=$F/inputs; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
S=s3://bagherilab-working/pohao/hfv_retrain_20261118; OLD=s3://bagherilab-working/pohao/share_space/05_results
export MPLBACKEND=Agg PYTHONPATH=$F/pylib:$F/scripts XDG_CACHE_HOME=$F/cache MPLCONFIGDIR=$F/cache/mpl
PY=$C/env/bin/python
module load coenv/aws/2.17.32
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
# ---- inputs (new folders only)
if [ ! -f $I/.ok ]; then
  mkdir -p $I/score $I/eval $I/logs $I/old/results $I/old/figs
  for g in fig2_crc33sec abl_crc33sec decomp_crc33sec loo_crc33sec; do
    aws s3 cp --only-show-errors $S/outputs/score_$g.tar - | tar -x -C $I/score || { echo "STOP input score_$g"; exit 1; }
  done
  aws s3 cp --only-show-errors $S/inputs/eval/evalpack_crc33sec.tar - | tar -x -C $I/eval || { echo "STOP input evalpack"; exit 1; }
  aws s3 cp --only-show-errors $S/inputs/eval/crc33sec_p300_unpaired.json $I/eval/ || exit 1
  for r in concat grouped perch bypass; do aws s3 cp --only-show-errors $S/outputs/log_tr_${r}_h200 $I/logs/log_tr_${r}_h200 || exit 1; done
  for p in a4_uni_probe a1_probe_mlp_spatial a6_cohort_invariance; do
    aws s3 sync --only-show-errors $OLD/results/$p $I/old/results/$p || { echo "STOP input old $p"; exit 1; }
  done
  aws s3 cp --only-show-errors $OLD/figures_pngs_updated/individual/ablation_analysis/06_unpaired_ablation_grid.png $I/old/figs/ || exit 1
  touch $I/.ok
fi
# ---- links into the checkout (PixCell hard-coded paths)
mkdir -p $R/data $R/inference_output/concat_ablation_1000 $R/results
ln -sfn $I/eval/orion-crc33sec $R/data/orion-crc33
ln -sfn $I/score/fig2_crc33sec $R/inference_output/si_a1_a2
for m in paired unpaired; do ln -sfn $I/score/abl_crc33sec/${m}_ablation $R/inference_output/concat_ablation_1000/${m}_ablation; done
mkdir -p $I/score/abl_crc33sec/unpaired_ablation/metadata
cp $I/eval/crc33sec_p300_unpaired.json $I/score/abl_crc33sec/unpaired_ablation/metadata/unpaired_mapping.json
ln -sfn $I/score/loo_crc33sec $R/inference_output/subchannel_loo_n300
ln -sfn $I/score/decomp_crc33sec $R/results/a2_decomposition
ln -sfn $I/old/results/a4_uni_probe $R/results/a4_uni_probe
for pair in concat:checkpoints/concat_95470_0 grouped:checkpoints/production_retrain_post_fix/full_seed_42 \
            perch:checkpoints/a1_per_channel/full_seed_42 bypass:checkpoints/a2_a3/a2_bypass/full_seed_42; do
  mkdir -p $R/${pair#*:}; ln -sfn $I/logs/log_tr_${pair%%:*}_h200 $R/${pair#*:}/train_log.jsonl
done
$PY $F/scripts/figrender.py guard $R || exit 1
$PY $F/scripts/figrender.py bands $I/score/fig2_crc33sec/cache.json $F/bands.json || exit 1
# ---- figure steps
fails=0
step () { local name=$1; shift; echo "### $(date -Is) step $name: $*"
  (cd $R && $PY -u $F/scripts/figrender.py run $R "$@") > $F/logs/step_$name.log 2>&1; local rc=$?
  echo "### $(date -Is) step $name rc $rc"; [ $rc -eq 0 ] || { fails=$((fails+1)); echo "$name rc $rc" >> $F/FIG_FAILED; tail -20 $F/logs/step_$name.log; }; }
rm -f $F/FIG_FAILED
step curves tools.ablation_a1_a2.build_cache -- --update-curves --cache-dir inference_output/si_a1_a2
step fig2 src.paper_figures.fig_combined_method_perf --bands $F/bands.json
step fig3 src.paper_figures.fig_combined_uni_decomp_v2
step si_assets src.paper_figures.build_si_assets --bands $F/bands.json
step si_a1a2 src.paper_figures.fig_si_a1_a2_unified
step grids figgrid -- $I/old/figs/06_unpaired_ablation_grid.png
# ---- upload
find $R/figures/pngs_updated $R/results/paper_figures -newer $I/.ok -type f 2>/dev/null | sort > $F/rendered.txt
cat $F/rendered.txt

aws s3 sync --only-show-errors $R/figures/pngs_updated $S/figures_rerender/figures/pngs_updated && \
  aws s3 cp --only-show-errors $F/bands.json $S/figures_rerender/bands.json && \
  aws s3 cp --only-show-errors $F/rendered.txt $S/figures_rerender/rendered.txt && echo "upload ok" || { echo "upload FAILED"; fails=$((fails+1)); }
[ -d $R/results/paper_figures ] && aws s3 sync --only-show-errors $R/results/paper_figures $S/figures_rerender/results/paper_figures
echo "### $(date -Is) done, $fails failed steps"
exit $([ $fails -eq 0 ] && echo 0 || echo 1)
