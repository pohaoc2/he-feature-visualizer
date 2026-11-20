#!/bin/bash
#SBATCH --job-name=ev-fig2ref
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=1:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_fig2ref_20261120/logs/%x-%j.out
# hfv_retrain_20261118 follow-up 2 (coordinator 2026-11-20): Fig 2 with D / E / G PQ and DICE against the PRIMARY
# reference (CellViT-256 on the real H&E, eval/realref.py tree from S3 outputs/realref_abl_crc33sec.tar) next to the
# unchanged Fig 2 (PQ / DICE against the Lin input cell mask, the secondary reference, figs_hyak.slurm job 43427989).
# Drawing code = eval/fig2_refvariant.py (copy of PixCell cb62fa7 fig_combined_method_perf.py, see its docstring), run
# twice on the same figs checkout and inputs as figs_hyak.slurm:
#   secondary: ablation metrics = F/inputs/score/abl_crc33sec (the tree the unchanged figure read), bands = F/bands.json
#   primary:   ablation metrics = realref_abl_crc33sec (same tiles, PQ / DICE / AJI / IoU / accuracy recomputed against
#              the primary reference, every other metric as scored), bands = Vanilla of realref_fig2_crc33sec cache.json
# Panels A-C (schematic, loss, delta LPIPS) and the F / H strips read the same inputs in both. Check: the secondary
# render must equal the unchanged PNG pixel for pixel outside the label strip above D. New folder W only; F is read.
# Run 1 (job 43429730, W = hfv_fig2ref_20261120) hung in the S3 download (aws s3 cp of a 7 MB tar at 0 % CPU for 37 min;
# the same copy from that node took 9 s afterwards); cancelled, rerun with FIG2REF_W=...20261120b (new folder):
#   sbatch --export=ALL,FIG2REF_W=/gscratch/cheme/chiu/hfv_fig2ref_20261120b -o .../hfv_fig2ref_20261120b/logs/%x-%j.out ...
set -uo pipefail
F=/gscratch/cheme/chiu/hfv_figs_20261118; R=$F/PixCell; I=$F/inputs; C=/gscratch/cheme/chiu/crc16-20261024/cellvit
W=${FIG2REF_W:-/gscratch/cheme/chiu/hfv_fig2ref_20261120}; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
export MPLBACKEND=Agg XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl PYTHONPATH=$R:$F/pylib
PY=$C/env/bin/python
module load coenv/aws/2.17.32
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
mkdir -p $W/in $W/out $W/cache/mpl
for t in realref_abl_crc33sec realref_fig2_crc33sec; do
  [ -d $W/in/${t#realref_} ] || aws s3 cp --only-show-errors $S/outputs/$t.tar - | tar -x -C $W/in || { echo "STOP input $t"; exit 1; }
done
n=$(find $W/in/abl_crc33sec -name metrics.json | wc -l); m=$(find $I/score/abl_crc33sec -name metrics.json | wc -l)
echo "metrics.json primary $n secondary $m"; [ "$n" = "$m" ] || { echo "STOP tile count differs"; exit 1; }
$PY $W/figrender.py guard $R || exit 1
$PY $W/figrender.py bands $W/in/fig2_crc33sec/cache.json $W/bands_primary.json || exit 1
cd $R
common="--images-root $I/score/abl_crc33sec --orion-root $I/eval/orion-crc33sec \
  --svg $R/figures/pngs_updated/methods/stage_2_svg.svg --cache $I/score/fig2_crc33sec/cache.json \
  --mapping $I/score/abl_crc33sec/unpaired_ablation/metadata/unpaired_mapping.json"
fails=0
$PY -u $W/fig2_refvariant.py $common --abl-root $I/score/abl_crc33sec --bands $F/bands.json --reference secondary \
  --out $W/out/fig2_architecture_performance_ref-secondary-lin.png || fails=$((fails+1))
$PY -u $W/fig2_refvariant.py $common --abl-root $W/in/abl_crc33sec --bands $W/bands_primary.json --reference primary \
  --out $W/out/fig2_architecture_performance_ref-primary-cellvit.png || fails=$((fails+1))
cp $R/figures/pngs_updated/concat/fig2_architecture_performance.png $W/out/fig2_architecture_performance_unchanged.png
$PY - $W/out <<'EOF' || fails=$((fails+1))
import sys, numpy as np
from PIL import Image
d = sys.argv[1]
u = np.array(Image.open(f"{d}/fig2_architecture_performance_unchanged.png").convert("RGB")).astype(int)
s = np.array(Image.open(f"{d}/fig2_architecture_performance_ref-secondary-lin.png").convert("RGB")).astype(int)
p = np.array(Image.open(f"{d}/fig2_architecture_performance_ref-primary-cellvit.png").convert("RGB")).astype(int)
print("shapes unchanged", u.shape, "secondary", s.shape, "primary", p.shape)
if u.shape != s.shape:
    print("CHECK FAILED: secondary render size differs from the unchanged PNG"); sys.exit(1)
rows = np.flatnonzero((u != s).any(axis=(1, 2)))
print("secondary vs unchanged: differing rows", len(rows), "range", (rows.min(), rows.max()) if len(rows) else None)
if p.shape == u.shape:
    pr = np.flatnonzero((u != p).any(axis=(1, 2)))
    print("primary vs unchanged: differing rows", len(pr), "range", (pr.min(), pr.max()) if len(pr) else None)
EOF
aws s3 sync --only-show-errors $W/out $S/figures_rerender/fig2_reference_variants/ && \
  aws s3 cp --only-show-errors $W/bands_primary.json $S/figures_rerender/fig2_reference_variants/bands_primary.json && echo "upload ok" || fails=$((fails+1))
echo "### $(date -Is) done, $fails failures"
exit $([ $fails -eq 0 ] && echo 0 || echo 1)
