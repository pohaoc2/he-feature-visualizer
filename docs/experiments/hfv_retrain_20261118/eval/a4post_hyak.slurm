#!/bin/bash
#SBATCH --job-name=ev-a4post
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_a4post_20261120/logs/%x-%j.out
# hfv_retrain_20261118 follow-up 1 (coordinator 2026-11-20): after the two Colab L4 shards (eval/boot_a4.sh + a4run.py,
# outputs/a4gen_crc33a4_shard{1,2}.tar), on the CPU, PixCell cb62fa7 unchanged:
#   1. merge the two shard folders (disjoint tiles; metrics.shard_01of02.csv / _02of02.csv per attribute) and concatenate
#      them into sweep/<attr>/metrics.csv with edit._aggregate_metrics (what run_sweep does at the end);
#   2. postprocess refresh-metrics (CellViT sidecars -> morpho.*, appearance recomputed, slope_summary.json), cwd = a dir
#      whose a4out -> the merged group, because the image paths in metrics.csv are relative (a4out/a4_uni_probe/...);
#   3. new results folder N = W/a4_rerun: sweep/ (merged), he/<tile>.png (the pack's real H&E of the 28 tiles, for the
#      Ref H&E row of panel F), probe_results.csv / .json + probe_directions/ copied from the folder the unchanged Fig 3
#      render was pointed at (F/inputs/old/results/a4_uni_probe = S3 share_space/05_results/results/a4_uni_probe), so
#      panel D is the same as before; then main.py specificity --out-dir N (specificity_full.csv,
#      specificity_morphology_summary.csv);
#   4. Fig 3: copy of the figs checkout F/PixCell (36 MB, its links kept) in W/fig3 with results/a4_uni_probe -> N and a
#      plain share-space-paper stand-in, then figrender.py run ... fig_combined_uni_decomp_v2 (as figs_hyak.slurm);
#   5. upload N -> S3 a4_rerun/, the PNG -> figures_rerender/fig3_a4_rerun/.
# Checks (STOP on failure): 1,008 PNG and 1,008 sidecars, 168 rows per attribute (28 tiles x 2 directions x 3 alphas),
# every row's sidecar found by refresh (no NaN morpho.nuclei_density). F and the shard tars are only read; new dir W.
set -uo pipefail
W=/gscratch/cheme/chiu/hfv_a4post_20261120; F=/gscratch/cheme/chiu/hfv_figs_20261118; I=$F/inputs
PY=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python; S=s3://bagherilab-working/pohao/hfv_retrain_20261118
export MPLBACKEND=Agg XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
module load coenv/aws/2.17.32
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname)"
stop() { echo "STOP $*"; exit 1; }
mkdir -p $W/cache/mpl $W/merge $W/run
P=$W/PixCell
[ -f $P/ok ] || { rm -rf $P; mkdir -p $P && aws s3 cp --only-show-errors $S/inputs/pixcell_cb62fa7.tar.gz - | tar -xz -C $P && touch $P/ok; } || stop PixCell
for k in 1 2; do
  [ -d $W/merge/s$k/a4_crc33a4 ] || { mkdir -p $W/merge/s$k && aws s3 cp --only-show-errors $S/outputs/a4gen_crc33a4_shard$k.tar - | tar -x -C $W/merge/s$k; } || stop "shard $k"
  cat $W/merge/s$k/a4_crc33a4/gpu.txt $W/merge/s$k/a4_crc33a4/speed_shard$k.json; echo
done
G=$W/merge/a4_crc33a4; rm -rf $G; mkdir -p $G
cp -a $W/merge/s1/a4_crc33a4/a4_uni_probe $G/
( cd $W/merge/s2/a4_crc33a4/a4_uni_probe && find sweep -name "metrics.shard_*" -o -path "sweep/*/*/*" -type f ) | while read -r f; do
  [ -e $G/a4_uni_probe/$f ] && { echo "STOP overlap $f"; exit 1; }; mkdir -p $(dirname $G/a4_uni_probe/$f); cp -a $W/merge/s2/a4_crc33a4/a4_uni_probe/$f $G/a4_uni_probe/$f; done || stop merge
O=$G/a4_uni_probe
np=$(find $O/sweep -name "alpha_*.png" | wc -l); ns=$(find $O/sweep -name "alpha_*.png.json" | wc -l)
echo "merged PNG $np sidecars $ns"; [ $np = 1008 ] && [ $ns = 1008 ] || stop "expected 1008 PNG and 1008 sidecars"
ln -sfn $G $W/run/a4out
cd $W/run
export PYTHONPATH=$P
$PY - $O <<'EOF' || stop aggregate
import sys, csv
from pathlib import Path
import src.a4_uni_probe.edit as edit
O = Path(sys.argv[1])
for d in sorted((O / "sweep").iterdir()):
    sh = sorted(d.glob("metrics.shard_*of*.csv"))
    m = edit._aggregate_metrics(d)
    rows = list(csv.DictReader(open(m)))
    tiles = {r["tile_id"] for r in rows}
    print(d.name, [s.name for s in sh], len(rows), "rows", len(tiles), "tiles")
    if len(sh) != 2 or len(rows) != 168 or len(tiles) != 28:
        sys.exit(f"STOP {d.name}: {len(sh)} shard files, {len(rows)} rows, {len(tiles)} tiles")
EOF
$PY -u -m src.a4_uni_probe.postprocess refresh-metrics --out-dir a4out/a4_uni_probe || stop refresh-metrics
$PY - $O <<'EOF' || stop "refresh check"
import sys, csv, math
from pathlib import Path
bad = 0
for m in sorted(Path(sys.argv[1]).glob("sweep/*/metrics.csv")):
    rows = list(csv.DictReader(open(m)))
    nan = sum(1 for r in rows if not r.get("morpho.nuclei_density") or math.isnan(float(r["morpho.nuclei_density"])))
    print(m.parent.name, len(rows), "rows, NaN morpho", nan, "| image_path e.g.", rows[0]["image_path"])
    bad += nan
sys.exit(1 if bad else 0)
EOF
N=$W/a4_rerun; rm -rf $N; mkdir -p $N/he
cp -a $O/sweep $N/
OLD=$I/old/results/a4_uni_probe
cp -a $OLD/probe_results.csv $OLD/probe_results.json $OLD/probe_directions $N/
md5sum $OLD/probe_results.csv $O/probe_results.csv
for f in $O/probe_directions/*.npy; do cmp -s $f $OLD/probe_directions/$(basename $f) || echo "probe direction differs from the D folder: $(basename $f)"; done
ids=$(ls $O/sweep/eccentricity_mean | grep -v '\.' )
[ $(echo "$ids" | wc -l) = 28 ] || stop "tile list"
aws s3 cp --only-show-errors $S/inputs/eval/evalpack_crc33a4.tar - | tar -x -C $W/merge orion-crc33a4/he || stop "pack he"
for t in $ids; do cp $W/merge/orion-crc33a4/he/$t.png $N/he/ || stop "he $t"; done
cd $P && $PY -u -m src.a4_uni_probe.main specificity --out-dir $N || stop specificity
ls -la $N; head -3 $N/specificity_full.csv
# ---- Fig 3
X=$W/fig3; rm -rf $X; mkdir -p $X/share-space-paper && cp -a $F/PixCell $X/ || stop "fig3 checkout"
R=$X/PixCell; rm -f $R/figures/pngs_updated/concat/fig3_uni_decomposition_v2.png
ln -sfn $N $R/results/a4_uni_probe
export PYTHONPATH=$R:$F/pylib
$PY $F/scripts/figrender.py guard $R || stop guard
(cd $R && $PY -u $F/scripts/figrender.py run $R src.paper_figures.fig_combined_uni_decomp_v2) > $W/logs/step_fig3.log 2>&1 || { tail -20 $W/logs/step_fig3.log; stop "fig3 render"; }
PNG=$R/figures/pngs_updated/concat/fig3_uni_decomposition_v2.png; [ -f $PNG ] || stop "fig3 png missing"
$PY -c "from PIL import Image; print('fig3', Image.open('$PNG').size)"
aws s3 sync --only-show-errors $N $S/a4_rerun/ && aws s3 cp --only-show-errors $PNG $S/figures_rerender/fig3_a4_rerun/fig3_uni_decomposition_v2.png \
  && aws s3 cp --only-show-errors $W/logs/step_fig3.log $S/figures_rerender/fig3_a4_rerun/step_fig3.log || stop upload
echo "### $(date -Is) done"
