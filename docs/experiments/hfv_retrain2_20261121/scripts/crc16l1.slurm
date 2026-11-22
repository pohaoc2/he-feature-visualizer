#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --job-name=r2-crc16l1
#SBATCH --output=/gscratch/cheme/chiu/hfv_retrain2_20261121/logs/%x-%j.out
# hfv_retrain2_20261121: CRC16 L1 eval inputs on the app 8 tiles (12 arm-B windows w220-w231, 192 central tiles).
#   1 msi_s3b_render.py render_l1 (sim-exp-validation copy in scripts/): L1 ch0-6 + L1 instance map, method C ch7-9 byte-copied
#     from app 8 kin/roots/orion-crc16e; gate: default render_tile == s3b_crc16b/render (byte-identical default path)
#   2 cells.py L1 (the training builder) on the same 192 tiles; gate: its PNG bytes == step 1 PNG bytes
# Output: R/eval/crc16_L1 (new). Any failure is a STOP: no retry, no requeue.
set -uo pipefail
R=/gscratch/cheme/chiu/hfv_retrain2_20261121; CC=/gscratch/cheme/chiu; C=$CC/crc16-20261024; A=$CC/he_celltype_audit_20261118
S=$CC/msi_perturb_20261119/s3b_crc16b; ENV=$CC/sim2real_apps_20261120/app8/kin/roots/orion-crc16e
O=$R/eval/crc16_L1; X=$R/eval/crc16_L1_cellspy; MASK=$C/p1t_prior_20261106/c723/lin/P37_S48-CRC16-mask.ome.tif
PY=$C/cellvit/env/bin/python
stop() { echo "$(date -Is) STOP $*"; exit 1; }
test -e $O && stop "$O exists"; test -e $X && stop "$X exists"
mkdir -p $R/eval
echo "### $(date -Is) job $SLURM_JOB_ID $(hostname) sev $(cat $R/sev3/COMMIT)"; sha256sum $R/scripts/msi_s3b_render.py $R/scripts/cells.py
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
S3B_L1=1 S3B_HFV=$CC/hfv_lintype_20261118/hfv S3B_SEV=$R/sev3/src PYTHONPATH=$A/pylib:$CC/slidesel-20261023/pylib:$C/pylib \
  $PY -u $R/scripts/msi_s3b_render.py render_l1 CRC16 $S $O $ENV 12 || stop render_l1
echo "### $(date -Is) render_l1 done"
HFV=$CC/hfv_lintype_20261118/hfv SEV_SRC=$R/sev3/src $PY -u $R/scripts/cells.py L1 crc16 $MASK $S none $O/tiles.txt exp_channels "" $X 12 \
  || stop cells.py
$PY - $O $X/cells_crc16_L1.tar <<'PYEOF' || stop "builder cross-check"
import sys, tarfile, json
from pathlib import Path
O, tar = Path(sys.argv[1]), sys.argv[2]
bad, n = [], 0
with tarfile.open(tar) as tf:
    for m in tf:
        if m.isfile():
            n += 1
            if tf.extractfile(m).read() != (O / m.name).read_bytes():
                bad.append(m.name)
tiles = (O / "tiles.txt").read_text().split()
r = dict(cellspy_members=n, expected=7 * len(tiles), png_bytes_differ=len(bad), examples=bad[:5])
(O / "crosscheck_cellspy.json").write_text(json.dumps(r, indent=1) + "\n")
print(r)
sys.exit(1 if bad or n != 7 * len(tiles) else 0)
PYEOF
sha256sum $O/manifest.json $O/render_l1.json
echo "### $(date -Is) DONE"
