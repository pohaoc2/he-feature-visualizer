#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/gscratch/cheme/chiu/hfv_crc34_20261118/logs/%x-%j.out
# crc34_render_20261118: vasculature / oxygen / glucose of the CRC34 kept tiles (tiles_crc34.json minus ink) with the mx
# agent's code (crc36_render_20261118/mx mx_channels.py + mx_lib.py, byte copies; md5 checked below against the mx
# agent's Hyak scripts) and the hfv code the mx agent imports (hfv_mx_20261118/code, copied to $N/mxcode).
# Parameters are NOT typed here: they are read from the CRC36 run's own summary (hfv_crc36_20261118/out/mx_channels.json
# "params"), so CRC34 uses exactly what CRC36 used. STOP if that summary does not exist yet.
# The mx agent's venv is used read only (no bytecode written); MPLCONFIGDIR is in this dir.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
W=/gscratch/cheme/chiu/hfv_mx_20261118      # mx agent: read only
N=/gscratch/cheme/chiu/hfv_crc34_20261118
S36=/gscratch/cheme/chiu/hfv_crc36_20261118/out/mx_channels.json
echo "== $(date -Is) $(hostname)"
test -s $S36 || { echo "STOP CRC36 mx summary missing ($S36)"; exit 1; }
(cd $W/scripts && md5sum mx_lib.py mx_channels.py) | (cd $N/mx && md5sum -c -)
mkdir -p $N/mxcode $N/tmp_mpl
if [ ! -e $N/mxcode/HFV_COMMIT ]; then cp -a $W/code/. $N/mxcode/; fi
(cd $W/code && find . -type f -not -path "*__pycache__*" | sort | xargs md5sum) | (cd $N/mxcode && md5sum -c --quiet -)
read DS O2L GLCL SMA < <($W/venv/bin/python -c "import json;p=json.load(open('$S36'))['params'];print(p['ds'],p['oxygen_L_um'],p['glucose_L_um'],'--sma-refine' if p['sma_refine'] else '-')")
[ "$SMA" = - ] && SMA=""
echo "params from CRC36 summary: ds=$DS oxygen_L_um=$O2L glucose_L_um=$GLCL sma=${SMA:-off}"
export HFV=$N/mxcode MPLCONFIGDIR=$N/tmp_mpl
$W/venv/bin/python -u $N/mx/mx_channels.py \
  --mx-tiff $N/lin/data__CRC34__P37_S77_Full_A24_C59qX_E15_20220302_235222_359806-zlib.ome.tiff \
  --markers $N/lin/data__CRC34__markers.csv --tiles $N/out/tiles_crc34.json --max-ink 0.0 \
  --out-tar $N/data/crc34_mx_channels.tar --prefix orion-crc34/exp_channels --summary $N/out/mx_channels.json \
  --ds $DS --o2-L-um $O2L --glc-L-um $GLCL $SMA --workers 16 --expect-mpp 0.325
$W/venv/bin/python - <<EOF
import json
a = json.load(open("$S36"))["params"]; b = json.load(open("$N/out/mx_channels.json"))["params"]
keys = [k for k in a if k != "coarse_shape"]
diff = {k: (a[k], b.get(k)) for k in keys if a[k] != b.get(k)}
print("params vs CRC36 (coarse_shape excluded, slide size):", "identical" if not diff else diff)
assert not diff, "STOP params differ from CRC36"
EOF
echo "== $(date -Is) done"
