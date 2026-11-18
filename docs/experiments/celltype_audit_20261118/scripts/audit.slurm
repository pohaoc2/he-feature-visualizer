#!/bin/bash
#SBATCH --account=cheme
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/gscratch/cheme/chiu/he_celltype_audit_20261118/logs/%x-%j.out
# celltype_audit_20261118 steps 2-4 (descriptive). Exits on the first failed gate (STOP, no retry).
#  2 gates : C33 column of Lin's sumAllsample table (lin_sumall_col.py, value-matched like C34), then lin_gate.py
#            (Lin CycIF_findcutoff GMM, FDR picked against Lin's published C33 positive fraction, quantile fallback)
#            on CRC33_01 + CRC33_02 pooled (C33 = pooled pieces); identity check (CSV means vs C33 means) reported for CRC33_01 and CRC33_02.
#  3 lookup: lookup.py (G-frame, G-id, mask matching, merge counts, ink gray values).
#  4 audit : audit.py -> out/audit.json.
set -euo pipefail
W=/gscratch/cheme/chiu/he_celltype_audit_20261118; S=$W/scripts; O=$W/out
LR=/gscratch/cheme/chiu/crc16-20261024/lin_reinit_20261106          # read only
PYC=/gscratch/cheme/chiu/crc16-20261024/cellvit/env/bin/python     # tifffile, cv2, h5py
PYG=/gscratch/cheme/chiu/cell-model-fit-20261021/envsmc/bin/python # pandas, sklearn
INK=/gscratch/cheme/chiu/crc16-20261024/darkdiag_20261031/ink.json  # read only
export PYTHONDONTWRITEBYTECODE=1
echo "== $(date -Is) $(hostname)"
if [ ! -s $O/lin_gates_crc33.json ]; then
  $PYC -u $S/lin_sumall_col.py $LR/matlab-sumAllsample-20220710.mat $LR/lin_sumall.json $O/lin_sumall_crc33.json C33 crc33
  LIN_SLIDE=crc33 $PYG -u $S/lin_gate.py $W/lin/CRC33_01__P37_S76_01-CRC33_01.csv,$W/lin/CRC33_02__P37_S76_02-CRC33_02.csv $O/lin_sumall_crc33.json $O/lin_gates_crc33.json > $O/lin_gate.log
fi
$PYG - $W <<'PY'
import json, sys, pandas as pd
w = sys.argv[1]; c = json.load(open(f"{w}/out/lin_sumall_crc33.json"))["cols"]
M = {"Area": "mean_Area", "Hoechst": "mean_Hoechst", "Pan-CK": "mean_Pan_CK", "CD45": "mean_CD45", "Ki67": "mean_Ki67", "E-cadherin": "mean_E_cadherin"}
out = {}
for piece, f in [("CRC33_01", "CRC33_01__P37_S76_01-CRC33_01.csv"), ("CRC33_02", "CRC33_02__P37_S76_02-CRC33_02.csv")]:
    d = pd.read_csv(f"{w}/lin/{f}", usecols=list(M))
    out[piece] = dict(n=len(d), rel_diff={m: round(float(d[m].mean() / c[k]["crc33"] - 1), 4) for m, k in M.items()})
json.dump(out, open(f"{w}/out/identity_c33.json", "w"), indent=1); print("IDENTITY", json.dumps(out))
PY
[ -s $O/lookup.npz ] || PYTHONPATH=$W/pylib $PYC -u $S/lookup.py $W   # pylib: imagecodecs 2026.5.10 (--no-deps; H&E is JPEG-tiled)
$PYG -u $S/audit.py $W $INK $O/audit.json
$PYG -u $S/chancheck.py $W $INK $O/chancheck.json
echo "== $(date -Is) done"
