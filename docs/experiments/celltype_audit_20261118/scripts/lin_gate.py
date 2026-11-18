# prereg_lin_reinit_20261106 §A2: per-marker GMM gates (Lin 2023 CycIF_findcutoff, FDR grid) on the whole-slide Lin CSV,
# FDR picked per marker against Lin's published CRC16 positive fraction (orion-crc sumAllsample table), quantile fallback.
#   lin_gate.py CSV LIN_SUMALL.json OUT.json
#   LIN_SLIDE=crc34 (§C5; default crc16) picks the sumall column the FDR is matched to.
# celltype_audit_20261118 copy, one change: CSV may be a comma-separated list of pieces that are pooled before gating
# (CRC33: Lin's C33 row equals the cell-weighted pool of CRC33_01 and CRC33_02, identity_c33.json), and the gates are
# also reported per piece (positive fraction of each piece at the pooled gate).
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import findcutoff as F

SEED = 20261106
SL = os.environ.get("LIN_SLIDE", "crc16")
FDRS = [0.01, 0.02, 0.05, 0.10, 0.20]
TYPING = ["Pan-CK", "E-cadherin", "CD45", "Ki67", "CD31", "SMA"]
OTHER = ["CD3e", "CD20", "CD68", "CD163", "CD4", "CD8a", "CD45RO", "FOXP3", "PD-1", "PD-L1"]
LIN = {"Pan-CK": "Pan_CK", "E-cadherin": "E_cadherin", "PD-1": "PD_1", "PD-L1": "PD_L1", "CD3e": "CD3e", "CD8a": "CD8a"}


def ok(f, ref):
    return abs(f - ref) <= max(0.02, 0.2 * ref)


def main(csv, sumall, out):
    cols = json.load(open(sumall))["cols"]
    parts = [pd.read_csv(c, usecols=["CellID", "Hoechst", "Area"] + TYPING + OTHER).assign(piece=i)
             for i, c in enumerate(csv.split(","))]
    df = pd.concat(parts, ignore_index=True)
    print("cells", len(df), flush=True)
    ident = {m: dict(csv=float(df[m].mean()), lin=cols[f"mean_{LIN.get(m, m).replace('-', '_')}"][SL])
             for m in ["Area", "Hoechst", "Pan-CK", "CD45", "Ki67", "E-cadherin"]}
    for v in ident.values():
        v["rel_diff"] = v["csv"] / v["lin"] - 1
    res = {}
    for m in TYPING + OTHER:
        ref = cols[f"mean_{LIN.get(m, m).replace('-', '_')}p"][SL]
        v = df[m].to_numpy(float)
        grid = {}
        for fdr in FDRS:
            g, pos = F.visualgate_shu(v, fdr, k=2, n_sample=50000, seed=SEED)
            grid[str(fdr)] = dict(gate_log=g, frac=float(pos.mean()))
            print(m, fdr, round(g, 3), round(float(pos.mean()), 4), "lin", ref, flush=True)
        best = min(grid, key=lambda k: abs(grid[k]["frac"] - ref))
        q = float(np.quantile(v, 1 - ref))
        r = dict(lin_frac=ref, grid=grid, best_fdr=float(best), best_frac=grid[best]["frac"], best_ok=ok(grid[best]["frac"], ref),
                 quantile_thr=q, typing=m in TYPING)
        r["final_thr"] = float(np.exp(grid[best]["gate_log"])) if r["best_ok"] else q
        r["final_source"] = "gmm" if r["best_ok"] else "quantile_fallback"
        r["final_frac"] = float((v > r["final_thr"]).mean())
        r["final_frac_by_piece"] = [float((v[df.piece.to_numpy() == i] > r["final_thr"]).mean()) for i in range(len(parts))]
        res[m] = r
    P = {m: df[m].to_numpy(float) > res[m]["final_thr"] for m in TYPING}
    imm = P["CD45"]; epi = P["Pan-CK"] | P["E-cadherin"]
    slide = dict(immune=float(imm.mean()), cancer=float((epi & ~imm).mean()), healthy=float((~imm & ~epi).mean()),
                 double_pos=float((imm & epi).mean()), prolif=float(P["Ki67"].mean()),
                 cancer_prolif=float(P["Ki67"][epi & ~imm].mean()))
    gate_ok = all(res[m]["best_ok"] or res[m]["final_source"] == "quantile_fallback" for m in TYPING)
    json.dump(dict(seed=SEED, fdr_grid=FDRS, n_cells=len(df), identity=ident, markers=res, slide_types=slide,
                   G_gate=dict(ok=gate_ok, gmm_pass=[m for m in TYPING if res[m]["best_ok"]],
                               fallback=[m for m in TYPING if not res[m]["best_ok"]])), open(out, "w"), indent=1)
    print(json.dumps(dict(identity=ident, slide=slide, fallback=[m for m in TYPING if not res[m]["best_ok"]]), indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
