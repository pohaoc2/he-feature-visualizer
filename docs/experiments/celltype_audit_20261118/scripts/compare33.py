"""celltype_audit_20261118 §4 (envsmc; descriptive): CELESTA and Astir on CRC33_01 vs the Lin-gate rules and hfv.
Class mapping identical to sim-exp-validation scripts/analysis/tcmvp/celltype_pub_compare.py (celltype_pub §1.5):
raw class E (Epithelial), S (Endothelial, SMA+ stromal; for the rules: non-epithelial non-immune), I (any immune type),
U (CELESTA 'Unknown'; Astir 'Unknown' + 'Other'); population: E -> cancer except in Lin NORM -> healthy, S -> healthy,
I -> immune, U kept apart ('unknown'), then merged either way (U -> healthy, U -> cancer) for kappa.
Scopes: 'tiles' = Lin cells inside the 10,379 training tiles (as audit.py); 'slide' = all CRC33_01 cells.
Bootstrap: 200 resamples of 1024-px squares (as celltype_pub_compare.py).
  compare33.py W OUT_JSON
"""
import glob
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image

W, OUT = sys.argv[1:3]
sys.path.insert(0, f"{W}/scripts")
from tcfix_rule import classify_fix  # noqa: E402

MK = ["Pan-CK", "E-cadherin", "CD45", "SMA", "CD31", "Ki67"]
GROUP = {0: "other", 1: "NORM", 2: "other", 3: "TUM", 4: "other", 5: "other", 6: "other", 7: "STR", 8: "LYM"}
E, S, I, U = 0, 1, 2, 3
H, C, IM, UN = 0, 1, 2, 3
POPN = ["healthy", "cancer", "immune", "unknown"]
RAWN = ["E", "S", "I", "U"]
CEL_RAW = {"Epithelial": E, "Endothelial": S, "SMA+ stromal": S, "Unknown": U}
AST_RAW = {"Epithelial": E, "Endothelial": S, "SMA+ stromal": S, "Unknown": U, "Other": U}
NB = 200
rng = np.random.default_rng(20261118)


def to_raw(names, table):
    return np.array([table.get(n, I) for n in names], np.int8)


def pop_of(raw, k):
    p = np.select([raw == E, raw == S, raw == I], [C, H, IM], UN).astype(np.int8)
    p[(raw == E) & (k == 1)] = H
    return p


def kappa(cm):
    n = cm.sum()
    if n == 0:
        return np.nan
    po = np.trace(cm) / n
    pe = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def merge(cm, to):
    M = np.zeros((4, 3)); M[0, 0] = M[1, 1] = M[2, 2] = 1; M[3, to] = 1
    return M.T @ cm @ M


def frac(v, n):
    return {n[i]: round(float((v == i).mean()), 4) for i in range(len(n))} if len(v) else {}


G = json.load(open(f"{W}/out/lin_gates_crc33.json"))["markers"]
thr = {m: G[m]["final_thr"] for m in MK}
cmap = np.array(Image.open(f"{W}/lin/CRC33_01_lin_he_class.tiff"))
P = 112.0 / 0.325
df = pd.read_csv(f"{W}/lin/CRC33_01__P37_S76_01-CRC33_01.csv", usecols=["CellID", "X_centroid", "Y_centroid", "Area"] + MK)
x, y = df.X_centroid.to_numpy(), df.Y_centroid.to_numpy()
k = cmap[np.clip((y / P).astype(int), 0, cmap.shape[0] - 1), np.clip((x / P).astype(int), 0, cmap.shape[1] - 1)]
grp = pd.Series(k).map(GROUP).to_numpy()
mean = {m: df[m].to_numpy(float) for m in MK}
ty0, tya, fixed = classify_fix(mean, thr, k)
raw = {"old": np.select([ty0 == 1, ty0 == 2], [E, I], S).astype(np.int8)}
raw["rule_a"] = raw["old"].copy(); raw["rule_a"][fixed] = E
TYP = ["old", "rule_a"]
names = {}
for t, pat, col, tab in [("celesta", "celesta33/out/tile_*.csv", "final", CEL_RAW),
                         ("celesta_whole", "celesta33_whole/celesta_whole.csv", "final", CEL_RAW),
                         ("astir", "astir33/astir_types.csv", "astir", AST_RAW)]:
    fs = sorted(glob.glob(f"{W}/{pat}"))
    if not fs:
        print("missing", t, flush=True)
        continue
    d = pd.concat([pd.read_csv(f) for f in fs]).set_index("CellID").reindex(df.CellID)
    assert d[col].notna().all(), f"{t} misses cells"
    names[t] = d[col].to_numpy()
    raw[t] = to_raw(names[t], tab)
    TYP.append(t)
pop = {t: pop_of(raw[t], k) for t in TYP}
assert (pop["rule_a"] == tya).all()
# hfv label of each Lin cell: hfv object whose centroid lies on this Lin cell (first one if several)
lk = np.load(f"{W}/out/lookup.npz")
assert (df.CellID.to_numpy() == lk["lin_id"]).all()
hl = pd.read_csv(f"{W}/raw/processed_crc33/cell_assignments.csv", usecols=["cell_type"]).cell_type.to_numpy()
hpop = pd.Series(hl).map({"healthy": H, "cancer": C, "immune": IM}).to_numpy()
lab = lk["hfv_label"]
pos_of = pd.Series(np.arange(len(df)), index=df.CellID.to_numpy())
hfv_on = np.full(len(df), -1, np.int8)
ok = lab > 0
li = pos_of.reindex(lab[ok]).to_numpy()
good = ~np.isnan(li)
hfv_on[li[good].astype(int)[::-1]] = hpop[ok][good][::-1]   # first hfv object wins
# training tiles
idx = json.load(open(f"{W}/raw/processed_crc33/index.json"))
ps = int(idx["patch_size"]); assert int(idx["stride"]) == ps
tiles = {(int(p["x0"]), int(p["y0"])) for p in idx["patches"]}
in_t = np.array([(a, b) in tiles for a, b in zip((x // ps).astype(int) * ps, (y // ps).astype(int) * ps)])
sq = (y // 1024).astype(np.int64) * 100000 + (x // 1024).astype(np.int64)
area = df.Area.to_numpy() * 0.325 ** 2
ki = mean["Ki67"] > thr["Ki67"]
R = dict(n_cells=int(len(df)), n_in_tiles=int(in_t.sum()), typings=TYP,
         method_counts={t: pd.Series(names[t]).value_counts().to_dict() for t in names})


def desc(m):
    return dict(n=int(m.sum()), area_um2=round(float(np.median(area[m])), 1) if m.any() else None,
                panck=round(float(np.median(mean["Pan-CK"][m])), 1) if m.any() else None,
                ki67p=round(float(ki[m].mean()), 4) if m.any() else None)


for scope, sm in [("tiles", in_t), ("slide", np.ones(len(df), bool))]:
    Sx = {}
    for g in ["NORM", "TUM", "STR", "LYM", "other", "all"]:
        m = sm & ((grp == g) if g != "all" else True)
        T = dict(n=int(m.sum()))
        for t in TYP:
            T[t] = dict(pop=frac(pop[t][m], POPN), raw=frac(raw[t][m], RAWN))
        if g in ("TUM", "NORM", "all"):
            usq, inv = np.unique(sq[m], return_inverse=True)
            Wb = np.stack([np.bincount(rng.integers(0, len(usq), len(usq)), minlength=len(usq)) for _ in range(NB)])
            T["kappa"] = {}
            for i, a in enumerate(TYP):
                for b in TYP[i + 1:]:
                    code = pop[a][m].astype(np.int64) * 4 + pop[b][m]
                    per = np.zeros((len(usq), 16)); np.add.at(per, (inv, code), 1)
                    tot = per.sum(0).reshape(4, 4); boot = (Wb @ per).reshape(NB, 4, 4)
                    kk = {}
                    for nm, to in (("U_to_healthy", H), ("U_to_cancer", C)):
                        kb = np.array([kappa(merge(cb, to)) for cb in boot])
                        kk[nm] = [round(float(kappa(merge(tot, to))), 4)] + np.nanpercentile(kb, [2.5, 97.5]).round(4).tolist()
                    T["kappa"][f"{a}~{b}"] = kk
        Sx[g] = T
    # cells rule (a) moved (TUM, non-immune, non-epithelial, not SMA/CD31+): what do CELESTA / Astir call them?
    mv = sm & fixed
    Sx["rule_a_moved"] = dict(n=int(mv.sum()), descriptors=desc(mv),
                              **{t: dict(raw=frac(raw[t][mv], RAWN),
                                         by_raw={RAWN[r]: desc(mv & (raw[t] == r)) for r in range(4)},
                                         names=pd.Series(names[t][mv]).value_counts(normalize=True).round(4).to_dict())
                                 for t in names})
    # TUM epithelium by the gates (reference group) and NORM epithelium
    for nm, mm in [("TUM_gate_epithelial", sm & (k == 3) & (ty0 == 1)), ("NORM_gate_epithelial", sm & (k == 1) & (ty0 == 1))]:
        Sx[nm] = dict(descriptors=desc(mm), **{t: frac(raw[t][mm], RAWN) for t in names})
    # agreement with hfv (Lin cells that carry an hfv object centroid)
    hm = sm & (hfv_on >= 0)
    Sx["hfv_agreement"] = dict(n=int(hm.sum()), **{
        t: {g: round(float((pop[t][hm & ((grp == g) if g != "all" else True)] ==
                            hfv_on[hm & ((grp == g) if g != "all" else True)]).mean()), 4)
            for g in ["NORM", "TUM", "STR", "LYM", "all"]} for t in TYP})
    R[scope] = Sx
    print(scope, json.dumps({g: {t: Sx[g][t]["pop"] for t in TYP} for g in ["NORM", "TUM", "all"]}), flush=True)
    print(scope, "moved", json.dumps(Sx["rule_a_moved"])[:1500], flush=True)
json.dump(R, open(OUT, "w"), indent=1)
