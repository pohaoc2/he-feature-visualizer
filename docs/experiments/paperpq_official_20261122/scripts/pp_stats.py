"""paperpq_official_20261122 (PREREG.md §8-9): paired tile bootstrap, gate, ranking rule. Runs locally on data/.

  python pp_stats.py DATA
DATA/per_tile.csv (pp_score.py), DATA/appear/<root>/{appear_per_tile.csv, fud.json, DONE} (pp_gen.py).
Bootstrap: 10,000 resamples of the 100 tiles, numpy default_rng(42), one index matrix for every generator, pipeline and
readout (paired); 95 % percentile intervals. Writes DATA/stats.json.
"""
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

DATA = Path(sys.argv[1])
B, SEED = 10_000, 42
NAME = {"a2_off_shelf": "PixCell off-the-shelf", "a1_concat": "Concat", "production": "Grouped",
        "a2_bypass_full_tme": "Grouped + bypass", "a1_per_channel": "Per-channel", "real": "real H&E"}
PUBLISHED = {"a2_off_shelf": 0.6987, "a1_concat": 0.7817, "a2_bypass_full_tme": 0.7943, "a1_per_channel": 0.0106}
PUB_GROUPED = 0.5694
READ = ["pq", "pq_inst", "dice", "a", "b", "c", "pq_x", "n_ref", "n_pred", "lpips", "style_hed"]
PRIMARY = ["a2_off_shelf", "a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel"]
SECONDARY = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel"]

rows = list(csv.DictReader(open(DATA / "per_tile.csv")))
ids = sorted({r["tile"] for r in rows})
pos = {t: i for i, t in enumerate(ids)}
assert len(ids) == 100
# arm key = (family, pipeline, variant); family P (CellViT-mask generators + off-the-shelf + real) or R (shape-library)
V = {}
for r in rows:
    fam = r["root"][0]
    k = (fam, r["pipeline"], r["variant"])
    V.setdefault(k, {q: np.full(100, np.nan) for q in READ})
    for q in READ[:9]:
        V[k][q][pos[r["tile"]]] = float(r[q])
fud = {}
for root in ("P_run", "P_off", "R_run", "R_off"):
    fam, pipe = root.split("_")
    for r in csv.DictReader(open(DATA / "appear" / root / "appear_per_tile.csv")):
        for q in ("lpips", "style_hed"):
            V[(fam, pipe, r["variant"])][q][pos[r["tile"]]] = float(r[q])
    for v, x in json.load(open(DATA / "appear" / root / "fud.json")).items():
        fud[f"{fam}|{pipe}|{v}"] = x
idx = np.random.default_rng(SEED).integers(0, 100, size=(B, 100))


def boot(x):
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    if not ok.all():   # readouts undefined on a tile (e.g. empty mask) are dropped from every resample of that readout
        m = np.where(ok[idx], x[idx], 0).sum(1) / ok[idx].sum(1)
    else:
        m = x[idx].mean(1)
    return dict(mean=float(np.nanmean(x)), lo=float(np.percentile(m, 2.5)), hi=float(np.percentile(m, 97.5)),
                n=int(ok.sum()))


def verdict(d):
    return "above" if d["lo"] > 0 else ("below" if d["hi"] < 0 else "not distinguishable")


out = dict(n_tiles=100, B=B, seed=SEED, arms={}, diffs={}, fud=fud)
for (fam, pipe, v), d in sorted(V.items()):
    out["arms"][f"{fam}|{pipe}|{v}"] = {q: boot(d[q]) for q in READ if not np.isnan(d[q]).all()}


def diff(k1, k2, q):
    return boot(V[k1][q] - V[k2][q])


DQ = ["pq", "pq_inst", "dice", "a", "b", "c", "lpips", "style_hed"]
for pipe in ("off", "run"):
    van = ("P", pipe, "a2_off_shelf")
    pairs = [(("P", pipe, v), van) for v in PRIMARY[1:]] + [(("R", pipe, v), van) for v in SECONDARY]
    pairs += [(("P", pipe, "a1_concat"), ("P", pipe, "a2_bypass_full_tme")), (("P", pipe, "a1_concat"), ("P", pipe, "production")),
              (("R", pipe, "a1_concat"), ("R", pipe, "a2_bypass_full_tme")), (("R", pipe, "a1_concat"), ("R", pipe, "production"))]
    for k1, k2 in pairs:
        out["diffs"][f"{'|'.join(k1)} - {'|'.join(k2)}"] = {q: diff(k1, k2, q) for q in DQ}

# ---- PREREG §9 gate (runner-consistent pipeline reproduces the published Fig 2 PQ)
gate = {}
for v, p in PUBLISHED.items():
    a = out["arms"][f"P|run|{v}"]["pq"]
    gate[v] = dict(published=p, mean=a["mean"], lo=a["lo"], hi=a["hi"], inside=a["lo"] <= p <= a["hi"])
gate["production(not gated)"] = dict(published=PUB_GROUPED, **{k: out["arms"]["P|run|production"]["pq"][k]
                                                               for k in ("mean", "lo", "hi")})
out["gate"] = dict(items=gate, passed=all(g["inside"] for k, g in gate.items() if k in PUBLISHED))


# ---- PREREG §8.3 ranking (pairwise, pq; consistency = dice, a, c same direction with CIs excluding 0)
def rank(fam_of, arms, pipe):
    res = []
    for x, y in combinations(arms, 2):
        k1, k2 = (fam_of(x), pipe, x), (fam_of(y), pipe, y)
        d = {q: diff(k1, k2, q) for q in ("pq", "dice", "a", "b", "c")}
        vd = verdict(d["pq"])
        cons = None
        if vd != "not distinguishable":
            cons = all(verdict(d[q]) == vd for q in ("dice", "a", "c"))
        res.append(dict(a=NAME[x], b=NAME[y], pq=d["pq"], verdict=vd,
                        consistent="all readouts agree" if cons else ("PQ only" if cons is False else None),
                        dice=verdict(d["dice"]), ha=verdict(d["a"]), hb=verdict(d["b"]), hc=verdict(d["c"])))
    return res


out["ranking"] = {}
for pipe in ("off", "run"):
    out["ranking"][f"primary|{pipe}"] = rank(lambda v: "P", PRIMARY, pipe)
    out["ranking"][f"secondary|{pipe}"] = rank(lambda v: "P" if v == "a2_off_shelf" else "R", ["a2_off_shelf"] + SECONDARY, pipe)

# ---- PREREG §8.4 sim2real flag (official pipeline; Grouped variant above same-family Concat on pq, a, c)
flags = []
for fam in ("P", "R"):
    for g in ("production", "a2_bypass_full_tme"):
        ok = all(diff((fam, "off", g), (fam, "off", "a1_concat"), q)["lo"] > 0 for q in ("pq", "a", "c"))
        flags.append(dict(family=fam, grouped=NAME[g], flagged=bool(ok)))
out["sim2real_flag"] = flags
json.dump(out, open(DATA / "stats.json", "w"), indent=1)
print(json.dumps(out["gate"], indent=1))
for k, r in out["ranking"].items():
    print(k)
    for x in r:
        print(f"  {x['a']:>22} vs {x['b']:<22} {x['pq']['mean']:+.3f} [{x['pq']['lo']:+.3f}, {x['pq']['hi']:+.3f}] {x['verdict']}"
              f" {x['consistent'] or ''}")
print(json.dumps(flags))
