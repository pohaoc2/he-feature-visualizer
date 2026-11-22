"""fidelity_variants_20261122 (PREREG.md §2, §3), local: paired tile-bootstrap differences and the decision rule.

  python fv_stats.py EXP_DIR
EXP_DIR/data/hema_{36,33}.csv     (fv_hema.py, Hyak)
EXP_DIR/../cellvit_official_20261122/data/fig2/tiles_{36,33}.csv   (secondary PQ only: pq_lin_*, ref_lin_*;
                                                                     primary PQ columns pq_main_* / pq_inst_* are not read)
Bootstrap: B = 10,000, default_rng(42).integers(0, n, (B, n)), tiles sorted by name, one matrix per slide shared by every
quantity and comparison; 95 % percentile interval of the mean difference. Missing values: a tile missing a quantity in any
arm is dropped for that quantity in all arms (paired); counts reported.
-> EXP_DIR/data/stats.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

E = Path(sys.argv[1])
F2 = E.parent / "cellvit_official_20261122/data/fig2"
B, SEED = 10000, 42
GROUPS = {"36": "CRC36", "33": "CRC33"}
VARS = ("van", "grp", "cat", "gbp", "pch")
PAIRS = (("cat", "grp"), ("cat", "gbp"), ("grp", "gbp"))
RULE_Q = ("pq_official", "a", "c")
QUANT = ("pq_official", "pq_runner", "a", "c", "b", "auc", "hmean", "a_tS", "b_tS")


def load(k):
    """-> tiles (sorted), {quantity: {arm: array over tiles}} with arm in VARS + ('real',)."""
    pq = list(csv.DictReader(open(F2 / f"tiles_{k}.csv")))
    hm = list(csv.DictReader(open(E / f"data/hema_{k}.csv")))
    tiles = sorted({r["tile"] for r in pq})
    if sorted({r["tile"] for r in hm}) != tiles:
        raise SystemExit(f"STOP {k}: hema tiles != tiles_{k}.csv tiles")
    ix = {t: i for i, t in enumerate(tiles)}
    n = len(tiles)
    Q = {q: {a: np.full(n, np.nan) for a in VARS + ("real",)} for q in QUANT}
    for r in pq:
        i = ix[r["tile"]]
        for p in ("official", "runner"):
            Q[f"pq_{p}"][r["variant"]][i] = float(r[f"pq_lin_{p}"])
            ref = float(r[f"ref_lin_{p}"])
            if not np.isnan(Q[f"pq_{p}"]["real"][i]) and Q[f"pq_{p}"]["real"][i] != ref:
                raise SystemExit(f"STOP {k}: ref_lin_{p} differs across variants on {r['tile']}")
            Q[f"pq_{p}"]["real"][i] = ref
    for r in hm:
        i = ix[r["tile"]]
        for q in QUANT[2:]:
            Q[q][r["arm"]][i] = float(r[q]) if r[q] not in ("", "nan") else np.nan
    return tiles, Q


def ci(x):
    lo, hi = np.percentile(x, [2.5, 97.5])
    return [float(lo), float(hi)]


def main():
    out = dict(rule="PREREG §3", B=B, seed=SEED, slides={})
    for k, name in GROUPS.items():
        tiles, Q = load(k)
        n = len(tiles)
        idx = np.random.default_rng(SEED).integers(0, n, (B, n))
        S = dict(n_tiles=n, quantities={})
        for q in QUANT:
            arr = np.stack([Q[q][a] for a in VARS + ("real",)])
            ok = ~np.isnan(arr).any(0)
            w = ok.astype(float)
            wb = w[idx].sum(1)

            def boot(d):
                d0 = np.where(ok, d, 0.0)
                return dict(mean=float(d0.sum() / w.sum()), ci95=ci(d0[idx].sum(1) / wb))

            R = dict(n_used=int(ok.sum()), n_dropped=int((~ok).sum()),
                     mean={a: float(np.nanmean(np.where(ok, Q[q][a], np.nan))) for a in VARS + ("real",)},
                     arm_ci95={a: boot(Q[q][a])["ci95"] for a in VARS + ("real",)}, diff={})
            for x, y in PAIRS:
                R["diff"][f"{x}-{y}"] = boot(Q[q][x] - Q[q][y])
            for v in VARS:
                R["diff"][f"{v}-real"] = boot(Q[q][v] - Q[q]["real"])
            S["quantities"][q] = R
        out["slides"][name] = S
    # decision rule (PREREG §3): Concat - G CI entirely < 0 on both slides for pq_official, a, c
    rule = {}
    for g in ("grp", "gbp"):
        cells = {f"{s}:{q}": out["slides"][s]["quantities"][q]["diff"][f"cat-{g}"]["ci95"][1] < 0
                 for s in GROUPS.values() for q in RULE_Q}
        rule[g] = dict(holds=all(cells.values()), cells=cells)
    out["rule_by_G"] = rule
    hit = [g for g in rule if rule[g]["holds"]]
    out["outcome"] = ("A: a Grouped variant follows masks better (" + ", ".join(hit) + ")") if hit else \
        "B: no evidence Grouped follows masks better"
    (E / "data/stats.json").write_text(json.dumps(out, indent=1))
    print(out["outcome"])


if __name__ == "__main__":
    main()
