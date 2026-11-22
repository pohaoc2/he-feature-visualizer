"""cellvit_official_20261122 (PREREG.md §4-§6), local: old vs new tables for RESULTS.md -> data/stats.json.

  python cvo_stats.py EXP_DIR
EXP_DIR/data/stage3/  (from Hyak $D/full: g3/score/{stats_r1.json,decision.json,tiles_*.csv,fud.json},
                       s3/{r2_app8_official.json,results_app8_official.json,r2_app8_*_b10k.json})
EXP_DIR/data/fig2/    tiles_{P,36,33}.csv (cvo_fig2.py), pub/ (published per-tile values, see PREREG §5.5)
old stage 3 = hfv_retrain2_20261121/data/stage3 (runner path).
Fig 2: tile bootstrap B = 10,000, default_rng(42).integers(0, n, (B, n)), one matrix per group (paper Vanilla 1000 and the
paper 100 separately), shared by every variant / path / quantity (paired); 95 % percentile interval of the mean.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

E = Path(sys.argv[1])
OLD = E.parent / "hfv_retrain2_20261121/data/stage3"
S3N = E / "data/stage3"
F2 = E / "data/fig2"
B, SEED = 10000, 42


def ci(bm):
    lo, hi = np.percentile(bm, [2.5, 97.5])
    return [float(lo), float(hi)]


# ------------------------------------------------------------------------------------------------------------ stage 3
def stage3():
    o1, n1 = (json.loads((d / "stats_r1.json").read_text()) for d in (OLD, S3N))
    oa, na = json.loads((OLD / "r2_app8.json").read_text()), json.loads((S3N / "r2_app8_official.json").read_text())
    od, nd = json.loads((OLD / "decision.json").read_text()), json.loads((S3N / "decision.json").read_text())
    ob, nb = (json.loads((S3N / f"r2_app8_{p}_b10k.json").read_text()) for p in ("runner", "official"))
    out = dict(decision=dict(old=od["decision_row"], new=nd["decision_row"], changed=od["decision_row"][0] != nd["decision_row"][0],
                             old_detail=od, new_detail=nd))
    r1 = {}
    for s in ("crc36ho", "crc33sec"):
        for q in ("fid_pq", "fid_f1", "pq2", "ref2", "f1", "pq1", "n_det", "count_ratio", "lpips", "hed"):
            r1[f"{s}:{q}"] = dict(
                old={m: o1["sets"][s]["per_model"][m][q] for m in "NK"} | {"N-K": o1["sets"][s]["n_minus_k"][q]},
                new={m: n1["sets"][s]["per_model"][m][q] for m in "NK"} | {"N-K": n1["sets"][s]["n_minus_k"][q]})
        for q in ("f1", "n_det", "count_ratio"):
            r1[f"{s}:real_{q}"] = dict(old=o1["sets"][s]["real"][q], new=n1["sets"][s]["real"][q])
    out["readout1"] = r1
    out["fud"] = dict(old=o1["fud"]["fud"], new=n1["fud"]["fud"])
    r3 = {}
    for a in ("R_L1", "NP", "ND", "R_s3b", "KP", "KD"):
        r3[a] = {k: dict(old=[oa["point"][a][k]] + oa["ci95"][a][k], new=[na["point"][a][k]] + na["ci95"][a][k])
                 for k in na["point"][a]}
    out["readout3_arms"] = r3
    out["readout3_n_minus_k"] = {k: dict(old=oa["n_minus_k"][k], new=na["n_minus_k"][k]) for k in na["n_minus_k"]}
    out["readout3_descriptive"] = {k: dict(old=oa["descriptive"][k], new=na["descriptive"][k]) for k in na["descriptive"]}
    out["readout3_b10k_seed42"] = {k: dict(old=ob["n_minus_k"][k], new=nb["n_minus_k"][k]) for k in nb["n_minus_k"]}
    a8o = json.loads((S3N / "results_app8_runner.json").read_text())
    a8n = json.loads((S3N / "results_app8_official.json").read_text())
    out["app8"] = dict(old_row=a8o["decision_row"], new_row=a8n["decision_row"],
                       primary={k: dict(old=a8o["primary"][k], new=a8n["primary"][k]) for k in a8n["primary"]},
                       point={a: {k: dict(old=a8o["point"][a][k], new=a8n["point"][a][k])
                                  for k in ("delta", "canc_neo", "heal_neo", "n_det_per_img")} for a in a8n["point"]})
    # gate: non-CellViT columns identical
    gate = {}
    for s in ("crc36ho", "crc33sec"):
        ro = list(csv.DictReader(open(S3N / f"old_tiles_{s}.csv")))  # old stage3/score/tiles_<set>.csv
        rn = list(csv.DictReader(open(S3N / f"tiles_{s}.csv")))
        cols = [c for c in ro[0] if c.split("_")[0] in ("lpips", "hed") or c.startswith("n_gt")]
        gate[s] = dict(columns=cols, identical=all(a["tile"] == b["tile"] and all(a[c] == b[c] for c in cols)
                                                   for a, b in zip(ro, rn)) and len(ro) == len(rn))
    gate["fud_identical"] = out["fud"]["old"] == out["fud"]["new"]
    out["gate_non_cellvit"] = gate
    return out


# -------------------------------------------------------------------------------------------------------------- Fig 2
PUB_PAPER = {"van": 0.7035106918895461, "van100": 0.6987351640533153, "cat": 0.7817372847706214,
             "gbp": 0.7943005168185073, "pch": 0.0106190677347819}
VMAP = {"van": "a2_off_shelf", "grp": "production", "cat": "a1_concat", "gbp": "a2_bypass_full_tme", "pch": "a1_per_channel"}


def rd(f):
    return list(csv.DictReader(open(f)))


def summ(x, idx):
    return dict(mean=float(np.mean(x)), ci95=ci(x[idx].mean(1)), n=int(len(x)))


def fig2():
    out = {}
    pq = json.loads((F2 / "pub/pqmatrix_crc33fig2.json").read_text())["res"]
    ids100 = [l.strip() for l in open(F2 / "pub/ids100.txt") if l.strip()]
    rows = rd(F2 / "tiles_P.csv")
    for sub in ("van1000", "100"):
        R = [r for r in rows if (r["variant"] == "van" if sub == "van1000" else r["in100"] == "1")]
        vs = ["van"] if sub == "van1000" else ["van", "cat", "gbp", "pch"]
        tiles = sorted({r["tile"] for r in R})
        n = len(tiles)
        idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
        g = {}
        for v in vs:
            rv = {r["tile"]: r for r in R if r["variant"] == v}
            col = lambda k: np.array([float(rv[t][k]) for t in tiles])
            d = {}
            for q in ("pq_main", "pq_inst"):
                rr, oo = col(f"{q}_runner"), col(f"{q}_official")
                d[q] = dict(runner=summ(rr, idx), official=summ(oo, idx), official_minus_runner=summ(oo - rr, idx))
            for p in ("runner", "official"):
                d[f"n_det_{p}"] = float(col(f"n_det_{p}").mean())
                d[f"n_real_{p}"] = float(col(f"n_real_{p}").mean())
            key = "van100" if (v == "van" and sub == "100") else v
            d["published"] = PUB_PAPER[key]
            if sub == "100":
                pt = dict(zip(ids100, pq[f"oldimg/{VMAP[v]}/oldGT"]["pq_tiles"]))
                pub = np.array([pt[t] for t in tiles])
                rr = col("pq_main_runner")
                d["repro"] = dict(published_mean=float(pub.mean()), runner_rerun_minus_published=float(rr.mean() - pub.mean()),
                                  tiles_identical=int(np.sum(np.isclose(rr, pub, atol=1e-9))), n=n,
                                  official_minus_published=summ(col("pq_main_official") - pub, idx))
            else:
                d["repro"] = dict(published_mean=PUB_PAPER["van"],
                                  runner_rerun_minus_published=float(col("pq_main_runner").mean() - PUB_PAPER["van"]))
            g[v] = d
        if sub == "100":
            for v in ("cat", "gbp", "pch"):
                for p in ("runner", "official"):
                    rv = {r["tile"]: r for r in R if r["variant"] == v}
                    r0 = {r["tile"]: r for r in R if r["variant"] == "van"}
                    x = np.array([float(rv[t][f"pq_main_{p}"]) - float(r0[t][f"pq_main_{p}"]) for t in tiles])
                    g[v][f"minus_van_{p}"] = summ(x, idx)
            # not in PREREG §5.5 (reported as unregistered): Grouped + bypass - Concat, the order of the two best variants
            rg = {r["tile"]: r for r in R if r["variant"] == "gbp"}
            rc = {r["tile"]: r for r in R if r["variant"] == "cat"}
            g["gbp_minus_cat_unregistered"] = {p: summ(np.array([float(rg[t][f"pq_main_{p}"]) - float(rc[t][f"pq_main_{p}"])
                                                                 for t in tiles]), idx) for p in ("runner", "official")}
        out[f"P_{sub}"] = g
    for k, s in (("36", "crc36ho"), ("33", "crc33sec")):
        R = rd(F2 / f"tiles_{k}.csv")
        tiles = sorted({r["tile"] for r in R})
        n = len(tiles)
        idx = np.random.default_rng(SEED).integers(0, n, size=(B, n))
        pubm = {(r["variant"], r["tile"]): float(r["pq"]) for r in rd(F2 / f"pub/realref_fig2_{s}.csv")}
        publ = {(r["variant"], r["tile"]): float(r["pq"]) for r in rd(F2 / f"pub/scoreres_fig2_{s}.csv")}
        g = {}
        by = {v: {r["tile"]: r for r in R if r["variant"] == v} for v in VMAP}
        for v in VMAP:
            col = lambda k_: np.array([float(by[v][t][k_]) for t in tiles])
            d = {}
            for q in ("pq_main", "pq_inst", "pq_lin"):
                rr, oo = col(f"{q}_runner"), col(f"{q}_official")
                d[q] = dict(runner=summ(rr, idx), official=summ(oo, idx), official_minus_runner=summ(oo - rr, idx))
                if v != "van":
                    for p in ("runner", "official"):
                        x = col(f"{q}_{p}") - np.array([float(by["van"][t][f"{q}_{p}"]) for t in tiles])
                        d[q][f"minus_van_{p}"] = summ(x, idx)
            for p in ("runner", "official"):
                d[f"n_det_{p}"] = float(col(f"n_det_{p}").mean())
            pm = np.array([pubm[(VMAP[v], t)] for t in tiles])
            pl = np.array([publ[(VMAP[v], t)] for t in tiles])
            d["published"] = dict(pq_main=float(pm.mean()), pq_lin=float(pl.mean()))
            d["repro"] = dict(main_runner_minus_published=float(col("pq_main_runner").mean() - pm.mean()),
                              main_tiles_identical=int(np.sum(np.isclose(col("pq_main_runner"), pm, atol=1e-9))),
                              lin_runner_minus_published=float(col("pq_lin_runner").mean() - pl.mean()),
                              lin_tiles_identical=int(np.sum(np.isclose(col("pq_lin_runner"), pl, atol=1e-9))), n=n)
            g[v] = d
        col = lambda k_: np.array([float(by["cat"][t][k_]) for t in tiles])
        rr, oo = col("ref_lin_runner"), col("ref_lin_official")
        g["ref_lin"] = dict(runner=summ(rr, idx), official=summ(oo, idx), official_minus_runner=summ(oo - rr, idx))
        g["n_real"] = {p: float(col(f"n_real_{p}").mean()) for p in ("runner", "official")}
        g["n_lin"] = float(col("n_lin").mean())
        # PREREG §5.5 gate: retrained Concat runner secondary PQ == stage-3 pq2_K per tile (same images, same runner)
        s3 = {r["tile"]: float(r["pq2_K"]) for r in rd(S3N / f"old_tiles_{s}.csv")}
        x = col("pq_lin_runner")
        y = np.array([s3[t] for t in tiles])
        g["gate_concat_lin_eq_stage3_pq2K"] = dict(n=n, identical=int(np.sum(np.isclose(x, y, atol=1e-12))),
                                                   max_abs_diff=float(np.max(np.abs(x - y))))
        out[k] = g
    return out


if __name__ == "__main__":
    res = dict(rule="cellvit_official_20261122 PREREG §4-§6 (cvo_stats.py)", B=B, seed=SEED)
    res["stage3"] = stage3()
    if (F2 / "tiles_P.csv").exists():
        res["fig2"] = fig2()
    (E / "data/stats.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["stage3"]["decision"] | {"app8": [res["stage3"]["app8"]["old_row"], res["stage3"]["app8"]["new_row"]]},
                     indent=1)[:3000])
