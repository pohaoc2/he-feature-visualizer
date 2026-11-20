"""Side-by-side of the PQ robustness readouts (eval/robust.py stats_robust.json) and the primary CellViT real-H&E-reference
statistics (results/stats_realref*.json): Fig 2 ranking and variant - Vanilla, ablation added effects, LOO drops,
decomposition vs full model. "agree" = same sign of the mean and the same verdict on whether the 95 % interval excludes 0.

  python robust_compare.py RESULTS_DIR OUT_JSON SET=stats_robust.json [SET=...]
"""
import json
import sys
from pathlib import Path

V5 = ["a1_concat", "production", "a2_bypass_full_tme", "a1_per_channel", "a2_off_shelf"]


def verdict(x):
    if x is None:
        return None
    return 0 if not x["excludes_0"] else (1 if x["mean"] > 0 else -1)


def pair(prim, rob):
    return dict(primary=prim, robust=rob,
                agree=None if prim is None or rob is None else verdict(prim) == verdict(rob))


def primary_for(res_dir, set_):
    d = json.load(open(res_dir / "stats_realref.json"))
    if set_ == "crc34eval":
        d = json.load(open(res_dir / "stats_realref_crc34eval.json"))
    return {k.rsplit("_", 1)[0]: v for k, v in d.items() if k.endswith("_" + set_)}


def compare(res_dir, set_, rob_path):
    R = json.load(open(rob_path))
    P = primary_for(res_dir, set_)
    out = dict(info=R["info"], crashes={k: R["crashes"][k] for k in ("convention", "dropped_ref_tiles", "max_frac")},
               n_crashed=len(R["crashes"]["crashed_images"]), real_line=R["main"]["real"])
    for analysis in ("main", "sensitivity_crashed_excluded"):
        M = R[analysis]
        o = {}
        if "fig2" in M and "fig2" in P:
            pm = P["fig2"]["metrics"]
            o["fig2_ranking"] = dict(cellvit_pq=sorted(V5, key=lambda v: -pm[v]["pq"]),
                                     stardist_pq=M["fig2"]["ranking"]["sd_pq"],
                                     cellvit_dice=sorted(V5, key=lambda v: -pm[v]["dice"]),
                                     stardist_dice=M["fig2"]["ranking"]["sd_dice"],
                                     lin_f1=M["fig2"]["ranking"]["f1"])
            o["fig2_means"] = {v: dict(cellvit_pq=pm[v]["pq"], cellvit_dice=pm[v]["dice"], **M["fig2"]["means"][v]) for v in V5}
            vv = M["fig2"]["vs_vanilla"]
            o["fig2_vs_vanilla"] = {f"{v}/{a}": pair(P["fig2"]["vs_vanilla"].get(f"{v}/{b}"), vv.get(f"{v}/{a}"))
                                    for v in V5[:4] for a, b in (("sd_pq", "pq"), ("sd_dice", "dice"), ("f1", "pq"))}
        if "abl" in P:
            for mode in ("paired", "unpaired"):
                if f"abl_{mode}" not in M:
                    continue
                pa, ra = P["abl"][mode]["added_effect"], M[f"abl_{mode}"]["added_effect"]
                groups = sorted({k.split("/")[0] for k in ra})
                o[f"abl_{mode}"] = {f"{g}/{a}": pair(pa.get(f"{g}/{b}"), ra.get(f"{g}/{a}"))
                                    for g in groups for a, b in (("sd_pq", "pq"), ("sd_dice", "dice"), ("f1", "pq"))}
        if "loo" in M and "loo" in P:
            pl, rl = P["loo"]["pq_drop"], M["loo"]["drop"]
            subs = sorted({k.split("/")[0] for k in rl})
            o["loo"] = {f"{s}/{a}": pair(pl.get(s), rl.get(f"{s}/{a}")) for s in subs for a in ("sd_pq", "f1")}
        if "decomp" in M and "decomp" in P:
            pv, rv = P["decomp"]["vs_full"], M["decomp"]["vs_full"]
            o["decomp_vs_full"] = {f"{md}/{a}": pair(pv.get(f"{md}/pq"), rv.get(f"{md}/{a}"))
                                   for md in ("uni_only", "tme_only", "neither") for a in ("sd_pq", "f1")}
        out[analysis] = o
    return out


def main():
    res_dir, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    out = {}
    for a in sys.argv[3:]:
        s, p = a.split("=", 1)
        out[s] = compare(res_dir, s, p)
    json.dump(out, open(out_json, "w"), indent=1)
    for s, o in out.items():
        m = o["main"]
        print(f"== {s}: radius {o['info']['match_radius_px']:.2f} px, crashed {o['n_crashed']}, dropped refs "
              f"{o['crashes']['dropped_ref_tiles']}, max crash frac {o['crashes']['max_frac']:.4f}")
        for k, v in m.get("fig2_ranking", {}).items():
            print(f"  rank {k:14s} {v}")
        for sec in ("fig2_vs_vanilla", "abl_paired", "abl_unpaired", "loo", "decomp_vs_full"):
            if sec not in m:
                continue
            ag = [v["agree"] for v in m[sec].values() if v["agree"] is not None]
            print(f"  {sec}: agree {sum(ag)}/{len(ag)}")
            for k, v in m[sec].items():
                p, r = v["primary"], v["robust"]
                f = lambda x: "NA" if x is None else f"{x['mean']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]"
                print(f"    {k:40s} cellvit {f(p)}  robust {f(r)}  {'' if v['agree'] else 'DIFF'}")


if __name__ == "__main__":
    main()
