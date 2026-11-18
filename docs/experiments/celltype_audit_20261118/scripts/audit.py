"""celltype_audit_20261118 step 4 (envsmc: pandas, numpy, scipy, PIL). Descriptive only; nothing is fitted or tuned.

Inputs (all under W): raw/processed_crc33/cell_assignments.csv (hfv labels = the generator's training labels),
lin/CRC33_01__P37_S76_01-CRC33_01.csv (Lin per-cell CSV), lin/CRC33_01_lin_he_class.tiff (Lin H&E class map),
out/lin_gates_crc33.json (Lin GMM / published-fraction gates for C33, lin_gate.py), out/lookup.npz|json (lookup.py),
INK = sim-exp-validation darkdiag ink.json (per-tile ink pixel fraction, style_pool.PIGMENT_LEVEL 60).

Labelings compared
  hfv       : he-feature-visualizer cell_type / cell_state (CellViT objects).
  dendro4   : Lin 2023 Ext Data Fig 1d, CRC panel: CD45+ immune | Pan-CK+ or E-cad+ tumour(epithelial) |
              CD31+ or aSMA+ stroma | none -> 'unidentified' (Lin Fig 3c 'not subtyped by the panel').
  old       : main-repo old rule = dendro4 with stroma + unidentified merged into healthy, epithelial -> cancer, no tissue step.
  old_t     : old + Lin H&E tissue step (epithelial in NORM -> healthy), = sim-exp-validation tcmvp_cells.
  rule_a    : old_t + in TUM a non-immune non-epithelial cell is healthy only if SMA+ or CD31+ (tcfix_rule.classify_fix).
  audit.py W INK OUT_JSON
"""
import json
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

W, INK, OUT = sys.argv[1:4]
sys.path.insert(0, f"{W}/scripts")
from tcfix_rule import classify_fix  # noqa: E402  (verbatim copy of sim-exp-validation scripts/analysis/tcmvp/tcfix_rule.py)

CLS = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]
GROUP = {0: "other", 1: "NORM", 2: "other", 3: "TUM", 4: "other", 5: "other", 6: "other", 7: "STR", 8: "LYM"}
GROUPS = ["NORM", "TUM", "STR", "LYM", "other"]
TYPING = ["Pan-CK", "E-cadherin", "CD45", "Ki67", "SMA", "CD31"]
LIN_LINEAGE = ["CD3e", "CD4", "CD8a", "CD20", "CD68", "CD163"]
PIGMENT_LEVEL = 60  # sim-exp-validation style_pool.PIGMENT_LEVEL (ink rule used for the 2026-10-31 bridge refit)


def fr(s):
    v = s.value_counts(normalize=True)
    return {k: round(float(v.get(k, 0.0)), 4) for k in sorted(s.unique())}


def main():
    look = json.load(open(f"{W}/out/lookup.json"))
    lk = np.load(f"{W}/out/lookup.npz")
    px = look["px_um"]["mask"]
    G = json.load(open(f"{W}/out/lin_gates_crc33.json"))["markers"]
    thr = {m: G[m]["final_thr"] for m in TYPING + LIN_LINEAGE}
    cmap = np.array(Image.open(f"{W}/lin/CRC33_01_lin_he_class.tiff"))
    res = dict(px_um=px, gates={m: dict(thr=thr[m], source=G[m]["final_source"], lin_frac=G[m]["lin_frac"],
                                         final_frac=G[m]["final_frac"]) for m in TYPING},
               class_map_shape=list(cmap.shape), mask_shape=look["mask_shape"])
    P = 112.0 / px

    def tissue(x, y):
        r = np.clip((y / P).astype(int), 0, cmap.shape[0] - 1)
        c = np.clip((x / P).astype(int), 0, cmap.shape[1] - 1)
        return cmap[r, c]

    # ---------------- Lin cells ----------------
    lin = pd.read_csv(f"{W}/lin/CRC33_01__P37_S76_01-CRC33_01.csv",
                      usecols=["CellID", "X_centroid", "Y_centroid", "Area", "Eccentricity", "Hoechst", "AF1"] + TYPING + LIN_LINEAGE)
    assert (lin.CellID.to_numpy() == lk["lin_id"]).all()
    lin["cls"] = tissue(lin.X_centroid.to_numpy(), lin.Y_centroid.to_numpy())
    lin["grp"] = lin.cls.map(GROUP)
    pos = {m: lin[m].to_numpy() > thr[m] for m in TYPING + LIN_LINEAGE}
    imm, epi, sv = pos["CD45"], pos["Pan-CK"] | pos["E-cadherin"], pos["SMA"] | pos["CD31"]
    lin["dendro4"] = np.where(imm, "immune", np.where(epi, "epithelial", np.where(sv, "stroma", "unidentified")))
    TY = {0: "healthy", 1: "cancer", 2: "immune"}
    mean = {m: lin[m].to_numpy() for m in TYPING}
    ty0, ty, fixed = classify_fix(mean, thr, lin.cls.to_numpy())
    lin["old"] = pd.Series(ty0).map(TY).to_numpy()
    lin["old_t"] = np.where((ty0 == 1) & (lin.cls.to_numpy() == 1), "healthy", lin["old"])
    lin["rule_a"] = pd.Series(ty).map(TY).to_numpy()
    lin["ki67p"] = pos["Ki67"]
    lin["svp"] = sv
    lin["area_um2"] = lin.Area * px * px
    lin["inked"] = lk["lin_gray"] < PIGMENT_LEVEL
    lin["n_hfv"] = lk["lin2hfv"]
    # compare like with like: only Lin cells inside the 10,379 training tiles (index.json patches, 256 px)
    idx = json.load(open(f"{W}/raw/processed_crc33/index.json"))
    ps = int(idx["patch_size"])
    tiles = {(int(p["x0"]), int(p["y0"])) for p in idx["patches"]}
    st = int(idx["stride"])
    assert st == ps, "tiles overlap; in-tile test below assumes a grid with stride = patch size"
    tx = (lin.X_centroid.to_numpy() // ps).astype(int) * ps
    ty_ = (lin.Y_centroid.to_numpy() // ps).astype(int) * ps
    lin["in_tiles"] = [(a, b) in tiles for a, b in zip(tx, ty_)]
    res["lin_slide_all"] = dict(n=len(lin), in_tiles=int(lin.in_tiles.sum()),
                                tissue_frac_all={CLS[i]: round(float((lin.cls == i).mean()), 4) for i in range(9)},
                                old_all=fr(lin.old), ki67p_all=round(float(lin.ki67p.mean()), 4))
    lin = lin[lin.in_tiles].reset_index(drop=True)
    res["lin"] = dict(n=len(lin), slide_tissue_frac={CLS[i]: round(float((lin.cls == i).mean()), 4) for i in range(9)},
                      dendro4=fr(lin.dendro4), old=fr(lin.old), rule_a=fr(lin.rule_a), ki67p=float(lin.ki67p.mean()))

    # ---------------- hfv objects ----------------
    h = pd.read_csv(f"{W}/raw/processed_crc33/cell_assignments.csv",
                    usecols=["patch_id", "type_cellvit", "type_codex", "type_codex_fine", "cell_type", "cell_state",
                             "centroid_x_global", "centroid_y_global", "Area_cellvit_px", "Ki67", "Pan-CK", "CD45",
                             "p_model_cancer", "p_model_immune", "p_model_healthy", "match_distance"])
    assert np.allclose(h.centroid_x_global.to_numpy(), lk["hfv_x"])
    h["cls"] = tissue(h.centroid_x_global.to_numpy(), h.centroid_y_global.to_numpy())
    h["grp"] = h.cls.map(GROUP)
    h["label"] = lk["hfv_label"]
    h["nmerge"] = lk["hfv_merge"]
    h["inked"] = lk["hfv_gray"] < PIGMENT_LEVEL
    h["area_um2"] = h.Area_cellvit_px * px * px
    h["state"] = h.cell_state.replace({"proliferative": "prolif", "nonproliferative": "nonprolif"})
    res["hfv"] = dict(n=len(h), type=fr(h.cell_type), state=fr(h.state))

    # hfv fusion audit: is the multiplex able to change the CellViT-prior class?
    names = {0: "background", 1: "neoplastic", 2: "inflammatory", 3: "connective", 4: "dead", 5: "epithelial"}
    # prior_only = assign_cells.match_cells falls back to the CellViT prior when max p_model < MIN_MODEL_CONFIDENCE (0.40);
    # for priors with a tie (types 0, 4: 1/3 each; 5: 0.5/0.5) _argmax_label then returns the first key ('cancer').
    pmax = h[["p_model_cancer", "p_model_immune", "p_model_healthy"]].max(axis=1)
    h["prior_only"] = pmax < 0.40
    res["hfv_fusion"] = {names[k]: dict(n=int(len(g)), frac=round(len(g) / len(h), 4), final=fr(g.cell_type),
                                        codex_model=fr(g.type_codex), prior_only=round(float(g.prior_only.mean()), 4),
                                        final_when_prior_only=fr(g.cell_type[g.prior_only]) if g.prior_only.any() else {},
                                        final_when_codex_used=fr(g.cell_type[~g.prior_only]) if (~g.prior_only).any() else {})
                         for k, g in h.groupby("type_cellvit")}
    res["hfv_fusion_match_distance_max"] = float(h.match_distance.max())

    # ---------------- matching ----------------
    lin_idx = pd.Series(np.arange(len(lin)), index=lin.CellID.to_numpy())
    m = h.label > 0
    h["li"] = -1
    h.loc[m, "li"] = lin_idx.reindex(h.label[m].to_numpy()).fillna(-1).astype(int).to_numpy()
    share = h[h.li >= 0].groupby("li").size()
    r_eq = float(np.sqrt(np.median(lin.Area) / np.pi))
    tree = cKDTree(np.c_[lin.X_centroid, lin.Y_centroid])
    dnn, inn = tree.query(np.c_[h.centroid_x_global, h.centroid_y_global])
    nnm = dnn <= r_eq
    agree = float((inn[(h.li >= 0) & nnm] == h.li[(h.li >= 0) & nnm]).mean())
    neo = h.type_cellvit.isin([1, 5])
    res["matching"] = dict(
        primary="Lin cell-mask label under the hfv centroid (no radius)",
        hfv_on_background=round(float((h.label == 0).mean()), 4),
        hfv_on_lin_cell_outside_tiles=round(float(((h.label > 0) & (h.li < 0)).mean()), 4),
        hfv_on_background_by_cellvit={names[k]: round(float((g.label == 0).mean()), 4) for k, g in h.groupby("type_cellvit")},
        hfv_sharing_a_lin_cell=round(float(h.li.isin(share[share > 1].index).mean()), 4),
        lin_without_hfv=round(float((lin.n_hfv == 0).mean()), 4),
        lin_without_hfv_by_dendro4={k: round(float((g.n_hfv == 0).mean()), 4) for k, g in lin.groupby("dendro4")},
        lin_without_hfv_by_tissue={k: round(float((g.n_hfv == 0).mean()), 4) for k, g in lin.groupby("grp")},
        hfv_merge_ge2=round(float((h.nmerge >= 2).mean()), 4),
        hfv_merge_ge2_neoplastic_or_epithelial=round(float((h.nmerge[neo] >= 2).mean()), 4),
        hfv_merge_ge2_by_tissue={k: round(float((g.nmerge >= 2).mean()), 4) for k, g in h.groupby("grp")},
        merge_unresolved=int((h.nmerge < 0).sum()),
        cells_per_tile_hfv=round(len(h) / h.patch_id.nunique(), 1),
        secondary=dict(rule="nearest Lin centroid within r_eq = sqrt(median Lin cell-mask Area / pi)", r_eq_px=round(r_eq, 2),
                       r_eq_um=round(r_eq * px, 2), nn_dist_px_q=[round(float(q), 2) for q in np.quantile(dnn, [.1, .25, .5, .75, .9])],
                       matched=round(float(nnm.mean()), 4), agree_with_primary=round(agree, 4)),
        lin_area_um2_median=round(float(lin.area_um2.median()), 2), hfv_area_um2_median=round(float(h.area_um2.median()), 2))
    hm = h[h.li >= 0].copy()
    L = lin.iloc[hm.li.to_numpy()].reset_index(drop=True)
    hm = hm.reset_index(drop=True)
    for c in ["dendro4", "old", "old_t", "rule_a", "ki67p", "svp", "Area", "area_um2", "Pan-CK", "E-cadherin", "CD45", "Ki67", "SMA", "CD31"]:
        hm["L_" + c] = L[c].to_numpy()
    hm["L_grp"] = L.grp.to_numpy()

    # ---------------- (a) class fractions by tissue ----------------
    a = {}
    for g in GROUPS + ["all"]:
        lg = lin if g == "all" else lin[lin.grp == g]
        hg = h if g == "all" else h[h.grp == g]
        a[g] = dict(n_lin=len(lg), n_hfv=len(hg), hfv=fr(hg.cell_type), old=fr(lg.old), old_t=fr(lg.old_t),
                    rule_a=fr(lg.rule_a), dendro4=fr(lg.dendro4),
                    prolif_hfv=round(float((hg.state == "prolif").mean()), 4), prolif_lin=round(float(lg.ki67p.mean()), 4))
    res["a_class_fractions"] = a
    # matched-pair confusion (hfv label vs the Lin cell it sits in)
    res["a_confusion_matched"] = {g: {lab: pd.crosstab(hg.cell_type, hg["L_" + lab], normalize="columns").round(4).to_dict()
                                      for lab in ["dendro4", "old_t", "rule_a"]}
                                  for g, hg in list(hm.groupby("grp")) + [("all", hm)]}
    res["a_agreement_matched"] = {g: {lab: round(float((hg.cell_type == hg["L_" + lab]).mean()), 4) for lab in ["old", "old_t", "rule_a"]}
                                  for g, hg in list(hm.groupby("grp")) + [("all", hm)]}
    res["state_agreement_matched"] = dict(
        hfv_prolif=round(float((hm.state == "prolif").mean()), 4), lin_ki67p=round(float(hm.L_ki67p.mean()), 4),
        both=round(float(((hm.state == "prolif") & hm.L_ki67p).mean()), 4),
        agree=round(float(((hm.state == "prolif") == hm.L_ki67p).mean()), 4))

    # ---------------- (b) keratin-weak tumour cells in TUM ----------------
    kw = (lin.grp == "TUM") & (lin.dendro4 == "unidentified")
    hk = hm[(hm.L_grp == "TUM") & (hm.L_dendro4 == "unidentified")]
    res["b_keratin_weak_TUM"] = dict(
        definition="Lin cell in Lin TUM tissue, CD45-, Pan-CK-, E-cad-, SMA-, CD31- (dendro4 'unidentified')",
        n=int(kw.sum()), frac_of_TUM_non_immune=round(float(kw.sum() / ((lin.grp == "TUM") & (lin.dendro4 != "immune")).sum()), 4),
        typed_healthy=dict(old=round(float((lin.old[kw] == "healthy").mean()), 4), old_t=round(float((lin.old_t[kw] == "healthy").mean()), 4),
                           rule_a=round(float((lin.rule_a[kw] == "healthy").mean()), 4),
                           hfv_matched=round(float((hk.cell_type == "healthy").mean()), 4)),
        hfv_matched_n=len(hk), hfv_matched_labels=fr(hk.cell_type),
        hfv_matched_by_cellvit={names[k]: dict(n=len(g), healthy=round(float((g.cell_type == "healthy").mean()), 4))
                                for k, g in hk.groupby("type_cellvit")},
        hfv_objects_in_TUM=fr(h[h.grp == "TUM"].cell_type))

    # ---------------- (c) normal epithelium in NORM ----------------
    ne = (lin.grp == "NORM") & (lin.dendro4 == "epithelial")
    hn = hm[(hm.L_grp == "NORM") & (hm.L_dendro4 == "epithelial")]
    res["c_normal_epithelium_NORM"] = dict(
        definition="Lin cell in Lin NORM tissue, CD45-, Pan-CK+ or E-cad+",
        n=int(ne.sum()), typed_cancer=dict(old=round(float((lin.old[ne] == "cancer").mean()), 4),
                                           old_t=round(float((lin.old_t[ne] == "cancer").mean()), 4),
                                           rule_a=round(float((lin.rule_a[ne] == "cancer").mean()), 4),
                                           hfv_matched=round(float((hn.cell_type == "cancer").mean()), 4)),
        hfv_matched_n=len(hn), hfv_matched_labels=fr(hn.cell_type), hfv_objects_in_NORM=fr(h[h.grp == "NORM"].cell_type))

    # ---------------- (d) ink ----------------
    ink = json.load(open(INK))
    inked_tiles = {t for t, f in ink.items() if f > 0}
    h["ink_tile"] = h.patch_id.isin(inked_tiles)
    sub = dict(inked_cells=h[h.inked], ink_tiles_not_inked_cells=h[h.ink_tile & ~h.inked], clean_tiles=h[~h.ink_tile])
    lsub = dict(inked_cells=lin[lin.inked], not_inked=lin[~lin.inked])
    res["d_ink"] = dict(
        rule=f"H&E grayscale < {PIGMENT_LEVEL} (style_pool.PIGMENT_LEVEL) at the 3x3 px around the centroid; tiles from ink.json (fraction > 0)",
        n_tiles=len(ink), n_ink_tiles=len(inked_tiles), n_ink_tiles_gt_0p005=sum(f > 0.005 for f in ink.values()),
        hfv={k: dict(n=len(v), type=fr(v.cell_type), state=fr(v.state), tissue=fr(v.cls.map(lambda i: CLS[i])))
             for k, v in sub.items()},
        lin={k: dict(n=len(v), old=fr(v.old), rule_a=fr(v.rule_a), dendro4=fr(v.dendro4), ki67p=round(float(v.ki67p.mean()), 4),
                     tissue=fr(v.cls.map(lambda i: CLS[i])),
                     median={mk: round(float(v[mk].median()), 1) for mk in ["Hoechst", "AF1", "Ki67", "CD45", "Pan-CK", "SMA"]})
             for k, v in lsub.items()})
    hi = hm[hm.inked.to_numpy()]
    res["d_ink"]["hfv_inked_matched_lin_ki67p"] = round(float(hi.L_ki67p.mean()), 4) if len(hi) else None

    # ---------------- (e) descriptors ----------------
    def desc(df, lab, area, panck, ki, sv):
        out = {}
        for k, g in df.groupby(lab):
            out[k] = dict(n=len(g), frac=round(len(g) / len(df), 4), area_um2=round(float(g[area].median()), 1),
                          panck=round(float(g[panck].median()), 1), ki67p=round(float(g[ki].mean()), 4),
                          sma_or_cd31p=round(float(g[sv].mean()), 4))
        return out
    e = {}
    for g in ["all", "TUM", "NORM", "STR"]:
        lg = lin if g == "all" else lin[lin.grp == g]
        hg = hm if g == "all" else hm[hm.L_grp == g]
        e[g] = dict(
            old=desc(lg, "old", "area_um2", "Pan-CK", "ki67p", "svp"),
            rule_a=desc(lg, "rule_a", "area_um2", "Pan-CK", "ki67p", "svp"),
            dendro4=desc(lg, "dendro4", "area_um2", "Pan-CK", "ki67p", "svp"),
            hfv_on_matched_lin=desc(hg, "cell_type", "L_area_um2", "L_Pan-CK", "L_ki67p", "L_svp"),
            hfv_cellvit_area_um2={k: round(float(v.area_um2.median()), 1) for k, v in hg.groupby("cell_type")})
    res["e_descriptors"] = e
    json.dump(res, open(OUT, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ["matching", "b_keratin_weak_TUM", "c_normal_epithelium_NORM"]}, indent=1))


if __name__ == "__main__":
    main()
