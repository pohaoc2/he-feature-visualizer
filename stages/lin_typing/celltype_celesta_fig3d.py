"""Cell typing "CELESTA + Lin 2023 Fig. 3d tissue-class fill-in" (user decision 2026-11-18; prereg_tcmvp2_20261118.md §10).

Pure functions (no file access): the per-cell CELESTA call (Zhang et al. Nat Methods 2022; default settings, prior
docs/experiments/tcmvp_20261118/celltype_pub/data/celesta_prior.csv, tiled run of celltype_pub README §1.4) and the Lin
112 um H&E tissue class at the nucleus centroid -> model population.

  CELESTA immune lineages (Immune, T cell, B cell, Macrophage, CD4 T cell, CD8 T cell, Treg) -> immune
  Endothelial, SMA+ stromal                                                    -> healthy
  Epithelial  -> healthy in a Lin NORM grid cell, cancer in every other grid class (tissue step, as the pipeline)
  Unknown, or no CELESTA call (nucleus not matched to a Lin CSV cell)          -> the Lin tissue class (Fig. 3d fill-in):
      TUM -> cancer; NORM, STR, MUS, ADI -> healthy; LYM -> immune; BACK, DEB, MUC -> healthy

Lin class codes (CRC16_lin_he_class.tiff; tcmvp_cells.LIN_L): 0 BACK, 1 NORM, 2 DEB, 3 TUM, 4 ADI, 5 MUC, 6 MUS, 7 STR, 8 LYM.
Population codes as the pipeline: 0 healthy, 1 cancer, 2 immune.
"""
import numpy as np

LIN_L = ["BACK", "NORM", "DEB", "TUM", "ADI", "MUC", "MUS", "STR", "LYM"]
NORM, TUM, LYM = 1, 3, 8
HEALTHY, CANCER, IMMUNE = 0, 1, 2
IMMUNE_TYPES = ("Immune", "T cell", "B cell", "Macrophage", "CD4 T cell", "CD8 T cell", "Treg")
STROMAL_TYPES = ("Endothelial", "SMA+ stromal")
EPI, UNKNOWN = "Epithelial", "Unknown"
KNOWN = frozenset(IMMUNE_TYPES + STROMAL_TYPES + (EPI, UNKNOWN))
# Fig. 3d fill-in by Lin tissue class (index = class code)
FILL = np.array([HEALTHY, HEALTHY, HEALTHY, CANCER, HEALTHY, HEALTHY, HEALTHY, HEALTHY, IMMUNE], np.int8)
FILL_LOW = (0, 2, 5)          # BACK, DEB, MUC: classes without a cell population of their own (reported apart)

# per-cell source codes (second return value)
SRC_IMMUNE, SRC_STROMAL, SRC_EPI, SRC_UNKNOWN, SRC_NOCALL = 0, 1, 2, 3, 4
SRC_N = ["immune", "stromal", "epithelial", "unknown_fill", "nocall_fill"]


def classify_celesta_fig3d(final, lin_cls):
    """final: per-cell CELESTA 'final' type (str; None / NaN / '' = no CELESTA call); lin_cls: Lin class code per cell.
    -> (ty int8 population, src int8 SRC_* code). Unknown and no-call cells take FILL[lin_cls]."""
    lin_cls = np.asarray(lin_cls).astype(np.int64)
    n = len(lin_cls)
    f = np.empty(n, object)
    f[:] = list(final) if n else []
    nocall = np.array([not isinstance(v, str) or v == "" for v in f], bool)
    bad = {v for v in f[~nocall] if v not in KNOWN}
    assert not bad, f"unknown CELESTA types {sorted(bad)}"
    assert n == 0 or (lin_cls.min() >= 0 and lin_cls.max() < len(LIN_L)), "Lin class out of range"
    imm = np.isin(f, IMMUNE_TYPES) & ~nocall
    stro = np.isin(f, STROMAL_TYPES) & ~nocall
    epi = (f == EPI) & ~nocall
    unk = (f == UNKNOWN) & ~nocall
    ty = FILL[lin_cls].copy() if n else np.zeros(0, np.int8)
    ty[imm] = IMMUNE
    ty[stro] = HEALTHY
    ty[epi] = np.where(lin_cls[epi] == NORM, HEALTHY, CANCER)
    src = np.select([imm, stro, epi, unk], [SRC_IMMUNE, SRC_STROMAL, SRC_EPI, SRC_UNKNOWN], SRC_NOCALL).astype(np.int8)
    return ty.astype(np.int8), src


def lookup(ids_sorted, final_sorted, query):
    """CELESTA 'final' for each CellID in query (None where absent). ids_sorted ascending int array."""
    query = np.asarray(query, np.int64)
    r = np.searchsorted(ids_sorted, query)
    r = np.minimum(r, len(ids_sorted) - 1)
    hit = ids_sorted[r] == query
    out = np.empty(len(query), object)
    out[:] = None
    out[hit] = final_sorted[r[hit]]
    return out
