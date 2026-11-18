"""celltype_fix rule (docs/experiments/tcmvp_20261118/celltype_fix/README.md §1; chosen = candidate (a)).

Old rule (cp_lin_cells.classify, Lin 2023 Ext Data Fig 1d; gates = lin_gates.json final_thr, slide quantile gates):
  CD45+ -> immune (2); else Pan-CK+ or E-cad+ -> epithelium (1); else -> healthy (0)
  then (tcmvp_cells.py, prereg_tcmvp §4.1): epithelium in Lin NORM tissue -> healthy (normal epithelium, pop 1).
New rule = old rule, plus ONE change, only for cells whose nucleus centroid lies in a Lin H&E TUM grid cell:
  a non-immune, non-epithelial cell is healthy only with positive stromal or vascular evidence (SMA+ or CD31+, same slide
  gates); every other such cell -> cancer (1).
No new threshold: the gates are the existing slide gates. NORM and every other tissue class: unchanged.
"""
import numpy as np

TUM, NORM = 3, 1


def classify_fix(mean, thr, lin_cls):
    """mean / thr: dicts with Pan-CK, E-cadherin, CD45, SMA, CD31 (arrays / gates); lin_cls: Lin class per cell.
    -> (ty0 old Lin-dendrogram type, ty new type, fixed bool = cells moved healthy -> cancer by the TUM rule).
    ty codes 0 healthy, 1 cancer, 2 immune. NORM relabel (epithelium -> healthy) is applied here too."""
    imm = mean["CD45"] > thr["CD45"]
    epi = (mean["Pan-CK"] > thr["Pan-CK"]) | (mean["E-cadherin"] > thr["E-cadherin"])
    sv = (mean["SMA"] > thr["SMA"]) | (mean["CD31"] > thr["CD31"])
    ty0 = np.where(imm, 2, np.where(epi, 1, 0)).astype(np.int8)
    ty = ty0.copy()
    ty[(ty0 == 1) & (lin_cls == NORM)] = 0
    fixed = (ty0 == 0) & (lin_cls == TUM) & ~sv
    ty[fixed] = 1
    return ty0, ty, fixed
