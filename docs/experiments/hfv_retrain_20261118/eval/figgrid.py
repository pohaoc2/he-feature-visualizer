"""ablation_grids_combined for the hfv_retrain_20261118 re-render (README figure table, coordinator 2026-11-19).

  PYTHONPATH=<this folder> python figrender.py run ROOT figgrid -- OLD_06_PNG
PixCell src/paper_figures/main.py builds this figure from 05_paired_ablation_grid.png (tile 10752_13824) and
06_unpaired_ablation_grid.png (tile 13056_27392, style 12032_15104). Panel a: new model, rendered here with PixCell's
build_representative_ablation_grid exactly as main.py calls it. Panel b: tile 13056_27392 has no new cell channels in
CRC33_01 (packgrid33 43393902, NO_CHANNELS), so README marks it "not redone (old-model input)" and the original
06_unpaired_ablation_grid.png (S3 share_space/05_results/figures_pngs_updated/individual/ablation_analysis/) is used as is.
Both go through PixCell's build_combined_ablation_grids_figure and save_figure_png to the same output path as main.py.
main.py itself is not run: it also builds the unpaired grid from the current tree, where that tile does not exist.
"""
import sys
from pathlib import Path

from src.paper_figures.fig_ablation_grid import build_representative_ablation_grid
from src.paper_figures.fig_combined_ablation_grids import build_combined_ablation_grids_figure
from src.paper_figures.main import ABLATION, PAIRED_DATASET_ROOT, PAIRED_METRICS_ROOT, PAIRED_REFERENCE_ROOT, SI_DIR
from src.paper_figures.style import apply_style
from tools.ablation_report.figures import save_figure_png

old06 = Path(sys.argv[1]).resolve()
apply_style()
ABLATION.mkdir(parents=True, exist_ok=True)
SI_DIR.mkdir(parents=True, exist_ok=True)
p05 = build_representative_ablation_grid(metrics_root=PAIRED_METRICS_ROOT, dataset_root=PAIRED_DATASET_ROOT,
                                         orion_root=PAIRED_REFERENCE_ROOT,
                                         out_png=ABLATION / "05_paired_ablation_grid.png", tile_id="10752_13824")
if p05 is None:
    sys.exit("STOP figgrid: paired grid for 10752_13824 not rendered")
fig = build_combined_ablation_grids_figure(ABLATION / "05_paired_ablation_grid.png", old06)
save_figure_png(fig, SI_DIR / "ablation_grids_combined.png")
print(f"figgrid: {SI_DIR / 'ablation_grids_combined.png'} (a new model {p05}; b old-model input {old06})")
