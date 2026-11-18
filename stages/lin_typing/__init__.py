"""Generator conditioning from Lin et al. 2023 (CRC33_01) cells, rendered with the sim2real renderer.

User decisions 2026-11-18 (he-feature-visualizer branch celltype-audit-20261118; not merged to main):
  objects  = Lin's cells (UNMICST2 nucleus dilated 5 px), not CellViT (option B);
  typing   = CELESTA + Lin Fig. 3d tissue-class fill-in (`celltype_celesta_fig3d`, byte-identical copy of
             sim-exp-validation scripts/analysis/tcmvp/celltype_celesta_fig3d.py at a3800e69);
  state    = Ki-67 above the Lin gate (pooled CRC33_01 + 02) -> prolif; CellViT dead (type 4) mapped by centroid -> dead;
  nuclei   = Lin cell mask eroded by the 5 px dilation (`nuclei`);
  channels = sim-exp-validation morphology_renderer.render_crop (outline arm) with a NucleusShapes library built from
             Lin nuclei (`render`), so training and inference channels come from the same code.
"""
