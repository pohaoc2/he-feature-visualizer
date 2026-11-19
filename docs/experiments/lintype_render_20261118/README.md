# lintype_render_20261118: generator conditioning from Lin cells, rendered with the sim2real renderer

Branch `celltype-audit-20261118` (not merged to main). Nothing was retrained. The user decisions of 2026-11-18 are the
spec. This continues `docs/experiments/celltype_audit_20261118`.

## 1. What was built

| Part | Rule | Code |
|---|---|---|
| Objects | Lin 2023 CRC33_01 cells (443,514) instead of CellViT objects (option B) | `stages/lin_typing/` |
| Population | CELESTA (tiled, default settings) + Lin Fig. 3d tissue-class fill-in | `celltype_celesta_fig3d.py`: byte-identical copy of sim-exp-validation `scripts/analysis/tcmvp/celltype_celesta_fig3d.py` @ a3800e69 (a test checks the sha256 and, when a checkout is present, identical output) |
| State | CellViT dead (type 4; its centroid falls on a Lin label) first; otherwise Ki-67 above the Lin gate (331.5, pooled CRC33_01 + 02) is prolif; otherwise nonprolif | `labels.py` |
| Nuclei | Lin cell mask (UNMICST2 nucleus dilated 5 px), eroded with a radius-5 disk | `nuclei.py` |
| Outline library | Lin nuclei of the cells in 400 random kept tiles; same 48 px / largest-part rule as `build_nucleus_shape_library.py`; NEW file `nucleus_shapes_lin_crc33_01.npz` | `nuclei.py`, `scripts/nuclei_render.py` |
| Channels | Each cell becomes one renderer agent (`NucleusShapes.pick(type, nucleus area, CellID)`, placed at the Lin centroid). The tile is painted by sim-exp-validation `morphology_renderer.render_crop(shape="outline", placement="jitter", allow_contact=True)` and written with `rasterize.write_channel_dir`. The renderer comes from sim-exp-validation e7c06626 (md5 in `data/PROVENANCE_e7c06626.md5`) | `render.py` |
| Tiles | 10,379 training tiles minus 961 ink tiles (`darkdiag ink.json` fraction > 0) = 9,418 | `render.ink_tiles` |
| Unchanged | vasculature / oxygen / glucose channels; H&E VAE and UNI features. The concat model's TME module ignores the mask VAE latent (`RawConditioningPassthrough.forward`: `del mask_latent`), so no latent needs recomputing | — |

Tests are in `tests/test_lin_typing.py`: 14 tests. On the Mac, 12 pass and 2 skip because skimage is missing; those 2 were
run and pass on the Hyak env.

### Erosion where cells touch (approximation; choice made from data)

Lin's mask never lets two labels share an edge. Touching cells are separated by a one-pixel background line, and 90.3 %
of the cells in the kept tiles touch another cell this way. Two variants differ only at those lines:

- **strict**: erode on all sides (the literal "erode by 5 px").
- **loose**: do not erode across the contact line.

Both were rendered. Paired with one-to-one CellViT objects (n = 151,036), the area ratio to CellViT is:

| Area ratio to CellViT (median) | strict | loose |
|---|---|---|
| isolated cells | 0.988 | 0.988 |
| touching cells | 0.741 | 1.216 |
| median abs log2 ratio | 0.468 | 0.37 |

The two variants bracket CellViT. On isolated cells, where the erosion is exact, both match CellViT's area. On the H&E
sample tiles, loose has the higher hematoxylin AUC (0.860 vs 0.837) and the higher Dice with CellViT (0.675 vs 0.535).
Strict is closer in total coverage (0.080 vs CellViT 0.084; loose 0.110).

**Default = loose**: `data/new_exp_channels.tar`. Strict: `data/new_exp_channels_strict.tar`.

## 2. Outputs (Hyak `/gscratch/cheme/chiu/hfv_lintype_20261118/`, new dir; per-cell data and images stay there)

| File | Content |
|---|---|
| `data/new_exp_channels.tar` (209 MB) | `orion-crc33/exp_channels/<7 cell channels>/<tile>.png` for 9,418 tiles, plus `orion-crc33/metadata/exp_index.hdf5` (key `exp_256`, 9,418 ids) — loose nuclei |
| `data/new_exp_channels_strict.tar` | same, strict nuclei |
| `out/nucleus_shapes_lin_crc33_01.npz` (+ `_strict`) | new outline libraries: 16,687 outlines (cancer 693, healthy 12,671, immune 3,323) |
| `out/cells_labels.npz`, `out/cells_nuclei.npz` | per-cell labels, nucleus areas, touching flag |
| `data/old_channels.tar` | current training channels (fetched for the comparison) |

## 3. Results (`data/*.json`, `figures/fig_lintype_render.png`)

Labels (cells in kept tiles: 398,137 Lin cells vs 246,754 CellViT objects):

| | current (hfv, CellViT objects) | new (Lin cells) |
|---|---|---|
| healthy / cancer / immune | 43.7 / 43.0 / 13.4 % | 77.2 / 3.2 / 19.6 % |
| prolif / nonprolif / dead | 26.4 / 73.3 / 0.23 % | 4.2 / 95.7 / 0.13 % |

- Where each CellViT object sits on a Lin cell (n = 225,199), the population agrees for 48.5 % of objects and the state
  for 75.3 %.
- Current cancer objects map to new healthy for 83 % of them, because 62 % of the kept-tile cells lie in Lin NORM.
- Lin TUM cells: 15,237. New cancer cells: 12,841 (epithelial outside NORM 10,582 + TUM fill-in 2,259).
- **Risk: CRC33_01 is mostly normal mucosa.** The cancer channel covers 0.6 % of a tile on average (was 6.2 %), so a
  generator retrained on this slide alone sees very few cancer examples. Adding tumour-rich slides (CRC34, CRC01, ...)
  with the same pipeline is the remedy; that is a user decision.
- Fill-in by tissue class (unknown → class): 66,982 cells. Of these, 8,729 come from BACK / DEB / MUC.

Mean area fraction per tile, kept tiles:

| Channel | current | new (loose) | new (strict) |
|---|---|---|---|
| mask | 0.124 | 0.168 | 0.093 |
| healthy | 0.046 | 0.129 | 0.075 |
| cancer | 0.062 | 0.006 | 0.003 |
| immune | 0.015 | 0.033 | 0.015 |
| prolif | 0.049 | 0.008 | 0.003 |
| nonprolif | 0.074 | 0.160 | 0.090 |
| dead | 0.0002 | 0.0002 | 0.0002 |

Agreement with the H&E (445 kept tiles of the audit's 500-tile random sample; median, bootstrap 95 % CI):

| Mask | hematoxylin AUC | Dice vs CellViT channel |
|---|---|---|
| CellViT (current channel) | 0.961 | — |
| Lin cell mask | 0.728 | 0.521 |
| Lin nuclei, strict | 0.837 | 0.535 |
| Lin nuclei, loose | 0.860 | 0.675 |
| rendered (loose; borrowed outlines) | 0.794 [0.788, 0.800] | 0.580 [0.570, 0.591] |
| rendered (strict) | 0.796 | 0.491 |

- CellViT outlines are traced on this H&E, so they align best.
- Lin's objects come from the registered Hoechst image.
- Borrowed outlines keep each cell's position and size but not its shape.
- In the loose rendering, 3.8 % of mask pixels are painted by two or more nuclei.

## 4. Deviations / notes

1. Erosion at contact lines: loose is the default; see §1. The strict variant is also built.
2. The library uses 400 tiles, the tile count of the sim-side builder. It is not filtered by tile edge, because whole
   objects are available.
3. Target nucleus area per cell is the eroded area of that cell. In strict mode, the 10,255 cells left empty by strict
   erosion take their loose area.
4. A first test run failed because 0 cells were found touching. That led to the gap-line finding in §1: the code was
   fixed and a test was added.

## 5. Retrain plan (not executed)

| Step | What |
|---|---|
| 1 | Unpack the current `orion-crc33` training data. Delete its 7 cell channel dirs under `exp_channels/` and its `metadata/exp_index.hdf5`, then untar `new_exp_channels.tar` over it. The index now lists 9,418 tiles, so the ink tiles drop out of training. vasculature / oxygen / glucose PNGs and the UNI / VAE features stay as they are |
| 2 | Use the same config, `PixCell/configs/config_controlnet_exp_a1_concat.py`, with the data root pointing at the new dir. Nothing else changes (bs 80, 20 epochs, AdamW lr 1e-5, warmup 200). Mask VAE latents are not needed, because the concat model drops them |
| 3 | Steps: 9,418 / 80 = 118 per epoch, × 20 = 2,360 steps (was 2,600). At the measured H200 rate (about 90 samples/s; 38 min for 2,600 steps), this is about 35 min, or about $0.55 on Tillicum (about $2 left) |
| 4 | Hyak l40s / a100 and Colab L4 throughput was never measured. Run a 50-step probe first: record samples/s and peak memory. If bs 80 does not fit, use gradient accumulation to keep the effective batch at 80. The total time is then 188,800 samples / measured rate |
| 5 | Train both variants (loose, strict) only if the user wants the erosion choice tested end to end. Otherwise train loose only |

## 6. Must be redone after a retrain

| Item | Why |
|---|---|
| Every generator-based sim2real output: sim H&E, UNI-based distance D, DED / ABC-SMC chains, tcmvp scoring | The generator changes |
| Sim side: switch `NucleusShapes` to `nucleus_shapes_lin_crc33_01.npz`, and recalibrate `NUCLEUS_RADIUS_PX` (8.4 px, calibrated to CellViT cell masks) | The training channels now come from Lin nuclei |
| Coverage-based archetype thresholds and real-tile channel statistics (exp_basis, preflight, style_pool) | Real channel fractions change (table in §3) |
| Decodability / channel-ablation analyses | New conditioning |
| hfv paper figures that show CellViT channels or hfv typing | Labels and objects changed |
| Tile lists and indices | 961 ink tiles are excluded |
| Mask VAE latents | Only if a grouped / TME model that uses them is retrained |

## 7. Data volume

- Fetched to Hyak: 79 MB, the current channels from own S3 by ranged reads. The 0.3 MB CellViT library was copied
  from the sim-exp-validation repo.
- Copied to the Mac: these json files and the figure only.
