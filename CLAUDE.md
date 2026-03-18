# H&E + Multiplex Feature Generator

Pipeline: H&E + multiplex OME-TIFF → patches → CellViT segmentation → cell type assignment → feature layers.

## Tech Stack

- Python 3.13, pytest
- Key libs: tifffile, Pillow, pandas, numpy, zarr, scipy, opencv (cv2)

## Project Structure

```
stages/patchify.py          # Stage 1: patches (MPP-ratio coordinate mapping, pre-aligned)
stages/assign_cells.py      # Stage 3: cell type assignment
stages/multiplex_layers.py  # Stage 4: vasculature/signaling layers
utils/normalize.py          # percentile_norm, percentile_to_uint8
utils/channels.py           # load_channel_metadata, resolve_channel_indices
utils/ome.py                # get_ome_mpp, open_zarr_store, read_overview_chw, get_image_dims
tests/                      # pytest suite
docs/plans/                 # Implementation plan docs
data/                       # Raw data (gitignored): CRC02-HE.ome.tif, CRC02.ome.tif, channel CSV
processed/                  # Stage 1 output (gitignored): he/, multiplex/, index.json
```

## Key Architecture

- **Coordinate mapping:** MPP ratio (`he_mpp / mx_mpp`) maps H&E patch coords to MX space. Images assumed pre-aligned.
- **Stage 2:** runs on Colab GPU via S3 bridge (`notebooks/cellvit_colab_stage2.ipynb`).

## User Preferences

- Use `data-engineering:data-engineer` agent for debugging scripts
- Prefer `--save-png PATH` for headless verification of GUI tools
- New plans: `docs/plans/YYYY-MM-DD-<name>.md`; mark **IMPLEMENTED** when done
- Use agents with `run_in_background=true` for parallelizable/isolated tasks
- Tests: `pytest`
