# Stage 2 Paper Figure Design

## Goal

Build a publication-ready stage-2 pipeline figure as a Python-rendered PNG that matches the stage-1 paper-figure workflow. The stage-2 code must live under `tools/paper/figures/stage2`, and the resulting figure outputs must live under `paper/figures/stage2`.

## Scope

This design covers one focused deliverable:

- a Python builder for the final stage-2 paper figure
- companion metadata for reproducibility
- relocation of the existing brainstorming HTML mockups into the stage-2 figure output directory as reference artifacts

This design does not cover stage 3, combined stage 2 + stage 3 figure assembly, or generalized PixCell result browsing across arbitrary experiments.

## Approved Decisions

- Final deliverable: Python-rendered publication figure, not HTML-first output.
- Example count: one representative patch row.
- Patch selection rule: auto-pick the first patch in the existing stage-1 selection manifest that has a matching external PixCell `all/generated_he.png`.
- Current verified match in the available data: `23296_21760`.
- TME example tiles must reuse the same visual treatment already established in stage 1.
- Reference H&E can be any H&E tile; default behavior should use the processed H&E tile matching the chosen patch.

## Inputs And Source Assets

### Required repo-local inputs

- Stage-1 selection manifest:
  `paper/figures/stage1/crc33_marker_high_patch_examples_selections.json`
- Processed H&E tiles:
  `processed_crc33/he/<patch_id>.png`
- Processed TME example tiles:
  `processed_crc33/cell_types/union/<patch_id>.png`
  `processed_crc33/cell_states/union/<patch_id>.png`
  `processed_crc33/vasculature/<patch_id>.png`
  `processed_crc33/oxygen/<patch_id>.png`
  `processed_crc33/glucose/<patch_id>.png`

### Required external input

- PixCell generated H&E output:
  `/home/pohaoc2/UW/bagherilab/PixCell/inference_output/paired_ablation/ablation_results/<patch_id>/all/generated_he.png`

The builder should treat the PixCell output root as configurable, but default it to the path above.

## Output Artifacts

The stage-2 builder should write:

- `paper/figures/stage2/crc33_stage2_training_pipeline.png`
- `paper/figures/stage2/crc33_stage2_training_pipeline.json`

The existing mockup files should be moved into the same output directory:

- `paper/figures/stage2/stage2_pipeline_designs.html`
- `paper/figures/stage2/stage2_pipeline_design_A_mod.html`

The JSON metadata should record:

- selected patch id
- selection group and manifest path
- resolved source paths for all figure assets
- PixCell output root and matched generated-H&E path
- output layout parameters
- note that patch selection used the first selection-manifest match with a PixCell generated image

## Package Layout

The implementation should create a dedicated stage-2 package:

- `tools/paper/figures/stage2/__init__.py`
- `tools/paper/figures/stage2/build_training_figure.py`
- `tools/paper/figures/stage2/assets.py`
- `tools/paper/figures/stage2/render.py`

### Responsibilities

`assets.py`

- resolve the representative patch id from the stage-1 selection manifest
- check external PixCell result availability
- resolve all required tile/image paths
- load the reference H&E, TME tiles, and generated H&E

`render.py`

- draw the compact stage-2 pipeline panel using PIL
- reuse stage-1 visual idioms where appropriate, especially header bands, tile treatment, and scale-bar helpers
- keep drawing logic separate from asset discovery and CLI handling

`build_training_figure.py`

- provide the CLI entrypoint
- orchestrate asset resolution, rendering, output saving, and metadata emission

## Figure Content

The final figure should be a compact single-row stage-2 panel suitable for placement to the left of stage 3.

### Left-side inputs

- one reference H&E tile
- four TME example tiles labeled `type`, `state`, `vas`, and `O2 / glu`

The TME tiles should reuse the same example-tile look already established in stage 1:

- cell type and cell state tiles should preserve the established palettes
- vasculature should render as the same red overlay style used in stage-1 feature tiles
- oxygen and glucose should reuse the existing metabolic tile styling rather than a newly invented appearance

### Middle model section

The figure should render the approved compact training layout:

- reference H&E feeds frozen UNI
- UNI outputs a labeled latent connection into the denoiser path
- the four TME tiles feed four separate CNN blocks
- the CNN outputs merge into an attention block
- the attention output enters the PixCell denoiser as the conditioning latent
- the PixCell denoiser is explicitly frozen and shown as `ControlNet + base transformer`
- a noisy latent `Z_t` label should enter the denoiser from above as text, not as an image

### Right-side decode section

- the denoiser outputs a two-line centered `Denoised` / `latent Z` text label
- the denoised latent then feeds a frozen `SD3.5 VAE` decoder block
- the decoder outputs one generated H&E tile loaded from PixCell external results

### Styling rules

- frozen modules use striped backgrounds
- trainable modules use the green-highlighted style established during the design iteration
- arrows should be orthogonal and clearly connected to block edges
- the loss path should be a simple dashed path that indicates updates only for CNNs + attention
- the width should stay approximately half the visual width of the stage-1 panel so a future stage-3 panel can sit beside it

## Selection Algorithm

The representative patch resolver should:

1. read the stage-1 selection manifest in manifest order
2. iterate groups in the order they appear in the manifest file
3. iterate each group's `selections` list in file order
4. for each `patch_id`, test whether the external PixCell path exists at:
   `/home/pohaoc2/UW/bagherilab/PixCell/inference_output/paired_ablation/ablation_results/<patch_id>/all/generated_he.png`
5. return the first matching patch

With the current verified data, that rule selects `23296_21760`.

This behavior should stay deterministic and should not depend on directory listing order.

## CLI Requirements

The main builder CLI should support:

- overriding the selection manifest path
- overriding the processed directory path
- overriding the PixCell output root
- overriding the output directory
- optionally forcing a specific patch id instead of auto-picking

Default behavior should require no extra arguments for the current CRC33 setup.

## Error Handling

Failures must be explicit and path-specific.

Required failure cases:

- selection manifest missing or malformed
- no selection-manifest patch has a matching PixCell `generated_he.png`
- chosen patch is missing one or more processed local tiles
- output directory cannot be created or written

If auto-pick fails, the error message should list the checked patch ids so the missing-match condition is obvious.

## Testing Strategy

Add focused tests that avoid dependence on the real external PixCell directory.

### Unit tests

- selection-order test: the resolver returns the first matching selected patch, not an arbitrary later match
- missing-match test: resolver raises a clear error when no selected patch has generated output
- metadata test: emitted JSON records the chosen patch and source paths

### Rendering tests

- synthetic asset test: builder can render a PNG from temporary fake local tiles and a fake PixCell result tree
- output-size smoke test: rendered canvas has the expected dimensions and non-empty content

Tests should use temporary directories and synthetic images for all external PixCell paths.

## Migration Notes

The current HTML mockups are brainstorming artifacts, not final deliverables. They should be moved under `paper/figures/stage2` so all stage-2 visual material is co-located, but the authoritative final figure should be produced by the Python builder.

## Out Of Scope

- SVG or browser-rendered final figure generation
- interactive stage-2 design tooling
- automatic assembly of stage 1 + stage 2 + stage 3 into one combined paper figure
- support for multiple representative rows in the first implementation