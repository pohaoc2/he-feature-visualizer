# Design: CellViT vs Astir Comparison Figures

**Date:** 2026-03-14  
**Status:** Approved in chat  
**Primary surfaces:** static scientific/QC figures  
**Primary files expected:** `stages/assign_cells.py`, `tools/scientific_vis_cellvit_mx.py`, new sample-level report tool under `tools/`

## Goal

Create side-by-side visual outputs that compare:

1. CellViT cell type assignment
2. Astir cell type assignment
3. Final fused cell assignment

The design must also add a separate, isolated visualization explaining Astir output using:

1. population-level cell type and probability distributions, and
2. representative per-cell examples showing marker evidence and class probabilities.

This work is for static figure generation first, not an interactive viewer.

## User Decisions Locked In

- Output surface: static figures first
- Reporting scope: both patch-level and sample-level outputs
- CellViT comparison view: mapped into the shared 3-class ontology
- Astir explanation surface: both population-level and representative-cell views
- Patch selection: automatic representative patch selection
- Structure: two figure generators, not one overloaded tool

## Existing Pipeline Context

The current pipeline already computes most of the required model outputs in Stage 3:

- Astir probabilities via `p_model_cancer`, `p_model_immune`, `p_model_healthy`
- final fused probabilities via `p_final_cancer`, `p_final_immune`, `p_final_healthy`
- final label via `cell_type`
- Astir top label via `type_astir`
- mapped CellViT prior label via `type_cellvit_prior`
- mismatch flag via `is_mismatch`
- confidence via `cell_type_confidence`
- sample summary via `cell_summary.json`
- patch overlays via `cell_types/*.png` and `cell_states/*.png`

The current scientific figure tool already renders a patch-level H&E/MX/CellViT/final/state figure, so the new design extends existing patterns instead of replacing them.

## Shared Label Ontology

All comparison figures use the same 3-class ontology:

- `cancer`
- `immune`
- `healthy`

Cell state remains:

- `dead`
- `proliferative`
- `quiescent`

`dead` stays in the state visualization only. It is not shown as a fourth type class in the CellViT/Astir/final comparison panels.

## CellViT Mapping for Comparison

For the comparison panels, CellViT must be mapped into the shared 3-class view:

- `Neoplastic` -> `cancer`
- `Inflammatory` -> `immune`
- `Connective` -> `healthy`
- `Epithelial` -> mapped through the existing split prior and exposed as the prior winner used by Stage 3
- `Dead` -> excluded from the type overlay and preserved in state reporting
- unknown/background -> neutral or omitted depending on current Stage 3 behavior

The figure labels should make it clear this is a mapped CellViT view, not the native CellViT ontology.

## Recommended Architecture

Use two figure generators powered by one shared normalized per-cell artifact.

### 1. Stage 3 shared artifact

Extend `stages/assign_cells.py` to write a normalized per-cell table, for example:

- `cell_assignments.csv`

One row represents one matched cell and contains:

- `patch_id`
- patch origin and local/global coordinates
- `cell_id` or stable row index if available
- `type_cellvit`
- `type_cellvit_prior`
- `type_astir`
- `cell_type`
- `cell_state`
- `cell_type_confidence`
- `is_mismatch`
- `p_model_cancer`, `p_model_immune`, `p_model_healthy`
- `p_final_cancer`, `p_final_immune`, `p_final_healthy`
- marker values used for Astir input
- optional distance-to-match metadata if already available cheaply

This table is the single analysis-ready interface for downstream visualization.

### 2. Patch-level figure generator

Keep `tools/scientific_vis_cellvit_mx.py` focused on patch-level spatial comparison.

Extend it, or add a closely related patch report tool, to consume:

- processed patch assets
- `cell_assignments.csv`
- `cell_summary.json`

### 3. Sample-level explanation figure generator

Add a new tool under `tools/` for sample-level Astir explanation and summary reporting.

It consumes:

- `cell_assignments.csv`
- `cell_summary.json`
- optional processed imagery for representative-cell crops

## Why This Architecture

This split is preferred over stuffing all logic into one tool because it:

- matches the repo's existing tool-oriented structure
- keeps spatial comparison and population explanation as separate concerns
- avoids re-deriving probabilistic outputs from partial artifacts
- creates a reusable data-engineering boundary: compute once, visualize many ways

## Patch-Level Figure Design

Each selected representative patch produces a scientific/QC panel figure with this layout:

1. `H&E`
2. `MX marker view`
3. `CellViT contours`
4. `CellViT mapped 3-class overlay`
5. `Astir top-class overlay`
6. `Final fused assignment overlay`
7. `Cell state overlay`
8. `Compact evidence panel`

### Patch figure requirements

- Use one consistent 3-class color palette across CellViT mapped, Astir, and final panels
- Keep state colors separate from type colors
- Show patch metadata in the title:
  - patch id
  - number of cells
  - mismatch rate
  - mean or median confidence
- Choose the MX channel automatically from the most discriminative available marker for the patch
- Support CLI override for a specific marker/channel

### Compact evidence panel

The final panel summarizes the patch with:

- class counts for `CellViT mapped`
- class counts for `Astir`
- class counts for `final`
- mismatch rate
- confidence distribution
- a short marker-evidence summary for the patch

### Representative patch selection

Patch selection is automatic by default.

Recommended ranking:

1. highest mismatch rate
2. highest count of low-confidence cells
3. high-confidence exemplar patches

This ensures the report shows both failure cases and clean examples.

Patches with no matched cells are skipped.

## Sample-Level Astir Explanation Figure Design

This is a separate, isolated figure or small figure set dedicated to explaining Astir behavior.

It has two components.

### 1. Population-level explanation

Required panels:

- class count bar chart for `CellViT mapped`, `Astir`, and `final`
- mismatch/confusion heatmap between `CellViT mapped` and `Astir`
- class probability distributions for Astir outputs
- per-marker distributions grouped by final class

Recommended plot types:

- grouped bar plots for class counts
- heatmap for mismatch/confusion
- violin or box plots for `p_model_*`
- box/violin/swarm overlays for marker distributions when readable

### 2. Representative-cell explanation

For each final class (`cancer`, `immune`, `healthy`), include representative cells:

- one high-confidence match
- one ambiguous or low-margin example
- one disagreement example when available

Each representative cell panel should include:

- small H&E crop centered on the cell
- cell contour overlay
- optional small MX evidence strip
- mapped CellViT label
- Astir top label
- final fused label
- Astir probability bar chart
- final fused probability bar chart
- marker values for the most relevant markers

## Evidence Language and Interpretability Guardrail

Astir provides class probabilities, but it does not inherently provide rigorous per-marker causal attribution suitable for strong explanatory claims.

To avoid overstating what the model knows, the visualization should present:

- marker evidence actually observed for the cell or class
- Astir class probability vectors
- final fused probability vectors
- whether the final call changed because of CellViT prior fusion

The report should avoid claiming exact causal attribution unless an explicit attribution method is added later.

In short, the explanation surface shows **evidence** for categorization, not a guaranteed causal decomposition.

## Astir Failure and Fallback Mode

Fallback mode already exists in the code path and should be reflected clearly in the figures.

### What counts as Astir failure

Astir is considered unavailable when Stage 3 cannot:

- import the `astir` package
- initialize the Astir model
- fit the type model
- retrieve cell-type probabilities

These cases are wrapped as `AstirUnavailableError`.

### What fallback mode means

If the user requested Astir mode and enabled `--allow-astir-fallback`, Stage 3 falls back to the built-in rule classifier instead of failing the whole run.

That rule classifier computes heuristic probabilities from marker evidence:

- `cancer`: mostly `PanCK` and `Ecadherin`
- `immune`: mostly `CD45` plus immune markers
- `healthy`: mostly `SMA` and `CD31`

The summary already records this as:

- `classifier_requested = "astir"`
- `classifier_used = "rule_fallback"`

### Figure behavior in fallback mode

- figures still render
- titles and subtitles must clearly state `rule_fallback`
- Astir-specific explanation panels should either:
  - be suppressed, or
  - be labeled as rule-based probability explanation, not Astir explanation

The design must never present fallback outputs as genuine Astir results.

## Data Flow

1. Stage 3 computes model probabilities and final assignments
2. Stage 3 writes:
   - `cell_types/*.png`
   - `cell_states/*.png`
   - `cell_summary.json`
   - new `cell_assignments.csv`
3. Patch report tool reads patch assets plus the normalized table
4. Sample report tool reads the normalized table plus summary JSON
5. Both tools save publication/QC-ready figures in PNG and PDF by default

## Error Handling

- If fallback mode was used, label outputs clearly and suppress Astir-only explanation claims
- If some markers are missing, render only available evidence panels and note reduced marker coverage
- If a class lacks representative disagreement cells, omit that slot instead of filling with a misleading substitute
- If a selected patch has no matched cells, skip it
- If image crops cannot be recovered for a representative cell, fall back to non-image probability/evidence panels for that example

## Testing Strategy

### Stage 3 tests

Add tests for:

- per-cell export table creation
- mapped CellViT 3-class export values
- `classifier_used` propagation into figure-consumable metadata

### Patch figure tests

Add smoke tests that:

- generate a patch comparison figure from synthetic or fixture data
- verify output files are written
- verify fallback labeling appears when classifier mode is `rule_fallback`

### Sample figure tests

Add tests for:

- representative patch ranking
- representative-cell bucket selection:
  - matched
  - ambiguous
  - disagreement
- sample-level figure generation smoke test

## Out of Scope

- interactive/browser viewer work
- claiming true feature attribution for Astir without a dedicated method
- changing the Stage 3 ontology away from `cancer`, `immune`, `healthy`
- replacing CellViT or Astir model behavior itself
- native CellViT class reporting as the main comparison surface

## Implementation Notes

- Prefer reusing the existing scientific visualization styling helper already used by `tools/scientific_vis_cellvit_mx.py`
- Keep visualization logic out of Stage 3 except for emitting the normalized per-cell table
- Keep the sample-level explanation tool isolated from the patch-level renderer

## Open Questions Deferred

These were intentionally deferred from the design because they do not block implementation planning:

- exact heuristic for "most discriminative" MX channel in a patch
- exact number of representative patches and representative cells per class
- exact final file names for the new sample-level report tool and output directory

These can be resolved in the implementation plan without changing the approved architecture.
