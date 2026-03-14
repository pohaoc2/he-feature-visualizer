# Design: Assign Cells Astir-First (Probabilistic) for CRC

**Date:** 2026-03-14  
**Status:** APPROVED (direction locked in chat: "Astir-first")  
**Primary files:** `stages/assign_cells.py`, `stages/extract_cell_features.py`, `tests/test_assign_cells.py`

## Motivation

The current Stage 3 logic is deterministic and threshold-heavy. For new marker panels,
we want a probabilistic classifier that:

1. handles marker variability better across slides,
2. exposes calibrated confidence,
3. keeps CellViT morphology as a secondary prior instead of the primary driver.

## Goals

- Use **Astir as the primary type model** for CRC 3-class typing:
  - `cancer`, `immune`, `healthy`
- Keep state schema:
  - `dead`, `proliferative`, `quiescent`
- Keep dead-state override from CellViT (`type_cellvit == 4`)
- Preserve dual input mode:
  - provided `--features-csv`, or
  - auto-generated CellViT+MX features CSV
- Preserve existing output artifacts:
  - `cell_types/*.png`, `cell_states/*.png`, `cell_summary.json`

## Label Ontology (Locked)

### Cell type

- `cancer`
- `immune`
- `healthy`

### Cell state

- `dead`
- `proliferative`
- `quiescent`

## CellViT Prior Mapping (for mismatch and low-confidence handling)

```python
CELLVIT_TYPE_NAMES = {
    0: "background",    # ignore
    1: "Neoplastic",    # cancer prior
    2: "Inflammatory",  # immune prior
    3: "Connective",    # healthy prior
    4: "Dead",          # dead state override
    5: "Epithelial",    # split prior
}
```

Prior probabilities:

- type 1 -> `{cancer:1.0, immune:0.0, healthy:0.0}`
- type 2 -> `{cancer:0.0, immune:1.0, healthy:0.0}`
- type 3 -> `{cancer:0.0, immune:0.0, healthy:1.0}`
- type 5 -> `{cancer:0.5, immune:0.0, healthy:0.5}` (split prior)
- type 0 -> ignored in reporting (or neutral prior if kept)
- type 4 -> type prior optional; state forced to dead

## Astir-First Type Modeling

## Marker panel (v1, locked to provided 19 channels)

Available channels:

- `Hoechst`
- `AF1`
- `CD31`
- `CD45`
- `CD68`
- `Argo550`
- `CD4`
- `FOXP3`
- `CD8a`
- `CD45RO`
- `CD20`
- `PD-L1`
- `CD3e`
- `CD163`
- `E-cadherin`
- `PD-1`
- `Ki67`
- `Pan-CK`
- `SMA`

Astir marker dictionary (v1):

- `cancer`: `Pan-CK`, `E-cadherin`
- `immune`: `CD45`, `CD3e`, `CD4`, `CD8a`, `CD20`, `CD68`, `CD163`, `FOXP3`, `CD45RO`, `PD-1`
- `healthy`: `SMA`, `CD31`

Aliases are resolved before modeling (e.g., `PanCK/Pan-CK`, `Ecadherin/E-cadherin`, `PD1/PD-1`).

Markers not used for v1 typing (tracked in summary for audit):
- `Hoechst`, `AF1`, `Argo550`, `PD-L1`

## Primary posterior

Let `P_astir(type | x)` be Astir type posterior per cell.

Astir is the primary signal, but we blend a small CellViT prior for stability:

```text
P_final(type) = 0.85 * P_astir(type) + 0.15 * P_cellvit_prior(type)
type_final = argmax P_final(type)
```

This is intentionally Astir-first.

## State Modeling (Astir-first compatible)

1. **Dead override**
   - If `type_cellvit == 4`, state is `dead` (hard override).

2. **Living-state assignment**
   - For non-dead cells, use marker evidence for proliferation:
   - Primary marker: `Ki67` (present in this panel).
   - `PCNA` is not part of the current 19-marker panel and is not used in v1.

v1 rule (locked from discussion):

```text
if Ki67_norm >= p75: proliferative
else: quiescent
```

Note: this keeps state deterministic while type is probabilistic; this is acceptable for v1
because dead/proliferative decisions remain interpretable and robust with limited state markers.

## Mismatch and Confidence

### Mismatch definition

- `type_astir = argmax P_astir`
- `type_cellvit_prior = argmax P_cellvit_prior`
- mismatch if `type_astir != type_cellvit_prior`

### Confidence

Use final posterior margin:

```text
margin = top1(P_final) - top2(P_final)
```

- `high`: `margin >= 0.25`
- `medium`: `0.12 <= margin < 0.25`
- `low`: `< 0.12`

If mismatch is true, downgrade one confidence level (high->medium, medium->low).

## Per-cell output fields (required)

- `cell_type` (`cancer|immune|healthy`)
- `cell_state` (`dead|proliferative|quiescent`)
- `cell_type_confidence` (`high|medium|low`)
- `type_astir`
- `type_cellvit_prior`
- `is_mismatch` (bool)
- `p_final_cancer`, `p_final_immune`, `p_final_healthy`

## Summary-level outputs (required)

`cell_summary.json` must include:

- type/state counts
- confidence distribution
- mismatch count/rate
- marker availability stats
- model mode metadata (`astir_first: true`)

## Pipeline Data Flow

1. Load features table
   - from `--features-csv`, or auto-extract from CellViT+MX
2. Canonicalize marker names and build Astir input matrix
3. Run Astir type model -> `P_astir`
4. Build CellViT type prior -> `P_cellvit_prior`
5. Blend posteriors (`0.85/0.15`) and assign `cell_type`
6. Assign `cell_state` with dead override + Ki67 p75
7. Rasterize cell type/state maps
8. Emit per-patch and global summary

## Error Handling and Fallbacks

- If Astir dependency/model is unavailable:
  - fail fast in strict mode, or
  - fallback to existing rule-based model when `--allow-astir-fallback` is set.
- If core markers are missing:
  - continue with available markers and log reduced-confidence warning.
- If Ki67 missing:
  - state for non-dead cells defaults to `quiescent` with warning.

## Out of Scope

- Subtyping within immune/healthy classes
- Learning dead state from MX only (dead remains CellViT override)
- Replacing Stage 2 CellViT inference

## Compatibility Notes

- Existing directory outputs are unchanged.
- Dual mode remains unchanged for CSV generation.
- This design supersedes threshold-first type assignment logic for CRC runs.
