# Stage 2 Pipeline Figure Redesign

**Date:** 2026-04-20  
**Target file:** `paper/figures/stage2/stage2_pipeline_design_A_mod.html`  
**Status:** Approved for implementation

---

## Problem

The current `stage2_pipeline_design_A_mod.html` has four issues:
1. Too tall — O2 and glucose were split into separate rows
2. Text overlaps arrows and blocks throughout
3. Python-rendered PNG is low-quality; HTML/SVG is the better output format
4. PixCell denoiser position inconsistent with the original `stage2_pipeline_designs.html` reference

---

## Architecture: What the diagram shows

The Stage 2 training pipeline has two input streams and a frozen denoising backbone:

**Input stream 1 — H&E encoder:**
- Reference H&E patch → UNI (frozen encoder) → UNI latent [B,1536]

**Input stream 2 — TME conditioning bank (9 trainable CNNs):**
- Stage 1 TME feature layers → 9 CNNs, grouped into 3 rows of CNN×3:
  - Cell types: tumor · healthy · immune (3 CNNs)
  - Cell states: prolif · nonprolif · dead (3 CNNs)
  - Microenv: vasculature · glucose · O₂ (3 CNNs)
- 9 CNN outputs → Attention fusion block → conditioning latent

**Frozen denoising backbone:**
- PixCell denoiser (ControlNet + base transformer, frozen)
  - Receives: UNI latent (top-left input), noisy Z_t (top input), conditioning latent (bottom-left input)
  - Produces: denoised Z

**Decoder:**
- Denoised Z → SD3.5 VAE (frozen) → generated H&E

**Loss:**
- v-loss on generated H&E; gradient flows back only to CNN×3 + Attention (backbone stays frozen)

---

## Layout

**Overall:** Layout B — left column feeding right pipeline.

```
[ref H&E]──[UNI]────────────────────────────[PixCell denoiser]──[SD3.5 VAE]
                     noisy Z_t ↓               ↑         ↑
[types]──[CNN×3 tumor/healthy/immune ]         |         |
[states]─[CNN×3 prolif/nonprolif/dead]──[Attention]──────┘     [gen H&E]
[micro]──[CNN×3 vas/glu/O₂           ]

          └──────────────── v-loss (dashed red) ──────────────────┘
```

**Page width:** max-width 640px (narrow, designed for side-by-side with Stage 3 panel).  
**SVG viewBox:** `0 0 510 258`

---

## Visual design

| Element | Fill | Stroke | Notes |
|---|---|---|---|
| Frozen blocks (UNI, PixCell, VAE) | `frozenStripe` pattern | `#8e8880` | 45° diagonal hatch |
| Trainable blocks (CNN rows, Attention) | `#dff4e3` | `#2d8c4f` | Green |
| H&E patches (input + output) | `#f6c8d5` | `#b45b77` | Pink with dot scatter |
| Gradient arrow | — | `#b33b34` dashed 5,4 | Red dashed |
| Data flow arrows | — | `#59544f` | Dark neutral |
| TME arrows | — | `#2d8c4f` | Green |

**Colors reused from existing `stage2_pipeline_design_A_mod.html`** CSS variables.

---

## Typography

All text inside PixCell denoiser block uses **small font** (≤8.5px) to keep the block visually compact:
- Block title "PixCell denoiser": font-size 11, bold
- Sub-labels "ControlNet + base transformer · frozen": font-size 7.5
- Internal sub-box "denoise Z_t → Z": font-size 8.5

CNN×3 rows:
- "CNN × 3": font-size 9.5, bold, left-center of box
- Sub-labels (variant names): font-size 7, right-center of box, two lines

---

## Components

| Component | SVG coords | Notes |
|---|---|---|
| Reference H&E patch | x=10, y=22, w=42, h=42 | Pink scatter dots |
| UNI (frozen) | x=72, y=29, w=58, h=28 | Striped fill |
| UNI latent arrow | (130,43)→(256,43) | Label "UNI latent [B,1536]" above |
| noisy Z_t | label at x=326, y=11; arrow down to y=22 | Enters PixCell top |
| **PixCell denoiser** | x=258, y=22, w=140, h=142 | Striped fill; contains denoise sub-box |
| Denoised Z arrow | (398,72)→(406,72) | Short; label above |
| SD3.5 VAE | x=408, y=50, w=44, h=44 | Striped fill |
| Generated H&E | x=408, y=120, w=44, h=44 | Pink scatter dots |
| TME thumbnails (×3) | x=10, y=100/140/180, w=30, h=26 | Blue/green/orange fills |
| CNN×3 type row | x=52, y=100, w=118, h=26 | |
| CNN×3 state row | x=52, y=140, w=118, h=26 | |
| CNN×3 micro row | x=52, y=180, w=118, h=26 | |
| Attention | x=198, y=139, w=52, h=28 | Green fill |
| Conditioning arrow | (250,153)→(258,153) | Green; "cond." label above |
| v-loss path | M430,164 L430,224 L224,224 L224,167 | Red dashed |

---

## What changes vs current file

1. **9 CNNs** shown as 3 compact "CNN×3" rows (replaces 4 separate CNN boxes)
2. **PixCell denoiser** smaller (140×142 vs 180×132) with smaller internal labels
3. **O2 and glucose** merged into "microenv" row (vas · glu · O₂) — eliminates extra height
4. **No Python rendering** — HTML/SVG is the deliverable; `build_training_figure.py` not invoked
5. **Text/arrow overlaps** eliminated by careful coordinate layout
6. **VAE + generated H&E** shifted left (~50px) vs current file

---

## Out of scope

- Changes to `tools/paper/figures/stage2/` Python code (not needed; figure is HTML-only)
- Stage 3 panel (separate figure)
- Actual patch image thumbnails (SVG placeholder icons retained)
