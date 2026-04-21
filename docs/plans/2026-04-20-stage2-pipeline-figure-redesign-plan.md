# Stage 2 Pipeline Figure Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `paper/figures/stage2/stage2_pipeline_design_A_mod.html` with a compact redesigned SVG (viewBox 510×258, max-width 640px) that collapses 4 CNNs into 3 "CNN×3" rows, merges O₂/glucose into a single microenv row, repositions PixCell denoiser, and eliminates text/arrow overlaps.

**Architecture:** Single HTML file edit — pure SVG/HTML, no Python. New layout uses viewBox `0 0 510 258`, a left TME column (x=10) feeding 3 compact CNN×3 rows (x=52) into an Attention block (x=198), which feeds PixCell (x=258, 140×142) alongside a UNI latent from the top-left. SD3.5 VAE and generated H&E shift left to x=408.

**Tech Stack:** HTML, SVG, pytest (string-content assertions)

---

## File Structure

- Modify: `paper/figures/stage2/stage2_pipeline_design_A_mod.html` — full SVG replacement
- Modify: `tests/test_stage2_paper_figure.py` — add structural assertions for new SVG

---

### Task 1: Add Failing Structural Tests

**Files:**
- Modify: `tests/test_stage2_paper_figure.py`

- [ ] **Step 1: Append structural tests**

Add the following test after the existing `test_stage2_package_and_mockups_exist` function in `tests/test_stage2_paper_figure.py`:

```python
def test_stage2_pipeline_redesign_svg_structure() -> None:
    """Validate the redesigned stage2_pipeline_design_A_mod.html matches spec."""
    repo = Path(__file__).resolve().parents[1]
    html = (repo / "paper/figures/stage2/stage2_pipeline_design_A_mod.html").read_text()

    # Viewport: narrower 510×258 viewBox
    assert 'viewBox="0 0 510 258"' in html, "viewBox must be 0 0 510 258"

    # Page width: max-width 640px
    assert "max-width: 640px" in html, "page max-width must be 640px"

    # CNN×3 rows: exactly 3 occurrences
    assert html.count("CNN \u00d7 3") == 3, "must have exactly 3 CNN \u00d7 3 labels"

    # Row labels present
    assert "tumor" in html and "healthy" in html and "immune" in html
    assert "prolif" in html and "nonprolif" in html and "dead" in html
    assert "vas" in html and "glu" in html

    # PixCell block at new x=258
    assert 'x="258"' in html, "PixCell rect must start at x=258"

    # SD3.5 VAE block at new x=408
    assert 'x="408"' in html, "VAE rect must start at x=408"

    # No old CNN1/CNN2/CNN3/CNN4 labels (replaced by CNN×3)
    assert "CNN1" not in html
    assert "CNN2" not in html
    assert "CNN3" not in html
    assert "CNN4" not in html
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/pohaoc2/UW/he-feature-visualizer
pytest tests/test_stage2_paper_figure.py::test_stage2_pipeline_redesign_svg_structure -v
```

Expected: FAIL — current HTML has viewBox `0 0 620 390`, CNN1–CNN4, max-width 700px.

- [ ] **Step 3: Commit failing tests**

```bash
git add tests/test_stage2_paper_figure.py
git commit -m "test: add structural assertions for stage2 pipeline figure redesign"
```

---

### Task 2: Rewrite stage2_pipeline_design_A_mod.html

**Files:**
- Modify: `paper/figures/stage2/stage2_pipeline_design_A_mod.html`

Fully replace the file contents with the redesigned HTML below. All coordinates follow the spec in `docs/plans/2026-04-20-stage2-pipeline-figure-redesign.md`.

**Key coordinate changes from current file:**

| Element | Old | New |
|---|---|---|
| `viewBox` | `0 0 620 390` | `0 0 510 258` |
| `max-width` | `700px` | `640px` |
| PixCell rect | `x=330 y=58 w=160 h=110` | `x=258 y=22 w=140 h=142` |
| SD3.5 VAE | `x=516 y=176 w=56 h=58` | `x=408 y=50 w=44 h=44` |
| Generated H&E | `x=512 y=274 w=64 h=60` | `x=408 y=120 w=44 h=44` |
| CNN boxes | 4 separate full-label boxes | 3 compact CNN×3 rows |
| TME thumbnails | 4 rows (type/state/vas/O2-glu) | 3 rows (types/states/micro) |

- [ ] **Step 1: Write the new HTML file**

Replace the entire contents of `paper/figures/stage2/stage2_pipeline_design_A_mod.html` with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Stage 2 Pipeline — Compact Design A</title>
<style>
  :root {
    --bg: #f3f1eb;
    --card: #fffdf9;
    --ink: #24211d;
    --muted: #6d675f;
    --line: #59544f;
    --frozen-stroke: #8e8880;
    --train-fill: #dff4e3;
    --train-stroke: #2d8c4f;
    --latent-fill: #eef2f9;
    --latent-stroke: #8aa0c3;
    --he-fill: #f6c8d5;
    --he-stroke: #b45b77;
    --loss: #b33b34;
  }

  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "Helvetica Neue", Arial, sans-serif;
    padding: 28px;
  }

  .page {
    max-width: 640px;
    margin: 0 auto;
  }

  h1 {
    margin: 0 0 8px;
    font-size: 22px;
    line-height: 1.2;
  }

  .subtitle {
    margin: 0 0 18px;
    color: var(--muted);
    font-size: 13px;
    line-height: 1.45;
  }

  .card {
    background: var(--card);
    border: 1px solid #d7d0c7;
    border-radius: 14px;
    padding: 18px 18px 14px;
    box-shadow: 0 8px 24px rgba(42, 35, 25, 0.08);
  }

  .caption {
    margin: 0 0 10px;
    color: var(--muted);
    font-size: 12px;
    line-height: 1.45;
  }

  svg {
    display: block;
    width: 100%;
    height: auto;
  }

  .legend {
    display: flex;
    gap: 18px;
    align-items: center;
    margin-top: 12px;
    color: var(--muted);
    font-size: 11px;
    flex-wrap: wrap;
  }

  .legend-item {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }

  .swatch {
    width: 18px;
    height: 12px;
    border-radius: 3px;
    border: 1px solid #aaa39a;
  }
</style>
</head>
<body>
<div class="page">
  <h1>Stage 2 — Compact Training Panel</h1>
  <p class="subtitle">Narrow panel for side-by-side layout with stage 3. Nine Stage 1 TME layers feed three CNN×3 groups; attention fuses them into a conditioning latent fed to the frozen PixCell denoising backbone alongside a UNI latent.</p>

  <div class="card">
    <p class="caption">Three trainable CNN×3 groups (cell types · cell states · microenv) condition the frozen PixCell denoiser. Gradient updates only CNN×3 + attention via v-loss.</p>

    <svg viewBox="0 0 510 258" xmlns="http://www.w3.org/2000/svg" aria-label="Stage 2 compact architecture diagram">
      <defs>
        <pattern id="frozenStripe" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <rect width="8" height="8" fill="#f7f4ef"/>
          <line x1="0" y1="0" x2="0" y2="8" stroke="#c6c0b7" stroke-width="3"/>
        </pattern>
        <marker id="arrowDark" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#59544f"/>
        </marker>
        <marker id="arrowGreen" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#2d8c4f"/>
        </marker>
        <marker id="arrowRed" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 Z" fill="#b33b34"/>
        </marker>
      </defs>

      <!-- ── Reference H&E patch ── -->
      <rect x="10" y="22" width="42" height="42" rx="6" fill="#f6c8d5" stroke="#b45b77" stroke-width="1.8"/>
      <circle cx="20" cy="34" r="2.2" fill="#93485f"/>
      <circle cx="34" cy="29" r="2.2" fill="#93485f"/>
      <circle cx="44" cy="42" r="2.2" fill="#93485f"/>
      <circle cx="24" cy="54" r="2.2" fill="#93485f"/>
      <circle cx="40" cy="58" r="2.2" fill="#93485f"/>
      <text x="31" y="78" text-anchor="middle" font-size="9" fill="#6d675f">ref H&amp;E</text>

      <!-- ── Arrow: ref H&E → UNI ── -->
      <line x1="52" y1="43" x2="70" y2="43" stroke="#59544f" stroke-width="1.8" marker-end="url(#arrowDark)"/>

      <!-- ── UNI frozen encoder ── -->
      <rect x="72" y="29" width="58" height="28" rx="7" fill="url(#frozenStripe)" stroke="#8e8880" stroke-width="1.8"/>
      <text x="101" y="45" text-anchor="middle" font-size="11" font-weight="700" fill="#24211d">UNI</text>
      <text x="101" y="53" text-anchor="middle" font-size="7.5" fill="#6d675f">frozen</text>

      <!-- ── Arrow: UNI → PixCell (UNI latent) ── -->
      <line x1="130" y1="43" x2="256" y2="43" stroke="#59544f" stroke-width="1.8" marker-end="url(#arrowDark)"/>
      <text x="193" y="38" text-anchor="middle" font-size="8" fill="#33435d">UNI latent [B,1536]</text>

      <!-- ── noisy Z_t input ── -->
      <text x="326" y="11" text-anchor="middle" font-size="9" fill="#33435d">noisy Z<tspan baseline-shift="sub" font-size="7">t</tspan></text>
      <line x1="326" y1="14" x2="326" y2="18" stroke="#59544f" stroke-width="1.8" marker-end="url(#arrowDark)"/>

      <!-- ── PixCell denoiser (frozen) ── -->
      <rect x="258" y="22" width="140" height="142" rx="10" fill="url(#frozenStripe)" stroke="#8e8880" stroke-width="2"/>
      <text x="328" y="40" text-anchor="middle" font-size="11" font-weight="700" fill="#24211d">PixCell denoiser</text>
      <text x="328" y="50" text-anchor="middle" font-size="7.5" fill="#6d675f">ControlNet + base transformer · frozen</text>
      <rect x="283" y="58" width="90" height="18" rx="5" fill="#ffffffc8" stroke="#bfb7ad" stroke-width="1"/>
      <text x="328" y="71" text-anchor="middle" font-size="8.5" fill="#59544f">denoise Z<tspan baseline-shift="sub" font-size="7">t</tspan> → Z</text>

      <!-- ── Arrow: PixCell → VAE (denoised Z) ── -->
      <line x1="398" y1="72" x2="406" y2="72" stroke="#59544f" stroke-width="1.8" marker-end="url(#arrowDark)"/>

      <!-- ── SD3.5 VAE (frozen) ── -->
      <rect x="408" y="50" width="44" height="44" rx="7" fill="url(#frozenStripe)" stroke="#8e8880" stroke-width="1.8"/>
      <text x="430" y="70" text-anchor="middle" font-size="9.5" font-weight="700" fill="#24211d">SD3.5</text>
      <text x="430" y="82" text-anchor="middle" font-size="9.5" font-weight="700" fill="#24211d">VAE</text>
      <text x="430" y="91" text-anchor="middle" font-size="7" fill="#6d675f">frozen</text>

      <!-- ── Arrow: VAE → generated H&E ── -->
      <line x1="430" y1="94" x2="430" y2="118" stroke="#59544f" stroke-width="1.8" marker-end="url(#arrowDark)"/>

      <!-- ── Generated H&E patch ── -->
      <rect x="408" y="120" width="44" height="44" rx="6" fill="#f6c8d5" stroke="#b45b77" stroke-width="1.8"/>
      <circle cx="418" cy="130" r="2" fill="#93485f"/>
      <circle cx="430" cy="126" r="2" fill="#93485f"/>
      <circle cx="441" cy="138" r="2" fill="#93485f"/>
      <circle cx="421" cy="153" r="2" fill="#93485f"/>
      <circle cx="437" cy="157" r="2" fill="#93485f"/>
      <text x="430" y="175" text-anchor="middle" font-size="9" fill="#6d675f">gen H&amp;E</text>

      <!-- ══════════════════════════════════════════════
           TME input column — 3 thumbnails (x=10)
           Row 1 y=100: cell types
           Row 2 y=140: cell states
           Row 3 y=180: microenv (vas · glu · O₂)
      ══════════════════════════════════════════════ -->

      <!-- TME thumbnail: cell types -->
      <rect x="10" y="100" width="30" height="26" rx="5" fill="#eef7fb" stroke="#7fa2b3" stroke-width="1.2"/>
      <circle cx="19" cy="111" r="2.8" fill="#2f74c0"/>
      <circle cx="28" cy="108" r="2.8" fill="#be3e3e"/>
      <circle cx="34" cy="118" r="2.8" fill="#2d8c4f"/>
      <circle cx="22" cy="120" r="2.8" fill="#be3e3e"/>

      <!-- TME thumbnail: cell states -->
      <rect x="10" y="140" width="30" height="26" rx="5" fill="#ebfaf4" stroke="#7ab79a" stroke-width="1.2"/>
      <circle cx="19" cy="151" r="2.2" fill="#3f9f79"/>
      <circle cx="28" cy="147" r="3.5" fill="#85d4b6"/>
      <circle cx="35" cy="158" r="2.2" fill="#3f9f79"/>
      <circle cx="22" cy="161" r="3" fill="#85d4b6"/>

      <!-- TME thumbnail: microenv (vas · glu · O₂) -->
      <rect x="10" y="180" width="30" height="26" rx="5" fill="#fff1e6" stroke="#d49d71" stroke-width="1.2"/>
      <path d="M14 200 C18 185, 28 188, 35 194" fill="none" stroke="#c56f3d" stroke-width="2" stroke-linecap="round"/>
      <ellipse cx="20" cy="198" rx="5" ry="3" fill="#efc56a"/>

      <!-- ── Arrows: TME thumbnails → CNN×3 rows ── -->
      <line x1="40" y1="113" x2="50" y2="113" stroke="#2d8c4f" stroke-width="1.8" marker-end="url(#arrowGreen)"/>
      <line x1="40" y1="153" x2="50" y2="153" stroke="#2d8c4f" stroke-width="1.8" marker-end="url(#arrowGreen)"/>
      <line x1="40" y1="193" x2="50" y2="193" stroke="#2d8c4f" stroke-width="1.8" marker-end="url(#arrowGreen)"/>

      <!-- ══════════════════════════════════════════════
           CNN×3 rows (x=52, w=118)
           Row 1 y=100: types  (tumor · healthy · immune)
           Row 2 y=140: states (prolif · nonprolif · dead)
           Row 3 y=180: micro  (vas · glu · O₂)
      ══════════════════════════════════════════════ -->

      <!-- CNN×3 row: cell types -->
      <rect x="52" y="100" width="118" height="26" rx="6" fill="#dff4e3" stroke="#2d8c4f" stroke-width="1.8"/>
      <text x="62" y="116" font-size="9.5" font-weight="700" fill="#205f37">CNN × 3</text>
      <text x="164" y="110" text-anchor="end" font-size="7" fill="#5a6a60">tumor · healthy</text>
      <text x="164" y="121" text-anchor="end" font-size="7" fill="#5a6a60">· immune</text>

      <!-- CNN×3 row: cell states -->
      <rect x="52" y="140" width="118" height="26" rx="6" fill="#dff4e3" stroke="#2d8c4f" stroke-width="1.8"/>
      <text x="62" y="156" font-size="9.5" font-weight="700" fill="#205f37">CNN × 3</text>
      <text x="164" y="150" text-anchor="end" font-size="7" fill="#5a6a60">prolif · nonprolif</text>
      <text x="164" y="161" text-anchor="end" font-size="7" fill="#5a6a60">· dead</text>

      <!-- CNN×3 row: microenv -->
      <rect x="52" y="180" width="118" height="26" rx="6" fill="#dff4e3" stroke="#2d8c4f" stroke-width="1.8"/>
      <text x="62" y="196" font-size="9.5" font-weight="700" fill="#205f37">CNN × 3</text>
      <text x="164" y="190" text-anchor="end" font-size="7" fill="#5a6a60">vas · glu</text>
      <text x="164" y="201" text-anchor="end" font-size="7" fill="#5a6a60">· O₂</text>

      <!-- ── CNN×3 rows → vertical spine → Attention ── -->
      <line x1="170" y1="113" x2="183" y2="113" stroke="#2d8c4f" stroke-width="1.8"/>
      <line x1="170" y1="153" x2="183" y2="153" stroke="#2d8c4f" stroke-width="1.8"/>
      <line x1="170" y1="193" x2="183" y2="193" stroke="#2d8c4f" stroke-width="1.8"/>
      <line x1="183" y1="113" x2="183" y2="193" stroke="#2d8c4f" stroke-width="1.8"/>
      <line x1="183" y1="153" x2="196" y2="153" stroke="#2d8c4f" stroke-width="1.8" marker-end="url(#arrowGreen)"/>

      <!-- ── Attention block ── -->
      <rect x="198" y="139" width="52" height="28" rx="7" fill="#dff4e3" stroke="#2d8c4f" stroke-width="2"/>
      <text x="224" y="153" text-anchor="middle" font-size="9.5" font-weight="700" fill="#205f37">Attn</text>
      <text x="224" y="163" text-anchor="middle" font-size="7" fill="#5a6a60">fusion</text>

      <!-- ── Arrow: Attention → PixCell (conditioning) ── -->
      <line x1="250" y1="153" x2="256" y2="153" stroke="#2d8c4f" stroke-width="1.8" marker-end="url(#arrowGreen)"/>
      <text x="252" y="148" text-anchor="middle" font-size="7.5" fill="#2d8c4f">cond.</text>

      <!-- ── v-loss gradient path (dashed red) ── -->
      <path d="M430,164 L430,224 L224,224 L224,167" fill="none" stroke="#b33b34" stroke-width="1.8" stroke-dasharray="5,4" marker-end="url(#arrowRed)"/>
      <text x="327" y="237" text-anchor="middle" font-size="8.5" fill="#b33b34">v-loss · updates CNN×3 + attention only</text>
    </svg>

    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background: repeating-linear-gradient(45deg, #f7f4ef, #f7f4ef 4px, #d6d0c7 4px, #d6d0c7 7px);"></span> frozen</span>
      <span class="legend-item"><span class="swatch" style="background:#dff4e3; border-color:#2d8c4f;"></span> trainable</span>
      <span class="legend-item"><svg width="32" height="12" aria-hidden="true"><line x1="1" y1="6" x2="31" y2="6" stroke="#b33b34" stroke-width="1.8" stroke-dasharray="5,4"/></svg> gradient</span>
    </div>
  </div>
</div>
</body>
</html>
```

- [ ] **Step 2: Run structural tests**

```bash
cd /home/pohaoc2/UW/he-feature-visualizer
pytest tests/test_stage2_paper_figure.py::test_stage2_pipeline_redesign_svg_structure tests/test_stage2_paper_figure.py::test_stage2_package_and_mockups_exist -v
```

Expected: PASS for both.

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
pytest tests/test_stage2_paper_figure.py -v
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add paper/figures/stage2/stage2_pipeline_design_A_mod.html
git commit -m "feat: redesign stage2 pipeline figure — 3×CNN×3 rows, 510×258 viewBox, microenv merge"
```

---

## Acceptance Criteria

- `viewBox="0 0 510 258"` present in the file
- `max-width: 640px` present
- Exactly 3 occurrences of `CNN × 3`
- Labels: tumor, healthy, immune, prolif, nonprolif, dead, vas, glu present
- PixCell rect at `x="258"`, VAE rect at `x="408"`
- No CNN1/CNN2/CNN3/CNN4 labels
- All `test_stage2_paper_figure.py` tests pass
