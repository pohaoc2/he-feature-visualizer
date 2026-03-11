# Stage 2 QC Visual Evidence (`processed_alt2_256`)

This file maps each QC claim to specific figures so the audience can verify by eye.

## Q1. HE and MX are from the intended matched region
- High-res registration overlays:
  - [vis_registered_check_ds1.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/vis_registered_check_ds1.png)
  - [vis_registered_check_ds2.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/vis_registered_check_ds2.png)
- Full alignment grid (all channels/overlays in one panel):
  - [alignment_grid.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/alignment_grid.png)
- Supporting metric from QC JSON:
  - `stage1_corners_inside_mx = 3/4` in [stage25_qc_post_stage2_pre_stage25.json](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/stage25_qc_post_stage2_pre_stage25.json)

## Q2. CSV points are biologically consistent with MX DNA
- 6-panel CSV + DNA figure:
  - [qc_vis_mx_dna_csv.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/qc_vis_mx_dna_csv.png)
- Segmentation-based crop check:
  - [c4r2_seg_csv.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/c4r2_seg_csv.png)
- Supporting metrics:
  - `csv_on_mask_crop_rate = 80.77%`
  - DNA intensity at CSV is much higher than random

## Q3. Transformed CSV still lands on transformed MX DNA
- Transformed panels in the 6-panel figure:
  - Panel `e/f` inside [qc_vis_mx_dna_csv.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/qc_vis_mx_dna_csv.png)
- HE-space combined overlays:
  - [c2r3_mx_csv.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/c2r3_mx_csv.png)
  - [c3r2_cv_csv_he.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/c3r2_cv_csv_he.png)

## Q4. Transformed MX DNA vs HE proximity
- High-res blends (visual):
  - [vis_registered_check_ds1.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/vis_registered_check_ds1.png)
  - [c3r1_blend.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/c3r1_blend.png)
- Reported proxy metrics (numeric) are in:
  - [stage25_qc_post_stage2_pre_stage25.json](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/stage25_qc_post_stage2_pre_stage25.json)

## Q5. Why matching may still be low
- Point-level assignment visualization:
  - [qc_vis_registration_debug.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/qc_vis_registration_debug.png)
- Matching diagnostics are in:
  - [stage25_qc_post_stage2_pre_stage25.json](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/stage25_qc_post_stage2_pre_stage25.json)

## Generated artifacts
- QC report JSON:
  - [stage25_qc_post_stage2_pre_stage25.json](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/stage25_qc_post_stage2_pre_stage25.json)
- Visual evidence pack:
  - [alignment_grid.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/alignment_grid.png)
  - [qc_vis_mx_dna_csv.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/qc_vis_mx_dna_csv.png)
  - [qc_vis_registration_debug.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/qc_vis_registration_debug.png)
  - [vis_registered_check_ds1.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/vis_registered_check_ds1.png)
  - [vis_registered_check_ds2.png](/home/pohaoc2/UW/bagherilab/he-feature-visualizer/processed_alt2_256/vis_registered_check_ds2.png)
