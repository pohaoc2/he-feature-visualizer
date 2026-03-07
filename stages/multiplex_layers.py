"""
multiplex_layers.py — Stage 4+5 of the histopathology analysis pipeline.

Stage 4 (spatial analysis) reads per-patch multiplex immunofluorescence arrays
produced by Stage 1 (patchify.py) and derives three biologically meaningful
proxy maps from the raw channel intensities:

    vasculature
        Binary segmentation of CD31+ vascular structures.  CD31 (PECAM-1) is an
        endothelial marker; the binarised mask delineates vessel lumens and
        capillaries in the tissue.  Rendered as red pixels on a transparent
        background for overlay on the H&E image.

    oxygen
        A spatial proxy for tissue oxygenation derived from the Euclidean
        distance transform of the CD31 mask.  Pixels at or near vessel walls
        are assumed to be well-oxygenated (blue in the RdYlBu colormap), while
        pixels far from vessels are assumed to be hypoxic (red).  This
        approximation follows the classical Krogh cylinder model of oxygen
        diffusion.

    glucose / metabolic demand
        A proxy for local metabolic activity based on the maximum of
        normalised Ki67 and PCNA intensities.  Ki67 marks cells in active
        proliferation (G1-M phases) and PCNA is a DNA-replication clamp loader
        also elevated during S-phase; their maximum highlights the most
        metabolically demanding regions.  Rendered with the 'hot' colormap
        (black → red → yellow/white as demand increases).

Stage 5 (visualisation) saves the three RGBA overlay PNGs to sub-directories
of the output folder so they can be served directly by server_patches.py and
composited on top of H&E tiles in viewer_patches.html.

Input
-----
processed/multiplex/{i}_{j}.npy   — (C, H, W) uint16 arrays from Stage 1

Output
------
{out}/vasculature/{i}_{j}.png     — RGBA: red vessels on transparent background
{out}/vasculature_mask/{i}_{j}.npy — bool: binary vessel mask for downstream analysis
{out}/oxygen/{i}_{j}.png          — RGBA: RdYlBu oxygenation map
{out}/glucose/{i}_{j}.png         — RGBA: hot metabolic-demand map
"""

import argparse
import json
import logging
import pathlib

import numpy as np
from PIL import Image

from stages.multiplex_layers_lib.channels import (
    extract_channel,
    get_channel_index,
    get_first_matching_channel_index,
    load_multiplex_patch,
)
from stages.multiplex_layers_lib.masks import (
    _validate_odd_kernel_size,
    apply_vessel_mask_quality_fallback,
    binarize_otsu,
    cleanup_vasculature_mask,
    refine_vasculature_with_sma,
)
from stages.multiplex_layers_lib.pde import (
    build_consumption_map,
    build_vessel_source_map,
    compute_metabolic_demand_map,
    solve_steady_state_diffusion,
)
from stages.multiplex_layers_lib.render import (
    apply_colormap,
    make_glucose_map,
    make_glucose_map_pde,
    make_oxygen_map,
    make_oxygen_map_pde,
    make_vasculature_overlay,
)
from utils.channels import resolve_channel_indices
from utils.normalize import percentile_norm

# ---------------------------------------------------------------------------
# Re-exported helper functions
# ---------------------------------------------------------------------------

# Helper implementations now live in stages/multiplex_layers_lib/*.py.
# They are imported above and remain available from this module for
# backward compatibility with existing tests and scripts.
__all__ = [
    "load_multiplex_patch",
    "get_channel_index",
    "get_first_matching_channel_index",
    "extract_channel",
    "binarize_otsu",
    "apply_colormap",
    "make_vasculature_overlay",
    "refine_vasculature_with_sma",
    "_validate_odd_kernel_size",
    "cleanup_vasculature_mask",
    "apply_vessel_mask_quality_fallback",
    "build_vessel_source_map",
    "build_consumption_map",
    "solve_steady_state_diffusion",
    "compute_metabolic_demand_map",
    "make_oxygen_map",
    "make_oxygen_map_pde",
    "make_glucose_map",
    "make_glucose_map_pde",
    "main",
]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description=(
            "Stage 4+5: Generate vasculature, oxygen, and glucose overlay PNGs "
            "from per-patch multiplex .npy files."
        )
    )
    parser.add_argument(
        "--multiplex-dir",
        required=True,
        help="Directory containing {i}_{j}.npy multiplex patch files.",
    )
    parser.add_argument(
        "--index",
        required=True,
        help="Path to processed/index.json (patch grid manifest).",
    )
    parser.add_argument(
        "--metadata-csv",
        required=True,
        help="Channel metadata CSV with 'Channel ID' and 'Target Name' columns.",
    )
    parser.add_argument(
        "--out",
        default="processed/",
        help="Output directory (default: processed/).",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=[
            "Keratin",
            "NaKATPase",
            "CDX2",
            "CD45",
            "CD3",
            "CD4",
            "CD8a",
            "CD20",
            "CD45RO",
            "CD68",
            "CD163",
            "FOXP3",
            "PD1",
            "aSMA",
            "CD31",
            "Desmin",
            "Collagen",
            "Ki67",
            "PCNA",
            "Vimentin",
            "Ecadherin",
        ],
        metavar="NAME",
        help="Multiplex channel names present in the .npy files, in order "
        "(must match --channels passed to patchify.py). "
        "Default: full Stage 3 marker panel, 21 channels.",
    )
    parser.add_argument(
        "--oxygen-model",
        choices=["distance", "pde"],
        default="distance",
        help=(
            "Oxygen proxy model. 'distance' is the current CD31 distance-transform "
            "model; 'pde' uses a steady-state diffusion-consumption solver."
        ),
    )
    parser.add_argument(
        "--glucose-model",
        choices=["max", "pde"],
        default="max",
        help=(
            "Glucose proxy model. 'max' uses max(norm(Ki67), norm(PCNA)); "
            "'pde' uses a steady-state diffusion-consumption solver."
        ),
    )
    parser.add_argument(
        "--pde-max-iters",
        type=int,
        default=500,
        help="Maximum PDE solver iterations.",
    )
    parser.add_argument(
        "--pde-tol",
        type=float,
        default=1e-4,
        help="PDE convergence tolerance (max absolute update).",
    )
    parser.add_argument(
        "--pde-diffusion",
        type=float,
        default=1.0,
        help="Diffusion coefficient used by PDE models.",
    )
    parser.add_argument(
        "--oxygen-consumption-base",
        type=float,
        default=0.1,
        help="Base oxygen consumption term used in oxygen PDE mode.",
    )
    parser.add_argument(
        "--oxygen-consumption-demand-weight",
        type=float,
        default=0.3,
        help="Demand weight for oxygen PDE consumption map.",
    )
    parser.add_argument(
        "--glucose-consumption-base",
        type=float,
        default=0.1,
        help="Base glucose consumption term used in glucose PDE mode.",
    )
    parser.add_argument(
        "--glucose-consumption-demand-weight",
        type=float,
        default=0.3,
        help="Demand weight for glucose PDE consumption map.",
    )
    parser.add_argument(
        "--vasc-sma-refine",
        action="store_true",
        help="Refine CD31 vessel mask by adding only SMA pixels adjacent to CD31.",
    )
    parser.add_argument(
        "--sma-channel-candidates",
        nargs="+",
        default=["aSMA", "Aortic smooth muscle actin"],
        metavar="NAME",
        help=(
            "Ordered list of channel names to try for SMA refinement. "
            "Used only with --vasc-sma-refine."
        ),
    )
    parser.add_argument(
        "--sma-adjacency-px",
        type=int,
        default=2,
        help=(
            "Pixel radius used to gate SMA support near CD31. "
            "Used only with --vasc-sma-refine."
        ),
    )
    parser.add_argument(
        "--vasc-open-kernel-size",
        type=int,
        default=0,
        help=(
            "Optional odd kernel size for vessel-mask morphological opening "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--vasc-close-kernel-size",
        type=int,
        default=0,
        help=(
            "Optional odd kernel size for vessel-mask morphological closing "
            "(0 disables)."
        ),
    )
    parser.add_argument(
        "--vasc-min-area",
        type=int,
        default=0,
        help=(
            "Optional minimum connected-component area for vessel masks "
            "(0 disables filtering)."
        ),
    )
    parser.add_argument(
        "--vasc-noisy-max-fraction",
        type=float,
        default=0.98,
        help=(
            "If vessel-mask coverage is >= this fraction, treat as noisy and "
            "fallback to CD31-only mask."
        ),
    )
    args = parser.parse_args()

    # Validate vessel-mask cleanup options once at startup.
    _validate_odd_kernel_size("vasc_open_kernel_size", args.vasc_open_kernel_size)
    _validate_odd_kernel_size("vasc_close_kernel_size", args.vasc_close_kernel_size)
    if args.vasc_min_area < 0:
        parser.error("--vasc-min-area must be >= 0")
    if not (0.0 < args.vasc_noisy_max_fraction <= 1.0):
        parser.error("--vasc-noisy-max-fraction must be in (0, 1]")

    multiplex_dir = pathlib.Path(args.multiplex_dir)
    index_path = pathlib.Path(args.index)
    out_dir = pathlib.Path(args.out)

    if args.oxygen_model == "pde":
        log.info(
            "Oxygen model: PDE (relative density proxy, not absolute concentration)."
        )
    if args.glucose_model == "pde":
        log.info(
            "Glucose model: PDE (relative density proxy, not absolute concentration)."
        )

    # ------------------------------------------------------------------
    # 1. Validate channel names against metadata CSV
    # ------------------------------------------------------------------
    log.info("Validating channel names against metadata CSV: %s", args.metadata_csv)
    resolve_channel_indices(args.metadata_csv, args.channels)
    log.info("  All requested channels found: %s", args.channels)

    # ------------------------------------------------------------------
    # 2. Resolve channel positions within the .npy arrays
    # ------------------------------------------------------------------
    cd31_idx = get_channel_index(args.channels, "CD31")
    ki67_idx = get_channel_index(args.channels, "Ki67")
    pcna_idx = get_channel_index(args.channels, "PCNA")
    sma_idx: int | None = None
    if args.vasc_sma_refine:
        sma_idx = get_first_matching_channel_index(
            args.channels, args.sma_channel_candidates
        )
    log.info(
        "  Channel indices — CD31: %d, Ki67: %d, PCNA: %d",
        cd31_idx,
        ki67_idx,
        pcna_idx,
    )
    if args.vasc_sma_refine and sma_idx is None:
        log.warning(
            "SMA refinement requested, but none of %s were found in --channels. "
            "Proceeding with CD31-only vessel masks.",
            args.sma_channel_candidates,
        )
    elif sma_idx is not None:
        log.info("  SMA refinement channel index: %d", sma_idx)

    # ------------------------------------------------------------------
    # 3. Load index.json, prepare output directories
    # ------------------------------------------------------------------
    log.info("Loading patch index: %s", index_path)
    with index_path.open(encoding="utf-8") as fh:
        index = json.load(fh)

    patches = index.get("patches", [])
    log.info("  %d patches in index.", len(patches))

    vasc_dir = out_dir / "vasculature"
    vasc_mask_dir = out_dir / "vasculature_mask"
    oxygen_dir = out_dir / "oxygen"
    glucose_dir = out_dir / "glucose"
    for d in (vasc_dir, vasc_mask_dir, oxygen_dir, glucose_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 4–6. Process each patch
    # ------------------------------------------------------------------
    processed = 0
    skipped = 0

    for patch_meta in patches:
        x0 = patch_meta["x0"]
        y0 = patch_meta["y0"]
        patch_id = f"{x0}_{y0}"

        npy_path = multiplex_dir / f"{patch_id}.npy"
        if not npy_path.exists():
            log.warning("Missing multiplex file: %s — skipping.", npy_path)
            skipped += 1
            continue

        # a. Load (C, H, W) uint16 array
        patch = load_multiplex_patch(str(npy_path))

        # b. Extract individual channels
        cd31_raw = extract_channel(patch, cd31_idx)
        ki67_raw = extract_channel(patch, ki67_idx)
        pcna_raw = extract_channel(patch, pcna_idx)
        sma_raw = extract_channel(patch, sma_idx) if sma_idx is not None else None

        # c. Vasculature overlay
        cd31_norm = percentile_norm(cd31_raw.astype(np.float32))
        cd31_base_mask = binarize_otsu(cd31_norm)
        vessel_mask = cd31_base_mask.copy()
        if sma_raw is not None:
            sma_norm = percentile_norm(sma_raw.astype(np.float32))
            sma_mask = binarize_otsu(sma_norm)
            vessel_mask = refine_vasculature_with_sma(
                vessel_mask,
                sma_mask,
                adjacency_px=args.sma_adjacency_px,
            )
        vessel_mask = cleanup_vasculature_mask(
            vessel_mask,
            open_kernel_size=args.vasc_open_kernel_size,
            close_kernel_size=args.vasc_close_kernel_size,
            min_area=args.vasc_min_area,
        )
        vessel_mask, vessel_status = apply_vessel_mask_quality_fallback(
            candidate_mask=vessel_mask,
            cd31_fallback_mask=cd31_base_mask,
            noisy_max_fraction=args.vasc_noisy_max_fraction,
        )
        if vessel_status == "empty_fallback":
            log.warning(
                "Patch %s: empty vessel mask after refinement/cleanup; "
                "falling back to CD31-only mask.",
                patch_id,
            )
        elif vessel_status == "noisy_fallback":
            log.warning(
                "Patch %s: vessel mask coverage exceeded noisy threshold "
                "(>= %.3f); falling back to CD31-only mask.",
                patch_id,
                args.vasc_noisy_max_fraction,
            )
        elif vessel_status == "empty":
            log.warning(
                "Patch %s: vessel mask is empty; proceeding with deterministic "
                "empty mask.",
                patch_id,
            )

        vasc_rgba = make_vasculature_overlay(vessel_mask)
        demand_map = compute_metabolic_demand_map(ki67_raw, pcna_raw)

        # d. Oxygen map
        if args.oxygen_model == "pde":
            oxygen_rgba = make_oxygen_map_pde(
                vessel_mask=vessel_mask,
                demand_map=demand_map,
                diffusion=args.pde_diffusion,
                max_iters=args.pde_max_iters,
                tol=args.pde_tol,
                base_consumption=args.oxygen_consumption_base,
                demand_weight=args.oxygen_consumption_demand_weight,
            )
        else:
            oxygen_rgba = make_oxygen_map(vessel_mask)

        # e. Glucose / metabolic demand map
        if args.glucose_model == "pde":
            glucose_rgba = make_glucose_map_pde(
                vessel_mask=vessel_mask,
                demand_map=demand_map,
                diffusion=args.pde_diffusion,
                max_iters=args.pde_max_iters,
                tol=args.pde_tol,
                base_consumption=args.glucose_consumption_base,
                demand_weight=args.glucose_consumption_demand_weight,
            )
        else:
            glucose_rgba = make_glucose_map(ki67_raw, pcna_raw)

        # f. Save outputs
        Image.fromarray(vasc_rgba, "RGBA").save(vasc_dir / f"{patch_id}.png")
        np.save(
            vasc_mask_dir / f"{patch_id}.npy",
            vessel_mask.astype(bool),
            allow_pickle=False,
        )
        Image.fromarray(oxygen_rgba, "RGBA").save(oxygen_dir / f"{patch_id}.png")
        Image.fromarray(glucose_rgba, "RGBA").save(glucose_dir / f"{patch_id}.png")

        processed += 1
        if processed % 50 == 0:
            log.info(
                "  Progress: %d patches processed, %d skipped …",
                processed,
                skipped,
            )

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    log.info("Done.")
    log.info("  Patches processed : %d", processed)
    log.info("  Patches skipped   : %d", skipped)
    log.info("  Vasculature PNGs  → %s", vasc_dir)
    log.info("  Vasculature masks → %s", vasc_mask_dir)
    log.info("  Oxygen PNGs       → %s", oxygen_dir)
    log.info("  Glucose PNGs      → %s", glucose_dir)


if __name__ == "__main__":
    main()
