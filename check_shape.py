#!/usr/bin/env python3
"""
check_shape.py -- Diagnostic: inspect OME-TIFF dimensions and alignment.

Usage:
    python check_shape.py --he-image data/CRC02-HE.ome.tif \
                          --multiplex-image data/CRC02.ome.tif

Prints pixel dimensions, physical pixel sizes (mpp), computes scale factors,
and verifies that physical extents agree between the two images.
"""

import argparse
import sys
import xml.etree.ElementTree as ET

import tifffile

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def inspect_tiff(path: str) -> dict:
    """Return a dict with shape/axes/mpp info for the given OME-TIFF."""
    info = {"path": path}
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = series.shape
        info["axes"] = axes
        info["shape"] = shape
        info["dtype"] = str(series.dtype)

        y_idx = axes.index("Y") if "Y" in axes else None
        x_idx = axes.index("X") if "X" in axes else None
        info["img_h"] = shape[y_idx] if y_idx is not None else None
        info["img_w"] = shape[x_idx] if x_idx is not None else None

        ome_xml = getattr(tif, "ome_metadata", None)
        if not ome_xml and tif.pages:
            ome_xml = getattr(tif.pages[0], "description", None)

        info["mpp_x"] = None
        info["mpp_y"] = None
        info["mpp_unit"] = None
        info["namespace"] = None

        if ome_xml:
            try:
                root = ET.fromstring(ome_xml)
                info["namespace"] = root.tag.split("}")[0].lstrip("{") if "}" in root.tag else None
                pixels = root.find(".//ome:Pixels", OME_NS)
                if pixels is not None:
                    info["mpp_x"] = _safe_float(pixels.get("PhysicalSizeX"))
                    info["mpp_y"] = _safe_float(pixels.get("PhysicalSizeY"))
                    info["mpp_unit"] = pixels.get("PhysicalSizeXUnit", "µm")
                else:
                    # Try without namespace
                    pixels_ns = root.find(".//Pixels")
                    if pixels_ns is not None:
                        info["mpp_x"] = _safe_float(pixels_ns.get("PhysicalSizeX"))
                        info["mpp_y"] = _safe_float(pixels_ns.get("PhysicalSizeY"))
                        info["mpp_unit"] = pixels_ns.get("PhysicalSizeXUnit", "µm")
            except ET.ParseError as e:
                info["xml_error"] = str(e)

    return info


def print_info(label: str, info: dict) -> None:
    print(f"\n{label}")
    print(f"  Path      : {info['path']}")
    print(f"  Shape     : {info['shape']}  axes={info['axes']}  dtype={info['dtype']}")
    print(f"  Dimensions: {info['img_w']} x {info['img_h']} px  (W x H)")
    if info["mpp_x"] is not None:
        print(f"  mpp       : x={info['mpp_x']} {info['mpp_unit']}/px  y={info['mpp_y']} {info['mpp_unit']}/px")
        phys_w = info["img_w"] * info["mpp_x"]
        phys_h = info["img_h"] * info["mpp_y"]
        print(f"  Physical  : {phys_w:.1f} x {phys_h:.1f} {info['mpp_unit']}")
    else:
        print("  mpp       : NOT FOUND in OME-XML")
    if info.get("namespace"):
        print(f"  Namespace : {info['namespace']}")
    if info.get("xml_error"):
        print(f"  XML error : {info['xml_error']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect OME-TIFF shapes and alignment.")
    parser.add_argument("--he-image", required=True, help="Path to H&E OME-TIFF")
    parser.add_argument("--multiplex-image", required=True, help="Path to multiplex OME-TIFF")
    args = parser.parse_args()

    he_info = inspect_tiff(args.he_image)
    mx_info = inspect_tiff(args.multiplex_image)

    print_info("H&E image", he_info)
    print_info("Multiplex image", mx_info)

    print("\nAlignment check:")
    he_w, he_h = he_info["img_w"], he_info["img_h"]
    mx_w, mx_h = mx_info["img_w"], mx_info["img_h"]

    if (he_w, he_h) == (mx_w, mx_h):
        print("  Pixel dimensions match — no scaling needed.")
        sys.exit(0)

    print(f"  Pixel dimensions differ: H&E={he_w}x{he_h}  Multiplex={mx_w}x{mx_h}")

    he_mpp_x, he_mpp_y = he_info["mpp_x"], he_info["mpp_y"]
    mx_mpp_x, mx_mpp_y = mx_info["mpp_x"], mx_info["mpp_y"]

    if None in (he_mpp_x, he_mpp_y, mx_mpp_x, mx_mpp_y):
        print("  WARNING: mpp missing for one or both images — cannot compute scale factors.")
        sys.exit(1)

    scale_x = he_mpp_x / mx_mpp_x
    scale_y = he_mpp_y / mx_mpp_y
    print(f"  Scale factors: x={scale_x:.6f}  y={scale_y:.6f}  (multiply H&E coords by this to get multiplex coords)")

    # Verify physical extents agree
    phys_he_w = he_w * he_mpp_x
    phys_he_h = he_h * he_mpp_y
    phys_mx_w = mx_w * mx_mpp_x
    phys_mx_h = mx_h * mx_mpp_y
    diff_w = abs(phys_he_w - phys_mx_w) / max(phys_he_w, phys_mx_w) * 100
    diff_h = abs(phys_he_h - phys_mx_h) / max(phys_he_h, phys_mx_h) * 100
    print(f"  Physical extents: H&E={phys_he_w:.1f}x{phys_he_h:.1f}µm  Multiplex={phys_mx_w:.1f}x{phys_mx_h:.1f}µm")
    print(f"  Extent difference: {diff_w:.2f}% (W)  {diff_h:.2f}% (H)")

    if diff_w > 5 or diff_h > 5:
        print("  WARNING: Physical extents differ by >5% — images may not be co-registered.")
    else:
        print("  Physical extents agree (within 5%) — alignment looks correct.")

    # Show example: what does patch (y0=0,x0=0) look like in multiplex coords?
    patch_size = 256
    size_mx_y = max(1, round(patch_size * scale_y))
    size_mx_x = max(1, round(patch_size * scale_x))
    print(f"\nExample: H&E patch {patch_size}x{patch_size} at (0,0) -> Multiplex region {size_mx_x}x{size_mx_y} px at (0,0), resized to {patch_size}x{patch_size}")


if __name__ == "__main__":
    main()
