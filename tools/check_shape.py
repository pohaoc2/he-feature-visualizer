#!/usr/bin/env python3
"""
check_shape.py -- Inspect OME-TIFF dimensions without downloading the full file.

Supports local paths and s3:// URLs.  For remote files only the BigTIFF header
(16 bytes) and the tail containing the IFDs / OME-XML are fetched via ranged
S3 GET requests.

Usage:
    # one or more images
    python check_shape.py s3://lin-2021-crc-atlas/data/WD-76845-096.ome.tif \\
                          s3://lin-2021-crc-atlas/data/WD-76845-097.ome.tif

    # named pair (H&E + multiplex) — also runs alignment check
    python check_shape.py --he-image   s3://lin-2021-crc-atlas/data/WD-76845-096.ome.tif \\
                          --multiplex-image s3://lin-2021-crc-atlas/data/WD-76845-097.ome.tif

    # local files still work
    python check_shape.py data/CRC02-HE.ome.tif data/CRC02.ome.tif
"""

import argparse
import struct
import sys
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import tifffile

OME_NS = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
TAIL_SIZE = 50 * 1024 * 1024  # 50 MB — enough to hold IFDs + OME-XML for known datasets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_float(x):
    try:
        return float(x) if x is not None else None
    except (ValueError, TypeError):
        return None


def _mpp_from_xml(xml_bytes: bytes) -> tuple[float | None, float | None, str | None]:
    """Extract (mpp_x, mpp_y, unit) from OME-XML bytes."""
    try:
        root = ET.fromstring(
            xml_bytes.rstrip(b"\x00").decode("utf-8", errors="replace")
        )
    except ET.ParseError:
        return None, None, None
    pixels = root.find(".//ome:Pixels", OME_NS) or root.find(".//Pixels")
    if pixels is None:
        return None, None, None
    return (
        _safe_float(pixels.get("PhysicalSizeX")),
        _safe_float(pixels.get("PhysicalSizeY")),
        pixels.get("PhysicalSizeXUnit", "µm"),
    )


def _channel_count_from_axes_shape(
    axes: str | None, shape: tuple[int, ...] | None
) -> int | None:
    """Infer channel count from axes/shape for local TIFF series metadata."""
    if not axes or not shape:
        return None
    axes_up = axes.upper()

    if "C" in axes_up:
        return int(shape[axes_up.index("C")])
    if "S" in axes_up:
        return int(shape[axes_up.index("S")])
    if "Y" in axes_up and "X" in axes_up:
        return 1
    return None


# ---------------------------------------------------------------------------
# S3 range-read helpers
# ---------------------------------------------------------------------------


def _s3_get_range(s3, bucket: str, key: str, start: int, end: int) -> bytes:
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end}")
    return resp["Body"].read()


def _s3_file_size(s3, bucket: str, key: str) -> int:
    return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]


# ---------------------------------------------------------------------------
# BigTIFF IFD parser (works on a bytes buffer that contains the IFD)
# ---------------------------------------------------------------------------


def _parse_ifd(buf: bytes, ifd_offset: int, buf_start: int, bo: str) -> dict:
    """Parse one BigTIFF IFD; return {tag: (type, count, value_or_offset)}."""
    pos = ifd_offset - buf_start
    n = struct.unpack_from(bo + "Q", buf, pos)[0]
    pos += 8
    tags: dict[int, tuple] = {}
    for _ in range(min(n, 300)):
        tag = struct.unpack_from(bo + "H", buf, pos)[0]
        typ = struct.unpack_from(bo + "H", buf, pos + 2)[0]
        cnt = struct.unpack_from(bo + "Q", buf, pos + 4)[0]
        val = struct.unpack_from(bo + "Q", buf, pos + 12)[0]
        tags[tag] = (typ, cnt, val)
        pos += 20
    return tags


# ---------------------------------------------------------------------------
# Core inspect functions
# ---------------------------------------------------------------------------


def inspect_tiff_local(path: str) -> dict:
    """Inspect a local OME-TIFF using tifffile (original behaviour)."""
    info: dict = {"path": path}
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        shape = series.shape
        info.update(axes=axes, shape=shape, dtype=str(series.dtype))
        info["img_h"] = shape[axes.index("Y")] if "Y" in axes else None
        info["img_w"] = shape[axes.index("X")] if "X" in axes else None
        info["n_channels"] = _channel_count_from_axes_shape(axes, shape)

        ome_xml = getattr(tif, "ome_metadata", None)
        if not ome_xml and tif.pages:
            ome_xml = getattr(tif.pages[0], "description", None)

        mpp_x = mpp_y = unit = None
        if ome_xml:
            xml_bytes = ome_xml.encode() if isinstance(ome_xml, str) else ome_xml
            mpp_x, mpp_y, unit = _mpp_from_xml(xml_bytes)

        info.update(mpp_x=mpp_x, mpp_y=mpp_y, mpp_unit=unit or "µm")
    return info


def inspect_tiff_s3(bucket: str, key: str) -> dict:
    """Inspect a remote OME-TIFF via two S3 range requests (header + tail)."""
    import boto3  # pylint: disable=import-outside-toplevel,import-error

    s3 = boto3.client("s3")

    path = f"s3://{bucket}/{key}"
    info: dict = {"path": path}

    file_size = _s3_file_size(s3, bucket, key)
    info["file_size"] = file_size

    # 1. Read BigTIFF header (16 bytes) → IFD offset
    hdr = _s3_get_range(s3, bucket, key, 0, 15)
    if len(hdr) < 16:
        raise ValueError(f"Could not read TIFF header from {path}")

    bo = "<" if hdr[:2] == b"II" else ">"
    magic = struct.unpack_from(bo + "H", hdr, 2)[0]
    if magic not in (42, 43):
        raise ValueError(f"Not a TIFF file: magic={magic:#06x}")
    is_bigtiff = magic == 43
    if not is_bigtiff:
        raise NotImplementedError(
            "Only BigTIFF remote read is implemented; file appears to be classic TIFF"
        )

    ifd_offset = struct.unpack_from(bo + "Q", hdr, 8)[0]

    # 2. Read tail (covers IFDs and OME-XML for known OME-TIFFs)
    tail_start = max(0, file_size - TAIL_SIZE)
    tail = _s3_get_range(s3, bucket, key, tail_start, file_size - 1)

    if ifd_offset < tail_start:
        raise ValueError(
            f"IFD offset {ifd_offset} is before tail start {tail_start}; "
            f"increase TAIL_SIZE (currently {TAIL_SIZE // 1024**2} MB)"
        )

    tags = _parse_ifd(tail, ifd_offset, tail_start, bo)

    # TIFF tags: 256=ImageWidth, 257=ImageLength, 270=ImageDescription
    width = tags.get(256, (None, None, None))[2]
    height = tags.get(257, (None, None, None))[2]
    samples_per_pixel = tags.get(277, (None, None, None))[2]
    info["img_w"] = width
    info["img_h"] = height
    info["shape"] = (height, width)
    info["n_channels"] = int(samples_per_pixel) if samples_per_pixel else 1
    info["axes"] = "YXS" if info["n_channels"] > 1 else "YX"
    info["dtype"] = "unknown"

    mpp_x = mpp_y = unit = None
    if 270 in tags:
        _, desc_len, desc_off = tags[270]
        # Try reading from tail buffer first
        local_off = desc_off - tail_start
        if 0 <= local_off < len(tail):
            xml_bytes = tail[local_off : local_off + desc_len]
        else:
            # Description stored elsewhere — do a targeted range fetch
            xml_bytes = _s3_get_range(
                s3, bucket, key, desc_off, desc_off + desc_len - 1
            )
        mpp_x, mpp_y, unit = _mpp_from_xml(xml_bytes)

    info.update(mpp_x=mpp_x, mpp_y=mpp_y, mpp_unit=unit or "µm")
    return info


def inspect_image(url_or_path: str) -> dict:
    """Dispatch to local or S3 inspector based on URL scheme."""
    parsed = urlparse(url_or_path)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        return inspect_tiff_s3(bucket, key)
    return inspect_tiff_local(url_or_path)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_info(label: str, info: dict) -> None:
    size_str = (
        f"  File size : {info['file_size'] / 1024**3:.2f} GiB\n"
        if "file_size" in info
        else ""
    )
    print(f"\n{label}")
    print(f"  Path      : {info['path']}")
    print(f"{size_str}", end="")
    print(f"  Dimensions: {info['img_w']} x {info['img_h']} px  (W x H)")
    n_channels = info.get("n_channels")
    if n_channels is None:
        n_channels = _channel_count_from_axes_shape(info.get("axes"), info.get("shape"))
    if n_channels is not None:
        print(f"  Channels  : {n_channels}")
    else:
        print("  Channels  : unknown")
    if info.get("mpp_x") is not None:
        u = info["mpp_unit"]
        phys_w = info["img_w"] * info["mpp_x"]
        phys_h = info["img_h"] * info["mpp_y"]
        print(f"  mpp       : x={info['mpp_x']} {u}/px  y={info['mpp_y']} {u}/px")
        print(f"  Physical  : {phys_w:.1f} x {phys_h:.1f} {u}")
    else:
        print("  mpp       : NOT FOUND in OME-XML")


def alignment_check(he: dict, mx: dict) -> None:
    print("\nAlignment check:")
    he_w, he_h = he["img_w"], he["img_h"]
    mx_w, mx_h = mx["img_w"], mx["img_h"]

    if (he_w, he_h) == (mx_w, mx_h):
        print("  Pixel dimensions match — no scaling needed.")
        return

    print(f"  Pixel dims differ: H&E={he_w}x{he_h}  Multiplex={mx_w}x{mx_h}")

    if None in (he["mpp_x"], he["mpp_y"], mx["mpp_x"], mx["mpp_y"]):
        print("  WARNING: mpp missing — cannot compute scale factors.")
        return

    sx = he["mpp_x"] / mx["mpp_x"]
    sy = he["mpp_y"] / mx["mpp_y"]
    print(f"  Scale (H&E → multiplex): x={sx:.6f}  y={sy:.6f}")

    phys_he_w = he_w * he["mpp_x"]
    phys_he_h = he_h * he["mpp_y"]
    phys_mx_w = mx_w * mx["mpp_x"]
    phys_mx_h = mx_h * mx["mpp_y"]
    dw = abs(phys_he_w - phys_mx_w) / max(phys_he_w, phys_mx_w) * 100
    dh = abs(phys_he_h - phys_mx_h) / max(phys_he_h, phys_mx_h) * 100
    u = he["mpp_unit"]
    print(
        f"  Physical extents: H&E={phys_he_w:.1f}x{phys_he_h:.1f}{u}"
        f"  Multiplex={phys_mx_w:.1f}x{phys_mx_h:.1f}{u}"
    )
    print(f"  Extent difference: {dw:.2f}% (W)  {dh:.2f}% (H)")

    if dw > 5 or dh > 5:
        print(
            "  WARNING: extents differ by >5% — images may not share the same origin."
        )
    else:
        print(
            "  Physical extents agree (within 5%) — scale-only alignment looks correct."
        )

    patch_size = 256
    mx_patch_w = round(patch_size * sx)
    mx_patch_h = round(patch_size * sy)
    print(
        f"\n  Example: H&E {patch_size}px patch"
        f" → multiplex {mx_patch_w}x{mx_patch_h}px region"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect OME-TIFF dimensions (local path or s3:// URL, no full download).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "images",
        nargs="*",
        help="One or more local paths or s3:// URLs to inspect individually.",
    )
    parser.add_argument("--he-image", help="H&E image (local path or s3:// URL)")
    parser.add_argument(
        "--multiplex-image", help="Multiplex image (local path or s3:// URL)"
    )
    args = parser.parse_args()

    if args.he_image and args.multiplex_image:
        he = inspect_image(args.he_image)
        mx = inspect_image(args.multiplex_image)
        print_info("H&E image", he)
        print_info("Multiplex image", mx)
        alignment_check(he, mx)
    elif args.images:
        for img in args.images:
            info = inspect_image(img)
            print_info(img, info)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
