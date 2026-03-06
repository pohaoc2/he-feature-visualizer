"""Parse multiplex channel metadata CSV files."""

from __future__ import annotations

import pandas as pd


def load_channel_metadata(csv_path: str) -> dict[str, tuple[int, str]]:
    """Parse a channel metadata CSV and return ``{lower_name: (0-based_index, original_name)}``.

    Supports two CSV layouts (auto-detected):
      * New format: ``Channel_Number`` (1-based int) + ``Marker_Name``
      * Old format: ``Channel ID`` (e.g. ``Channel:0:N``, 0-based) + ``Target Name``
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    seen: dict[str, tuple[int, str]] = {}

    if "Channel_Number" in df.columns and "Marker_Name" in df.columns:
        for _, row in df.iterrows():
            target = str(row["Marker_Name"]).strip()
            key = target.lower()
            if key in seen:
                continue
            try:
                idx = int(row["Channel_Number"]) - 1
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Cannot parse integer from Channel_Number '{row['Channel_Number']}'"
                ) from exc
            seen[key] = (idx, target)

    elif "Channel ID" in df.columns and "Target Name" in df.columns:
        for _, row in df.iterrows():
            channel_id = str(row["Channel ID"]).strip()
            target = str(row["Target Name"]).strip()
            key = target.lower()
            if key in seen:
                continue
            parts = channel_id.split(":")
            try:
                idx = int(parts[-1])
            except (ValueError, IndexError) as exc:
                raise ValueError(
                    f"Cannot parse integer index from Channel ID '{channel_id}'"
                ) from exc
            seen[key] = (idx, target)

    else:
        raise ValueError(
            f"Unrecognised metadata CSV format. Expected either "
            f"'Channel_Number'+'Marker_Name' or 'Channel ID'+'Target Name'. "
            f"Found columns: {list(df.columns)}"
        )

    return seen


def resolve_channel_indices(
    csv_path: str,
    channel_names: list[str],
) -> tuple[list[int], list[str]]:
    """Resolve requested marker names to 0-based channel indices.

    Returns ``(indices, resolved_names)``.  Raises :class:`ValueError` if any
    name is not found (case-insensitive match).
    """
    seen = load_channel_metadata(csv_path)

    indices: list[int] = []
    resolved: list[str] = []
    missing: list[str] = []

    for name in channel_names:
        key = name.lower()
        if key not in seen:
            missing.append(name)
        else:
            idx, orig = seen[key]
            indices.append(idx)
            resolved.append(orig)

    if missing:
        available = sorted(seen.keys())
        raise ValueError(
            f"Channel(s) not found in metadata CSV: {missing}\n"
            f"Available target names: {available}"
        )

    return indices, resolved
