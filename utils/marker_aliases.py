"""Marker-name normalization and alias resolution helpers."""

from __future__ import annotations

from collections.abc import Iterable


def normalize_marker_name(name: str) -> str:
    """Normalize marker names for robust matching across punctuation/spacing."""
    return "".join(ch.lower() for ch in str(name).strip() if ch.isalnum())


# Canonical marker names used by Stage 3 logic.
_CANONICAL_ALIASES: dict[str, tuple[str, ...]] = {
    "Keratin": ("Keratin", "Pan-CK", "PanCK", "Pan CK"),
    "PanCK": ("PanCK", "Pan-CK", "Pan CK", "Keratin"),
    "aSMA": ("aSMA", "SMA", "Aortic smooth muscle actin"),
    "PD1": ("PD1", "PD-1"),
    "Ecadherin": ("Ecadherin", "E-cadherin", "E cadherin"),
    "CD3": ("CD3", "CD3e"),
    "CD3e": ("CD3e", "CD3"),
    "NaKATPase": ("NaKATPase", "Na/K ATPase", "Na-K-ATPase"),
}

_NORMALIZED_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in _CANONICAL_ALIASES.items():
    for alias in aliases:
        _NORMALIZED_TO_CANONICAL[normalize_marker_name(alias)] = canonical
_NORMALIZED_TO_CANONICAL.update(
    {normalize_marker_name(name): name for name in _CANONICAL_ALIASES}
)


def canonicalize_marker_name(name: str) -> str:
    """Return canonical Stage 3 marker name for known aliases."""
    key = normalize_marker_name(name)
    return _NORMALIZED_TO_CANONICAL.get(key, str(name).strip())


def marker_candidates(marker: str) -> list[str]:
    """Return canonical marker name plus known aliases."""
    canonical = canonicalize_marker_name(marker)
    aliases = _CANONICAL_ALIASES.get(canonical, (canonical,))
    ordered = [canonical, *aliases]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        key = normalize_marker_name(item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def resolve_first_present_column(columns: Iterable[str], marker: str) -> str | None:
    """Resolve the first matching column name for a canonical marker."""
    by_norm = {normalize_marker_name(col): col for col in columns}
    for candidate in marker_candidates(marker):
        resolved = by_norm.get(normalize_marker_name(candidate))
        if resolved is not None:
            return resolved
    return None
