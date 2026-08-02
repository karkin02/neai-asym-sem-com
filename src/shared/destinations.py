"""Canonical warehouse destination roles and physical-target compatibility."""

from __future__ import annotations


DESTINATIONS = frozenset(("conveyor", "inspection_tray", "rejection_tray", "outbound_bin"))

_ALIASES = {
    "conveyor": "conveyor",
    "inspection_tray": "inspection_tray",
    "manual_inspection": "inspection_tray",
    "left_tray": "inspection_tray",
    "blue_tray": "inspection_tray",
    "rejection_tray": "rejection_tray",
    "reject_tray": "rejection_tray",
    "right_tray": "rejection_tray",
    "yellow_tray": "rejection_tray",
    "outbound_bin": "outbound_bin",
}

PHYSICAL_TARGETS = {
    "conveyor": "conveyor",
    "inspection_tray": "left_tray",
    "rejection_tray": "right_tray",
    "outbound_bin": "outbound_bin",
}


def normalize_destination(value: object) -> str | None:
    """Return a semantic role for canonical or legacy destination names."""
    if value is None:
        return None
    return _ALIASES.get(str(value).strip().lower())


def physical_target(value: object) -> str:
    """Resolve a semantic/legacy destination to the local MuJoCo target name."""
    role = normalize_destination(value)
    if role is None:
        raise ValueError(f"Unknown destination role: {value!r}")
    return PHYSICAL_TARGETS[role]
