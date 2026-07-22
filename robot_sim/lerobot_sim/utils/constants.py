"""
Shared constants and type aliases for the lerobot_sim package.

Mirrors the key naming conventions used by the upstream LeRobot project so
that observations, actions, and features produced by this simulator are
directly compatible with LeRobot policies and datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Observation / action key names (match upstream LeRobot constants)
# ---------------------------------------------------------------------------
ACTION: str = "action"
OBS_STATE: str = "observation.state"
OBS_IMAGE: str = "observation.image"
OBS_IMAGES: str = "observation.images"
OBS_ENV_STATE: str = "observation.environment_state"

# ---------------------------------------------------------------------------
# Default rendering dimensions
# ---------------------------------------------------------------------------
DEFAULT_RENDER_WIDTH: int = 384
DEFAULT_RENDER_HEIGHT: int = 384
DEFAULT_FPS: int = 10

# ---------------------------------------------------------------------------
# Joint-limit defaults for a 6-DOF SO-100 style arm (radians)
# ---------------------------------------------------------------------------
SO100_NUM_JOINTS: int = 6
SO100_JOINT_LOWER: Tuple[float, ...] = (-3.14, -1.57, -1.57, -3.14, -1.57, -3.14)
SO100_JOINT_UPPER: Tuple[float, ...] = (3.14, 1.57, 1.57, 3.14, 1.57, 3.14)
SO100_LINK_LENGTHS: Tuple[float, ...] = (0.10, 0.15, 0.15, 0.05, 0.05, 0.03)

# ---------------------------------------------------------------------------
# Color palette (RGB 0-255) used by the 2-D renderers
# ---------------------------------------------------------------------------
COLOR_BACKGROUND: Tuple[int, int, int] = (240, 240, 240)
COLOR_ROBOT: Tuple[int, int, int] = (66, 133, 244)
COLOR_TARGET: Tuple[int, int, int] = (219, 68, 55)
COLOR_OBJECT: Tuple[int, int, int] = (244, 180, 0)
COLOR_SUCCESS: Tuple[int, int, int] = (15, 157, 88)
COLOR_GRID: Tuple[int, int, int] = (200, 200, 200)
COLOR_TEXT: Tuple[int, int, int] = (50, 50, 50)


class FeatureType(Enum):
    """Enumeration of observation feature types, matching LeRobot upstream."""

    ACTION = "action"
    STATE = "state"
    VISUAL = "visual"
    ENV = "env"


@dataclass(frozen=True)
class PolicyFeature:
    """Describes a single feature consumed or produced by a policy.

    Attributes:
        type: The semantic category of the feature.
        shape: Tuple of integers describing the tensor shape (excluding batch).
    """

    type: FeatureType
    shape: Tuple[int, ...]
