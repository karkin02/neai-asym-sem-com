"""Safety handling for model-predicted joint-position chunks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SO101_ACTION_LOW = np.array((-1.8, -1.7, -2.0, -1.8, -2.8, 0.0))
SO101_ACTION_HIGH = np.array((1.8, 1.5, 1.8, 1.8, 2.8, 0.025))
# The blue-tray expert trajectory intentionally saturates shoulder_pan. The
# executable values are still clipped to SO101_ACTION_HIGH before simulation.
SO101_ACTION_BOUND_TOLERANCE = np.array(
    (0.12, 0.02, 0.02, 0.02, 0.02, 0.0005)
)


@dataclass(frozen=True)
class BoundedActionChunk:
    values: np.ndarray
    accepted: bool
    clipped_rows: int
    maximum_overshoot: float
    violation: dict[str, float | int] | None


def bound_action_chunk(
    values: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    tolerance: np.ndarray,
) -> BoundedActionChunk:
    """Clip negligible boundary overshoot and reject material violations.

    The returned values never exceed the physical limits. ``tolerance`` only
    decides whether a raw prediction is close enough to clip safely; it does
    not expand the executable range.
    """
    chunk = np.asarray(values)
    below = np.maximum(action_low - chunk, 0.0)
    above = np.maximum(chunk - action_high, 0.0)
    overshoot = np.maximum(below, above)
    maximum = float(np.max(overshoot, initial=0.0))
    material = overshoot > tolerance
    violation = None
    if np.any(material):
        row, joint = np.unravel_index(np.argmax(overshoot), overshoot.shape)
        raw = float(chunk[row, joint])
        limit = float(action_low[joint] if raw < action_low[joint] else action_high[joint])
        violation = {
            "command_index": int(row),
            "joint_index": int(joint),
            "raw_value": raw,
            "limit": limit,
            "overshoot": float(overshoot[row, joint]),
        }
        return BoundedActionChunk(chunk, False, 0, maximum, violation)

    clipped = np.clip(chunk, action_low, action_high)
    clipped_rows = int(np.count_nonzero(np.any(overshoot > 0.0, axis=1)))
    return BoundedActionChunk(clipped, True, clipped_rows, maximum, None)
