from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VLAConfidence:
    confidence: float
    chunk_smoothness: float
    temporal_consistency: float
    maximum_normalized_step: float
    prediction_disagreement: float


def estimate_vla_confidence(
    action_chunk: np.ndarray,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    previous_chunk: np.ndarray | None = None,
) -> VLAConfidence:
    """Estimate uncertainty from action stability; this is not model probability."""
    chunk = np.asarray(action_chunk, dtype=float)
    low = np.asarray(action_low, dtype=float)
    high = np.asarray(action_high, dtype=float)
    if chunk.ndim != 2 or chunk.shape[1] != low.size or high.shape != low.shape:
        raise ValueError("Action chunks and action bounds must have matching dimensions.")
    if not np.isfinite(chunk).all():
        return VLAConfidence(0.0, 0.0, 0.0, float("inf"), float("inf"))

    span = np.maximum(high - low, 1e-9)
    steps = np.diff(chunk, axis=0)
    maximum_step = float(np.max(np.abs(steps) / span)) if steps.size else 0.0
    smoothness = float(np.exp(-4.0 * maximum_step))

    disagreement = 0.0
    temporal = 1.0
    if previous_chunk is not None:
        previous = np.asarray(previous_chunk, dtype=float)
        overlap = min(len(chunk), len(previous))
        if previous.ndim != 2 or previous.shape[1] != chunk.shape[1] or overlap == 0:
            raise ValueError("Previous action chunk must match the current action dimension.")
        disagreement = float(
            np.mean(np.abs(chunk[:overlap] - previous[:overlap]) / span)
        )
        temporal = float(np.exp(-6.0 * disagreement))

    return VLAConfidence(
        confidence=min(smoothness, temporal),
        chunk_smoothness=smoothness,
        temporal_consistency=temporal,
        maximum_normalized_step=maximum_step,
        prediction_disagreement=disagreement,
    )
