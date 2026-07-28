"""Uncertainty-gated routing for Architecture C.

The gate routes an instruction to the local SmolVLA policy only when BOTH:

1. the CLIP grounding confidence is at/above a threshold, and
2. the instruction is *recognized* — a configurable keyword allowlist of verbs
   the local policy was trained for AND a known object (named in the instruction
   or present in the detections).

Otherwise it escalates over the channel to GPT-4o-mini (the Architecture B
path). Everything here is a pure function of its inputs, so it is deterministic
and unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class RoutingConfig:
    """Gate configuration.

    Attributes:
        clip_threshold: Minimum CLIP confidence to consider local handling.
        known_verbs: Task verbs the local policy is trained for.
        known_objects: Object labels the local policy is trained for.
    """

    clip_threshold: float = 0.60
    known_verbs: tuple[str, ...] = ("pick", "place", "put", "move", "grasp", "drop", "route")
    known_objects: tuple[str, ...] = ("sample", "package", "cube", "block", "box", "cup", "bottle")


@dataclass(frozen=True)
class RouteDecision:
    """Result of the gate.

    Attributes:
        route: ``"local"`` (SmolVLA) or ``"escalate"`` (cloud).
        escalated: True when routed to the cloud.
        recognized: Whether the instruction matched the keyword allowlist.
        clip_ok: Whether CLIP confidence met the threshold.
        clip_confidence: The confidence considered.
        reason: Human-readable explanation.
    """

    route: str
    escalated: bool
    recognized: bool
    clip_ok: bool
    clip_confidence: float
    reason: str


def is_recognized(instruction: str, detected_labels: Sequence[str], config: RoutingConfig) -> bool:
    """True if the instruction has a known verb and a known object.

    The object may be named in the instruction or appear in the detections.
    """
    text = instruction.lower()
    has_verb = any(verb in text for verb in config.known_verbs)
    known = {o.lower() for o in config.known_objects}
    object_in_text = any(obj in text for obj in known)
    object_detected = any(label.lower() in known for label in detected_labels)
    return has_verb and (object_in_text or object_detected)


def decide_route(
    instruction: str,
    clip_confidence: float,
    detected_labels: Sequence[str],
    config: RoutingConfig = RoutingConfig(),
) -> RouteDecision:
    """Decide local vs escalate for one instruction."""
    recognized = is_recognized(instruction, detected_labels, config)
    clip_ok = clip_confidence >= config.clip_threshold
    if clip_ok and recognized:
        return RouteDecision(
            route="local",
            escalated=False,
            recognized=True,
            clip_ok=True,
            clip_confidence=clip_confidence,
            reason=f"clip {clip_confidence:.2f} >= {config.clip_threshold} and instruction recognized",
        )
    reasons = []
    if not clip_ok:
        reasons.append(f"clip {clip_confidence:.2f} < {config.clip_threshold}")
    if not recognized:
        reasons.append("instruction not recognized")
    return RouteDecision(
        route="escalate",
        escalated=True,
        recognized=recognized,
        clip_ok=clip_ok,
        clip_confidence=clip_confidence,
        reason="; ".join(reasons),
    )
