"""Local visual condition gate for Architecture A warehouse routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from shared.perception import Detection, detect_package_damage_mark


DESTINATION_INSTRUCTIONS = {
    "conveyor": "Pick up the package and place it on the conveyor.",
    "inspection_tray": "The package barcode is missing. Place it in the blue inspection tray.",
    "rejection_tray": "The package is damaged. Place it in the yellow rejection tray.",
}


@dataclass(frozen=True)
class InspectionDecision:
    destination: str | None
    instruction: str | None
    should_escalate: bool
    reason: str
    barcode_confidence: float
    damage_confidence: float
    package_confidence: float


def _maximum(detections: Sequence[Detection], label: str) -> float:
    return max(
        (float(item.confidence) for item in detections if item.label == label),
        default=0.0,
    )


def decide_condition_route(
    views: Mapping[str, Sequence[Detection]],
    *,
    barcode_threshold: float = 0.80,
    damage_threshold: float = 0.40,
    package_threshold: float = 0.80,
) -> InspectionDecision:
    """Resolve the warehouse destination from close visual evidence only.

    Damage has priority over barcode state. A well-grounded package with no
    visible barcode is conservatively routed to inspection. If the package
    itself is not grounded, A stops/escalates rather than interpreting a failed
    camera view as a missing barcode.
    """
    barcode_view = tuple(views.get("barcode", ()))
    damage_view = tuple(views.get("damage", ()))
    # Both are certified poses of the same physical wrist camera. Accept a
    # barcode visible in either pose; overhead detections remain ineligible.
    barcode = max(
        _maximum(barcode_view, "barcode"),
        _maximum(damage_view, "barcode"),
    )
    damage = _maximum(damage_view, "damage_mark")
    package = max(
        _maximum(barcode_view, "package"),
        _maximum(damage_view, "package"),
    )
    if damage >= damage_threshold:
        destination = "rejection_tray"
        reason = "damage mark detected"
    elif barcode >= barcode_threshold:
        destination = "conveyor"
        reason = "barcode detected and no damage detected"
    elif package >= package_threshold:
        destination = "inspection_tray"
        reason = "package grounded but barcode absent"
    else:
        return InspectionDecision(
            destination=None,
            instruction=None,
            should_escalate=True,
            reason="inspection inconclusive: package not grounded",
            barcode_confidence=barcode,
            damage_confidence=damage,
            package_confidence=package,
        )
    return InspectionDecision(
        destination=destination,
        instruction=DESTINATION_INSTRUCTIONS[destination],
        should_escalate=False,
        reason=reason,
        barcode_confidence=barcode,
        damage_confidence=damage,
        package_confidence=package,
    )


def inspect_environment(env, detector, *, width: int = 320, height: int = 240):
    frames = env.capture_condition_views(width=width, height=height)
    detections = {name: detector.detect(frame) for name, frame in frames.items()}
    damage_cue = detect_package_damage_mark(frames["damage"], detections["damage"])
    if damage_cue is not None:
        detections["damage"] = [*detections["damage"], damage_cue]
    return decide_condition_route(detections), frames, detections
