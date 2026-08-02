"""Scene-graph construction from object detections.

Converts a flat list of :class:`~shared.perception.detector.Detection` objects
into a structured, JSON-serialisable scene graph describing objects and their
pairwise spatial relations (``left_of`` / ``above`` / ``near``) plus optional
object-to-zone relations (e.g. an object ``near`` or ``in`` a tray).

This is a pure function of the detections and the frame size — no model, no
heavy dependencies — so it is fully deterministic and unit-testable. The output
shape is intentionally close to ``SO101MuJoCoEnvironment.scene_graph()`` (keys
``task`` / ``objects`` / ``relations``) so it can travel through the same
Architecture B handoff payloads.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from .detector import Detection


def _object_id(label: str, index: int) -> str:
    """Stable per-scene id that disambiguates duplicate labels."""
    return f"{label}_{index}"


def _pairwise_relation(a_center, b_center):
    """Return the dominant-axis relation from ``a`` to ``b``.

    Uses the larger of the horizontal/vertical offsets so each pair yields one
    directional relation. Image coordinates have y increasing downward, so a
    smaller ``cy`` means physically higher → ``above``.
    """
    dx = b_center[0] - a_center[0]
    dy = b_center[1] - a_center[1]
    if abs(dx) >= abs(dy):
        # a is left of b when b is further right (dx >= 0)
        return ("left_of", True) if dx >= 0 else ("left_of", False)
    return ("above", True) if dy >= 0 else ("above", False)


def build_scene_graph(
    detections: Sequence[Detection],
    image_width: int,
    image_height: int,
    zones: Optional[Mapping[str, Any]] = None,
    task: Optional[str] = None,
    near_fraction: float = 0.15,
) -> dict[str, Any]:
    """Build a scene graph from detections.

    Args:
        detections: Detected objects.
        image_width: Frame width in pixels (for normalisation + near threshold).
        image_height: Frame height in pixels.
        zones: Optional mapping ``{zone_name: spec}`` where ``spec`` is either
            ``{"bbox": [x1, y1, x2, y2]}`` or ``{"center": [cx, cy]}`` (or a bare
            ``[cx, cy]``). Produces object-to-zone ``in``/``near`` relations.
        task: Optional instruction/task string carried in the payload.
        near_fraction: "near" distance threshold as a fraction of the image
            diagonal.

    Returns:
        A JSON-serialisable dict with keys ``task``, ``image_size``,
        ``objects``, ``relations`` and a human-readable ``summary`` list.
    """
    diag = math.hypot(image_width, image_height) or 1.0
    near_threshold = near_fraction * diag

    objects: list[dict[str, Any]] = []
    for index, det in enumerate(detections):
        cx, cy = det.center
        objects.append(
            {
                "id": _object_id(det.label, index),
                "label": det.label,
                "confidence": round(float(det.confidence), 4),
                "bbox": [round(float(v), 1) for v in det.bbox],
                "center": [round(cx, 1), round(cy, 1)],
            }
        )

    relations: list[dict[str, str]] = []

    # Pairwise object-object relations (each unordered pair once).
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            ci = objects[i]["center"]
            cj = objects[j]["center"]
            relation, a_is_subject = _pairwise_relation(ci, cj)
            subject, obj = (i, j) if a_is_subject else (j, i)
            relations.append(
                {
                    "subject": objects[subject]["id"],
                    "relation": relation,
                    "object": objects[obj]["id"],
                }
            )
            if math.dist(ci, cj) < near_threshold:
                relations.append(
                    {"subject": objects[i]["id"], "relation": "near", "object": objects[j]["id"]}
                )

    # Object-zone relations.
    for zone_name, spec in (zones or {}).items():
        for obj in objects:
            rel = _zone_relation(obj["center"], spec, near_threshold)
            if rel is not None:
                relations.append(
                    {"subject": obj["id"], "relation": rel, "object": zone_name}
                )

    summary = [f"{r['subject']} {r['relation']} {r['object']}" for r in relations]

    return {
        "task": task,
        "image_size": [int(image_width), int(image_height)],
        "objects": objects,
        "relations": relations,
        "summary": summary,
    }


def _zone_relation(center, spec: Any, near_threshold: float) -> Optional[str]:
    """Return ``"in"``/``"near"``/``None`` for an object centre versus a zone."""
    cx, cy = center
    if isinstance(spec, Mapping) and "bbox" in spec:
        x1, y1, x2, y2 = spec["bbox"]
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            return "in"
        zone_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    elif isinstance(spec, Mapping) and "center" in spec:
        zone_center = tuple(spec["center"])
    else:
        zone_center = tuple(spec)  # bare [cx, cy]
    return "near" if math.dist((cx, cy), zone_center) < near_threshold else None
