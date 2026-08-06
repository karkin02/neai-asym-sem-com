"""Scene-representation payloads at multiple compression levels.

Architecture B benchmarks the bandwidth/quality tradeoff by sending the scene
to the cloud planner at one of three levels:

- ``full_json``    — the complete scene graph (objects + relations + metadata).
- ``scene_graph``  — a compact relational summary (labels + relation strings).
- ``raw_image``    — the PNG-encoded camera frame (base64), no scene graph.

Each payload reports its on-wire ``num_bytes`` for the network-cost metric.
Pillow is imported lazily and only for ``raw_image``, so the JSON levels have
no heavy dependency.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import numpy as np


BARCODE_CONFIDENCE_THRESHOLD = 0.50
DAMAGE_MARK_CONFIDENCE_THRESHOLD = 0.45


class CompressionLevel(str, Enum):
    FULL_JSON = "full_json"
    SCENE_GRAPH = "scene_graph"
    RAW_IMAGE = "raw_image"


@dataclass(frozen=True)
class Payload:
    """A serialised scene representation ready for the channel + planner.

    Attributes:
        level: The compression level used.
        instruction: The task instruction (always sent alongside the scene).
        content: Planner-facing content — a dict for the JSON levels, or
            ``{"image_b64": ...}`` for ``raw_image``.
        num_bytes: On-wire size in bytes (what the channel bills).
    """

    level: CompressionLevel
    instruction: str
    content: dict[str, Any]
    num_bytes: int


def visual_observations_for_views(
    view_labels: dict[str, list[Any]],
    *,
    barcode_threshold: float = BARCODE_CONFIDENCE_THRESHOLD,
    damage_threshold: float = DAMAGE_MARK_CONFIDENCE_THRESHOLD,
    package_threshold: float = 0.50,
) -> dict[str, Any]:
    """Summarize multi-view detector labels without treating an unseen view as evidence."""
    normalized: dict[str, set[str]] = {}
    confidence_by_view: dict[str, dict[str, float]] = {}
    for view, items in view_labels.items():
        normalized[view] = set()
        confidence_by_view[view] = {}
        for item in items:
            raw_label = getattr(item, "label", item)
            label = str(raw_label).strip().lower()
            label = "package" if label in {"sample", "box", "parcel"} else label
            confidence = float(getattr(item, "confidence", 1.0))
            normalized[view].add(label)
            confidence_by_view[view][label] = max(
                confidence, confidence_by_view[view].get(label, 0.0)
            )
    package_views = [
        view
        for view, labels in normalized.items()
        if "package" in labels
        and confidence_by_view[view].get("package", 0.0) >= package_threshold
    ]
    barcode_views = [view for view, labels in normalized.items() if "barcode" in labels]
    damage_views = [view for view, labels in normalized.items() if "damage_mark" in labels]
    obstacle_views = [view for view, labels in normalized.items() if "obstacle" in labels]
    inspected_views = list(normalized)
    certified_barcode_views = [view for view in ("barcode", "damage") if view in normalized]
    barcode_evidence_views = certified_barcode_views or list(normalized)
    damage_evidence_views = ["damage"] if "damage" in normalized else list(normalized)
    barcode_detected_by = [
        view
        for view in barcode_views
        if view in barcode_evidence_views
        and confidence_by_view[view].get("barcode", 0.0) >= barcode_threshold
    ]
    damage_detected_by = [
        view
        for view in damage_views
        if view in damage_evidence_views
        and confidence_by_view[view].get("damage_mark", 0.0) >= damage_threshold
    ]
    return {
        "views_inspected": inspected_views,
        "package_detected": bool(package_views),
        "package_detected_by": package_views,
        "barcode_detected": bool(barcode_detected_by),
        "barcode_detected_by": barcode_detected_by,
        "damage_mark_detected": bool(damage_detected_by),
        "damage_mark_detected_by": damage_detected_by,
        "obstacle_detected": bool(obstacle_views),
        "obstacle_detected_by": obstacle_views,
        "inspection_complete": bool(inspected_views) and len(package_views) == len(inspected_views),
        "barcode_inspection_complete": (
            confidence_by_view.get("barcode", {}).get("package", 0.0) >= package_threshold
            if "barcode" in normalized
            else bool(inspected_views) and len(package_views) == len(inspected_views)
        ),
        "damage_inspection_complete": (
            confidence_by_view.get("damage", {}).get("package", 0.0) >= package_threshold
            if "damage" in normalized
            else bool(inspected_views) and len(package_views) == len(inspected_views)
        ),
        "class_confidence_by_view": confidence_by_view,
        "barcode_max_confidence": max(
            (confidence_by_view[view].get("barcode", 0.0) for view in barcode_evidence_views), default=0.0
        ),
        "damage_mark_max_confidence": max(
            (confidence_by_view[view].get("damage_mark", 0.0) for view in damage_evidence_views), default=0.0
        ),
        "barcode_confidence_threshold": float(barcode_threshold),
        "damage_mark_confidence_threshold": float(damage_threshold),
        "package_confidence_threshold": float(package_threshold),
    }


def _compact_scene_graph(scene_graph: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full scene graph to labels + relation summary only."""
    labels = [str(o["label"]).strip().lower() for o in scene_graph.get("objects", [])]
    normalized = ["package" if label in {"sample", "box", "parcel"} else label for label in labels]
    derived_observations = {
        "package_detected": "package" in normalized,
        "barcode_detected": "barcode" in normalized,
        "damage_mark_detected": "damage_mark" in normalized,
        "obstacle_detected": "obstacle" in normalized,
    }
    return {
        "objects": [{"id": o["id"], "label": o["label"]} for o in scene_graph.get("objects", [])],
        "relations": scene_graph.get("summary", []),
        "visual_observations": scene_graph.get("visual_observations", derived_observations),
    }


def build_payload(
    level: CompressionLevel,
    scene_graph: dict[str, Any],
    instruction: str,
    frame: Optional[np.ndarray] = None,
) -> Payload:
    """Serialise the scene at ``level`` and measure its byte size.

    Args:
        level: Which compression level to build.
        scene_graph: Output of :func:`shared.perception.build_scene_graph`.
        instruction: The task instruction.
        frame: The camera frame; required for ``raw_image``.

    Returns:
        A :class:`Payload` with planner content and ``num_bytes``.
    """
    level = CompressionLevel(level)

    if level is CompressionLevel.RAW_IMAGE:
        if frame is None:
            raise ValueError("raw_image compression requires a frame.")
        image_b64 = _encode_png_b64(frame)
        content = {"instruction": instruction, "image_b64": image_b64}
        # Bill the raw PNG bytes (the dominant on-wire cost), not the base64.
        num_bytes = (len(image_b64) * 3) // 4
        return Payload(level, instruction, content, int(num_bytes))

    if level is CompressionLevel.SCENE_GRAPH:
        body = {"instruction": instruction, "scene": _compact_scene_graph(scene_graph)}
    else:  # FULL_JSON
        body = {"instruction": instruction, "scene": scene_graph}

    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    return Payload(level, instruction, body, len(encoded))


def _encode_png_b64(frame: np.ndarray) -> str:
    """PNG-encode an image frame to a base64 string (lazy Pillow import)."""
    from io import BytesIO

    from PIL import Image

    array = np.asarray(frame)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
