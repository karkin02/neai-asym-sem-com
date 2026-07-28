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


def _compact_scene_graph(scene_graph: dict[str, Any]) -> dict[str, Any]:
    """Reduce a full scene graph to labels + relation summary only."""
    return {
        "objects": [{"id": o["id"], "label": o["label"]} for o in scene_graph.get("objects", [])],
        "relations": scene_graph.get("summary", []),
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
