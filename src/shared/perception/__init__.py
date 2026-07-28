"""Shared perception pipeline for Architectures B and C.

Real (not simulated) YOLOv8n detection, scene-graph construction, and CLIP
grounding. Each component is a standalone, injectable unit so it can be tested
without the heavy vision stack installed.
"""

from __future__ import annotations

from .detector import Detection, YoloDetector, crop_bbox, parse_yolo_result
from .scene_graph import build_scene_graph
from .clip_grounding import ClipGrounder, GroundingResult

__all__ = [
    "Detection",
    "YoloDetector",
    "crop_bbox",
    "parse_yolo_result",
    "build_scene_graph",
    "ClipGrounder",
    "GroundingResult",
]
