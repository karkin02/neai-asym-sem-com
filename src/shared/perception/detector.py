"""YOLOv8n object detection for the shared B/C perception pipeline.

The detector is a thin, dependency-light wrapper around Ultralytics YOLOv8.
``ultralytics`` is imported lazily inside :meth:`YoloDetector._ensure_model`
so this module can be imported (and unit-tested with an injected fake model)
in an environment that does not have the heavy vision stack installed.

Output contract (stable across Architectures B and C)::

    detect(frame) -> list[Detection]
    Detection(label: str, confidence: float, bbox: (x1, y1, x2, y2))

``bbox`` is pixel coordinates in the input frame's own resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Detection:
    """A single detected object.

    Attributes:
        label: Class name (e.g. ``"cup"``).
        confidence: Detection confidence in ``[0, 1]``.
        bbox: Pixel box ``(x1, y1, x2, y2)`` in the source frame.
    """

    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Return the ``(cx, cy)`` pixel centre of the box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "label": self.label,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 1) for v in self.bbox],
        }


def detect_package_damage_mark(
    frame: np.ndarray,
    detections: Sequence[Detection],
    *,
    package_confidence: float = 0.50,
) -> Optional[Detection]:
    """Verify a narrow saturated-red damage stripe in a YOLO package crop."""
    packages = [
        item
        for item in detections
        if item.label == "package" and item.confidence >= package_confidence
    ]
    if not packages:
        return None
    package = max(packages, key=lambda item: item.confidence)
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in package.bbox)
    pad = 3
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(width, x2 + pad), min(height, y2 + pad)
    crop = np.asarray(frame[y1:y2, x1:x2])
    if crop.size == 0:
        return None
    red = crop[:, :, 0].astype(np.int16)
    green = crop[:, :, 1].astype(np.int16)
    blue = crop[:, :, 2].astype(np.int16)
    mask = (
        (red >= 140)
        & (green <= 55)
        & (blue <= 55)
        & (red >= green + 90)
        & (red >= blue + 90)
    )
    ys, xs = np.where(mask)
    if len(xs) < 30:
        return None
    component_width = int(xs.max() - xs.min() + 1)
    component_height = int(ys.max() - ys.min() + 1)
    if component_width > 12 or component_height < 20:
        return None
    if component_height / max(component_width, 1) < 2.5:
        return None
    confidence = min(0.99, 0.45 + (len(xs) - 30) / 200.0)
    return Detection(
        label="damage_mark",
        confidence=float(confidence),
        bbox=(
            float(x1 + xs.min()),
            float(y1 + ys.min()),
            float(x1 + xs.max() + 1),
            float(y1 + ys.max() + 1),
        ),
    )


def parse_yolo_result(result: Any, conf_threshold: float) -> List[Detection]:
    """Convert one Ultralytics result object into :class:`Detection` records.

    Kept as a standalone function so the parsing/threshold logic is unit-testable
    with a minimal fake result (no real model or weights required).

    Args:
        result: An Ultralytics ``Results`` object exposing ``.boxes`` (each box
            with ``.conf``, ``.cls``, ``.xyxy``) and ``.names`` (id -> label).
        conf_threshold: Minimum confidence to keep a detection.

    Returns:
        Detections at or above ``conf_threshold``, in the model's own order.
    """
    detections: List[Detection] = []
    names = getattr(result, "names", {})
    for box in result.boxes:
        confidence = float(box.conf[0])
        if confidence < conf_threshold:
            continue
        label = names[int(box.cls[0])]
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append(
            Detection(label=str(label), confidence=confidence, bbox=(x1, y1, x2, y2))
        )
    return detections


class YoloDetector:
    """YOLOv8n detector with a lazy model load and an injectable backend.

    Args:
        model_path: Ultralytics weights path/name (default ``"yolov8n.pt"``,
            auto-downloaded by Ultralytics on first real use).
        conf_threshold: Minimum confidence to report.
        model: Optional pre-built model. Any callable matching the Ultralytics
            ``model(frame, verbose=False) -> [Results]`` contract works; passing
            one here (e.g. a fake in tests) skips the lazy import entirely.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: Optional[str] = None,
        model: Optional[Any] = None,
    ) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device
        self._model = model

    def _ensure_model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO  # lazy: heavy import only when needed

            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a single BGR/RGB frame.

        Args:
            frame: ``(H, W, 3)`` image array.

        Returns:
            A list of :class:`Detection` above the confidence threshold.
        """
        model = self._ensure_model()
        if self.device is not None and hasattr(model, "predict"):
            result = model.predict(
                source=frame,
                verbose=False,
                device=self.device,
                conf=self.conf_threshold,
            )[0]
        else:
            result = model(frame, verbose=False)[0]
        return parse_yolo_result(result, self.conf_threshold)


def crop_bbox(frame: np.ndarray, bbox: Sequence[float]) -> np.ndarray:
    """Return the sub-image inside ``bbox`` (clamped to the frame).

    Used to produce object crops for CLIP grounding. Returns a ``(1, 1, C)``
    fallback if the clamped box is empty, so downstream code never crashes.

    Args:
        frame: ``(H, W, C)`` image array.
        bbox: ``(x1, y1, x2, y2)`` pixel coordinates.

    Returns:
        The cropped image array.
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1i = int(max(0, min(width, round(x1))))
    x2i = int(max(0, min(width, round(x2))))
    y1i = int(max(0, min(height, round(y1))))
    y2i = int(max(0, min(height, round(y2))))
    if x2i <= x1i or y2i <= y1i:
        return frame[0:1, 0:1]
    return frame[y1i:y2i, x1i:x2i]
