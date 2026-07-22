"""
Vision-based teleoperation interface using real-world object detection.

Translates a detected object's 2-D pixel position (from a live webcam
feed processed by YOLO) into continuous action vectors, matching the
same interface as KeyboardTeleop so it can be swapped in directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VisionTeleop:
    """Maps a detected object's screen position to robot actions.

    Attributes:
        detector: A Detector instance (from visual_assistant/detector.py).
        action_dim: Dimensionality of the output action vector (matches env.action_space).
        target_label: The YOLO class label to track (e.g. "cup", "bottle").
        speed: Magnitude of the velocity command toward the target.
        grasp_distance: Normalized distance threshold below which the gripper closes.
    """

    detector: object
    action_dim: int = 7
    target_label: str = "cup"
    speed: float = 0.5
    grasp_distance: float = 0.08

    def __post_init__(self) -> None:
        self._last_frame: Optional[np.ndarray] = None

    def set_frame(self, frame: np.ndarray) -> None:
        """Store the latest webcam frame to run detection on."""
        self._last_frame = frame

    def get_action(self) -> np.ndarray:
        """Return an action vector derived from the detected object's position.

        Returns:
            1-D float32 array of shape (action_dim,).
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        if self._last_frame is None:
            return action

        detections = self.detector.detect(self._last_frame)
        target = next((d for d in detections if d.get("label") == self.target_label), None)
        if target is None:
            return action

        x1, y1, x2, y2 = target["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        h, w = self._last_frame.shape[:2]
        norm_x = (cx / w) * 2 - 1
        norm_y = 1 - (cy / h) * 2

        action[0] = np.clip(norm_x, -1.0, 1.0) * self.speed
        if self.action_dim > 1:
            action[1] = np.clip(norm_y, -1.0, 1.0) * self.speed

        dist_from_center = np.sqrt(norm_x**2 + norm_y**2)
        if self.action_dim > 6 and dist_from_center < self.grasp_distance:
            action[6] = 1.0

        return action

    def process_pygame_events(self) -> bool:
        """Pump Pygame events just to detect window close.

        Returns:
            False if a QUIT event was received; True otherwise.

        Raises:
            ImportError: If Pygame is not installed.
        """
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("Pygame required: pip install pygame") from exc
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
