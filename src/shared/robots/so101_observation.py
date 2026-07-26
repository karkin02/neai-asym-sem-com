from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol

import numpy as np

from .so101_adapter import JOINT_NAMES, JointCalibration


class CameraSource(Protocol):
    def capture_rgb(self, camera: str) -> np.ndarray: ...


class JointStateSource(Protocol):
    def read_joint_positions(self) -> dict[str, float]: ...


@dataclass(frozen=True)
class SO101ObservationPacket:
    overhead: np.ndarray
    wrist: np.ndarray
    state: tuple[float, ...]
    task: str

    def as_lerobot_input(self) -> dict[str, object]:
        return {
            "observation.images.overhead": self.overhead,
            "observation.images.wrist": self.wrist,
            "observation.state": np.asarray(self.state, dtype=np.float32),
            "task": self.task,
        }


class SO101ObservationAdapter:
    """Validate physical sensors and map them into model-space observations."""

    def __init__(
        self,
        camera_source: CameraSource,
        state_source: JointStateSource,
        *,
        calibration: dict[str, JointCalibration],
    ) -> None:
        if set(calibration) != set(JOINT_NAMES):
            raise ValueError("Calibration must contain exactly the six SO-101 joints.")
        self._camera_source = camera_source
        self._state_source = state_source
        self._calibration = calibration.copy()

    def read(self, *, task: str) -> SO101ObservationPacket:
        if not task.strip():
            raise ValueError("Task instruction cannot be empty.")
        overhead = self._validate_image(
            self._camera_source.capture_rgb("overhead"),
            camera="overhead",
        )
        wrist = self._validate_image(
            self._camera_source.capture_rgb("wrist"),
            camera="wrist",
        )
        if overhead.shape != wrist.shape:
            raise ValueError(
                "Overhead and wrist cameras must use the same image shape."
            )

        hardware_state = self._state_source.read_joint_positions()
        if set(hardware_state) != set(JOINT_NAMES):
            raise ValueError("Joint state must contain exactly the six SO-101 joints.")
        numeric = {name: float(hardware_state[name]) for name in JOINT_NAMES}
        if not all(isfinite(value) for value in numeric.values()):
            raise ValueError("Joint state must contain only finite values.")
        state = tuple(
            self._calibration[name].unmap_value(numeric[name])
            for name in JOINT_NAMES
        )
        return SO101ObservationPacket(
            overhead=overhead,
            wrist=wrist,
            state=state,
            task=task,
        )

    @staticmethod
    def _validate_image(image: np.ndarray, *, camera: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"{camera} camera must provide an HWC RGB image.")
        if array.dtype != np.uint8:
            raise ValueError(f"{camera} camera image must use uint8 pixels.")
        if array.shape[0] < 32 or array.shape[1] < 32:
            raise ValueError(f"{camera} camera image is unexpectedly small.")
        return np.ascontiguousarray(array)
