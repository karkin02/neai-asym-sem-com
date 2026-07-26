from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class JointCommandTransport(Protocol):
    def write_joint_positions(
        self,
        positions: dict[str, float],
    ) -> None: ...


@dataclass(frozen=True)
class JointCalibration:
    """Map one model-space joint value into hardware-space units."""

    offset: float = 0.0
    scale: float = 1.0
    minimum: float = float("-inf")
    maximum: float = float("inf")

    def map_value(self, value: float) -> float:
        mapped = self.offset + self.scale * value
        if not self.minimum <= mapped <= self.maximum:
            raise ValueError(
                f"Mapped command {mapped:.6f} is outside "
                f"[{self.minimum:.6f}, {self.maximum:.6f}]."
            )
        return mapped

    def unmap_value(self, value: float) -> float:
        if self.scale == 0.0:
            raise ValueError("Calibration scale cannot be zero.")
        if not self.minimum <= value <= self.maximum:
            raise ValueError(
                f"Hardware state {value:.6f} is outside "
                f"[{self.minimum:.6f}, {self.maximum:.6f}]."
            )
        return (value - self.offset) / self.scale


class DryRunTransport:
    """Record commands without connecting to or moving a robot."""

    def __init__(self) -> None:
        self.commands: list[dict[str, float]] = []

    def write_joint_positions(
        self,
        positions: dict[str, float],
    ) -> None:
        self.commands.append(positions.copy())


class SO101ActionAdapter:
    """Safety boundary from six model actions to an SO-101 transport."""

    def __init__(
        self,
        transport: JointCommandTransport,
        *,
        calibration: dict[str, JointCalibration],
        dry_run: bool = True,
    ) -> None:
        missing = set(JOINT_NAMES) - set(calibration)
        extra = set(calibration) - set(JOINT_NAMES)
        if missing or extra:
            raise ValueError(
                f"Calibration keys must exactly match SO-101 joints; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )
        self._transport = transport
        self._calibration = calibration.copy()
        self._dry_run = dry_run
        self._emergency_stopped = False
        self._last_command: dict[str, float] | None = None

    def emergency_stop(self) -> None:
        self._emergency_stopped = True

    def reset_emergency_stop(self, *, operator_confirmed: bool) -> None:
        if not operator_confirmed:
            raise PermissionError(
                "An operator confirmation is required to reset emergency stop."
            )
        self._emergency_stopped = False

    def command(self, values: Sequence[float]) -> dict[str, float]:
        if self._emergency_stopped:
            raise RuntimeError("Emergency stop is latched; command rejected.")
        if len(values) != len(JOINT_NAMES):
            raise ValueError(
                f"Expected {len(JOINT_NAMES)} joint values, got {len(values)}."
            )
        numeric = tuple(float(value) for value in values)
        if not all(isfinite(value) for value in numeric):
            raise ValueError("Joint commands must contain only finite values.")

        mapped = {
            name: self._calibration[name].map_value(value)
            for name, value in zip(JOINT_NAMES, numeric, strict=True)
        }
        self._last_command = mapped.copy()
        if not self._dry_run:
            self._transport.write_joint_positions(mapped)
        return mapped

    @property
    def last_command(self) -> dict[str, float] | None:
        return self._last_command.copy() if self._last_command else None
