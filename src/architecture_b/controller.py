"""Scripted controller: structured target -> MuJoCo joint commands.

Architecture B/C decide *what* to do (a :class:`~architecture_b.planner.ActionTarget`);
this controller turns that into six-joint commands and executes them through
Architecture A's existing environment — it never reimplements physics. It reuses
``env.solve_ik(...)`` to get joint targets and ``env.step(...)`` to move, exactly
as the warehouse scripted demo does.

The env is duck-typed (only ``sample_position``, ``target_positions``,
``solve_ik`` and ``step`` are needed), so this unit is testable with a fake env
and imports without ``mujoco``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from .planner import ActionTarget
from shared.destinations import physical_target

# Gripper actuator values in metres (env range 0.0 .. 0.025). solve_ik takes a
# float gripper command, so the planner's "open"/"close" intent maps to these.
GRIPPER_OPEN = 0.02
GRIPPER_CLOSE = 0.0


def gripper_value(intent: str) -> float:
    """Map a planner gripper intent (``"open"``/``"close"``) to a float command."""
    return GRIPPER_CLOSE if str(intent).lower() == "close" else GRIPPER_OPEN


@dataclass(frozen=True)
class JointAction:
    """Minimal action matching ``architecture_a.contracts.Action`` (name, values).

    Duck-typed so ``SO101MuJoCoEnvironment.step`` accepts it without importing
    Architecture A's package here.
    """

    name: str = "joint_position"
    values: tuple[float, ...] = ()


@dataclass
class ExecutionResult:
    success: bool
    steps: int
    attempts: int = 0
    final_info: dict[str, Any] = field(default_factory=dict)
    joint_commands: list[list[float]] = field(default_factory=list)
    failure_reason: Optional[str] = None


@dataclass
class ExecutionSession:
    """Resumable pick/place state; one phase runs between visual checkpoints."""

    target: ActionTarget
    result: ExecutionResult = field(default_factory=lambda: ExecutionResult(False, 0, attempts=1))
    phase: str = "approach"
    object_position: tuple[float, ...] | None = None


class ScriptedController:
    """Executes an :class:`ActionTarget` as a pick -> carry -> release sequence.

    Args:
        env: An ``SO101MuJoCoEnvironment`` (or any object exposing
            ``sample_position``, ``target_positions``, ``solve_ik`` and ``step``).
        destination_positions: Optional override mapping destination name ->
            3D world position. Defaults to ``env.target_positions``.
    """

    def __init__(
        self,
        env: Any,
        destination_positions: Optional[dict[str, Sequence[float]]] = None,
        max_attempts: int = 2,
    ) -> None:
        self._env = env
        self._destinations = destination_positions
        self._max_attempts = max(1, int(max_attempts))
        self._visual_object_position: tuple[float, ...] | None = None
        self._previous_link_latency = 0.0
        self._local_motion_horizon = 0.0
        self._world_advances_during_network = True
        self._visual_progress: float | None = None
        self._visual_direction = 1.0

    def set_world_advances_during_network(self, enabled: bool) -> None:
        """Configure whether link delay also advances external machinery."""
        self._world_advances_during_network = bool(enabled)

    def set_local_motion_horizon(self, seconds: float) -> None:
        """Lead the next visual target by a known local actuation duration."""
        self._local_motion_horizon = max(0.0, float(seconds))

    def observe_visual(self, detections: Sequence[Any], *, camera: str, width: int, height: int) -> None:
        candidates = [d for d in detections if str(d.label) in {"package", "box", "sample", "parcel"}
                      and d.bbox[2] - d.bbox[0] < 50 and d.bbox[3] - d.bbox[1] < 50]
        projector = getattr(self._env, "camera_pixel_to_world", None)
        if not candidates or not callable(projector):
            return
        item = max(candidates, key=lambda d: float(d.confidence))
        world = projector(item.center, camera=camera, width=width, height=height)
        r, cy, speed = 0.12, -0.08, 0.010
        x, y = float(world[0]), float(world[1])
        angle = np.arctan2(-(y - cy), x)
        if angle < np.pi:
            angle += 2 * np.pi
        arc_length = np.pi * r
        progress = min(arc_length, max(0.0, (angle - np.pi) * r))
        if self._visual_progress is not None:
            delta = progress - self._visual_progress
            endpoint_reversal = (
                self._visual_progress > arc_length - 0.015 and delta < 0.0
            ) or (
                self._visual_progress < 0.015 and delta > 0.0
            )
            if abs(delta) > 0.002 or endpoint_reversal:
                self._visual_direction = 1.0 if delta > 0 else -1.0
        self._visual_progress = progress
        prediction_delay = (
            self._previous_link_latency if self._world_advances_during_network else 0.0
        ) + self._local_motion_horizon
        self._local_motion_horizon = 0.0
        progress += self._visual_direction * speed * prediction_delay
        while progress < 0.0 or progress > arc_length:
            if progress > arc_length:
                progress = 2.0 * arc_length - progress
                self._visual_direction = -1.0
            elif progress < 0.0:
                progress = -progress
                self._visual_direction = 1.0
        angle = np.pi + progress / r
        predicted = (r * np.cos(angle), cy - r * np.sin(angle), float(world[2]))
        self._visual_object_position = tuple(float(v) for v in predicted)

    def note_link_latency(self, seconds: float) -> None:
        self._previous_link_latency = max(0.0, float(seconds))

    def _destination_position(self, destination: Optional[str]) -> Sequence[float]:
        table = self._destinations if self._destinations is not None else dict(self._env.target_positions)
        target_name = physical_target(destination)
        if target_name not in table:
            raise ValueError(f"Physical destination {target_name!r} is unavailable.")
        return table[target_name]

    def _step(self, joints: Sequence[float], result: ExecutionResult) -> Any:
        values = tuple(float(v) for v in joints)
        result.joint_commands.append([round(v, 5) for v in values])
        step_result = self._env.step(JointAction(values=values))
        result.steps += 1
        return step_result

    def start(self, target: ActionTarget) -> ExecutionSession:
        """Create a stopped session. No joint command is issued until ``advance``."""
        return ExecutionSession(target=target)

    def advance(self, session: ExecutionSession, target: ActionTarget | None = None) -> ExecutionSession:
        """Execute exactly one checkpointed phase, accepting a safe reroute target."""
        if target is not None:
            session.target = target
        env, result = self._env, session.result
        try:
            if session.phase == "approach":
                session.object_position = self._visual_object_position or tuple(float(v) for v in env.sample_position)
                joints = env.solve_ik(session.object_position, gripper=GRIPPER_OPEN)
                self._step(joints, result)
                session.phase = "grasp"
            elif session.phase == "grasp":
                if self._visual_object_position is not None:
                    session.object_position = self._visual_object_position
                    self._step(env.solve_ik(session.object_position, gripper=GRIPPER_OPEN), result)
                self._step(env.solve_ik(session.object_position, gripper=GRIPPER_CLOSE), result)
                session.phase = "carry"
            elif session.phase == "carry":
                destination = tuple(float(v) for v in self._destination_position(session.target.destination))
                if self._visual_object_position is not None:
                    above = (destination[0], destination[1], destination[2] + 0.14)
                    self._step(env.solve_ik(above, gripper=GRIPPER_CLOSE), result)
                self._step(env.solve_ik(destination, gripper=GRIPPER_CLOSE), result)
                session.phase = "release"
            elif session.phase == "release":
                destination = tuple(float(v) for v in self._destination_position(session.target.destination))
                if self._visual_object_position is not None:
                    self._step(env.solve_ik(destination, gripper=GRIPPER_CLOSE), result)
                released = self._step(env.solve_ik(destination, gripper=GRIPPER_OPEN), result)
                retreat = (destination[0], destination[1], destination[2] + 0.14)
                self._step(env.solve_ik(retreat, gripper=GRIPPER_OPEN), result)
                result.final_info = dict(getattr(released, "info", {}) or {})
                result.success = bool(result.final_info.get("success", False))
                result.failure_reason = None if result.success else result.final_info.get(
                    "failure_reason", "placement_failed"
                )
                session.phase = "complete"
        except Exception as exc:
            result.failure_reason = f"execution_error: {exc}"
            session.phase = "failed"
        return session

    def execute(self, target: ActionTarget) -> ExecutionResult:
        """Pick the target object and place it at the target destination.

        Sequence: approach (gripper open) -> grasp (close at object) -> carry
        (to destination) -> release (open). Success is read from the env's own
        final ``StepResult.info``.
        """
        result = ExecutionResult(success=False, steps=0)
        env = self._env
        dest_pos = tuple(float(v) for v in self._destination_position(target.destination))

        for attempt in range(1, self._max_attempts + 1):
            result.attempts = attempt
            # Re-observe the object before every bounded recovery attempt. A
            # failed local policy or first grasp can move it substantially.
            object_pos = tuple(float(v) for v in env.sample_position)
            try:
                self._step(env.solve_ik(object_pos, gripper=GRIPPER_OPEN), result)
                self._step(env.solve_ik(object_pos, gripper=GRIPPER_CLOSE), result)
                self._step(env.solve_ik(dest_pos, gripper=GRIPPER_CLOSE), result)
                # A planner may select the semantic destination but cannot keep
                # the physical gripper closed or issue joint commands.
                released = self._step(env.solve_ik(dest_pos, gripper=GRIPPER_OPEN), result)
                retreat = (dest_pos[0], dest_pos[1], dest_pos[2] + 0.14)
                self._step(env.solve_ik(retreat, gripper=GRIPPER_OPEN), result)
            except Exception as exc:  # keep a failed IK/step from crashing the trial
                result.failure_reason = f"execution_error: {exc}"
                continue

            info = dict(getattr(released, "info", {}) or {})
            result.final_info = info
            result.success = bool(info.get("success", False))
            result.failure_reason = (
                None
                if result.success
                else info.get("failure_reason") or "placement_failed"
            )
            if result.success:
                break
        return result
