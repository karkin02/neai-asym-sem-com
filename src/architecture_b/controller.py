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
                last = self._step(env.solve_ik(dest_pos, gripper=GRIPPER_OPEN), result)
            except Exception as exc:  # keep a failed IK/step from crashing the trial
                result.failure_reason = f"execution_error: {exc}"
                continue

            info = dict(getattr(last, "info", {}) or {})
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
