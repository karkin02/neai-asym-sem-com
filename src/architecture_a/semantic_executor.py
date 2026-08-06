"""Execute cloud-planned semantic destinations through Architecture A.

The cloud planner selects *what* destination to use. This adapter keeps all
joint generation and validation local: SmolVLA predicts a chunk, Architecture
A bounds it, MuJoCo previews its outcome, and only then is it executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from architecture_a.action_safety import (
    SO101_ACTION_BOUND_TOLERANCE,
    SO101_ACTION_HIGH,
    SO101_ACTION_LOW,
    bound_action_chunk,
)
from shared.destinations import normalize_destination
from shared.smolvla_runtime import SmolVlaRuntime, rollout_chunk


DESTINATION_INSTRUCTIONS = {
    "conveyor": "Pick up the package and place it on the conveyor.",
    "inspection_tray": "The package barcode is missing. Place it in the blue inspection tray.",
    "rejection_tray": "The package is damaged. Place it in the yellow rejection tray.",
}


@dataclass
class SemanticExecutionResult:
    success: bool
    steps: int
    attempts: int = 0
    final_info: dict[str, Any] = field(default_factory=dict)
    joint_commands: list[list[float]] = field(default_factory=list)
    failure_reason: str | None = None


@dataclass
class SemanticExecutionSession:
    """Resumable SmolVLA chunk execution state for visual replanning boundaries."""

    target: Any
    result: SemanticExecutionResult = field(default_factory=lambda: SemanticExecutionResult(False, 0))
    phase: str = "chunk"


class ArchitectureASmolVlaController:
    """Architecture A motion/safety adapter for a semantic planner target."""

    def __init__(
        self,
        env: Any,
        runtime: SmolVlaRuntime,
        *,
        max_attempts: int = 3,
        max_steps: int = 50,
        record_camera: str | None = None,
        initial_record_camera: str | None = None,
    ) -> None:
        self._env = env
        self._runtime = runtime
        self._max_attempts = max(1, int(max_attempts))
        self._max_steps = max(1, int(max_steps))
        self._seed = 0
        self._record_camera = record_camera
        self._initial_record_camera = initial_record_camera or record_camera
        self.frames: list[Any] = []

    def prepare(self, seed: int) -> None:
        self._seed = int(seed)
        self.frames = []
        if self._runtime.available():
            self._runtime.reset(self._seed)

    def start(self, target: Any) -> SemanticExecutionSession:
        """Start a stopped session; the first validated chunk runs on ``advance``."""
        return SemanticExecutionSession(target=target)

    def advance(
        self, session: SemanticExecutionSession, target: Any | None = None
    ) -> SemanticExecutionSession:
        """Run one bounded, previewed SmolVLA chunk between replanning checkpoints."""
        if target is not None:
            session.target = target
        original_attempts = self._max_attempts
        self._max_attempts = 1
        try:
            chunk_result = self.execute(session.target)
        finally:
            self._max_attempts = original_attempts
        session.result.attempts += 1
        session.result.steps += chunk_result.steps
        session.result.joint_commands.extend(chunk_result.joint_commands)
        session.result.final_info = chunk_result.final_info
        session.result.success = chunk_result.success
        session.result.failure_reason = chunk_result.failure_reason
        if chunk_result.success:
            session.phase = "complete"
        elif session.result.attempts >= original_attempts:
            session.phase = "failed"
        return session

    def execute(self, target: Any) -> SemanticExecutionResult:
        destination = normalize_destination(getattr(target, "destination", None))
        instruction = DESTINATION_INSTRUCTIONS.get(destination)
        if instruction is None:
            return SemanticExecutionResult(
                False, 0, failure_reason="unsupported_semantic_destination"
            )
        if not self._runtime.available():
            return SemanticExecutionResult(False, 0, failure_reason="local_runtime_unavailable")

        result = SemanticExecutionResult(False, 0)
        if self._initial_record_camera:
            self.frames.append(
                self._env.capture_rgb(
                    camera=self._initial_record_camera, width=480, height=360
                )
            )

        def record_frame() -> None:
            if self._record_camera:
                self.frames.append(
                    self._env.capture_rgb(camera=self._record_camera, width=480, height=360)
                )

        for attempt in range(1, self._max_attempts + 1):
            result.attempts = attempt
            overhead = self._env.capture_rgb(camera="overhead", width=320, height=240)
            wrist = self._env.capture_rgb(camera="wrist", width=320, height=240)
            state = tuple(float(value) for value in self._env.joint_positions)
            chunk = self._runtime.predict_chunk(overhead, wrist, state, instruction)
            bounded = bound_action_chunk(
                chunk,
                action_low=SO101_ACTION_LOW,
                action_high=SO101_ACTION_HIGH,
                tolerance=SO101_ACTION_BOUND_TOLERANCE,
            )
            result.final_info = {
                "action_bound_clipped_rows": bounded.clipped_rows,
                "maximum_action_bound_overshoot": bounded.maximum_overshoot,
                "rejected_action_violation": bounded.violation,
                "semantic_destination": destination,
            }
            if not bounded.accepted:
                result.failure_reason = "action_chunk_out_of_bounds"
                continue

            preview = getattr(self._env, "preview_action_success", None)
            if not callable(preview) or not preview(bounded.values[: self._max_steps]):
                result.failure_reason = "preexecution_outcome_validation_failed"
                continue

            success, steps, info, _ = rollout_chunk(
                self._env,
                bounded.values,
                max_steps=self._max_steps,
                on_step=record_frame if self._record_camera else None,
            )
            result.steps += steps
            result.final_info.update(info)
            result.success = success
            result.failure_reason = None if success else info.get("failure_reason", "placement_failed")
            if success:
                break
        return result
