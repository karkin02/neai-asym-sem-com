from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .so101_adapter import SO101ActionAdapter
from .so101_observation import (
    SO101ObservationAdapter,
    SO101ObservationPacket,
)


class SO101JointPolicy(Protocol):
    def predict(self, observation: SO101ObservationPacket) -> Sequence[float]: ...


@dataclass(frozen=True)
class ControlCycleResult:
    command: dict[str, float] | None
    latency_ms: float
    escalation_recommended: bool
    failure_reason: str | None


class SO101DryRunController:
    """Run one validated observation-policy-action cycle."""

    def __init__(
        self,
        observation_adapter: SO101ObservationAdapter,
        policy: SO101JointPolicy,
        action_adapter: SO101ActionAdapter,
        *,
        maximum_policy_latency_ms: float = 1_000.0,
    ) -> None:
        if maximum_policy_latency_ms <= 0.0:
            raise ValueError("Maximum policy latency must be positive.")
        self._observation_adapter = observation_adapter
        self._policy = policy
        self._action_adapter = action_adapter
        self._maximum_policy_latency_ms = maximum_policy_latency_ms

    def cycle(self, *, task: str) -> ControlCycleResult:
        try:
            observation = self._observation_adapter.read(task=task)
        except (ValueError, RuntimeError) as error:
            return ControlCycleResult(
                command=None,
                latency_ms=0.0,
                escalation_recommended=True,
                failure_reason=f"observation_rejected:{error}",
            )

        start = time.perf_counter()
        try:
            values = self._policy.predict(observation)
        except Exception as error:
            return ControlCycleResult(
                command=None,
                latency_ms=round((time.perf_counter() - start) * 1_000.0, 3),
                escalation_recommended=True,
                failure_reason=f"policy_failed:{error}",
            )
        latency_ms = (time.perf_counter() - start) * 1_000.0
        if latency_ms > self._maximum_policy_latency_ms:
            return ControlCycleResult(
                command=None,
                latency_ms=round(latency_ms, 3),
                escalation_recommended=True,
                failure_reason="policy_timeout",
            )

        try:
            command = self._action_adapter.command(values)
        except (ValueError, RuntimeError) as error:
            return ControlCycleResult(
                command=None,
                latency_ms=round(latency_ms, 3),
                escalation_recommended=True,
                failure_reason=f"action_rejected:{error}",
            )
        return ControlCycleResult(
            command=command,
            latency_ms=round(latency_ms, 3),
            escalation_recommended=False,
            failure_reason=None,
        )
