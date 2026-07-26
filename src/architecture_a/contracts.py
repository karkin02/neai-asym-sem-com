from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class Observation:
    """Model-facing state supplied by a local environment adapter."""

    image: Any | None
    robot_state: tuple[float, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """Environment-neutral action with an optional robot command vector."""

    name: str
    values: tuple[float, ...] = ()
    arguments: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyDecision:
    """Action plus confidence proxies needed by the future hybrid policy."""

    action: Action
    can_handle_instruction: bool
    grounding_confidence: float
    policy_confidence: float
    reason: str | None = None

    @property
    def should_escalate(self) -> bool:
        return (
            not self.can_handle_instruction
            or self.grounding_confidence < 0.60
            or self.policy_confidence < 0.55
        )


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)


class Environment(Protocol):
    def reset(self, *, seed: int, instruction: str) -> Observation: ...

    def step(self, action: Action) -> StepResult: ...


class LocalPolicy(Protocol):
    def reset(self) -> None: ...

    def decide(self, observation: Observation, instruction: str) -> PolicyDecision: ...

