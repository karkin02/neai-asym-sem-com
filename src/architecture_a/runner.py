from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import Environment, LocalPolicy


@dataclass(frozen=True)
class EpisodeResult:
    episode_id: int
    seed: int
    instruction: str
    success: bool
    steps: int
    latency_ms: float
    escalation_recommended: bool
    minimum_grounding_confidence: float
    minimum_policy_confidence: float
    failure_reason: str | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ArchitectureA:
    def __init__(
        self,
        environment: Environment,
        policy: LocalPolicy,
        *,
        max_steps: int = 20,
    ) -> None:
        self._environment = environment
        self._policy = policy
        self._max_steps = max_steps

    def run_episode(
        self,
        *,
        episode_id: int,
        seed: int,
        instruction: str,
    ) -> EpisodeResult:
        observation = self._environment.reset(seed=seed, instruction=instruction)
        self._policy.reset()
        start = time.perf_counter()
        grounding_scores: list[float] = []
        policy_scores: list[float] = []
        escalation_recommended = False
        success = False
        failure_reason: str | None = None
        steps = 0

        for steps in range(1, self._max_steps + 1):
            decision = self._policy.decide(observation, instruction)
            grounding_scores.append(decision.grounding_confidence)
            policy_scores.append(decision.policy_confidence)
            escalation_recommended |= decision.should_escalate

            if not decision.can_handle_instruction:
                failure_reason = decision.reason or "unsupported_instruction"
                break

            transition = self._environment.step(decision.action)
            observation = transition.observation
            success = bool(transition.info.get("success", False))
            failure_reason = transition.info.get("failure_reason")
            if transition.terminated or transition.truncated:
                break
        else:
            failure_reason = "maximum_steps_exceeded"

        latency_ms = (time.perf_counter() - start) * 1000.0
        return EpisodeResult(
            episode_id=episode_id,
            seed=seed,
            instruction=instruction,
            success=success,
            steps=steps,
            latency_ms=round(latency_ms, 3),
            escalation_recommended=escalation_recommended,
            minimum_grounding_confidence=min(grounding_scores, default=0.0),
            minimum_policy_confidence=min(policy_scores, default=0.0),
            failure_reason=None if success else failure_reason,
        )

