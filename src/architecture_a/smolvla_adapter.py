from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from .contracts import Action, Observation, PolicyDecision


@dataclass(frozen=True)
class ModelOutput:
    action: Sequence[float]
    confidence: float | None = None


class SmolVLABackend(Protocol):
    def reset(self) -> None: ...

    def predict(self, batch: dict[str, Any]) -> ModelOutput: ...


@dataclass(frozen=True)
class SmolVLAFeatures:
    image_key: str = "observation.image"
    state_key: str = "observation.state"
    task_key: str = "task"
    action_dimension: int = 6


class SmolVLAPolicyAdapter:
    """Translate Architecture A observations into the LeRobot policy contract."""

    def __init__(
        self,
        backend: SmolVLABackend,
        *,
        features: SmolVLAFeatures = SmolVLAFeatures(),
        device: str = "cpu",
    ) -> None:
        self._backend = backend
        self._features = features
        self._device = device

    def reset(self) -> None:
        self._backend.reset()

    def decide(self, observation: Observation, instruction: str) -> PolicyDecision:
        if observation.image is None:
            return PolicyDecision(
                action=Action("noop"),
                can_handle_instruction=False,
                grounding_confidence=0.0,
                policy_confidence=0.0,
                reason="missing_camera_observation",
            )

        batch = self._build_batch(observation, instruction)
        output = self._backend.predict(batch)
        values = tuple(float(value) for value in output.action)
        if len(values) != self._features.action_dimension:
            raise ValueError(
                "SmolVLA action dimension mismatch: "
                f"expected {self._features.action_dimension}, got {len(values)}. "
                "Use a checkpoint trained for this robot embodiment."
            )

        confidence = (
            float(output.confidence) if output.confidence is not None else 0.65
        )
        return PolicyDecision(
            action=Action("joint_position", values=values),
            can_handle_instruction=True,
            grounding_confidence=1.0,
            policy_confidence=max(0.0, min(confidence, 1.0)),
            reason=None,
        )

    def _build_batch(
        self,
        observation: Observation,
        instruction: str,
    ) -> dict[str, Any]:
        image = torch.as_tensor(observation.image, device=self._device)
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError("Expected an HWC RGB or RGBA camera observation.")
        image = image[..., :3].permute(2, 0, 1).contiguous().float() / 255.0
        state = torch.tensor(
            observation.robot_state,
            dtype=torch.float32,
            device=self._device,
        )
        return {
            self._features.image_key: image.unsqueeze(0),
            self._features.state_key: state.unsqueeze(0),
            self._features.task_key: [instruction],
        }


class LeRobotSmolVLABackend:
    """Lazy wrapper around LeRobot so the core project stays importable."""

    def __init__(
        self,
        checkpoint: str,
        *,
        device: str = "cpu",
    ) -> None:
        try:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        except ImportError as error:
            raise RuntimeError(
                "LeRobot with SmolVLA support is not installed. Install the "
                "matching LeRobot release before loading a checkpoint."
            ) from error

        self._policy = SmolVLAPolicy.from_pretrained(checkpoint)
        self._policy.to(device)
        self._policy.eval()

    def reset(self) -> None:
        reset = getattr(self._policy, "reset", None)
        if reset is not None:
            reset()

    @torch.inference_mode()
    def predict(self, batch: dict[str, Any]) -> ModelOutput:
        action = self._policy.select_action(batch)
        if hasattr(action, "detach"):
            action = action.detach().cpu()
        if hasattr(action, "squeeze"):
            action = action.squeeze(0)
        if hasattr(action, "tolist"):
            action = action.tolist()
        return ModelOutput(action=action)

