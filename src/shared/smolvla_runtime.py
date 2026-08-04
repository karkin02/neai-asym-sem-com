"""Importable wrapper around Architecture A's production SmolVLA inference.

Architecture A's real (two-camera, open-loop 50-action-chunk) inference lives
inline in ``scripts/evaluate_smolvla.py:main()`` and is not importable, so
Architecture C cannot call it directly. This module replicates that exact load
+ predict recipe (same `lerobot` classes, same batch keys, same
``predict_action_chunk`` call) in an importable, injectable form — **without
modifying any Architecture A file**.

``lerobot``/``torch`` are imported lazily inside the predictor, and the
checkpoint is checked with :meth:`SmolVlaRuntime.available`, so Architecture C
degrades gracefully to escalation when the ~0.72 GB checkpoint or the lerobot
stack is absent (mirroring A's "escalation when local capability is
unavailable" behaviour).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

# Mirror scripts/evaluate_smolvla.py defaults.
DEFAULT_CHECKPOINT = Path("outputs/train/smolvla_pickplace_50/checkpoints/last/pretrained_model")
DEFAULT_VLM = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"


class ChunkPredictor(Protocol):
    def predict_chunk(
        self, overhead: np.ndarray, wrist: np.ndarray, robot_state, instruction: str
    ) -> np.ndarray: ...


class SmolVlaRuntime:
    """Loads SmolVLA once and predicts an action chunk from two camera views.

    Args:
        checkpoint: Path to the ``pretrained_model`` directory.
        vlm: Path to the local VLM cache (``config.vlm_model_name``).
        device: Torch device string.
        predictor: Optional injected :class:`ChunkPredictor` (a fake in tests);
            bypasses the lazy lerobot load entirely.
    """

    def __init__(
        self,
        checkpoint: Path = DEFAULT_CHECKPOINT,
        vlm: str | Path = DEFAULT_VLM,
        device: str = "cpu",
        predictor: Optional[ChunkPredictor] = None,
    ) -> None:
        self.checkpoint = Path(checkpoint)
        # Keep Hub model IDs as POSIX-style strings on Windows. Converting a
        # model ID to Path would produce backslashes and make Transformers
        # reject it as an invalid repository name.
        self.vlm = str(vlm)
        self.device = device
        self._predictor = predictor

    def available(self) -> bool:
        """True if local inference can run (injected predictor, or checkpoint on disk)."""
        return self._predictor is not None or self.checkpoint.exists()

    def _ensure_predictor(self) -> ChunkPredictor:
        if self._predictor is None:
            self._predictor = _LeRobotSmolVlaPredictor(self.checkpoint, self.vlm, self.device)
        return self._predictor

    def predict_chunk(self, overhead, wrist, robot_state, instruction: str) -> np.ndarray:
        """Return an ``(N, 6)`` absolute-joint-position chunk for the observation."""
        chunk = self._ensure_predictor().predict_chunk(overhead, wrist, robot_state, instruction)
        return np.asarray(chunk, dtype=float)

    def reset(self, seed: int) -> None:
        """Reset stochastic policy state for a reproducible episode."""
        predictor = self._ensure_predictor()
        reset = getattr(predictor, "reset", None)
        if callable(reset):
            reset(seed)


def rollout_chunk(
    env: Any, chunk: np.ndarray, max_steps: int = 50, on_step: Any | None = None
) -> tuple[bool, int, dict, Any | None]:
    """Execute a chunk open-loop through A's env (reuses ``env.step``).

    One chunk row per step, stopping on success/termination. Returns
    ``(success, steps, final_info, final_observation)``. This only changes
    *where* the action came from (SmolVLA); physical execution is entirely A's
    ``env.step``. Returning the observation permits bounded closed-loop
    replanning without reaching into an environment's private state.
    """
    from architecture_b.controller import JointAction  # duck-typed Action(name, values)

    success = False
    steps = 0
    info: dict = {}
    observation = None
    for row in np.asarray(chunk)[:max_steps]:
        result = env.step(JointAction(values=tuple(float(v) for v in row)))
        if on_step is not None:
            on_step()
        steps += 1
        observation = getattr(result, "observation", observation)
        info = dict(getattr(result, "info", {}) or {})
        if info.get("success"):
            success = True
            break
        if getattr(result, "terminated", False) or getattr(result, "truncated", False):
            break
    return success, steps, info, observation


def _image_tensor(image: np.ndarray, torch: Any):
    """Replicate ``scripts/evaluate_smolvla.py:image_tensor`` (HWC uint8 -> CHW float [0,1])."""
    return (
        torch.from_numpy(np.asarray(image).copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


class _LeRobotSmolVlaPredictor:
    """Default predictor: replicates A's evaluate_smolvla load + predict_action_chunk."""

    def __init__(self, checkpoint: Path, vlm: str, device: str) -> None:
        import torch  # lazy
        import truststore  # lazy

        truststore.inject_into_ssl()
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        config = PreTrainedConfig.from_pretrained(str(checkpoint))
        config.device = device
        config.vlm_model_name = str(vlm)
        policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=config)
        policy.to(device)
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={
                "device_processor": {"device": device},
                "tokenizer_processor": {"tokenizer_name": str(vlm)},
            },
        )
        self._torch = torch
        self.device = device
        self._policy = policy
        self._pre = preprocessor
        self._post = postprocessor

    def reset(self, seed: int) -> None:
        np.random.seed(seed)
        self._torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            self._torch.cuda.manual_seed_all(seed)
        self._policy.reset()

    def predict_chunk(self, overhead, wrist, robot_state, instruction: str) -> np.ndarray:
        torch = self._torch
        batch = self._pre(
            {
                "observation.images.overhead": _image_tensor(overhead, torch),
                "observation.images.wrist": _image_tensor(wrist, torch),
                "observation.state": torch.tensor(robot_state, dtype=torch.float32),
                "task": instruction,
            }
        )
        chunk = self._post(self._policy.predict_action_chunk(batch))
        return chunk.squeeze(0).cpu().numpy()
