"""
2-D PushT simulation environment (Gymnasium-compatible).

The agent controls a circular pusher that must slide a T-shaped block on
a 2-D surface until it aligns with a target outline.  Observations are
returned in the LeRobot-standard dictionary format.

Classes:
    PushTSimEnv: Gymnasium environment for the PushT task.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot_sim.envs.configs import PushTSimConfig
from lerobot_sim.utils.constants import (
    COLOR_BACKGROUND,
    COLOR_OBJECT,
    COLOR_ROBOT,
    COLOR_SUCCESS,
    COLOR_TARGET,
)


class PushTSimEnv(gym.Env):
    """Gymnasium environment for the 2-D PushT task.

    The agent emits (dx, dy) actions to move a pusher disc.  A T-shaped
    block is pushed until it overlaps with a target outline.  The
    environment produces pixel observations and/or agent-position states
    in the LeRobot dictionary format.

    Attributes:
        metadata: Gymnasium metadata with supported render modes.
        cfg: ``PushTSimConfig`` controlling episode length, resolution, etc.
    """

    metadata: Dict[str, Any] = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: PushTSimConfig | None = None) -> None:
        """Initialise the PushT environment.

        Args:
            cfg: Optional configuration; a default ``PushTSimConfig`` is used
                when *None*.
        """
        super().__init__()
        self.cfg = cfg or PushTSimConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._step_count = 0
        self._init_spaces()
        self._init_entities()

    # ------------------------------------------------------------------
    # Initialisation helpers (called by __init__)
    # ------------------------------------------------------------------

    def _init_spaces(self) -> None:
        """Define action and observation Gymnasium spaces."""
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        obs_dict: Dict[str, spaces.Space] = {}
        obs_dict["agent_pos"] = spaces.Box(low=0.0, high=1.0, shape=(self.cfg.state_dim,), dtype=np.float32)
        if "pixels" in self.cfg.obs_type:
            h, w = self.cfg.observation_height, self.cfg.observation_width
            obs_dict["pixels"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_dict)

    def _init_entities(self) -> None:
        """Create initial positions for the pusher, T-block, and target."""
        self._agent_pos = np.array([0.5, 0.5], dtype=np.float32)
        self._block_pos = np.array([0.3, 0.3], dtype=np.float32)
        self._block_angle: float = 0.0
        self._target_pos = np.array([0.7, 0.7], dtype=np.float32)
        self._target_angle: float = 0.0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _sample_block_position(self) -> None:
        """Randomise the T-block starting position and angle."""
        self._block_pos = self._rng.uniform(0.2, 0.8, size=2).astype(np.float32)
        self._block_angle = float(self._rng.uniform(0.0, 2.0 * np.pi))

    def _sample_target_position(self) -> None:
        """Randomise the target outline position and angle."""
        self._target_pos = self._rng.uniform(0.3, 0.7, size=2).astype(np.float32)
        self._target_angle = float(self._rng.uniform(0.0, 2.0 * np.pi))

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional seed for the random number generator.
            options: If it contains ``"object_world_pos"``, the block (the object
                to push) is placed there (e.g. reconstructed from a real camera)
                instead of being randomly sampled. The push target is always
                sampled independently.

        Returns:
            Tuple of (observation dict, info dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._agent_pos = np.array([0.5, 0.5], dtype=np.float32)
        obj = (options or {}).get("object_world_pos")
        if obj is not None:
            self._block_pos = np.clip(
                np.asarray(obj, dtype=np.float32)[:2], 0.0, 1.0
            ).astype(np.float32)
            self._block_angle = float(self._rng.uniform(0.0, 2.0 * np.pi))
        else:
            self._sample_block_position()
        self._sample_target_position()
        return self._build_observation(), {}

    def _move_agent(self, action: np.ndarray) -> None:
        """Apply the (dx, dy) action to the agent and clamp to [0, 1].

        Args:
            action: 2-D displacement vector.
        """
        raw = self._agent_pos + action * 0.05
        self._agent_pos = np.clip(raw, 0.0, 1.0).astype(np.float32)

    def _push_block(self) -> None:
        """If the agent overlaps the block, push the block away."""
        diff = self._block_pos - self._agent_pos
        dist = float(np.linalg.norm(diff))
        if dist < 0.06:
            direction = diff / max(dist, 1e-8)
            self._block_pos += (direction * 0.02).astype(np.float32)
            self._block_pos = np.clip(self._block_pos, 0.0, 1.0).astype(np.float32)

    def _compute_reward(self) -> Tuple[float, bool]:
        """Compute the reward and success flag.

        Returns:
            Tuple of (scalar reward, done boolean).
        """
        pos_dist = float(np.linalg.norm(self._block_pos - self._target_pos))
        reward = 1.0 - pos_dist
        success = pos_dist < 0.08
        return reward, success

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action: 2-D array of (dx, dy) displacements in [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float32)
        self._move_agent(action)
        self._push_block()
        self._step_count += 1
        reward, success = self._compute_reward()
        truncated = self._step_count >= self.cfg.episode_length
        return (
            self._build_observation(),
            reward,
            success,
            truncated,
            {"is_success": success},
        )

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Assemble the observation dictionary.

        Returns:
            Dictionary with ``'agent_pos'`` (6-D) and optionally ``'pixels'``.
        """
        state = np.concatenate(
            [
                self._agent_pos,
                self._block_pos,
                self._target_pos,
            ]
        ).astype(np.float32)
        obs: Dict[str, np.ndarray] = {"agent_pos": state}
        if "pixels" in self.cfg.obs_type:
            obs["pixels"] = self.render()
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_background(self, canvas: np.ndarray) -> None:
        """Fill *canvas* with the background colour.

        Args:
            canvas: Mutable (H, W, 3) uint8 array.
        """
        canvas[:] = COLOR_BACKGROUND

    def _draw_target(self, canvas: np.ndarray) -> None:
        """Draw the target T-outline on *canvas*.

        Args:
            canvas: Mutable (H, W, 3) uint8 array.
        """
        h, w = canvas.shape[:2]
        cx, cy = int(self._target_pos[0] * w), int(self._target_pos[1] * h)
        half = int(0.04 * w)
        canvas[max(cy - half, 0) : cy + half, max(cx - half * 2, 0) : cx + half * 2] = COLOR_TARGET

    def _draw_block(self, canvas: np.ndarray) -> None:
        """Draw the T-shaped block on *canvas*.

        Args:
            canvas: Mutable (H, W, 3) uint8 array.
        """
        h, w = canvas.shape[:2]
        cx, cy = int(self._block_pos[0] * w), int(self._block_pos[1] * h)
        half = int(0.03 * w)
        canvas[max(cy - half, 0) : cy + half, max(cx - half * 2, 0) : cx + half * 2] = COLOR_OBJECT

    def _draw_agent(self, canvas: np.ndarray) -> None:
        """Draw the circular pusher on *canvas*.

        Args:
            canvas: Mutable (H, W, 3) uint8 array.
        """
        h, w = canvas.shape[:2]
        cx, cy = int(self._agent_pos[0] * w), int(self._agent_pos[1] * h)
        rr, cc = np.ogrid[:h, :w]
        mask = (rr - cy) ** 2 + (cc - cx) ** 2 < (0.02 * w) ** 2
        canvas[mask] = COLOR_ROBOT

    def render(self) -> np.ndarray:
        """Render the current scene as an RGB image.

        Returns:
            (H, W, 3) uint8 NumPy array.
        """
        h, w = self.cfg.observation_height, self.cfg.observation_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self._draw_background(canvas)
        self._draw_target(canvas)
        self._draw_block(canvas)
        self._draw_agent(canvas)
        return canvas
