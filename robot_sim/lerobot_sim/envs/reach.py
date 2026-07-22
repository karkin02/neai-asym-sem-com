"""
3-D reaching simulation environment (Gymnasium-compatible).

A 6-DOF robot arm must move its end-effector to a randomly sampled target
sphere.  This is the simplest environment in the suite and is ideal for
verifying that policies can learn basic continuous control.

Classes:
    ReachSimEnv: Gymnasium environment for the reaching task.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot_sim.envs.configs import ReachSimConfig
from lerobot_sim.robots.sim_robot_arm import SimRobotArm
from lerobot_sim.utils.constants import (
    COLOR_BACKGROUND,
    COLOR_ROBOT,
    COLOR_SUCCESS,
    COLOR_TARGET,
)


class ReachSimEnv(gym.Env):
    """Gymnasium environment for a 3-D reaching task.

    The agent controls a 6-DOF arm through joint-velocity commands.  The
    goal is to position the end-effector within a small sphere centred on
    a randomly sampled target.  Observations include joint positions,
    end-effector position, and optional pixel renderings.

    Attributes:
        metadata: Gymnasium metadata with supported render modes.
        cfg: ``ReachSimConfig`` controlling episode length, resolution, etc.
    """

    metadata: Dict[str, Any] = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: ReachSimConfig | None = None) -> None:
        """Initialise the Reach environment.

        Args:
            cfg: Optional configuration; a default ``ReachSimConfig`` is used
                when *None*.
        """
        super().__init__()
        self.cfg = cfg or ReachSimConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._arm = SimRobotArm()
        self._step_count = 0
        self._init_spaces()
        self._target_pos = np.zeros(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _init_spaces(self) -> None:
        """Define action and observation Gymnasium spaces."""
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.cfg.action_dim,), dtype=np.float32)
        obs_dict: Dict[str, spaces.Space] = {}
        obs_dict["agent_pos"] = spaces.Box(low=-5.0, high=5.0, shape=(self.cfg.state_dim,), dtype=np.float32)
        if "pixels" in self.cfg.obs_type:
            h, w = self.cfg.observation_height, self.cfg.observation_width
            obs_dict["pixels"] = spaces.Box(low=0, high=255, shape=(h, w, 3), dtype=np.uint8)
        self.observation_space = spaces.Dict(obs_dict)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _sample_target(self) -> None:
        """Sample a random target position in the reachable workspace."""
        self._target_pos = self._rng.uniform([-0.3, -0.3, 0.1], [0.3, 0.3, 0.5]).astype(np.float64)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional RNG seed.
            options: If it contains ``"object_world_pos"``, the target is placed
                there (e.g. reconstructed from a real camera) instead of being
                randomly sampled.

        Returns:
            Tuple of (observation dict, info dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._arm.reset()
        obj = (options or {}).get("object_world_pos")
        if obj is not None:
            self._target_pos = np.clip(
                np.asarray(obj, dtype=np.float64), [-0.3, -0.3, 0.1], [0.3, 0.3, 0.5]
            )
        else:
            self._sample_target()
        return self._build_observation(), {}

    def _compute_reward(self) -> Tuple[float, bool]:
        """Compute the shaped reward and success flag.

        Returns:
            Tuple of (scalar reward, success boolean).
        """
        # The renderer is 2-D (x, y only) and this arm's z-kinematics cannot
        # reach the object plane, so success is measured in the x-y plane —
        # i.e. "the dots overlap on screen" is the success condition.
        dist = float(np.linalg.norm((self._arm.ee_position - self._target_pos)[:2]))
        reward = -dist
        success = dist < 0.03
        return reward, success

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action: 6-D array of joint-velocity commands in [-1, 1].

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float64)
        self._arm.step(action)
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

    def _build_state_vector(self) -> np.ndarray:
        """Concatenate joint positions, ee position, and target position.

        Returns:
            1-D float32 array of length ``state_dim``.
        """
        joints = self._arm.joint_positions
        ee = self._arm.ee_position
        return np.concatenate([joints, ee])[: self.cfg.state_dim].astype(np.float32)

    def _build_observation(self) -> Dict[str, np.ndarray]:
        """Assemble the observation dictionary.

        Returns:
            Dictionary with ``'agent_pos'`` and optionally ``'pixels'``.
        """
        obs: Dict[str, np.ndarray] = {"agent_pos": self._build_state_vector()}
        if "pixels" in self.cfg.obs_type:
            obs["pixels"] = self.render()
        return obs

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _world_to_pixel(self, pos: np.ndarray, h: int, w: int) -> Tuple[int, int]:
        """Map a 3-D world position to 2-D pixel coordinates.

        Args:
            pos: World-space [x, y, z].
            h: Canvas height.
            w: Canvas width.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        px = int((pos[0] + 0.5) / 1.0 * w)
        py = int((1.0 - (pos[1] + 0.5) / 1.0) * h)
        return np.clip(px, 0, w - 1), np.clip(py, 0, h - 1)

    def _draw_background(self, canvas: np.ndarray) -> None:
        """Fill the canvas with the background colour.

        Args:
            canvas: Mutable (H, W, 3) uint8 array.
        """
        canvas[:] = COLOR_BACKGROUND

    def _draw_circle(
        self,
        canvas: np.ndarray,
        pos: np.ndarray,
        colour: Tuple[int, int, int],
        radius_frac: float,
    ) -> None:
        """Draw a filled circle on the canvas at a world position.

        Args:
            canvas: Mutable (H, W, 3) uint8 array.
            pos: 3-D world position.
            colour: RGB colour tuple.
            radius_frac: Circle radius as a fraction of canvas width.
        """
        h, w = canvas.shape[:2]
        cx, cy = self._world_to_pixel(pos, h, w)
        rr, cc = np.ogrid[:h, :w]
        mask = (rr - cy) ** 2 + (cc - cx) ** 2 < (radius_frac * w) ** 2
        canvas[mask] = colour

    def render(self) -> np.ndarray:
        """Render the current scene as an RGB image.

        Returns:
            (H, W, 3) uint8 NumPy array.
        """
        h, w = self.cfg.observation_height, self.cfg.observation_width
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self._draw_background(canvas)
        self._draw_circle(canvas, self._target_pos, COLOR_TARGET, 0.03)
        self._draw_circle(canvas, self._arm.ee_position, COLOR_ROBOT, 0.02)
        return canvas
