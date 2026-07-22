"""
3-D pick-and-place simulation environment (Gymnasium-compatible).

A 6-DOF robot arm must grasp a cube from a randomised start position and
place it on a target pad.  The environment wraps ``SimRobotArm`` for
kinematics and produces LeRobot-standard observation dictionaries.

Classes:
    PickPlaceSimEnv: Gymnasium environment for the pick-and-place task.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from lerobot_sim.envs.configs import PickPlaceSimConfig
from lerobot_sim.robots.sim_robot_arm import SimRobotArm
from lerobot_sim.utils.constants import (
    COLOR_BACKGROUND,
    COLOR_OBJECT,
    COLOR_ROBOT,
    COLOR_SUCCESS,
    COLOR_TARGET,
)


class PickPlaceSimEnv(gym.Env):
    """Gymnasium environment for a 3-D pick-and-place task.

    The agent controls a 6-DOF arm through joint-velocity commands plus a
    gripper toggle.  A cube must be picked up and placed at a target
    location.  Observations include the arm state vector and optional pixel
    renderings.

    Attributes:
        metadata: Gymnasium metadata with supported render modes.
        cfg: ``PickPlaceSimConfig`` controlling episode length, resolution, etc.
    """

    metadata: Dict[str, Any] = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, cfg: PickPlaceSimConfig | None = None) -> None:
        """Initialise the PickPlace environment.

        Args:
            cfg: Optional configuration; a default ``PickPlaceSimConfig`` is
                used when *None*.
        """
        super().__init__()
        self.cfg = cfg or PickPlaceSimConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._arm = SimRobotArm()
        self._step_count = 0
        self._init_spaces()
        self._init_entities()

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

    def _init_entities(self) -> None:
        """Create initial cube and target positions."""
        self._cube_pos = np.array([0.2, 0.0, 0.02], dtype=np.float64)
        self._target_pos = np.array([0.3, 0.2, 0.02], dtype=np.float64)
        self._cube_grasped = False

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def _randomise_cube(self) -> None:
        """Sample a random cube starting position in the reachable workspace."""
        self._cube_pos = self._rng.uniform([-0.3, -0.3, 0.02], [0.3, 0.3, 0.02]).astype(np.float64)

    def _randomise_target(self) -> None:
        """Sample a random target pad position in the reachable workspace."""
        self._target_pos = self._rng.uniform([-0.3, -0.3, 0.02], [0.3, 0.3, 0.02]).astype(np.float64)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional RNG seed.
            options: If it contains ``"object_world_pos"``, the cube (the object
                to pick) is placed there (e.g. reconstructed from a real camera)
                instead of being randomly sampled. The place target is always
                randomised independently.

        Returns:
            Tuple of (observation dict, info dict).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._cube_grasped = False
        self._arm.reset()
        obj = (options or {}).get("object_world_pos")
        if obj is not None:
            p = np.asarray(obj, dtype=np.float64)
            self._cube_pos = np.array(
                [np.clip(p[0], -0.3, 0.3), np.clip(p[1], -0.3, 0.3), 0.02],
                dtype=np.float64,
            )
        else:
            self._randomise_cube()
        self._randomise_target()
        return self._build_observation(), {}

    def _update_grasp(self) -> None:
        """Attach or detach the cube based on gripper state and proximity."""
        # x-y plane proximity: the renderer is 2-D and the arm cannot reach the
        # object's z, so grasping triggers on planar distance (gripper closed).
        dist = float(np.linalg.norm((self._arm.ee_position - self._cube_pos)[:2]))
        if not self._arm.gripper_open and dist < 0.06:
            self._cube_grasped = True
        if self._arm.gripper_open:
            self._cube_grasped = False

    def _move_cube_with_gripper(self) -> None:
        """If the cube is grasped, snap it to the end-effector position."""
        if self._cube_grasped:
            self._cube_pos = self._arm.ee_position.copy()

    def _compute_reward(self) -> Tuple[float, bool]:
        """Compute shaped reward and success flag.

        Returns:
            Tuple of (scalar reward, success boolean).
        """
        # Measured in the x-y plane (see _update_grasp for why).
        cube_target_dist = float(np.linalg.norm((self._cube_pos - self._target_pos)[:2]))
        ee_cube_dist = float(np.linalg.norm((self._arm.ee_position - self._cube_pos)[:2]))
        reward = -ee_cube_dist - cube_target_dist
        success = cube_target_dist < 0.05
        return reward, success

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Advance the environment by one step.

        Args:
            action: 7-D array of joint velocities (6) + gripper command (1).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        action = np.asarray(action, dtype=np.float64)
        self._arm.step(action)
        self._update_grasp()
        self._move_cube_with_gripper()
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
        """Concatenate arm state, cube position, and target position.

        Returns:
            1-D float32 array of length ``state_dim``.
        """
        arm_state = self._arm.get_state()
        return np.concatenate([arm_state[: self.cfg.state_dim - 0]]).astype(np.float32)[: self.cfg.state_dim]

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
        self._draw_circle(canvas, self._target_pos, COLOR_TARGET, 0.04)
        self._draw_circle(canvas, self._cube_pos, COLOR_OBJECT, 0.03)
        self._draw_circle(canvas, self._arm.ee_position, COLOR_ROBOT, 0.02)
        return canvas
