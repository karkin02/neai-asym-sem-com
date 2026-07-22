"""
Dataclass configurations for every simulation environment.

Each config mirrors the pattern used by upstream LeRobot ``EnvConfig``
subclasses so that policies/tools built for LeRobot can consume these
configurations without modification.

Classes:
    SimEnvConfig: Abstract base configuration shared by all sim envs.
    PushTSimConfig: Configuration for the 2-D PushT task.
    PickPlaceSimConfig: Configuration for the 3-D pick-and-place task.
    ReachSimConfig: Configuration for the 3-D reaching task.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, Tuple

from lerobot_sim.utils.constants import (
    ACTION,
    DEFAULT_FPS,
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    FeatureType,
    OBS_IMAGE,
    OBS_STATE,
    PolicyFeature,
)


@dataclass
class SimEnvConfig(abc.ABC):
    """Base configuration shared by all lerobot_sim environments.

    Attributes:
        task: Human-readable task identifier.
        fps: Simulation frames per second.
        episode_length: Maximum steps per episode.
        obs_type: Observation mode (``'pixels_agent_pos'``, ``'pixels'``).
        render_mode: Gymnasium render mode (``'rgb_array'``, ``'human'``).
        observation_height: Pixel height of rendered observations.
        observation_width: Pixel width of rendered observations.
        seed: Random seed for reproducibility.
        features: Mapping of feature key to ``PolicyFeature`` metadata.
        features_map: Mapping of raw env keys to LeRobot-standard keys.
    """

    task: str = "base"
    fps: int = DEFAULT_FPS
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    observation_height: int = DEFAULT_RENDER_HEIGHT
    observation_width: int = DEFAULT_RENDER_WIDTH
    seed: int = 42
    features: Dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: Dict[str, str] = field(default_factory=dict)

    @property
    def env_type(self) -> str:
        """Return a short string identifying the environment type.

        Returns:
            The ``task`` field value.
        """
        return self.task

    @property
    @abc.abstractmethod
    def gym_kwargs(self) -> dict:
        """Return keyword arguments forwarded to ``gymnasium.make()``.

        Returns:
            A dictionary of environment-specific kwargs.
        """
        raise NotImplementedError


@dataclass
class PushTSimConfig(SimEnvConfig):
    """Configuration for the 2-D PushT simulation environment.

    The agent controls a circular pusher that must slide a T-shaped block
    onto a target outline.  Observations include the agent (x, y) position
    and an optional pixel rendering.

    Attributes:
        task: Fixed to ``'PushT-Sim-v0'``.
        fps: Runs at 10 Hz by default.
        episode_length: 300 steps per episode.
        action_dim: Dimensionality of the action vector (dx, dy).
        state_dim: Dimensionality of the agent state (x, y).
    """

    task: str = "PushT-Sim-v0"
    fps: int = 10
    episode_length: int = 300
    action_dim: int = 2
    state_dim: int = 6

    def __post_init__(self) -> None:
        """Populate ``features`` and ``features_map`` based on ``obs_type``."""
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features_map[ACTION] = ACTION
        self.features_map["agent_pos"] = OBS_STATE
        self.features_map["pixels"] = OBS_IMAGE
        self._add_state_features()
        self._add_visual_features()

    def _add_state_features(self) -> None:
        """Register the agent-position state feature."""
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))

    def _add_visual_features(self) -> None:
        """Register pixel observation features when obs_type requires them."""
        if "pixels" in self.obs_type:
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )

    @property
    def gym_kwargs(self) -> dict:
        """Return PushT-specific Gymnasium kwargs.

        Returns:
            Dictionary with ``obs_type``, ``render_mode``, and ``max_episode_steps``.
        """
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class PickPlaceSimConfig(SimEnvConfig):
    """Configuration for the 3-D pick-and-place simulation environment.

    A 6-DOF arm must grasp a cube and place it on a target location.
    Observations include joint positions, end-effector pose, gripper state,
    and an optional pixel rendering.

    Attributes:
        task: Fixed to ``'PickPlace-Sim-v0'``.
        fps: Runs at 20 Hz by default.
        episode_length: 400 steps per episode.
        action_dim: Number of joint-velocity commands plus gripper (7).
        state_dim: Full proprioceptive state dimension (joint + ee + gripper).
    """

    task: str = "PickPlace-Sim-v0"
    fps: int = 20
    episode_length: int = 400
    action_dim: int = 7
    state_dim: int = 13

    def __post_init__(self) -> None:
        """Populate ``features`` and ``features_map`` based on ``obs_type``."""
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features_map[ACTION] = ACTION
        self.features_map["agent_pos"] = OBS_STATE
        self.features_map["pixels"] = OBS_IMAGE
        self._add_state_features()
        self._add_visual_features()

    def _add_state_features(self) -> None:
        """Register proprioceptive state features."""
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))

    def _add_visual_features(self) -> None:
        """Register pixel observation features when obs_type requires them."""
        if "pixels" in self.obs_type:
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )

    @property
    def gym_kwargs(self) -> dict:
        """Return PickPlace-specific Gymnasium kwargs.

        Returns:
            Dictionary with ``obs_type``, ``render_mode``, and ``max_episode_steps``.
        """
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@dataclass
class ReachSimConfig(SimEnvConfig):
    """Configuration for the 3-D reaching simulation environment.

    A 6-DOF arm must move its end-effector to a randomly sampled target
    position.  Observations include joint positions, end-effector xyz, and
    an optional pixel rendering.

    Attributes:
        task: Fixed to ``'Reach-Sim-v0'``.
        fps: Runs at 20 Hz by default.
        episode_length: 200 steps per episode.
        action_dim: Number of joint-velocity commands (6).
        state_dim: Proprioceptive state dimension (joints + ee xyz).
    """

    task: str = "Reach-Sim-v0"
    fps: int = 20
    episode_length: int = 200
    action_dim: int = 6
    state_dim: int = 9

    def __post_init__(self) -> None:
        """Populate ``features`` and ``features_map`` based on ``obs_type``."""
        self.features[ACTION] = PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))
        self.features_map[ACTION] = ACTION
        self.features_map["agent_pos"] = OBS_STATE
        self.features_map["pixels"] = OBS_IMAGE
        self._add_state_features()
        self._add_visual_features()

    def _add_state_features(self) -> None:
        """Register proprioceptive state features."""
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,))

    def _add_visual_features(self) -> None:
        """Register pixel observation features when obs_type requires them."""
        if "pixels" in self.obs_type:
            self.features["pixels"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(self.observation_height, self.observation_width, 3),
            )

    @property
    def gym_kwargs(self) -> dict:
        """Return Reach-specific Gymnasium kwargs.

        Returns:
            Dictionary with ``obs_type``, ``render_mode``, and ``max_episode_steps``.
        """
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }
