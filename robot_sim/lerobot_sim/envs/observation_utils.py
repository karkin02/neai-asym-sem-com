"""
Observation preprocessing utilities for LeRobot compatibility.

Converts raw environment observations into the tensor format expected by
LeRobot policies: channel-first float32 images normalised to [0, 1] and
flat state vectors.

Functions:
    preprocess_observation: Convert a raw observation dict to LeRobot format.
    env_to_policy_features: Map env config features to policy feature dicts.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from lerobot_sim.utils.constants import (
    ACTION,
    FeatureType,
    OBS_ENV_STATE,
    OBS_IMAGE,
    OBS_STATE,
    PolicyFeature,
)


def _convert_image_to_chw(image: np.ndarray) -> np.ndarray:
    """Transpose an (H, W, C) uint8 image to (C, H, W) float32 in [0, 1].

    Args:
        image: Channel-last uint8 image.

    Returns:
        Channel-first float32 image normalised to [0, 1].
    """
    chw = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw)


def _add_batch_dim(arr: np.ndarray) -> np.ndarray:
    """Prepend a batch dimension if the array is unbatched.

    Args:
        arr: Array that may or may not have a batch axis.

    Returns:
        Array with shape ``(1, ...original_shape)``.
    """
    if arr.ndim < 2:
        return arr[np.newaxis]
    return arr


def _process_pixels(obs: Dict[str, np.ndarray], result: Dict[str, np.ndarray]) -> None:
    """Extract and convert pixel observations into LeRobot format.

    Args:
        obs: Raw observation dictionary.
        result: Output dictionary to populate.
    """
    if "pixels" in obs:
        img = _convert_image_to_chw(obs["pixels"])
        result[OBS_IMAGE] = _add_batch_dim(img)


def _process_state(obs: Dict[str, np.ndarray], result: Dict[str, np.ndarray]) -> None:
    """Extract and convert agent state observations.

    Args:
        obs: Raw observation dictionary.
        result: Output dictionary to populate.
    """
    if "agent_pos" in obs:
        state = obs["agent_pos"].astype(np.float32)
        result[OBS_STATE] = _add_batch_dim(state)


def _process_env_state(obs: Dict[str, np.ndarray], result: Dict[str, np.ndarray]) -> None:
    """Extract and convert environment state observations.

    Args:
        obs: Raw observation dictionary.
        result: Output dictionary to populate.
    """
    if "environment_state" in obs:
        env_state = obs["environment_state"].astype(np.float32)
        result[OBS_ENV_STATE] = _add_batch_dim(env_state)


def preprocess_observation(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Convert a raw environment observation dict to LeRobot format.

    Mirrors the behaviour of ``lerobot.envs.utils.preprocess_observation``
    but operates entirely with NumPy arrays (no PyTorch dependency).

    Args:
        obs: Dictionary of observation arrays as returned by ``env.step()``.

    Returns:
        Dictionary with LeRobot-standard keys and preprocessed arrays.
    """
    result: Dict[str, np.ndarray] = {}
    _process_pixels(obs, result)
    _process_state(obs, result)
    _process_env_state(obs, result)
    return result


def _get_channel_first_shape(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Convert an (H, W, C) shape to (C, H, W).

    Args:
        shape: Tuple of (height, width, channels).

    Returns:
        Tuple of (channels, height, width).
    """
    if len(shape) != 3:
        return shape
    return (shape[2], shape[0], shape[1])


def env_to_policy_features(
    features: Dict[str, PolicyFeature],
    features_map: Dict[str, str],
) -> Dict[str, PolicyFeature]:
    """Map environment features to the keys expected by a policy.

    Visual features are converted from (H, W, C) to (C, H, W).

    Args:
        features: Mapping from env key to ``PolicyFeature``.
        features_map: Mapping from env key to policy key.

    Returns:
        Dictionary keyed by policy-side names.
    """
    policy_features: Dict[str, PolicyFeature] = {}
    for env_key, ft in features.items():
        policy_key = features_map.get(env_key, env_key)
        shape = _get_channel_first_shape(ft.shape) if ft.type is FeatureType.VISUAL else ft.shape
        policy_features[policy_key] = PolicyFeature(type=ft.type, shape=shape)
    return policy_features
