"""
EnvHub entry point for loading lerobot_sim via the HF Hub.

This file follows the LeRobot EnvHub convention so that this project
can be loaded remotely with::

    from lerobot.envs.factory import make_env
    envs = make_env("your-user/lerobot-sim", trust_remote_code=True)

Functions:
    make_env: Create vectorised simulation environments.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import gymnasium as gym

from lerobot_sim.envs.configs import SimEnvConfig
from lerobot_sim.envs.factory import (
    _build_vector_env,
    _env_class_for_config,
    _resolve_config,
    _validate_n_envs,
)


def _default_config() -> SimEnvConfig:
    """Return the default PushT environment configuration.

    Returns:
        A ``PushTSimConfig`` instance.
    """
    from lerobot_sim.envs.configs import PushTSimConfig

    return PushTSimConfig()


def make_env(
    n_envs: int = 1,
    use_async_envs: bool = False,
    cfg: Optional[SimEnvConfig] = None,
) -> Dict[str, Dict[int, gym.vector.VectorEnv]]:
    """Create vectorised simulation environments (EnvHub API).

    Args:
        n_envs: Number of parallel environments.
        use_async_envs: Use ``AsyncVectorEnv`` if True.
        cfg: Optional ``SimEnvConfig``; defaults to PushT.

    Returns:
        ``{suite_name: {0: VectorEnv}}`` matching LeRobot convention.
    """
    resolved = cfg if cfg is not None else _default_config()
    _validate_n_envs(n_envs)
    env_cls = _env_class_for_config(resolved)
    vec = _build_vector_env(env_cls, resolved, n_envs, use_async_envs)
    return {resolved.env_type: {0: vec}}
