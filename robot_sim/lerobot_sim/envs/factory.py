"""
Factory function for creating simulation environments.

Mirrors the upstream ``lerobot.envs.factory.make_env`` API so that
callers can instantiate any simulation environment by config or by name
and receive a Gymnasium ``VectorEnv`` wrapped in the standard mapping.

Functions:
    make_sim_env: Create one or more vectorised simulation environments.
"""

from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym

from lerobot_sim.envs.configs import (
    PickPlaceSimConfig,
    PushTSimConfig,
    ReachSimConfig,
    SimEnvConfig,
)

# ---------------------------------------------------------------------------
# Config look-up table (name -> default config constructor)
# ---------------------------------------------------------------------------
_ENV_REGISTRY: Dict[str, type] = {
    "pusht": PushTSimConfig,
    "pick_place": PickPlaceSimConfig,
    "reach": ReachSimConfig,
}


def _resolve_config(cfg: SimEnvConfig | str) -> SimEnvConfig:
    """Convert a string name to its default config, or pass through a config.

    Args:
        cfg: Either a ``SimEnvConfig`` instance or one of
            ``'pusht'``, ``'pick_place'``, ``'reach'``.

    Returns:
        A concrete ``SimEnvConfig`` instance.

    Raises:
        ValueError: If the string name is not in the registry.
    """
    if isinstance(cfg, SimEnvConfig):
        return cfg
    if cfg not in _ENV_REGISTRY:
        raise ValueError(f"Unknown env '{cfg}'. Choose from {list(_ENV_REGISTRY)}")
    return _ENV_REGISTRY[cfg]()


def _env_class_for_config(cfg: SimEnvConfig) -> type:
    """Return the Gymnasium env class matching *cfg*.

    Args:
        cfg: A concrete ``SimEnvConfig`` instance.

    Returns:
        The corresponding ``gym.Env`` subclass.

    Raises:
        ValueError: If the config type is not recognised.
    """
    from lerobot_sim.envs.pick_place import PickPlaceSimEnv
    from lerobot_sim.envs.push_t import PushTSimEnv
    from lerobot_sim.envs.reach import ReachSimEnv

    dispatch = {
        PushTSimConfig: PushTSimEnv,
        PickPlaceSimConfig: PickPlaceSimEnv,
        ReachSimConfig: ReachSimEnv,
    }
    env_cls = dispatch.get(type(cfg))
    if env_cls is None:
        raise ValueError(f"No env class registered for config type {type(cfg).__name__}")
    return env_cls


def _validate_n_envs(n_envs: int) -> None:
    """Raise if *n_envs* is less than one.

    Args:
        n_envs: Requested number of parallel environments.

    Raises:
        ValueError: When ``n_envs < 1``.
    """
    if n_envs < 1:
        raise ValueError("`n_envs` must be at least 1")


def _build_vector_env(env_cls: type, cfg: SimEnvConfig, n_envs: int, use_async: bool) -> gym.vector.VectorEnv:
    """Construct a Gymnasium vector environment.

    Args:
        env_cls: The single-env Gymnasium class.
        cfg: Environment configuration forwarded to the constructor.
        n_envs: Number of parallel copies.
        use_async: If *True*, use ``AsyncVectorEnv``; otherwise ``SyncVectorEnv``.

    Returns:
        A ``VectorEnv`` wrapping *n_envs* instances.
    """
    wrapper_cls = gym.vector.AsyncVectorEnv if use_async else gym.vector.SyncVectorEnv
    fns = [lambda c=cfg: env_cls(c) for _ in range(n_envs)]
    return wrapper_cls(fns)


def make_sim_env(
    cfg: SimEnvConfig | str,
    n_envs: int = 1,
    use_async_envs: bool = False,
) -> Dict[str, Dict[int, gym.vector.VectorEnv]]:
    """Create vectorised simulation environments matching the LeRobot API.

    The return format mirrors ``lerobot.envs.factory.make_env``: a mapping
    from suite name to ``{task_id: VectorEnv}`` so that downstream
    evaluation scripts work without modification.

    Args:
        cfg: Either a ``SimEnvConfig`` instance or a string name
            (``'pusht'``, ``'pick_place'``, ``'reach'``).
        n_envs: Number of parallel environments (default 1).
        use_async_envs: Whether to use ``AsyncVectorEnv`` (default *False*).

    Returns:
        ``{suite_name: {0: VectorEnv}}`` mapping.
    """
    resolved_cfg = _resolve_config(cfg)
    _validate_n_envs(n_envs)
    env_cls = _env_class_for_config(resolved_cfg)
    vec = _build_vector_env(env_cls, resolved_cfg, n_envs, use_async_envs)
    suite_name = resolved_cfg.env_type
    return {suite_name: {0: vec}}
