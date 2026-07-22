"""
Small stateless helpers used across the lerobot_sim package.

Provides functions for image format conversion, numerical clamping,
seeding, and channel-order transposition.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def clamp(value: float, lo: float, hi: float) -> float:
    """Return *value* clamped to the closed interval [*lo*, *hi*].

    Args:
        value: The scalar to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        The clamped scalar.
    """
    return max(lo, min(hi, value))


def _validate_image_ndim(image: np.ndarray) -> None:
    """Raise if *image* does not have exactly three dimensions.

    Args:
        image: Array to validate.

    Raises:
        ValueError: When ndim != 3.
    """
    if image.ndim != 3:
        raise ValueError(f"Expected 3-D image, got ndim={image.ndim}")


def channel_last_to_first(image: np.ndarray) -> np.ndarray:
    """Transpose an (H, W, C) image to (C, H, W).

    Args:
        image: NumPy array with shape (H, W, C).

    Returns:
        Transposed array with shape (C, H, W).
    """
    _validate_image_ndim(image)
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def channel_first_to_last(image: np.ndarray) -> np.ndarray:
    """Transpose a (C, H, W) image to (H, W, C).

    Args:
        image: NumPy array with shape (C, H, W).

    Returns:
        Transposed array with shape (H, W, C).
    """
    _validate_image_ndim(image)
    return np.ascontiguousarray(np.transpose(image, (1, 2, 0)))


def normalize_to_range(value: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Scale *value* from [*lo*, *hi*] into [0, 1].

    Args:
        value: Array of values in the original range.
        lo: Minimum of the original range.
        hi: Maximum of the original range.

    Returns:
        Normalized array with values in [0, 1].
    """
    span = hi - lo
    safe_span = span if span != 0.0 else 1.0
    return (value - lo) / safe_span


def seed_rngs(seed: int) -> np.random.Generator:
    """Create and return a NumPy random generator seeded with *seed*.

    Args:
        seed: The integer seed value.

    Returns:
        A seeded ``numpy.random.Generator``.
    """
    return np.random.default_rng(seed)


def get_channel_first_shape(height: int, width: int, channels: int = 3) -> Tuple[int, int, int]:
    """Return the (C, H, W) shape tuple from spatial dimensions.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        channels: Number of colour channels (default 3).

    Returns:
        Tuple ``(channels, height, width)``.
    """
    return (channels, height, width)
