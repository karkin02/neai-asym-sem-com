"""Simulated network channel for Architecture B (and C's escalation path).

Models the cost of shipping a JSON/image payload to the cloud planner: a fixed
latency plus a bandwidth-limited transmission time, with a ``clean`` (oracle)
mode and a ``degraded`` mode that also drops packets. It is a standalone,
deterministic unit — it returns the modelled latency (and only sleeps if
``realtime=True``) so trials stay fast and reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ChannelConfig:
    """Channel parameters.

    Attributes:
        name: Condition label recorded in metrics (e.g. ``"clean"``).
        bandwidth_bytes_per_sec: Uplink capacity; transmission time is
            ``payload_bytes / bandwidth``.
        latency_seconds: Fixed one-way latency added to every transmission.
        drop_probability: Per-transmission packet-loss probability in ``[0, 1]``.
    """

    name: str
    bandwidth_bytes_per_sec: float
    latency_seconds: float
    drop_probability: float = 0.0


# Oracle: effectively free and lossless (the "no network cost" reference).
CLEAN = ChannelConfig(
    name="clean", bandwidth_bytes_per_sec=1.0e9, latency_seconds=0.0, drop_probability=0.0
)
# Degraded: ~1 Mbit/s, 300 ms latency, 20% loss.
DEGRADED = ChannelConfig(
    name="degraded", bandwidth_bytes_per_sec=125_000.0, latency_seconds=0.30, drop_probability=0.20
)

PRESETS = {"clean": CLEAN, "degraded": DEGRADED}


@dataclass(frozen=True)
class TransmissionResult:
    """Outcome of one payload transmission.

    Attributes:
        delivered: True if the payload arrived (False when dropped).
        dropped: True if the degraded channel lost the packet.
        payload_bytes: On-wire size that was billed.
        latency_seconds: Total modelled delay (fixed latency + bandwidth time).
        condition: The channel condition name.
    """

    delivered: bool
    dropped: bool
    payload_bytes: int
    latency_seconds: float
    condition: str


class ChannelSimulator:
    """Applies a :class:`ChannelConfig` to outgoing payloads.

    Args:
        config: The channel condition.
        seed: RNG seed for deterministic drop decisions.
        realtime: If True, actually ``time.sleep`` the modelled latency (for a
            live demo). Default False so trials/tests run instantly.
    """

    def __init__(self, config: ChannelConfig = CLEAN, seed: int = 0, realtime: bool = False) -> None:
        self.config = config
        self._rng = random.Random(seed)
        self.realtime = realtime

    def transmit(self, payload: Union[bytes, bytearray, int]) -> TransmissionResult:
        """Transmit a payload (or its byte count) across the channel.

        Args:
            payload: The bytes to send, or an int byte count.

        Returns:
            A :class:`TransmissionResult`. On a drop, ``delivered`` is False and
            only the fixed latency is billed (the send failed mid-flight).
        """
        num_bytes = payload if isinstance(payload, int) else len(payload)
        dropped = self._rng.random() < self.config.drop_probability
        if dropped:
            latency = self.config.latency_seconds
            result = TransmissionResult(False, True, int(num_bytes), latency, self.config.name)
        else:
            transfer = num_bytes / self.config.bandwidth_bytes_per_sec
            latency = self.config.latency_seconds + transfer
            result = TransmissionResult(True, False, int(num_bytes), latency, self.config.name)
        if self.realtime and latency > 0:
            import time

            time.sleep(latency)
        return result


def get_channel(name: str, seed: int = 0, realtime: bool = False) -> ChannelSimulator:
    """Build a :class:`ChannelSimulator` from a preset name (``clean``/``degraded``)."""
    if name not in PRESETS:
        raise ValueError(f"Unknown channel '{name}'. Choose from {sorted(PRESETS)}.")
    return ChannelSimulator(PRESETS[name], seed=seed, realtime=realtime)
