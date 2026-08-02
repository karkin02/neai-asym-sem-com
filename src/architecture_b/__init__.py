"""Architecture B — fully networked / split control.

Compressed scene representation -> simulated channel -> GPT-4o-mini planner ->
scripted controller -> Architecture A's MuJoCo environment.
"""

from __future__ import annotations

from .channel import CLEAN, DEGRADED, ChannelConfig, ChannelSimulator, get_channel
from .controller import ScriptedController
from .payload import CompressionLevel, Payload, build_payload
from .planner import ActionTarget, GptPlanner, HeuristicPlanner, get_planner


def __getattr__(name: str):
    if name == "run_trial":
        from .runner import run_trial

        return run_trial
    raise AttributeError(name)

__all__ = [
    "CLEAN",
    "DEGRADED",
    "ChannelConfig",
    "ChannelSimulator",
    "get_channel",
    "ScriptedController",
    "CompressionLevel",
    "Payload",
    "build_payload",
    "ActionTarget",
    "GptPlanner",
    "HeuristicPlanner",
    "get_planner",
    "run_trial",
]
