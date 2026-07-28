"""Architecture C — hybrid/adaptive control.

A CLIP-confidence + keyword gate routes recognized, high-confidence instructions
to local SmolVLA (zero network cost) and escalates everything else over the
Architecture B channel + GPT-4o-mini path.
"""

from __future__ import annotations

from .router import RouteDecision, RoutingConfig, decide_route, is_recognized
from .runner import run_trial

__all__ = [
    "RouteDecision",
    "RoutingConfig",
    "decide_route",
    "is_recognized",
    "run_trial",
]
