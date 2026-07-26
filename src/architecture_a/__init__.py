"""Co-located edge VLA architecture."""

from .contracts import Action, Observation, PolicyDecision
from .runner import ArchitectureA, EpisodeResult

__all__ = [
    "Action",
    "ArchitectureA",
    "EpisodeResult",
    "Observation",
    "PolicyDecision",
]

