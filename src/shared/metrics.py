"""Per-trial metrics shared by Architectures A, B and C.

Architecture A already writes its own ``metrics.json`` (via
``scripts/evaluate_smolvla.py``); there is no shared writer to reuse. This
module provides a :class:`TrialRecord` and :func:`write_metrics` for B and C
that are **column-compatible** with A's per-episode records — same
``episode_id`` / ``seed`` / ``success`` / ``latency_seconds`` / ``failure_reason``
fields — while adding the B/C-only columns the proposal requires:
``network_payload_bytes`` (0 for A), ``clip_confidence`` and ``escalated``.

The combined comparison (``scripts/run_comparison.py``) reads every
architecture's ``metrics.json`` ``results`` list through :func:`load_results`
and unions the columns, so all three merge into one table.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

# Columns emitted into the combined comparison CSV, in order. Architectures
# that lack a column leave it blank/default (e.g. A has 0 payload bytes).
COMPARISON_COLUMNS = [
    "architecture",
    "episode_id",
    "seed",
    "instruction",
    "scene",
    "success",
    "steps",
    "latency_seconds",
    "network_payload_bytes",
    "clip_confidence",
    "escalated",
    "route",
    "channel_condition",
    "compression_level",
    "failure_reason",
]


@dataclass
class TrialRecord:
    """One trial's outcome, comparable across architectures.

    ``network_payload_bytes`` is 0 for the co-located Architecture A;
    ``clip_confidence`` is ``None`` where grounding was not computed; ``escalated``
    is meaningful for Architecture C (``True``/``False``) and ``None`` elsewhere.
    """

    architecture: str
    episode_id: int
    seed: int
    instruction: str
    scene: str
    success: bool
    steps: int = 0
    latency_seconds: float = 0.0
    network_payload_bytes: int = 0
    clip_confidence: Optional[float] = None
    escalated: Optional[bool] = None
    route: Optional[str] = None
    channel_condition: Optional[str] = None
    compression_level: Optional[str] = None
    failure_reason: Optional[str] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.latency_seconds is not None:
            data["latency_seconds"] = round(float(self.latency_seconds), 3)
        if self.clip_confidence is not None:
            data["clip_confidence"] = round(float(self.clip_confidence), 4)
        return data


def summarize(architecture: str, records: Sequence[TrialRecord], config: Optional[dict] = None) -> dict[str, Any]:
    """Build the ``metrics.json`` summary dict (mirrors A's top-level shape)."""
    episodes = len(records)
    successes = sum(1 for r in records if r.success)
    escalations = sum(1 for r in records if r.escalated)
    latencies = [r.latency_seconds for r in records if r.latency_seconds is not None]
    payloads = [r.network_payload_bytes for r in records]
    return {
        "architecture": architecture,
        "episodes": episodes,
        "successes": successes,
        "success_rate": (successes / episodes) if episodes else 0.0,
        "escalations": escalations,
        "escalation_rate": (escalations / episodes) if episodes else 0.0,
        "mean_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "total_network_payload_bytes": int(sum(payloads)),
        "config": config or {},
        "results": [r.as_dict() for r in records],
    }


def write_metrics(
    run_dir: Path,
    architecture: str,
    records: Sequence[TrialRecord],
    config: Optional[dict] = None,
) -> Path:
    """Write ``metrics.json`` under ``run_dir`` and return its path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "metrics.json"
    path.write_text(json.dumps(summarize(architecture, records, config), indent=2), encoding="utf-8")
    return path


def load_results(metrics_path: Path, architecture_fallback: Optional[str] = None) -> list[dict[str, Any]]:
    """Load a ``metrics.json`` and return its per-episode ``results`` rows.

    Works for A's native file and B/C files. Each row is tagged with its
    architecture (from the summary, else ``architecture_fallback``) so rows from
    all three merge cleanly.
    """
    data = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    architecture = data.get("architecture", architecture_fallback)
    rows = []
    for item in data.get("results", []):
        row = dict(item)
        row.setdefault("architecture", architecture)
        rows.append(row)
    return rows
