"""Merge Architecture A/B/C metrics into one comparison table.

Reads each architecture's ``metrics.json`` (A's native file and B/C's
A-compatible files), unions the per-episode columns, and writes a combined
``comparison.csv`` plus a per-architecture ``comparison_summary.csv``.

Run (from the repo root, with ``src`` importable, i.e. after ``pip install -e .``):

    python scripts/run_comparison.py \
      --a outputs/architecture_a_demo/smolvla-<ts> \
      --b outputs/architecture_b_demo/architecture_b-<ts> \
      --c outputs/architecture_c_demo/architecture_c-<ts> \
      --output outputs/comparison

Each ``--a/--b/--c`` accepts a ``metrics.json`` file OR a directory (the run
directory, or a parent containing timestamped run directories — the newest with
a ``metrics.json`` is used). Any architecture may be omitted.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

import sys

# Allow running as a plain script (python scripts/run_comparison.py) without an
# editable install by adding src/ to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from shared.metrics import COMPARISON_COLUMNS, load_results  # noqa: E402

_DEFAULTS = {"network_payload_bytes": 0}


def resolve_metrics_path(path: Path) -> Optional[Path]:
    """Resolve a file/dir into a ``metrics.json`` path (newest run if a parent)."""
    path = Path(path)
    if path.is_file():
        return path
    if (path / "metrics.json").is_file():
        return path / "metrics.json"
    candidates = sorted(path.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _row_for_csv(raw: dict[str, Any]) -> dict[str, Any]:
    """Project one result row onto the fixed comparison columns."""
    return {col: raw.get(col, _DEFAULTS.get(col, "")) for col in COMPARISON_COLUMNS}


def build_rows(paths_by_arch: dict[str, Path]) -> list[dict[str, Any]]:
    """Load and normalise per-episode rows from each architecture's metrics."""
    rows: list[dict[str, Any]] = []
    for architecture, metrics_path in paths_by_arch.items():
        resolved = resolve_metrics_path(metrics_path)
        if resolved is None:
            print(f"[comparison] no metrics.json found for {architecture} at {metrics_path}")
            continue
        for raw in load_results(resolved, architecture_fallback=architecture):
            rows.append(_row_for_csv(raw))
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows per architecture (success rate, latency, bytes, escalation)."""
    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        arch = row.get("architecture", "?")
        agg = summary.setdefault(
            arch, {"architecture": arch, "episodes": 0, "successes": 0, "latency_sum": 0.0, "bytes_sum": 0, "escalations": 0}
        )
        agg["episodes"] += 1
        agg["successes"] += 1 if row.get("success") in (True, "True", "true", 1) else 0
        agg["latency_sum"] += float(row.get("latency_seconds") or 0.0)
        agg["bytes_sum"] += int(row.get("network_payload_bytes") or 0)
        if row.get("escalated") in (True, "True", "true", 1):
            agg["escalations"] += 1

    out = []
    for agg in summary.values():
        n = agg["episodes"] or 1
        out.append(
            {
                "architecture": agg["architecture"],
                "episodes": agg["episodes"],
                "success_rate": round(agg["successes"] / n, 4),
                "mean_latency_seconds": round(agg["latency_sum"] / n, 3),
                "total_network_payload_bytes": agg["bytes_sum"],
                "escalation_rate": round(agg["escalations"] / n, 4),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Merge A/B/C metrics into one comparison table.")
    parser.add_argument("--a", type=Path, help="Architecture A metrics.json or run dir.")
    parser.add_argument("--b", type=Path, help="Architecture B metrics.json or run dir.")
    parser.add_argument("--c", type=Path, help="Architecture C metrics.json or run dir.")
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison"))
    args = parser.parse_args(argv)

    paths_by_arch = {a: p for a, p in (("A", args.a), ("B", args.b), ("C", args.c)) if p is not None}
    if not paths_by_arch:
        parser.error("Provide at least one of --a/--b/--c.")

    rows = build_rows(paths_by_arch)
    summary = summarize_rows(rows)

    comparison_path = args.output / "comparison.csv"
    summary_path = args.output / "comparison_summary.csv"
    _write_csv(comparison_path, rows, COMPARISON_COLUMNS)
    _write_csv(summary_path, summary, list(summary[0].keys()) if summary else ["architecture"])

    print(f"[comparison] {len(rows)} rows -> {comparison_path}")
    print(f"[comparison] summary -> {summary_path}")
    for entry in summary:
        print(
            f"  {entry['architecture']}: success={entry['success_rate']:.2f} "
            f"latency={entry['mean_latency_seconds']:.3f}s "
            f"bytes={entry['total_network_payload_bytes']} "
            f"escalation={entry['escalation_rate']:.2f}"
        )


if __name__ == "__main__":
    main()
