"""Unit tests for the A/B/C comparison merger (scripts/run_comparison.py)."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from shared.metrics import TrialRecord, write_metrics

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "run_comparison.py"
_spec = importlib.util.spec_from_file_location("run_comparison", _SCRIPT)
run_comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_comparison)


class ComparisonTest(unittest.TestCase):
    def _write_a(self, root: Path) -> Path:
        # Architecture A's native metrics.json shape (no "architecture" key).
        run_dir = root / "smolvla-000"
        run_dir.mkdir(parents=True)
        payload = {
            "scene": "pick_place",
            "results": [
                {"episode_id": 0, "seed": 1, "success": True, "latency_seconds": 0.5, "failure_reason": None},
                {"episode_id": 1, "seed": 2, "success": False, "latency_seconds": 0.6, "failure_reason": "grasp_failed"},
            ],
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")
        return run_dir

    def test_merge_and_summary(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a_dir = self._write_a(root)
            b_dir = root / "b"
            c_dir = root / "c"
            write_metrics(b_dir, "B", [
                TrialRecord("B", 0, 1, "pick", "warehouse_normal", True, latency_seconds=1.2,
                            network_payload_bytes=800, compression_level="scene_graph", channel_condition="clean"),
            ])
            write_metrics(c_dir, "C", [
                TrialRecord("C", 0, 1, "pick", "warehouse_normal", True, latency_seconds=0.4,
                            clip_confidence=0.9, escalated=False, route="local", network_payload_bytes=0),
                TrialRecord("C", 1, 2, "pick", "warehouse_normal", True, latency_seconds=1.5,
                            clip_confidence=0.2, escalated=True, route="escalated", network_payload_bytes=800),
            ])

            rows = run_comparison.build_rows({"A": a_dir, "B": b_dir, "C": c_dir})
            self.assertEqual(len(rows), 5)  # 2 A + 1 B + 2 C

            a_rows = [r for r in rows if r["architecture"] == "A"]
            self.assertEqual(a_rows[0]["network_payload_bytes"], 0)  # default filled for A
            self.assertEqual(a_rows[0]["escalated"], "")             # not applicable to A

            summary = {s["architecture"]: s for s in run_comparison.summarize_rows(rows)}
            self.assertEqual(summary["A"]["episodes"], 2)
            self.assertAlmostEqual(summary["A"]["success_rate"], 0.5)
            self.assertEqual(summary["C"]["escalation_rate"], 0.5)  # 1 of 2 escalated
            self.assertEqual(summary["B"]["total_network_payload_bytes"], 800)

    def test_resolve_metrics_path_variants(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            run.mkdir()
            metrics = run / "metrics.json"
            metrics.write_text("{}", encoding="utf-8")
            self.assertEqual(run_comparison.resolve_metrics_path(metrics), metrics)
            self.assertEqual(run_comparison.resolve_metrics_path(run), metrics)
            self.assertEqual(run_comparison.resolve_metrics_path(root), metrics)  # parent -> newest


if __name__ == "__main__":
    unittest.main()
