"""Standalone unit tests for Architecture C (routing gate + hybrid run_trial).

All external systems are faked, so these run on any environment with numpy.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from architecture_b.channel import ChannelConfig, ChannelSimulator
from architecture_b.controller import ScriptedController
from architecture_b.payload import CompressionLevel
from architecture_b.planner import HeuristicPlanner
from architecture_c.router import RoutingConfig, decide_route, is_recognized
from architecture_c.runner import run_trial
from shared.perception import Detection
from shared.smolvla_runtime import SmolVlaRuntime


# ---- Fakes ----------------------------------------------------------------
class _FakeEnv:
    target_positions = {
        "left_tray": (-0.14, 0.25, 0.075),
        "right_tray": (0.14, 0.25, 0.075),
        "conveyor": (0.0, 0.30, 0.075),
    }

    def __init__(self, success_on_step=1):
        self.sample_position = (0.1, 0.2, 0.05)
        self._steps = 0
        self._success_on = success_on_step
        self.calls = []

    def reset(self, *, seed, instruction):
        self._steps = 0
        return types.SimpleNamespace(robot_state=(0.0,) * 6)

    def capture_rgb(self, *, camera="overhead", width=320, height=240):
        return np.zeros((height, width, 3), dtype=np.uint8)

    def solve_ik(self, target_position, *, gripper, max_iterations=120):
        return (0.0, 0.0, 0.0, 0.0, 0.0, float(gripper))

    def step(self, action):
        self.calls.append(action)
        self._steps += 1
        success = self._steps >= self._success_on
        return types.SimpleNamespace(
            info={"success": success, "failure_reason": None if success else "placement_failed"},
            terminated=success,
            truncated=False,
        )


class _FakeDetector:
    def detect(self, frame):
        return [Detection("sample", 0.9, (10, 40, 30, 60)), Detection("left_tray", 0.8, (200, 40, 240, 60))]


class _FakeGrounder:
    def __init__(self, confidence):
        self._confidence = confidence

    def score(self, expression, crops):
        return types.SimpleNamespace(confidence=self._confidence)


class _FakePredictor:
    def __init__(self, chunk):
        self._chunk = chunk

    def predict_chunk(self, overhead, wrist, robot_state, instruction):
        return self._chunk


INSTRUCTION = "Pick up the sample and place it in the left tray."


def _run(clip_confidence, runtime, channel=None):
    return run_trial(
        env=_FakeEnv(success_on_step=1),
        detector=_FakeDetector(),
        grounder=_FakeGrounder(clip_confidence),
        local_runtime=runtime,
        planner=HeuristicPlanner(),
        controller=ScriptedController(_FakeEnv(success_on_step=4)),
        channel=channel or ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)),
        compression_level=CompressionLevel.SCENE_GRAPH,
        instruction=INSTRUCTION,
        seed=1,
        episode_id=0,
        scene="warehouse_normal",
        routing_config=RoutingConfig(clip_threshold=0.6),
    )


# ---- Router ---------------------------------------------------------------
class RouterTest(unittest.TestCase):
    def test_high_confidence_recognized_routes_local(self):
        d = decide_route(INSTRUCTION, 0.9, ["sample"], RoutingConfig(clip_threshold=0.6))
        self.assertEqual(d.route, "local")
        self.assertFalse(d.escalated)

    def test_low_confidence_escalates(self):
        d = decide_route(INSTRUCTION, 0.2, ["sample"], RoutingConfig(clip_threshold=0.6))
        self.assertEqual(d.route, "escalate")
        self.assertTrue(d.escalated)
        self.assertFalse(d.clip_ok)

    def test_unrecognized_instruction_escalates(self):
        d = decide_route("teleport the widget to orbit", 0.99, [], RoutingConfig())
        self.assertEqual(d.route, "escalate")
        self.assertFalse(d.recognized)

    def test_recognized_via_detected_label(self):
        # object not named in text but present in detections
        self.assertTrue(is_recognized("pick it up", ["sample"], RoutingConfig()))
        self.assertFalse(is_recognized("pick it up", ["banana"], RoutingConfig()))


# ---- Hybrid run_trial -----------------------------------------------------
class HybridRunTest(unittest.TestCase):
    def test_local_route_zero_network_cost(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        record = _run(clip_confidence=0.95, runtime=runtime)
        self.assertEqual(record.route, "local")
        self.assertFalse(record.escalated)
        self.assertEqual(record.network_payload_bytes, 0)
        self.assertTrue(record.success)

    def test_low_confidence_escalates_with_payload(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        record = _run(clip_confidence=0.10, runtime=runtime)
        self.assertEqual(record.route, "escalated")
        self.assertTrue(record.escalated)
        self.assertGreater(record.network_payload_bytes, 0)

    def test_unavailable_runtime_forces_escalation(self):
        # No injected predictor and a checkpoint path that does not exist.
        runtime = SmolVlaRuntime(checkpoint="does/not/exist")
        self.assertFalse(runtime.available())
        record = _run(clip_confidence=0.95, runtime=runtime)  # gate says local, but unavailable
        self.assertEqual(record.route, "escalated")
        self.assertIn("unavailable", record.extra["routing_reason"])

    def test_channel_drop_on_escalation_is_failure(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        drop = ChannelSimulator(ChannelConfig("degraded", 125_000, 0.3, drop_probability=1.0))
        record = _run(clip_confidence=0.10, runtime=runtime, channel=drop)
        self.assertFalse(record.success)
        self.assertEqual(record.failure_reason, "channel_drop")


if __name__ == "__main__":
    unittest.main()
