"""Standalone unit tests for Architecture B (payload, planner, controller, run_trial).

All external systems (MuJoCo env, OpenAI, YOLO) are faked, so these run on any
environment with numpy.
"""

from __future__ import annotations

import types
import unittest

import numpy as np

from architecture_b.channel import ChannelConfig, ChannelSimulator
from architecture_b.controller import GRIPPER_CLOSE, GRIPPER_OPEN, ScriptedController
from architecture_b.payload import CompressionLevel, build_payload
from architecture_b.planner import ActionTarget, GptPlanner, HeuristicPlanner
from architecture_b.runner import run_trial
from shared.perception import Detection, build_scene_graph


def _sample_graph():
    dets = [
        Detection("sample", 0.9, (10, 40, 30, 60)),
        Detection("left_tray", 0.8, (200, 40, 240, 60)),
    ]
    return build_scene_graph(dets, 320, 240, task="pick"), dets


# ---- Fakes ----------------------------------------------------------------
class _StepResult:
    def __init__(self, success):
        self.info = {"success": success, "failure_reason": None if success else "placement_failed"}
        self.terminated = success
        self.truncated = False


class _FakeEnv:
    target_positions = {
        "left_tray": (-0.14, 0.25, 0.075),
        "right_tray": (0.14, 0.25, 0.075),
        "conveyor": (0.0, 0.30, 0.075),
    }

    def __init__(self):
        self.sample_position = (0.1, 0.2, 0.05)
        self.calls = []
        self._steps = 0

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
        return _StepResult(success=(self._steps == 4))  # succeed on the release step


class _FakeDetector:
    def detect(self, frame):
        return [
            Detection("sample", 0.9, (10, 40, 30, 60)),
            Detection("left_tray", 0.8, (200, 40, 240, 60)),
        ]


def _fake_openai_client(content):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    completion = types.SimpleNamespace(choices=[choice])
    completions = types.SimpleNamespace(create=lambda **kwargs: completion)
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(chat=chat)


# ---- Payload --------------------------------------------------------------
class PayloadTest(unittest.TestCase):
    def test_scene_graph_smaller_than_full_json(self):
        graph, _ = _sample_graph()
        full = build_payload(CompressionLevel.FULL_JSON, graph, "pick")
        compact = build_payload(CompressionLevel.SCENE_GRAPH, graph, "pick")
        self.assertGreater(full.num_bytes, compact.num_bytes)
        self.assertEqual(compact.level, CompressionLevel.SCENE_GRAPH)

    def test_raw_image_requires_frame(self):
        graph, _ = _sample_graph()
        with self.assertRaises(ValueError):
            build_payload(CompressionLevel.RAW_IMAGE, graph, "pick", frame=None)


# ---- Planner --------------------------------------------------------------
class PlannerTest(unittest.TestCase):
    def test_heuristic_routes_by_keyword(self):
        graph, _ = _sample_graph()
        payload = build_payload(CompressionLevel.FULL_JSON, graph, "reject the damaged package")
        target = HeuristicPlanner().plan(payload)
        self.assertEqual(target.destination, "right_tray")
        self.assertEqual(target.target_object, "sample")
        self.assertEqual(target.source, "heuristic")

    def test_gpt_planner_parses_injected_client(self):
        graph, _ = _sample_graph()
        payload = build_payload(CompressionLevel.SCENE_GRAPH, graph, "pick")
        client = _fake_openai_client(
            '{"target_object":"sample","destination":"left_tray","gripper":"close","reasoning":"ok"}'
        )
        target = GptPlanner(client=client).plan(payload)
        self.assertEqual(target.destination, "left_tray")
        self.assertEqual(target.gripper, "close")
        self.assertEqual(target.source, "gpt-4o-mini")


# ---- Controller -----------------------------------------------------------
class ControllerTest(unittest.TestCase):
    def test_pick_carry_release_sequence(self):
        env = _FakeEnv()
        controller = ScriptedController(env)
        target = ActionTarget("sample", "left_tray", "open", "r", "test")
        result = controller.execute(target)
        self.assertEqual(result.steps, 4)
        grippers = [a.values[5] for a in env.calls]
        self.assertEqual(grippers, [GRIPPER_OPEN, GRIPPER_CLOSE, GRIPPER_CLOSE, GRIPPER_OPEN])
        self.assertTrue(result.success)


# ---- run_trial ------------------------------------------------------------
class RunTrialTest(unittest.TestCase):
    def _run(self, channel):
        return run_trial(
            env=_FakeEnv(),
            detector=_FakeDetector(),
            planner=HeuristicPlanner(),
            controller=ScriptedController(_FakeEnv()),
            channel=channel,
            compression_level=CompressionLevel.SCENE_GRAPH,
            instruction="place the sample in the left tray",
            seed=1,
            episode_id=0,
            scene="warehouse_normal",
        )

    def test_trial_records_bytes_and_route(self):
        record = self._run(ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)))
        self.assertEqual(record.architecture, "B")
        self.assertEqual(record.route, "cloud")
        self.assertEqual(record.compression_level, "scene_graph")
        self.assertGreater(record.network_payload_bytes, 0)
        self.assertIsNone(record.failure_reason)

    def test_channel_drop_is_a_failure(self):
        record = self._run(ChannelSimulator(ChannelConfig("degraded", 125_000, 0.3, drop_probability=1.0)))
        self.assertFalse(record.success)
        self.assertEqual(record.failure_reason, "channel_drop")


if __name__ == "__main__":
    unittest.main()
