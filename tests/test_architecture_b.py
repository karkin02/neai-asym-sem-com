"""Standalone unit tests for Architecture B (payload, planner, controller, run_trial).

All external systems (MuJoCo env, OpenAI, YOLO) are faked, so these run on any
environment with numpy.
"""

from __future__ import annotations

import types
import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from architecture_b.channel import ChannelConfig, ChannelSimulator
from architecture_b.controller import GRIPPER_CLOSE, GRIPPER_OPEN, ScriptedController
from architecture_b.payload import CompressionLevel, build_payload, visual_observations_for_views
from architecture_b.planner import ActionTarget, GptPlanner, HeuristicPlanner, MAX_COMPLETION_TOKENS
from architecture_b.runner import checkpoint_for_scene, run_trial, vlm_for_runtime
from architecture_b.service import ServiceTask, atomic_write_json
from shared.instructions import instruction_for_scenario
from shared.perception import Detection, build_scene_graph


class VlmResolutionTests(unittest.TestCase):
    def test_explicit_vlm_wins(self):
        self.assertEqual(vlm_for_runtime("D:/models/vlm", Path("missing")), "D:/models/vlm")

    def test_existing_local_vlm_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            expected = str(Path(directory).resolve())
            self.assertEqual(vlm_for_runtime(None, Path(directory)), expected)

    def test_hub_id_is_portable_fallback(self):
        self.assertEqual(
            vlm_for_runtime(None, Path("definitely-missing-vlm")),
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        )

    def test_checkpoint_resolves_from_real_manifest(self):
        self.assertEqual(
            checkpoint_for_scene("warehouse_normal", None),
            Path("outputs/train/warehouse_v3_150/checkpoints/002000/pretrained_model"),
        )


class ServiceTests(unittest.TestCase):
    def test_service_task_validates_scene_and_request_id(self):
        task = ServiceTask.from_dict(
            {"request_id": "job-1", "scene": "package_damaged", "seed": 60000}
        )
        self.assertEqual(task.scene, "package_damaged")
        with self.assertRaises(ValueError):
            ServiceTask.from_dict(
                {"request_id": "../escape", "scene": "warehouse_normal", "seed": 1}
            )
        with self.assertRaises(ValueError):
            ServiceTask.from_dict({"request_id": "job-2", "scene": "unknown", "seed": 1})

    def test_service_response_write_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response_job.json"
            atomic_write_json(path, {"status": "complete"})
            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "status": "complete"\n}')
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_windows_utf8_bom_task_is_parseable(self):
        raw = '\ufeff{"request_id":"job-1","scene":"warehouse_normal","seed":40000}'
        value = json.loads(raw.encode("utf-8").decode("utf-8-sig"))
        self.assertEqual(ServiceTask.from_dict(value).request_id, "job-1")


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
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return completion

    completions = types.SimpleNamespace(create=create)
    chat = types.SimpleNamespace(completions=completions)
    return types.SimpleNamespace(chat=chat, calls=calls)


# ---- Payload --------------------------------------------------------------
class PayloadTest(unittest.TestCase):
    def test_multiview_negative_requires_package_in_every_view(self):
        complete = visual_observations_for_views(
            {"overhead": ["package"], "wrist": ["package"]}
        )
        incomplete = visual_observations_for_views(
            {"overhead": ["package"], "wrist": ["left_tray"]}
        )
        self.assertTrue(complete["inspection_complete"])
        self.assertFalse(complete["barcode_detected"])
        self.assertFalse(incomplete["inspection_complete"])

    def test_multiview_positive_records_camera_provenance(self):
        observations = visual_observations_for_views(
            {"overhead": ["package"], "wrist": ["package", "damage_mark"]}
        )
        self.assertTrue(observations["damage_mark_detected"])
        self.assertEqual(observations["damage_mark_detected_by"], ["wrist"])

    def test_low_confidence_damage_is_not_a_positive(self):
        observations = visual_observations_for_views(
            {
                "overhead": [types.SimpleNamespace(label="package", confidence=0.9)],
                "damage": [
                    types.SimpleNamespace(label="package", confidence=0.9),
                    types.SimpleNamespace(label="damage_mark", confidence=0.3),
                ],
            }
        )
        self.assertFalse(observations["damage_mark_detected"])
        self.assertEqual(observations["damage_mark_max_confidence"], 0.3)

    def test_scene_graph_smaller_than_full_json(self):
        graph, _ = _sample_graph()
        full = build_payload(CompressionLevel.FULL_JSON, graph, "pick")
        compact = build_payload(CompressionLevel.SCENE_GRAPH, graph, "pick")
        self.assertGreater(full.num_bytes, compact.num_bytes)
        self.assertEqual(compact.level, CompressionLevel.SCENE_GRAPH)
        observations = compact.content["scene"]["visual_observations"]
        self.assertFalse(observations["barcode_detected"])
        self.assertTrue(observations["package_detected"])

    def test_raw_image_requires_frame(self):
        graph, _ = _sample_graph()
        with self.assertRaises(ValueError):
            build_payload(CompressionLevel.RAW_IMAGE, graph, "pick", frame=None)


# ---- Planner --------------------------------------------------------------
class PlannerTest(unittest.TestCase):
    def test_warehouse_default_instruction_targets_conveyor(self):
        instruction = instruction_for_scenario("warehouse_normal")
        self.assertIn("conveyor", instruction.lower())
        graph = {
            "objects": [
                {"id": "package_0", "label": "package"},
                {"id": "barcode_0", "label": "barcode"},
            ],
            "summary": [],
        }
        target = HeuristicPlanner().plan(
            build_payload(CompressionLevel.SCENE_GRAPH, graph, instruction)
        )
        self.assertEqual(target.destination, "conveyor")

    def test_heuristic_prefers_package_over_infrastructure(self):
        graph = {
            "objects": [
                {"id": "outbound_bin_0", "label": "outbound_bin"},
                {"id": "package_0", "label": "package"},
                {"id": "conveyor_0", "label": "conveyor"},
            ],
            "summary": [],
        }
        payload = build_payload(CompressionLevel.SCENE_GRAPH, graph, "route normally")
        self.assertEqual(HeuristicPlanner().plan(payload).target_object, "package")

    def test_heuristic_routes_by_keyword(self):
        graph, _ = _sample_graph()
        payload = build_payload(CompressionLevel.FULL_JSON, graph, "reject the damaged package")
        target = HeuristicPlanner().plan(payload)
        self.assertEqual(target.destination, "rejection_tray")
        self.assertEqual(target.target_object, "package")
        self.assertEqual(target.source, "heuristic")

    def test_gpt_planner_parses_injected_client(self):
        graph, _ = _sample_graph()
        payload = build_payload(CompressionLevel.SCENE_GRAPH, graph, "pick")
        client = _fake_openai_client(
            '{"command":"REROUTE","target_object":"sample","destination":"left_tray",'
            '"gripper":"close","confidence":0.9,"evidence":["barcode absent"],'
            '"reasoning":"ok"}'
        )
        target = GptPlanner(client=client).plan(payload)
        self.assertEqual(target.destination, "inspection_tray")
        self.assertEqual(target.gripper, "close")
        self.assertEqual(target.source, "gpt-4o-mini")
        self.assertEqual(target.command, "REROUTE")
        self.assertEqual(target.confidence, 0.9)
        self.assertEqual(client.calls[0]["max_completion_tokens"], MAX_COMPLETION_TOKENS)


# ---- Controller -----------------------------------------------------------
class ControllerTest(unittest.TestCase):
    def test_pick_carry_release_sequence(self):
        env = _FakeEnv()
        controller = ScriptedController(env)
        target = ActionTarget("sample", "inspection_tray", "open", "r", "test")
        result = controller.execute(target)
        self.assertEqual(result.steps, 4)
        grippers = [a.values[5] for a in env.calls]
        self.assertEqual(grippers, [GRIPPER_OPEN, GRIPPER_CLOSE, GRIPPER_CLOSE, GRIPPER_OPEN])
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)

    def test_failed_release_reobserves_and_retries_once(self):
        env = _FakeEnv()

        def succeed_on_eighth(action):
            env.calls.append(action)
            env._steps += 1
            return _StepResult(success=(env._steps == 8))

        env.step = succeed_on_eighth
        result = ScriptedController(env, max_attempts=2).execute(
            ActionTarget("sample", "conveyor", "open", "r", "test")
        )
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.steps, 8)

    def test_planner_cannot_prevent_destination_release(self):
        env = _FakeEnv()
        controller = ScriptedController(env)
        target = ActionTarget("sample", "conveyor", "close", "r", "test")
        result = controller.execute(target)
        self.assertEqual(env.calls[-1].values[5], GRIPPER_OPEN)
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

    def test_obstacle_holds_then_stops_without_controller_actions(self):
        env = _FakeEnv()
        record = run_trial(
            env=env,
            detector=_FakeDetector(),
            planner=HeuristicPlanner(),
            controller=ScriptedController(env),
            channel=ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)),
            compression_level=CompressionLevel.SCENE_GRAPH,
            instruction="Stop because an unexpected obstacle blocks the robot path.",
            seed=1,
            episode_id=0,
            scene="unexpected_obstacle",
        )
        self.assertTrue(record.success)
        self.assertEqual(record.steps, 0)
        self.assertEqual(env.calls, [])
        self.assertEqual(record.extra["recovery_command"], "STOP")
        self.assertEqual(record.extra["obstacle_rechecks"], 3)
        self.assertEqual(record.extra["joint_commands_executed"], 0)


if __name__ == "__main__":
    unittest.main()
