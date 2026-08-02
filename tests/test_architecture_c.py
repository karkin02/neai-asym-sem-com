"""Standalone unit tests for Architecture C (routing gate + hybrid run_trial).

All external systems are faked, so these run on any environment with numpy.
"""

from __future__ import annotations

import types
import unittest
from pathlib import Path

import numpy as np

from architecture_b.channel import ChannelConfig, ChannelSimulator
from architecture_b.controller import ScriptedController
from architecture_b.payload import CompressionLevel
from architecture_b.planner import HeuristicPlanner
from architecture_c.router import RoutingConfig, decide_route, is_recognized
from architecture_c.runner import checkpoint_for_scene, run_trial
from shared.perception import Detection
from shared.smolvla_runtime import DEFAULT_VLM, SmolVlaRuntime


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
            observation=types.SimpleNamespace(robot_state=(0.0,) * 6),
            info={"success": success, "failure_reason": None if success else "placement_failed"},
            terminated=success,
            truncated=False,
        )


class _FakeDetector:
    def detect(self, frame):
        return [Detection("sample", 0.9, (10, 40, 30, 60)), Detection("left_tray", 0.8, (200, 40, 240, 60))]


class _EmptyDetector:
    def detect(self, frame):
        return []


class _PreviewRejectEnv(_FakeEnv):
    def preview_action_success(self, chunk):
        return False


class _PreviewAcceptsSecondEnv(_FakeEnv):
    def __init__(self, success_on_step=1):
        super().__init__(success_on_step=success_on_step)
        self.preview_calls = 0

    def preview_action_success(self, chunk):
        self.preview_calls += 1
        return self.preview_calls >= 2


class _WristFallbackDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls == 1:
            return []
        return [Detection("package", 0.8, (100, 80, 140, 130))]


class _FakeGrounder:
    def __init__(self, confidence, margin=0.1):
        self._confidence = confidence
        self._margin = margin

    def score(self, expression, crops):
        self.crop_shapes = [crop.shape for crop in crops]
        return types.SimpleNamespace(confidence=self._confidence, margin=self._margin)


class _FakePredictor:
    def __init__(self, chunk):
        self._chunk = chunk

    def predict_chunk(self, overhead, wrist, robot_state, instruction):
        return self._chunk


INSTRUCTION = "Pick up the sample and place it in the left tray."


def _run(clip_confidence, runtime, channel=None, detector=None, grounder=None):
    return run_trial(
        env=_FakeEnv(success_on_step=1),
        detector=detector or _FakeDetector(),
        grounder=grounder or _FakeGrounder(clip_confidence),
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
    def test_checkpoint_for_scene_uses_architecture_a_manifest(self):
        selected = checkpoint_for_scene("warehouse_normal", None)
        self.assertEqual(
            selected,
            Path(
                "outputs/train/warehouse_v3_150/"
                "checkpoints/002000/pretrained_model"
            ),
        )

    def test_explicit_checkpoint_overrides_manifest(self):
        self.assertEqual(
            checkpoint_for_scene("warehouse_normal", Path("custom/model")),
            Path("custom/model"),
        )
    def test_default_vlm_is_portable_hugging_face_model_id(self):
        runtime = SmolVlaRuntime(checkpoint="does/not/exist")
        self.assertEqual(DEFAULT_VLM, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        self.assertEqual(runtime.vlm, DEFAULT_VLM)

    def test_high_confidence_recognized_routes_local(self):
        d = decide_route(INSTRUCTION, 0.9, ["sample"], RoutingConfig(clip_threshold=0.6))
        self.assertEqual(d.route, "local")
        self.assertFalse(d.escalated)

    def test_calibrated_default_margin_accepts_heldout_operating_point(self):
        d = decide_route(
            INSTRUCTION,
            0.9,
            ["package"],
            RoutingConfig(),
            clip_margin=0.01,
        )
        self.assertEqual(d.route, "local")

        rejected = decide_route(
            INSTRUCTION,
            0.9,
            ["package"],
            RoutingConfig(),
            clip_margin=-0.024,
        )
        self.assertEqual(rejected.route, "escalate")

    def test_low_confidence_escalates(self):
        d = decide_route(INSTRUCTION, 0.2, ["sample"], RoutingConfig(clip_threshold=0.6))
        self.assertEqual(d.route, "escalate")
        self.assertTrue(d.escalated)
        self.assertFalse(d.clip_ok)

    def test_ambiguous_clip_margin_escalates(self):
        d = decide_route(
            INSTRUCTION,
            0.9,
            ["package", "left_tray"],
            RoutingConfig(clip_threshold=0.6, clip_margin_threshold=0.02),
            clip_margin=0.005,
        )
        self.assertEqual(d.route, "escalate")
        self.assertIn("margin", d.reason)

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
    def test_wrist_fallback_when_overhead_misses_package(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        detector = _WristFallbackDetector()
        record = _run(0.8, runtime, detector=detector)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(record.extra["perception_view"], "wrist_fallback")

    def test_no_detections_uses_full_frame_for_clip(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        grounder = _FakeGrounder(0.8)
        record = _run(
            clip_confidence=0.8,
            runtime=runtime,
            detector=_EmptyDetector(),
            grounder=grounder,
        )
        self.assertEqual(grounder.crop_shapes, [(240, 320, 3)])
        self.assertEqual(record.route, "escalated")
        self.assertIn("no YOLO detections", record.extra["routing_reason"])

    def test_local_route_zero_network_cost(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        record = _run(clip_confidence=0.95, runtime=runtime)
        self.assertEqual(record.route, "local")
        self.assertFalse(record.escalated)
        self.assertEqual(record.network_payload_bytes, 0)
        self.assertTrue(record.success)

    def test_failed_local_execution_escalates_to_recovery(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((2, 6))))
        env = _FakeEnv(success_on_step=99)
        record = run_trial(
            env=env,
            detector=_FakeDetector(),
            grounder=_FakeGrounder(0.95),
            local_runtime=runtime,
            planner=HeuristicPlanner(),
            controller=ScriptedController(env),
            channel=ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)),
            compression_level=CompressionLevel.SCENE_GRAPH,
            instruction=INSTRUCTION,
            seed=1,
            episode_id=0,
            scene="warehouse_normal",
            routing_config=RoutingConfig(clip_threshold=0.6),
            max_local_steps=2,
        )
        self.assertEqual(record.route, "escalated")
        self.assertTrue(record.escalated)
        self.assertGreater(record.network_payload_bytes, 0)
        self.assertEqual(record.extra["local_failure_reason"], "placement_failed")
        self.assertIn("local execution failed", record.extra["routing_reason"])

    def test_failed_forward_preview_escalates_before_local_execution(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((2, 6))))
        env = _PreviewRejectEnv(success_on_step=4)
        record = run_trial(
            env=env,
            detector=_FakeDetector(),
            grounder=_FakeGrounder(0.95),
            local_runtime=runtime,
            planner=HeuristicPlanner(),
            controller=ScriptedController(env),
            channel=ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)),
            compression_level=CompressionLevel.SCENE_GRAPH,
            instruction=INSTRUCTION,
            seed=1,
            episode_id=0,
            scene="warehouse_normal",
        )
        self.assertTrue(record.escalated)
        self.assertEqual(record.extra["local_steps"], 0)
        self.assertEqual(
            record.extra["local_failure_reason"],
            "preexecution_outcome_validation_failed",
        )

    def test_failed_preview_retries_fresh_chunk_before_escalation(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((2, 6))))
        env = _PreviewAcceptsSecondEnv(success_on_step=1)
        record = run_trial(
            env=env,
            detector=_FakeDetector(),
            grounder=_FakeGrounder(0.95),
            local_runtime=runtime,
            planner=HeuristicPlanner(),
            controller=ScriptedController(env),
            channel=ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)),
            compression_level=CompressionLevel.SCENE_GRAPH,
            instruction=INSTRUCTION,
            seed=1,
            episode_id=0,
            scene="warehouse_normal",
            max_local_attempts=2,
        )
        self.assertTrue(record.success)
        self.assertEqual(record.route, "local")
        self.assertEqual(record.extra["local_attempts"], 2)
        self.assertEqual(env.preview_calls, 2)

    def test_local_route_replans_with_fresh_chunk(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((2, 6))))
        env = _FakeEnv(success_on_step=3)
        record = run_trial(
            env=env,
            detector=_FakeDetector(),
            grounder=_FakeGrounder(0.95),
            local_runtime=runtime,
            planner=HeuristicPlanner(),
            controller=ScriptedController(env),
            channel=ChannelSimulator(ChannelConfig("clean", 1e9, 0.0)),
            compression_level=CompressionLevel.SCENE_GRAPH,
            instruction=INSTRUCTION,
            seed=1,
            episode_id=0,
            scene="warehouse_normal",
            max_local_steps=2,
            max_local_attempts=2,
        )
        self.assertTrue(record.success)
        self.assertEqual(record.route, "local")
        self.assertEqual(record.steps, 3)
        self.assertEqual(record.extra["local_attempts"], 2)

    def test_low_confidence_escalates_with_payload(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        record = _run(clip_confidence=0.10, runtime=runtime)
        self.assertEqual(record.route, "escalated")
        self.assertTrue(record.escalated)
        self.assertGreater(record.network_payload_bytes, 0)

    def test_obstacle_escalation_holds_then_stops_without_joint_commands(self):
        runtime = SmolVlaRuntime(predictor=_FakePredictor(np.zeros((3, 6))))
        env = _FakeEnv(success_on_step=1)
        record = run_trial(
            env=env,
            detector=_EmptyDetector(),
            grounder=_FakeGrounder(0.10),
            local_runtime=runtime,
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
        self.assertTrue(record.escalated)
        self.assertEqual(record.steps, 0)
        self.assertEqual(record.extra["recovery_command"], "STOP")
        self.assertEqual(record.extra["obstacle_rechecks"], 3)
        self.assertEqual(record.extra["joint_commands_executed"], 0)

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
