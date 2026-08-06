from __future__ import annotations

import types
import unittest

import numpy as np

from architecture_a.semantic_executor import ArchitectureASmolVlaController
from architecture_b.planner import ActionTarget
from shared.smolvla_runtime import SmolVlaRuntime


class _Predictor:
    def __init__(self) -> None:
        self.instructions: list[str] = []

    def predict_chunk(self, overhead, wrist, robot_state, instruction):
        self.instructions.append(instruction)
        return np.zeros((2, 6), dtype=float)


class _Env:
    joint_positions = (0.0,) * 6

    def __init__(self, preview: bool = True) -> None:
        self.preview = preview
        self.steps = 0

    def capture_rgb(self, *, camera="overhead", width=320, height=240):
        return np.zeros((height, width, 3), dtype=np.uint8)

    def preview_action_success(self, chunk):
        return self.preview

    def step(self, action):
        self.steps += 1
        return types.SimpleNamespace(
            observation=types.SimpleNamespace(robot_state=(0.0,) * 6),
            info={"success": True, "failure_reason": None},
            terminated=True,
            truncated=False,
        )


class SemanticExecutorTest(unittest.TestCase):
    def test_semantic_destination_uses_color_aligned_local_instruction(self):
        predictor = _Predictor()
        env = _Env()
        controller = ArchitectureASmolVlaController(
            env, SmolVlaRuntime(predictor=predictor)
        )
        result = controller.execute(
            ActionTarget("package", "inspection_tray", "open", "test", "fake")
        )
        self.assertTrue(result.success)
        self.assertEqual(env.steps, 1)
        self.assertIn("blue inspection tray", predictor.instructions[0])

    def test_failed_preview_never_executes_joint_commands(self):
        env = _Env(preview=False)
        controller = ArchitectureASmolVlaController(
            env, SmolVlaRuntime(predictor=_Predictor()), max_attempts=3
        )
        result = controller.execute(
            ActionTarget("package", "conveyor", "open", "test", "fake")
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "preexecution_outcome_validation_failed")
        self.assertEqual(result.attempts, 3)
        self.assertEqual(env.steps, 0)

    def test_unknown_destination_is_rejected_locally(self):
        env = _Env()
        controller = ArchitectureASmolVlaController(
            env, SmolVlaRuntime(predictor=_Predictor())
        )
        result = controller.execute(
            ActionTarget("package", "moon", "open", "test", "fake")
        )
        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, "unsupported_semantic_destination")
        self.assertEqual(env.steps, 0)

    def test_resumable_chunk_waits_for_advance(self):
        env = _Env()
        controller = ArchitectureASmolVlaController(
            env, SmolVlaRuntime(predictor=_Predictor())
        )
        session = controller.start(
            ActionTarget("package", "conveyor", "open", "test", "fake")
        )
        self.assertEqual(env.steps, 0)
        controller.advance(session)
        self.assertEqual(session.phase, "complete")
        self.assertTrue(session.result.success)


if __name__ == "__main__":
    unittest.main()
