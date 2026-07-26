import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from architecture_a.mock import MockLocalPolicy, MockPickPlaceEnvironment
from architecture_a.runner import ArchitectureA
from architecture_a.smolvla_adapter import (
    ModelOutput,
    SmolVLAFeatures,
    SmolVLAPolicyAdapter,
)


class FakeSmolVLABackend:
    def __init__(self, action=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5)) -> None:
        self.action = action
        self.last_batch = None

    def reset(self) -> None:
        pass

    def predict(self, batch):
        self.last_batch = batch
        return ModelOutput(action=self.action, confidence=0.8)


class ArchitectureATest(unittest.TestCase):
    def test_vla_confidence_detects_unstable_and_disagreeing_chunks(self) -> None:
        from architecture_a.vla_confidence import estimate_vla_confidence

        low = np.full(6, -1.0)
        high = np.full(6, 1.0)
        stable = np.zeros((4, 6))
        unstable = stable.copy()
        unstable[2] = 1.0
        stable_score = estimate_vla_confidence(stable, action_low=low, action_high=high)
        unstable_score = estimate_vla_confidence(
            unstable, action_low=low, action_high=high, previous_chunk=stable
        )
        self.assertEqual(stable_score.confidence, 1.0)
        self.assertLess(unstable_score.chunk_smoothness, stable_score.chunk_smoothness)
        self.assertLess(unstable_score.temporal_consistency, 1.0)

    def test_so101_hardware_adapter_is_safe_in_dry_run(self) -> None:
        from shared.robots import (
            DryRunTransport,
            JointCalibration,
            SO101ActionAdapter,
        )

        transport = DryRunTransport()
        calibration = {
            name: JointCalibration(minimum=-2.0, maximum=2.0)
            for name in (
                "shoulder_pan",
                "shoulder_lift",
                "elbow_flex",
                "wrist_flex",
                "wrist_roll",
                "gripper",
            )
        }
        adapter = SO101ActionAdapter(
            transport,
            calibration=calibration,
            dry_run=True,
        )
        mapped = adapter.command((0.0, 0.1, 0.2, 0.3, 0.4, 0.5))

        self.assertEqual(mapped["elbow_flex"], 0.2)
        self.assertEqual(transport.commands, [])
        adapter.emergency_stop()
        with self.assertRaisesRegex(RuntimeError, "Emergency stop"):
            adapter.command((0.0,) * 6)

    def test_so101_observation_adapter_maps_sensors_to_model_space(self) -> None:
        from shared.robots import (
            JointCalibration,
            SO101ObservationAdapter,
        )

        class Cameras:
            def capture_rgb(self, camera: str) -> np.ndarray:
                value = 10 if camera == "overhead" else 20
                return np.full((48, 64, 3), value, dtype=np.uint8)

        class State:
            def read_joint_positions(self) -> dict[str, float]:
                return {
                    "shoulder_pan": 1.0,
                    "shoulder_lift": 1.0,
                    "elbow_flex": 1.0,
                    "wrist_flex": 1.0,
                    "wrist_roll": 1.0,
                    "gripper": 1.0,
                }

        calibration = {
            name: JointCalibration(
                offset=1.0,
                scale=2.0,
                minimum=-10.0,
                maximum=10.0,
            )
            for name in State().read_joint_positions()
        }
        packet = SO101ObservationAdapter(
            Cameras(),
            State(),
            calibration=calibration,
        ).read(task="Pick up the sample.")

        self.assertEqual(packet.state, (0.0,) * 6)
        self.assertEqual(packet.overhead.shape, (48, 64, 3))
        self.assertEqual(
            packet.as_lerobot_input()["observation.state"].shape,
            (6,),
        )

    def test_so101_dry_run_controller_completes_safe_cycle(self) -> None:
        from shared.robots import (
            DryRunTransport,
            JointCalibration,
            SO101ActionAdapter,
            SO101DryRunController,
            SO101ObservationAdapter,
        )

        names = (
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        )
        calibration = {
            name: JointCalibration(minimum=-2.0, maximum=2.0)
            for name in names
        }

        class Sensors:
            def capture_rgb(self, camera: str) -> np.ndarray:
                return np.zeros((48, 64, 3), dtype=np.uint8)

            def read_joint_positions(self) -> dict[str, float]:
                return {name: 0.0 for name in names}

        class Policy:
            def predict(self, observation: object) -> tuple[float, ...]:
                return (0.1,) * 6

        sensors = Sensors()
        controller = SO101DryRunController(
            SO101ObservationAdapter(
                sensors,
                sensors,
                calibration=calibration,
            ),
            Policy(),
            SO101ActionAdapter(
                DryRunTransport(),
                calibration=calibration,
                dry_run=True,
            ),
        )
        result = controller.cycle(task="Pick up the sample.")

        self.assertFalse(result.escalation_recommended)
        self.assertEqual(result.command["shoulder_pan"], 0.1)

    def test_shared_so101_environment_uses_architecture_a_implementation(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment
        from shared.environments import (
            SO101MuJoCoEnvironment as SharedSO101Environment,
        )

        self.assertIs(SharedSO101Environment, SO101MuJoCoEnvironment)

    def setUp(self) -> None:
        self.runner = ArchitectureA(
            MockPickPlaceEnvironment(),
            MockLocalPolicy(),
            max_steps=4,
        )

    def test_supported_instruction_succeeds_without_escalation(self) -> None:
        result = self.runner.run_episode(
            episode_id=0,
            seed=2,
            instruction="Pick up the red_sample and place it in the left_tray.",
        )

        environment = MockPickPlaceEnvironment()
        observation = environment.reset(seed=2, instruction="")
        expected_object = observation.metadata["visible_objects"][0]
        expected_target = observation.metadata["visible_targets"][0]
        result = self.runner.run_episode(
            episode_id=0,
            seed=2,
            instruction=(
                f"Pick up the {expected_object} and place it in the {expected_target}."
            ),
        )

        self.assertTrue(result.success)
        self.assertEqual(result.steps, 2)
        self.assertFalse(result.escalation_recommended)

    def test_unknown_instruction_recommends_escalation(self) -> None:
        result = self.runner.run_episode(
            episode_id=1,
            seed=3,
            instruction="Inspect the sample for cracks.",
        )

        self.assertFalse(result.success)
        self.assertTrue(result.escalation_recommended)
        self.assertEqual(
            result.failure_reason,
            "instruction_outside_pick_place_skill",
        )

    def test_bad_grounding_recommends_escalation(self) -> None:
        result = self.runner.run_episode(
            episode_id=2,
            seed=4,
            instruction="Pick up the yellow_sample and place it in the left_tray.",
        )

        self.assertFalse(result.success)
        self.assertTrue(result.escalation_recommended)
        self.assertEqual(result.minimum_grounding_confidence, 0.0)

    def test_pybullet_mock_renders_headlessly(self) -> None:
        from architecture_a.pybullet_env import PyBulletMockEnvironment

        environment = PyBulletMockEnvironment(gui=False, realtime=False)
        try:
            observation = environment.reset(seed=2, instruction="")
            pixels = environment.capture_rgb(width=64, height=48)
            self.assertEqual(len(pixels), 64 * 48 * 4)
            self.assertIn("visible_objects", observation.metadata)
        finally:
            environment.close()

    def test_mujoco_environment_completes_pick_place(self) -> None:
        from architecture_a.mujoco_env import MuJoCoPickPlaceEnvironment

        environment = MuJoCoPickPlaceEnvironment(gui=False, realtime=False)
        try:
            observation = environment.reset(seed=2, instruction="")
            object_name = observation.metadata["visible_objects"][0]
            target_name = observation.metadata["visible_targets"][0]
            result = ArchitectureA(
                environment,
                MockLocalPolicy(),
                max_steps=4,
            ).run_episode(
                episode_id=0,
                seed=2,
                instruction=(
                    f"Pick up the {object_name} and place it in the {target_name}."
                ),
            )
            frame = environment.capture_rgb(width=64, height=48)
            self.assertTrue(result.success, result.failure_reason)
            self.assertEqual(frame.shape, (48, 64, 3))
        finally:
            environment.close()

    def test_smolvla_adapter_builds_lerobot_batch(self) -> None:
        from architecture_a.contracts import Observation

        backend = FakeSmolVLABackend()
        adapter = SmolVLAPolicyAdapter(backend)
        decision = adapter.decide(
            Observation(
                image=np.zeros((48, 64, 3), dtype=np.uint8),
                robot_state=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            ),
            "Pick up the red sample.",
        )

        self.assertEqual(decision.action.name, "joint_position")
        self.assertEqual(len(decision.action.values), 6)
        self.assertEqual(
            tuple(backend.last_batch["observation.image"].shape),
            (1, 3, 48, 64),
        )
        self.assertEqual(backend.last_batch["task"], ["Pick up the red sample."])

    def test_smolvla_adapter_rejects_wrong_embodiment(self) -> None:
        from architecture_a.contracts import Observation

        adapter = SmolVLAPolicyAdapter(
            FakeSmolVLABackend(action=(0.0, 0.1, 0.2, 0.3)),
            features=SmolVLAFeatures(action_dimension=6),
        )
        with self.assertRaisesRegex(ValueError, "action dimension mismatch"):
            adapter.decide(
                Observation(
                    image=np.zeros((8, 8, 3), dtype=np.uint8),
                    robot_state=(0.0,) * 6,
                ),
                "Pick up the sample.",
            )

    def test_so101_environment_has_six_joint_policy_boundary(self) -> None:
        from architecture_a.contracts import Action
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=True,
        )
        try:
            observation = environment.reset(seed=0, instruction="")
            transition = environment.step(
                Action(
                    "joint_position",
                    values=(0.2, -0.5, 0.8, 0.4, 0.1, 0.015),
                )
            )
            wrist = environment.capture_rgb(
                camera="wrist",
                width=64,
                height=48,
            )
            self.assertEqual(len(observation.robot_state), 6)
            self.assertEqual(
                observation.metadata["joint_names"],
                environment.JOINT_NAMES,
            )
            self.assertEqual(wrist.shape, (48, 64, 3))
            self.assertFalse(transition.terminated)
        finally:
            environment.close()

    def test_warehouse_environment_exports_missing_barcode_scene_graph(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            scenario="barcode_missing",
        )
        try:
            observation = environment.reset(seed=1010, instruction="Inspect package.")
            graph = environment.scene_graph()
            self.assertEqual(observation.metadata["problem"], "barcode_missing")
            self.assertEqual(graph["problem"], "barcode_missing")
            self.assertEqual(graph["objects"][0]["barcode"], "missing")
            self.assertEqual(graph["zones"]["left_tray_blue"], "manual inspection")
            self.assertNotIn("image", graph)
        finally:
            environment.close()

    def test_warehouse_normal_has_no_exception(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            scenario="warehouse_normal",
        )
        try:
            environment.reset(seed=1010, instruction="Move package to conveyor.")
            graph = environment.scene_graph()
            self.assertIsNone(graph["problem"])
            self.assertEqual(graph["objects"][0]["barcode"], "present")
            self.assertEqual(graph["objects"][0]["condition"], "intact")
            self.assertEqual(
                graph["zones"]["outbound_bin_green"],
                "completed normal shipments",
            )
        finally:
            environment.close()

    def test_warehouse_normal_targets_conveyor(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            observation_images=False,
            scenario="warehouse_normal",
        )
        try:
            observation = environment.reset(seed=1001, instruction="")
            self.assertEqual(observation.metadata["target_name"], "conveyor")
            np.testing.assert_allclose(
                observation.metadata["target_position"],
                environment.CONVEYOR_POSITION,
            )
        finally:
            environment.close()

    def test_warehouse_obstacle_stops_local_execution(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            scenario="unexpected_obstacle",
        )
        try:
            environment.reset(seed=1010, instruction="Pick package.")
            graph = environment.scene_graph()
            self.assertEqual(graph["problem"], "unexpected_obstacle")
            self.assertTrue(graph["robot"]["blocked"])
            self.assertIsNone(graph["robot"]["holding"])
            self.assertEqual(graph["objects"][-1]["id"], "obstacle_01")
        finally:
            environment.close()

    def test_action_chunk_predicts_obstacle_contact_before_execution(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            kinematic_control=True,
            scenario="unexpected_obstacle",
        )
        try:
            observation = environment.reset(seed=1010, instruction="")
            sample = np.asarray(observation.metadata["sample_position"])
            chunk = np.asarray(
                [
                    environment.solve_ik(sample + (0.0, 0.0, 0.14), gripper=0.02),
                    environment.solve_ik(sample + (0.0, 0.0, 0.055), gripper=0.02),
                ]
            )
            self.assertEqual(environment.predict_obstacle_collision(chunk), 1)
        finally:
            environment.close()

    def test_warehouse_environment_exports_damaged_package_scene_graph(self) -> None:
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            scenario="package_damaged",
        )
        try:
            environment.reset(seed=1010, instruction="Inspect package.")
            graph = environment.scene_graph()
            self.assertEqual(graph["problem"], "package_damaged")
            self.assertEqual(graph["objects"][0]["condition"], "damaged")
        finally:
            environment.close()

    def test_so101_ik_reaches_workspace_target(self) -> None:
        from architecture_a.contracts import Action
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            kinematic_control=True,
        )
        try:
            observation = environment.reset(seed=4, instruction="")
            target = np.asarray(observation.metadata["sample_position"]) + (
                0.0,
                0.0,
                0.15,
            )
            joints = environment.solve_ik(target, gripper=0.02)
            environment.step(Action("joint_position", values=joints))
            error = np.linalg.norm(environment.end_effector_position - target)
            self.assertLess(error, 0.04)
        finally:
            environment.close()

    def test_demonstration_recorder_aligns_modalities(self) -> None:
        from architecture_a.demonstrations import collect_episode
        from architecture_a.so101_env import SO101MuJoCoEnvironment

        environment = SO101MuJoCoEnvironment(
            gui=False,
            realtime=False,
            observation_images=False,
            kinematic_control=True,
        )
        try:
            with TemporaryDirectory() as directory:
                result = collect_episode(
                    environment,
                    episode_id=0,
                    seed=100,
                    output_root=Path(directory),
                )
                overhead = np.load(
                    result.output_dir / "observation_overhead.npy"
                )
                wrist = np.load(result.output_dir / "observation_wrist.npy")
                states = np.load(result.output_dir / "observation_state.npy")
                actions = np.load(result.output_dir / "action.npy")
                self.assertTrue(result.success)
                self.assertEqual(overhead.shape, (48, 240, 320, 3))
                self.assertEqual(wrist.shape, overhead.shape)
                self.assertEqual(states.shape, (48, 6))
                self.assertEqual(actions.shape, (48, 6))
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()
