from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import Action, Observation
from .so101_env import SO101MuJoCoEnvironment

FPS = 20
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
STEPS_PER_WAYPOINT = 6


@dataclass(frozen=True)
class DemonstrationResult:
    episode_id: int
    success: bool
    steps: int
    instruction: str
    output_dir: Path


class ScriptedIKPickPlacePolicy:
    """Waypoint policy used only to generate synthetic demonstrations."""

    def __init__(
        self,
        environment: SO101MuJoCoEnvironment,
        *,
        variation_seed: int | None = None,
        vary_speed: bool = False,
    ) -> None:
        self._environment = environment
        self._variation_seed = variation_seed
        self._vary_speed = vary_speed
        self._phase = 0
        self._segment_step = 0
        self._steps_per_waypoint = STEPS_PER_WAYPOINT
        self._sample: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._waypoints: tuple[tuple[np.ndarray, float], ...] = ()
        self._variation: dict[str, object] = {"enabled": False}
        self._segment_start: np.ndarray | None = None
        self._segment_target: np.ndarray | None = None

    def reset(self, observation: Observation) -> None:
        self._phase = 0
        self._segment_step = 0
        self._sample = np.asarray(observation.metadata["sample_position"])
        self._target = np.asarray(observation.metadata["target_position"])
        self._build_waypoints()
        self._segment_start = None
        self._segment_target = None

    def _build_waypoints(self) -> None:
        assert self._sample is not None and self._target is not None
        if self._variation_seed is None:
            approach_offset = np.zeros(3)
            transfer_offset = np.zeros(3)
            approach_height = 0.16
            lift_height = 0.18
            grasp_height = 0.015
            place_height = 0.025
        else:
            rng = np.random.default_rng(self._variation_seed)
            approach_offset = np.array(
                (rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), 0.0)
            )
            transfer_offset = np.array(
                (rng.uniform(-0.05, 0.05), rng.uniform(-0.04, 0.04), 0.0)
            )
            approach_height = float(rng.uniform(0.14, 0.20))
            lift_height = float(rng.uniform(0.16, 0.21))
            grasp_height = float(rng.uniform(0.012, 0.022))
            place_height = float(rng.uniform(0.020, 0.035))
            if self._vary_speed:
                self._steps_per_waypoint = int(rng.choice((5, 6)))

        self._waypoints = (
            (
                self._sample
                + approach_offset
                + np.array((0.0, 0.0, approach_height)),
                0.020,
            ),
            (self._sample + (0.0, 0.0, grasp_height), 0.020),
            (self._sample + (0.0, 0.0, grasp_height), 0.002),
            (self._sample + (0.0, 0.0, lift_height), 0.002),
            (
                self._target
                + transfer_offset
                + np.array((0.0, 0.0, lift_height)),
                0.002,
            ),
            (self._target + (0.0, 0.0, place_height), 0.002),
            (self._target + (0.0, 0.0, place_height), 0.020),
            (self._target + (0.0, 0.0, lift_height), 0.020),
        )
        self._variation = {
            "enabled": self._variation_seed is not None,
            "seed": self._variation_seed,
            "vary_speed": self._vary_speed,
            "steps_per_waypoint": self._steps_per_waypoint,
            "approach_offset": approach_offset.tolist(),
            "transfer_offset": transfer_offset.tolist(),
            "approach_height": approach_height,
            "lift_height": lift_height,
            "grasp_height": grasp_height,
            "place_height": place_height,
        }

    def action(self) -> Action:
        if self._phase >= len(self._waypoints):
            raise StopIteration("Demonstration trajectory is complete.")
        target, gripper = self._waypoints[self._phase]
        if self._segment_step == 0:
            self._segment_start = np.asarray(self._environment.joint_positions)
            self._segment_target = np.asarray(
                self._environment.solve_ik(target, gripper=gripper)
            )
        assert self._segment_start is not None and self._segment_target is not None
        alpha = (self._segment_step + 1) / self._steps_per_waypoint
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        joints = self._segment_start + (
            self._segment_target - self._segment_start
        ) * smooth
        self._segment_step += 1
        if self._segment_step == self._steps_per_waypoint:
            self._phase += 1
            self._segment_step = 0
        return Action(
            "joint_position",
            values=tuple(float(value) for value in joints),
        )

    @property
    def total_steps(self) -> int:
        return 8 * self._steps_per_waypoint

    @property
    def variation(self) -> dict[str, object]:
        return self._variation.copy()


def collect_episode(
    environment: SO101MuJoCoEnvironment,
    *,
    episode_id: int,
    seed: int,
    output_root: Path,
    variation_seed: int | None = None,
    vary_speed: bool = False,
) -> DemonstrationResult:
    probe = environment.reset(seed=seed, instruction="")
    target_name = str(probe.metadata["target_name"])
    instruction = f"Pick up the red sample and place it in the {target_name}."
    observation = environment.reset(seed=seed, instruction=instruction)
    initial_sample_position = list(observation.metadata["sample_position"])
    policy = ScriptedIKPickPlacePolicy(
        environment,
        variation_seed=variation_seed,
        vary_speed=vary_speed,
    )
    policy.reset(observation)
    frames_overhead = []
    frames_wrist = []
    states = []
    actions = []
    success = False

    for _ in range(policy.total_steps):
        action = policy.action()
        frames_overhead.append(
            environment.capture_rgb(
                camera="overhead",
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
            )
        )
        frames_wrist.append(
            environment.capture_rgb(
                camera="wrist",
                width=FRAME_WIDTH,
                height=FRAME_HEIGHT,
            )
        )
        states.append(observation.robot_state)
        actions.append(action.values)
        transition = environment.step(action)
        observation = transition.observation
        success = bool(transition.info["success"])

    episode_dir = output_root / f"episode_{episode_id:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)
    np.save(episode_dir / "observation_overhead.npy", np.asarray(frames_overhead))
    np.save(episode_dir / "observation_wrist.npy", np.asarray(frames_wrist))
    np.save(episode_dir / "observation_state.npy", np.asarray(states))
    np.save(episode_dir / "action.npy", np.asarray(actions))
    metadata = {
        "episode_id": episode_id,
        "seed": seed,
        "instruction": instruction,
        "initial_sample_position": initial_sample_position,
        "target_name": target_name,
        "success": success,
        "joint_names": environment.JOINT_NAMES,
        "frames": len(actions),
        "fps": FPS,
        "image_size": [FRAME_HEIGHT, FRAME_WIDTH, 3],
        "format": "architecture_a_raw_v1",
        "trajectory_variation": policy.variation,
    }
    (episode_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return DemonstrationResult(
        episode_id=episode_id,
        success=success,
        steps=len(actions),
        instruction=instruction,
        output_dir=episode_dir,
    )


def select_hard_case_seeds(
    environment: SO101MuJoCoEnvironment,
    *,
    start_seed: int,
    count: int,
) -> list[int]:
    """Balance low-depth starts across target side and object x-sign."""

    buckets: dict[tuple[str, str], list[int]] = {
        (target, side): []
        for target in ("left_tray", "right_tray")
        for side in ("negative_x", "positive_x")
    }
    per_bucket = (count + len(buckets) - 1) // len(buckets)
    candidate = start_seed
    while sum(len(values) for values in buckets.values()) < count:
        observation = environment.reset(seed=candidate, instruction="")
        sample = np.asarray(observation.metadata["sample_position"])
        target = str(observation.metadata["target_name"])
        side = "negative_x" if sample[0] < 0 else "positive_x"
        bucket = buckets[(target, side)]
        if sample[1] <= 0.11 and len(bucket) < per_bucket:
            bucket.append(candidate)
        candidate += 1
        if candidate - start_seed > 100_000:
            raise RuntimeError("Could not find enough balanced hard-case seeds.")

    selected: list[int] = []
    for index in range(per_bucket):
        for key in sorted(buckets):
            if index < len(buckets[key]) and len(selected) < count:
                selected.append(buckets[key][index])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect scripted SO-101 demos.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--diverse",
        action="store_true",
        help="Vary approach paths and heights per episode.",
    )
    parser.add_argument(
        "--vary-speed",
        action="store_true",
        help="Also vary trajectory speed; this may blur action-chunk timing.",
    )
    parser.add_argument(
        "--hard-cases",
        action="store_true",
        help="Select balanced low-depth object starts instead of consecutive seeds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/demonstrations"),
    )
    args = parser.parse_args()
    environment = SO101MuJoCoEnvironment(
        gui=False,
        realtime=False,
        observation_images=False,
        kinematic_control=True,
    )
    try:
        seeds = (
            select_hard_case_seeds(
                environment,
                start_seed=args.seed,
                count=args.episodes,
            )
            if args.hard_cases
            else [args.seed + index for index in range(args.episodes)]
        )
        results = [
            collect_episode(
                environment,
                episode_id=index,
                seed=seed,
                output_root=args.output,
                variation_seed=seed if args.diverse else None,
                vary_speed=args.vary_speed,
            )
            for index, seed in enumerate(seeds)
        ]
    finally:
        environment.close()
    summary = {
        "episodes": len(results),
        "successful": sum(result.success for result in results),
        "success_rate": (
            sum(result.success for result in results) / len(results)
            if results
            else 0.0
        ),
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
