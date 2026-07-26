from __future__ import annotations

import argparse
import json
from pathlib import Path

from .mock import MockLocalPolicy, MockPickPlaceEnvironment
from .runner import ArchitectureA


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Architecture A rollouts.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--render",
        choices=(
            "none",
            "pybullet",
            "pybullet-headless",
            "mujoco",
            "mujoco-headless",
        ),
        default="none",
        help="Render the deterministic mock task with PyBullet.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/a_results.jsonl"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    environment = _build_environment(args.render)
    runner = ArchitectureA(
        environment,
        MockLocalPolicy(),
        max_steps=4,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        with args.output.open("w", encoding="utf-8") as output:
            for episode_id in range(args.episodes):
                seed = args.seed + episode_id
                probe = MockPickPlaceEnvironment()
                observation = probe.reset(seed=seed, instruction="")
                object_name = observation.metadata["visible_objects"][0]
                target_name = observation.metadata["visible_targets"][0]
                instruction = (
                    f"Pick up the {object_name} and place it in the {target_name}."
                )
                result = runner.run_episode(
                    episode_id=episode_id,
                    seed=seed,
                    instruction=instruction,
                )
                results.append(result)
                output.write(json.dumps(result.as_dict()) + "\n")
    finally:
        close = getattr(environment, "close", None)
        if close is not None:
            close()

    successes = sum(result.success for result in results)
    escalations = sum(result.escalation_recommended for result in results)
    summary = {
        "episodes": len(results),
        "success_rate": successes / len(results) if results else 0.0,
        "escalation_rate": escalations / len(results) if results else 0.0,
        "output": str(args.output),
    }
    print(json.dumps(summary, indent=2))


def _build_environment(render_mode: str):
    if render_mode == "none":
        return MockPickPlaceEnvironment()

    if render_mode.startswith("mujoco"):
        from .mujoco_env import MuJoCoPickPlaceEnvironment

        return MuJoCoPickPlaceEnvironment(
            gui=render_mode == "mujoco",
            realtime=render_mode == "mujoco",
        )

    from .pybullet_env import PyBulletMockEnvironment

    return PyBulletMockEnvironment(
        gui=render_mode == "pybullet",
        realtime=render_mode == "pybullet",
    )


if __name__ == "__main__":
    main()
