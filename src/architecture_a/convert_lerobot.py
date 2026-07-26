from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .demonstrations import FPS
from .so101_env import SO101MuJoCoEnvironment


def dataset_features(
    image_shape: tuple[int, int, int],
    *,
    use_videos: bool,
) -> dict[str, dict[str, Any]]:
    height, width, channels = image_shape
    visual_dtype = "video" if use_videos else "image"
    return {
        "observation.images.overhead": {
            "dtype": visual_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist": {
            "dtype": visual_dtype,
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(SO101MuJoCoEnvironment.JOINT_NAMES),
        },
        "action": {
            "dtype": "float32",
            "shape": (6,),
            "names": list(SO101MuJoCoEnvironment.JOINT_NAMES),
        },
    }


def episode_paths(
    raw_roots: Path | Sequence[Path],
) -> list[Path]:
    roots = [raw_roots] if isinstance(raw_roots, Path) else list(raw_roots)
    episodes = [
        episode
        for root in roots
        for episode in sorted(root.glob("episode_*"))
    ]
    if not episodes:
        joined = ", ".join(str(root) for root in roots)
        raise ValueError(f"No episodes found under: {joined}.")
    return episodes


def inspect_raw_dataset(
    raw_root: Path | Sequence[Path],
    *,
    use_videos: bool = False,
) -> dict[str, Any]:
    episodes = episode_paths(raw_root)
    total_frames = 0
    successful = 0
    expected_shape: tuple[int, int, int] | None = None

    for episode in episodes:
        metadata = json.loads((episode / "metadata.json").read_text("utf-8"))
        overhead = np.load(episode / "observation_overhead.npy", mmap_mode="r")
        wrist = np.load(episode / "observation_wrist.npy", mmap_mode="r")
        states = np.load(episode / "observation_state.npy", mmap_mode="r")
        actions = np.load(episode / "action.npy", mmap_mode="r")
        frame_count = len(actions)
        if not (
            len(overhead) == len(wrist) == len(states) == frame_count
        ):
            raise ValueError(f"Misaligned modalities in {episode.name}.")
        if states.shape[1:] != (6,) or actions.shape[1:] != (6,):
            raise ValueError(f"Expected 6-D states and actions in {episode.name}.")
        image_shape = tuple(int(value) for value in overhead.shape[1:])
        if wrist.shape[1:] != overhead.shape[1:]:
            raise ValueError(f"Camera shapes differ in {episode.name}.")
        if expected_shape is None:
            expected_shape = image_shape
        elif image_shape != expected_shape:
            raise ValueError(f"Image shape changed in {episode.name}.")
        total_frames += frame_count
        successful += int(bool(metadata["success"]))

    assert expected_shape is not None
    return {
        "episodes": len(episodes),
        "successful_episodes": successful,
        "frames": total_frames,
        "fps": FPS,
        "image_shape": expected_shape,
        "storage": "videos" if use_videos else "images",
        "features": dataset_features(expected_shape, use_videos=use_videos),
    }


def convert(
    *,
    raw_root: Path | Sequence[Path],
    output_root: Path,
    repo_id: str,
    use_videos: bool = False,
) -> dict[str, Any]:
    summary = inspect_raw_dataset(raw_root, use_videos=use_videos)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as error:
        raise RuntimeError(
            "LeRobot's dataset dependencies are required for conversion. "
            "The raw dataset passed validation; install "
            "'lerobot[dataset]>=0.6,<0.7' and rerun this command."
        ) from error

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_root,
        fps=FPS,
        robot_type="so101_compatible_mujoco",
        features=summary["features"],
        use_videos=use_videos,
    )
    try:
        for episode in episode_paths(raw_root):
            metadata = json.loads((episode / "metadata.json").read_text("utf-8"))
            overhead = np.load(episode / "observation_overhead.npy")
            wrist = np.load(episode / "observation_wrist.npy")
            states = np.load(episode / "observation_state.npy")
            actions = np.load(episode / "action.npy")
            for index in range(len(actions)):
                dataset.add_frame(
                    {
                        "observation.images.overhead": overhead[index],
                        "observation.images.wrist": wrist[index],
                        "observation.state": states[index].astype(np.float32),
                        "action": actions[index].astype(np.float32),
                        "task": metadata["instruction"],
                    }
                )
            dataset.save_episode()
    finally:
        finalize = getattr(dataset, "finalize", None)
        if finalize is not None:
            finalize()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw demos to LeRobot v3.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=[Path("outputs/demonstrations_training")],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/lerobot_dataset"),
    )
    parser.add_argument(
        "--repo-id",
        default="local/architecture_a_so101_pickplace",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate raw episodes without requiring LeRobot.",
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Encode camera streams as videos. Images are the portable default.",
    )
    args = parser.parse_args()
    if args.validate_only:
        summary = inspect_raw_dataset(args.input, use_videos=args.videos)
    else:
        summary = convert(
            raw_root=args.input,
            output_root=args.output,
            repo_id=args.repo_id,
            use_videos=args.videos,
        )
    print(json.dumps(summary, indent=2, default=list))


if __name__ == "__main__":
    main()
