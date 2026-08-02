"""Capture overhead/wrist YOLO confidences without GPT or robot execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from architecture_b.payload import visual_observations_for_views
from shared.environments import SO101MuJoCoEnvironment
from shared.perception import YoloDetector, detect_package_damage_mark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--warehouse-layout", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument(
        "--yolo-weights",
        default="outputs/train/warehouse_yolov8n_v9_single_wrist_damage_balanced/weights/best.pt",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/b_multiview_diagnostics"))
    args = parser.parse_args()

    env = SO101MuJoCoEnvironment(
        gui=False,
        realtime=False,
        observation_images=False,
        kinematic_control=True,
        scenario=args.scene,
        warehouse_layout=args.warehouse_layout,
    )
    detector = YoloDetector(
        model_path=args.yolo_weights,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )
    try:
        records = []
        for episode in range(args.episodes):
            seed = args.seed + episode
            env.reset(seed=seed, instruction="Inspect package visually.")
            views = {
                "overhead": list(
                    detector.detect(env.capture_rgb(camera="overhead", width=320, height=240))
                )
            }
            for camera, frame in env.capture_condition_views(width=320, height=240).items():
                views[camera] = list(detector.detect(frame))
                if camera == "damage":
                    damage_cue = detect_package_damage_mark(frame, views[camera])
                    if damage_cue is not None:
                        views[camera].append(damage_cue)
            observations = visual_observations_for_views(views)
            records.append({"seed": seed, "visual_observations": observations})
        payload = {
            "scene": args.scene,
            "seed_start": args.seed,
            "episodes": args.episodes,
            "results": records,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        path = args.output / f"{args.scene}-{args.seed}-{args.episodes}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2))
        print(f"wrote {path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
