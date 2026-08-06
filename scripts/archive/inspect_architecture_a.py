"""Run Architecture A's local barcode/damage condition gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from architecture_a.inspection import inspect_environment
from architecture_a.so101_env import SO101MuJoCoEnvironment
from shared.perception import YoloDetector


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture A close visual inspection")
    parser.add_argument(
        "--scene",
        choices=("warehouse_normal", "barcode_missing", "package_damaged"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--warehouse-layout", choices=("v1", "v2", "v3"), default="v1")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--weights",
        default="outputs/train/warehouse_yolov8n_v9_single_wrist_damage_balanced/weights/best.pt",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/architecture_a_inspection"))
    args = parser.parse_args()

    env = SO101MuJoCoEnvironment(
        realtime=False,
        observation_images=False,
        kinematic_control=True,
        scenario=args.scene,
        warehouse_layout=args.warehouse_layout,
    )
    detector = YoloDetector(
        model_path=args.weights,
        conf_threshold=0.05,
        device=args.device,
    )
    try:
        env.reset(seed=args.seed, instruction="local visual condition inspection")
        decision, frames, detections = inspect_environment(env, detector)
    finally:
        env.close()

    run_dir = args.output / f"{args.scene}-{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        Image.fromarray(frame).save(run_dir / f"{name}.png")
    result = {
        "scene": args.scene,
        "seed": args.seed,
        "destination": decision.destination,
        "instruction": decision.instruction,
        "should_escalate": decision.should_escalate,
        "reason": decision.reason,
        "barcode_confidence": decision.barcode_confidence,
        "damage_confidence": decision.damage_confidence,
        "package_confidence": decision.package_confidence,
        "detections": {
            name: [item.as_dict() for item in items]
            for name, items in detections.items()
        },
    }
    path = run_dir / "inspection.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(path.resolve())


if __name__ == "__main__":
    main()
