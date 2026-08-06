"""Persistent mixed-task Architecture B service using a durable file inbox."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from architecture_a.semantic_executor import ArchitectureASmolVlaController
from shared.metrics import TrialRecord, write_metrics
from shared.perception import YoloDetector
from shared.smolvla_runtime import SmolVlaRuntime

from .channel import get_channel
from .payload import CompressionLevel
from .planner import get_planner
from .runner import (
    GENERIC_WAREHOUSE_INSTRUCTION,
    _build_env,
    checkpoint_for_scene,
    run_trial,
    vlm_for_runtime,
)

ALLOWED_SCENES = frozenset(
    ("warehouse_normal", "barcode_missing", "package_damaged", "unexpected_obstacle")
)
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True)
class ServiceTask:
    request_id: str
    scene: str
    seed: int
    instruction: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ServiceTask":
        request_id = str(value.get("request_id", ""))
        scene = str(value.get("scene", ""))
        if not REQUEST_ID_PATTERN.fullmatch(request_id):
            raise ValueError("request_id must be 1-64 safe filename characters")
        if scene not in ALLOWED_SCENES:
            raise ValueError(f"unsupported scene: {scene!r}")
        seed = int(value.get("seed"))
        instruction = value.get("instruction")
        return cls(request_id, scene, seed, str(instruction) if instruction else None)


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _instruction(task: ServiceTask) -> str:
    if task.instruction:
        return task.instruction
    if task.scene == "unexpected_obstacle":
        return "Stop because an unexpected obstacle blocks the robot path."
    return GENERIC_WAREHOUSE_INSTRUCTION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inbox", type=Path, default=Path("runtime/architecture_b_tasks"))
    parser.add_argument("--output", type=Path, default=Path("outputs/architecture_b_service"))
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("--once", action="store_true", help="Process current tasks, then exit.")
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Record the original close angled observer view for every frame.",
    )
    parser.add_argument("--planner", choices=("gpt", "heuristic"), default="gpt")
    parser.add_argument("--channel", choices=("clean", "throttled", "restricted", "delayed", "degraded", "practical", "stressed", "extreme", "level1", "level2", "level3", "level4", "level5"), default="clean")
    parser.add_argument("--compression", choices=[item.value for item in CompressionLevel], default="scene_graph")
    parser.add_argument("--warehouse-layout", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vlm", default=None)
    parser.add_argument("--local-attempts", type=int, default=3)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument(
        "--yolo-weights",
        default="outputs/train/warehouse_yolov8n_v9_single_wrist_damage_balanced/weights/best.pt",
    )
    args = parser.parse_args()

    checkpoint = checkpoint_for_scene("warehouse_normal", args.checkpoint)
    if checkpoint is None:
        raise FileNotFoundError("No Architecture A warehouse checkpoint is configured.")
    runtime = SmolVlaRuntime(checkpoint=checkpoint, vlm=vlm_for_runtime(args.vlm), device=args.device)
    detector = YoloDetector(
        model_path=args.yolo_weights,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )
    planner = get_planner(args.planner)
    args.inbox.mkdir(parents=True, exist_ok=True)
    run_dir = args.output / time.strftime("architecture_b_service-%Y%m%d-%H%M%S")
    config = {
        "mode": "persistent_mixed_task_service",
        "planner": args.planner,
        "device": args.device,
        "warehouse_layout": args.warehouse_layout,
        "checkpoint": str(checkpoint),
        "inbox": str(args.inbox),
        "record_demo": bool(args.record_demo),
    }
    records: list[TrialRecord] = []
    write_metrics(run_dir, "B", records, config)
    print(f"[B-service] ready; watching {args.inbox.resolve()}", flush=True)

    try:
        while True:
            task_files = sorted(args.inbox.glob("task_*.json"))
            for task_path in task_files:
                fallback_id = task_path.stem.removeprefix("task_")
                fallback_response = args.inbox / f"response_{fallback_id}.json"
                if fallback_response.exists():
                    continue
                try:
                    task = ServiceTask.from_dict(json.loads(task_path.read_text(encoding="utf-8-sig")))
                    if task.request_id != fallback_id:
                        raise ValueError("request_id must match the task_<request_id>.json filename")
                    response_path = args.inbox / f"response_{task.request_id}.json"
                    if response_path.exists():
                        continue
                    selected = checkpoint_for_scene(task.scene, args.checkpoint)
                    if task.scene != "unexpected_obstacle" and (
                        selected is None or selected.resolve() != checkpoint.resolve()
                    ):
                        raise ValueError("task requires a different checkpoint; service refused hot model replacement")
                    env = _build_env(task.scene, False, False, args.warehouse_layout)
                    try:
                        controller = ArchitectureASmolVlaController(
                            env,
                            runtime,
                            max_attempts=args.local_attempts,
                            record_camera="observer" if args.record_demo else None,
                            initial_record_camera="observer" if args.record_demo else None,
                        )
                        record = run_trial(
                            env=env,
                            detector=detector,
                            planner=planner,
                            controller=controller,
                            channel=get_channel(args.channel, seed=task.seed, realtime=False),
                            compression_level=CompressionLevel(args.compression),
                            instruction=_instruction(task),
                            seed=task.seed,
                            episode_id=len(records),
                            scene=task.scene,
                        )
                    finally:
                        env.close()
                    records.append(record)
                    if args.record_demo and controller.frames:
                        from PIL import Image

                        images = [Image.fromarray(frame) for frame in controller.frames]
                        images[0].save(
                            run_dir / f"{task.request_id}.gif",
                            save_all=True,
                            append_images=images[1:],
                            duration=50,
                            loop=0,
                        )
                    write_metrics(run_dir, "B", records, config)
                    atomic_write_json(
                        response_path,
                        {"request_id": task.request_id, "status": "complete", "result": record.as_dict()},
                    )
                    print(
                        f"[B-service] {task.request_id}: scene={task.scene} success={record.success} "
                        f"latency={record.latency_seconds:.3f}s",
                        flush=True,
                    )
                except Exception as error:  # keep service alive; response contains no secret values
                    atomic_write_json(
                        fallback_response,
                        {"request_id": fallback_id, "status": "error", "error": str(error)},
                    )
                    print(f"[B-service] {fallback_id}: error={error}", flush=True)
            if args.once:
                break
            time.sleep(max(0.05, args.poll_seconds))
    except KeyboardInterrupt:
        print("\n[B-service] stop requested", flush=True)
    finally:
        write_metrics(run_dir, "B", records, config)
        print(f"[B-service] preserved {len(records)} completed task(s) in {run_dir}", flush=True)


if __name__ == "__main__":
    main()
