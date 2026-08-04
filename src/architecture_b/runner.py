"""Architecture B trial runner: perception -> channel -> GPT-4o-mini -> execute.

``run_trial`` is fully dependency-injected (env, detector, planner, controller,
channel are passed in) so it is unit-testable with fakes and no ``mujoco``.
``main`` wires the real components and writes an A-compatible ``metrics.json``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from shared.metrics import TrialRecord, write_metrics
from shared.perception import build_scene_graph, crop_bbox, normalize_object_class
from shared.instructions import instruction_for_scenario

from .channel import ChannelSimulator, get_channel
from .controller import ScriptedController
from .payload import CompressionLevel, build_payload, visual_observations_for_views
from .planner import Planner, get_planner

DEFAULT_INSTRUCTION = "Pick up the sample and place it in the left tray."
GENERIC_WAREHOUSE_INSTRUCTION = "Inspect and sort the detected package using visual evidence."
LOCAL_VLM = Path(".hf-cache/checkpoints/smolvlm2_500m_video_instruct")


def checkpoint_for_scene(
    scene: str,
    explicit: Path | None,
    manifest_path: Path = Path("config/architecture_a_model.json"),
) -> Path | None:
    if explicit is not None or not manifest_path.exists():
        return explicit
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = manifest.get("scenario_checkpoints", {}).get(scene, {})
        selected = entry.get("checkpoint") or manifest.get("checkpoint")
        return Path(selected) if selected else None
    except (OSError, ValueError, TypeError):
        return explicit


def vlm_for_runtime(explicit: str | None, local_path: Path = LOCAL_VLM) -> str:
    """Prefer an explicit VLM, then the bundled cache, then the portable Hub ID."""
    if explicit:
        return explicit
    if local_path.is_dir():
        return str(local_path.resolve())
    from shared.smolvla_runtime import DEFAULT_VLM

    return DEFAULT_VLM


def run_trial(
    *,
    env: Any,
    detector: Any,
    planner: Planner,
    controller: ScriptedController,
    channel: ChannelSimulator,
    compression_level: CompressionLevel,
    instruction: str,
    seed: int,
    episode_id: int,
    scene: str,
    grounder: Optional[Any] = None,
    referring_expression: Optional[str] = None,
    camera: str = "overhead",
    frame_size: tuple[int, int] = (320, 240),
) -> TrialRecord:
    """Run one Architecture B trial and return an A-comparable record."""
    width, height = frame_size
    start = time.perf_counter()

    env.reset(seed=seed, instruction=instruction)
    frame = env.capture_rgb(camera=camera, width=width, height=height)

    detections = detector.detect(frame)
    scene_graph = build_scene_graph(detections, width, height, task=instruction)
    capture_conditions = getattr(env, "capture_condition_views", None)
    if callable(capture_conditions):
        condition_frames = capture_conditions(width=width, height=height)
        view_detections = {
            camera: list(detections),
            **{view: list(detector.detect(image)) for view, image in condition_frames.items()},
        }
        from shared.perception import detect_package_damage_mark

        if "damage" in condition_frames:
            damage_cue = detect_package_damage_mark(
                condition_frames["damage"], view_detections["damage"]
            )
            if damage_cue is not None:
                view_detections["damage"].append(damage_cue)
    else:
        wrist_frame = env.capture_rgb(camera="wrist", width=width, height=height)
        view_detections = {camera: list(detections), "wrist": list(detector.detect(wrist_frame))}
    scene_graph["visual_observations"] = visual_observations_for_views(view_detections)

    clip_confidence = None
    if grounder is not None:
        crops = [crop_bbox(frame, d.bbox) for d in detections]
        clip_confidence = grounder.score(referring_expression or instruction, crops).confidence

    payload = build_payload(compression_level, scene_graph, instruction, frame=frame)
    transmission = channel.transmit(payload.num_bytes)

    record = TrialRecord(
        architecture="B",
        episode_id=episode_id,
        seed=seed,
        instruction=instruction,
        scene=scene,
        success=False,
        network_payload_bytes=payload.num_bytes,
        clip_confidence=clip_confidence,
        route="cloud",
        channel_condition=transmission.condition,
        compression_level=str(payload.level.value),
    )

    if not transmission.delivered:
        # Degraded-channel failure: payload never reached the planner.
        record.failure_reason = "channel_drop"
        record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
        return record

    # Keep an obstacle escalation stationary while perception is rechecked.
    # A missing detection is not positive proof that a reported obstacle has
    # cleared, so the local safety layer ends the hold with STOP and never
    # forwards an unsafe cloud target to the joint controller.
    obstacle_reported = "obstacle" in instruction.lower() or any(
        normalize_object_class(str(item.label)) == "obstacle" for item in detections
    )
    if obstacle_reported:
        recheck_labels: list[list[str]] = []
        for _ in range(3):
            hold_frame = env.capture_rgb(camera=camera, width=width, height=height)
            hold_detections = detector.detect(hold_frame)
            recheck_labels.append([str(item.label) for item in hold_detections])
        record.success = True
        record.steps = 0
        record.failure_reason = None
        record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
        record.extra = {
            "recovery_command": "STOP",
            "safety_hold": True,
            "obstacle_rechecks": len(recheck_labels),
            "obstacle_recheck_labels": recheck_labels,
            "joint_commands_executed": 0,
            "stop_reason": "reported_obstacle_not_positively_verified_clear",
        }
        return record

    target = planner.plan(payload)
    command = str(getattr(target, "command", "STOP")).upper()
    confidence = float(getattr(target, "confidence", 0.0))
    if command == "STOP" or confidence < 0.5 or target.destination is None:
        record.steps = 0
        record.failure_reason = "planner_safe_stop"
        record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
        record.extra = {
            "action_target": target.as_dict(),
            "visual_observations": scene_graph["visual_observations"],
            "joint_commands_executed": 0,
            "stop_reason": "invalid_or_low_confidence_semantic_plan",
        }
        return record
    prepare = getattr(controller, "prepare", None)
    if callable(prepare):
        prepare(seed)
    execution = controller.execute(target)

    record.success = execution.success
    record.steps = execution.steps
    record.failure_reason = execution.failure_reason
    record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
    record.extra = {
        "action_target": target.as_dict(),
        "visual_observations": scene_graph["visual_observations"],
    }
    return record


def _build_env(scene: str, gui: bool, realtime: bool, warehouse_layout: str) -> Any:
    from shared.environments import SO101MuJoCoEnvironment  # lazy: needs mujoco

    return SO101MuJoCoEnvironment(
        gui=gui,
        realtime=realtime,
        observation_images=False,
        kinematic_control=True,
        scenario=scene,
        warehouse_layout=warehouse_layout,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Architecture B — networked GPT-4o-mini control")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Keep models loaded and run until Ctrl+C; checkpoint metrics after every episode.",
    )
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--scene", default="warehouse_normal")
    parser.add_argument("--warehouse-layout", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--compression", choices=[c.value for c in CompressionLevel], default="scene_graph")
    parser.add_argument("--channel", choices=["clean", "degraded"], default="clean")
    parser.add_argument("--planner", choices=["gpt", "heuristic"], default="gpt")
    parser.add_argument(
        "--executor",
        choices=("architecture_a", "scripted"),
        default="architecture_a",
        help="Architecture A SmolVLA is production; scripted is benchmark-only.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--vlm", default=None)
    parser.add_argument("--local-attempts", type=int, default=3)
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument(
        "--yolo-weights",
        default="outputs/train/warehouse_yolov8n_v9_single_wrist_damage_balanced/weights/best.pt",
    )
    parser.add_argument(
        "--device",
        "--perception-device",
        dest="perception_device",
        default="cpu",
        help="YOLO/CLIP inference device.",
    )
    parser.add_argument("--clip", action="store_true", help="Also compute a CLIP confidence for logging.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Record the original close angled observer view for every frame.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/architecture_b_demo"))
    args = parser.parse_args(argv)
    instruction = args.instruction or (
        instruction_for_scenario(args.scene)
        if args.scene == "unexpected_obstacle"
        else GENERIC_WAREHOUSE_INSTRUCTION
    )

    from shared.perception import ClipGrounder, YoloDetector  # lazy: heavy vision stack

    env = _build_env(args.scene, args.gui, args.realtime, args.warehouse_layout)
    detector = YoloDetector(
        model_path=args.yolo_weights,
        conf_threshold=args.conf_threshold,
        device=args.perception_device,
    )
    grounder = ClipGrounder(device=args.perception_device) if args.clip else None
    planner = get_planner(args.planner)
    if args.executor == "architecture_a":
        from architecture_a.semantic_executor import ArchitectureASmolVlaController
        from shared.smolvla_runtime import SmolVlaRuntime

        checkpoint = checkpoint_for_scene(args.scene, args.checkpoint)
        if checkpoint is None:
            raise FileNotFoundError("No Architecture A checkpoint is configured.")
        runtime = SmolVlaRuntime(
            checkpoint=checkpoint,
            vlm=vlm_for_runtime(args.vlm),
            device=args.perception_device,
        )
        controller = ArchitectureASmolVlaController(
            env,
            runtime,
            max_attempts=args.local_attempts,
            record_camera="observer" if args.record_demo else None,
            initial_record_camera="observer" if args.record_demo else None,
        )
    else:
        controller = ScriptedController(env)

    run_dir = args.output / time.strftime("architecture_b-%Y%m%d-%H%M%S")
    config = {
        "scene": args.scene,
        "compression": args.compression,
        "channel": args.channel,
        "planner": args.planner,
        "executor": args.executor,
        "clip": bool(args.clip),
        "perception_device": args.perception_device,
        "warehouse_layout": args.warehouse_layout,
        "yolo_weights": args.yolo_weights,
        "continuous": bool(args.continuous),
    }

    records = []
    path = write_metrics(run_dir, "B", records, config)
    episode_indices = itertools.count() if args.continuous else range(args.episodes)
    try:
        for i in episode_indices:
            channel = get_channel(args.channel, seed=args.seed + i, realtime=args.realtime)
            record = run_trial(
                env=env,
                detector=detector,
                planner=planner,
                controller=controller,
                channel=channel,
                compression_level=CompressionLevel(args.compression),
                instruction=instruction,
                seed=args.seed + i,
                episode_id=i,
                scene=args.scene,
                grounder=grounder,
            )
            records.append(record)
            if args.record_demo and getattr(controller, "frames", None):
                from PIL import Image

                images = [Image.fromarray(frame) for frame in controller.frames]
                images[0].save(
                    run_dir / f"episode_{i:04d}.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=50,
                    loop=0,
                )
            path = write_metrics(run_dir, "B", records, config)
            print(f"[B] episode {i}: success={record.success} bytes={record.network_payload_bytes} "
                  f"latency={record.latency_seconds:.3f}s channel={record.channel_condition}", flush=True)
    except KeyboardInterrupt:
        print(f"\n[B] stop requested; preserved {len(records)} completed episode(s).", flush=True)
    finally:
        path = write_metrics(run_dir, "B", records, config)
        if hasattr(env, "close"):
            env.close()

    print(f"[B] wrote {path}")


if __name__ == "__main__":
    main()
