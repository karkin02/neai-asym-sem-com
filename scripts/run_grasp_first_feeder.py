"""Compact MuJoCo feeder demo: inspect once at the apex, grasp, then sort."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from architecture_b.channel import get_channel
from architecture_b.controller import ScriptedController
from architecture_b.payload import CompressionLevel, build_payload, visual_observations_for_views
from architecture_b.planner import get_planner
from architecture_b.runner import _build_env
from shared.perception import (
    YoloDetector,
    build_scene_graph,
    detect_package_damage_mark,
)


def compact_packages(detections: list[Any]) -> list[Any]:
    return [
        item
        for item in detections
        if str(item.label) in {"package", "box", "sample", "parcel"}
        and item.bbox[2] - item.bbox[0] < 50
        and item.bbox[3] - item.bbox[1] < 50
    ]


def majority_fuse(batches: list[list[Any]]) -> list[Any]:
    """Keep each label only when it appears in a majority of temporal frames."""
    if not batches:
        return []
    quorum = len(batches) // 2 + 1
    labels = {str(item.label) for batch in batches for item in batch}
    fused = []
    for label in labels:
        representatives = [
            max(
                (item for item in batch if str(item.label) == label),
                key=lambda item: float(item.confidence),
                default=None,
            )
            for batch in batches
        ]
        present = [item for item in representatives if item is not None]
        if len(present) >= quorum:
            fused.append(max(present, key=lambda item: float(item.confidence)))
    return fused


def wait_for_pick_zone(env: Any, detector: Any, controller: ScriptedController) -> None:
    """Wait causally until overhead vision places the package near the front apex."""
    for _ in range(45):
        frame = env.capture_rgb(camera="overhead", width=320, height=240)
        detections = list(detector.detect(frame))
        packages = compact_packages(detections)
        if packages:
            item = max(packages, key=lambda value: float(value.confidence))
            position = env.camera_pixel_to_world(
                item.center, camera="overhead", width=320, height=240
            )
            controller.note_link_latency(0.0)
            controller.observe_visual(
                detections, camera="overhead", width=320, height=240
            )
            if abs(float(position[0])) <= 0.065:
                return
        env.advance_idle(1.0)
    raise RuntimeError("Package did not enter the visual pickup zone within 45 seconds.")


def wait_for_inspection_zone(env: Any, detector: Any, controller: ScriptedController) -> None:
    """Acquire the package on the inbound-left arc before it reaches the apex."""
    for _ in range(45):
        frame = env.capture_rgb(camera="overhead", width=320, height=240)
        detections = list(detector.detect(frame))
        packages = compact_packages(detections)
        if packages:
            item = max(packages, key=lambda value: float(value.confidence))
            position = env.camera_pixel_to_world(
                item.center, camera="overhead", width=320, height=240
            )
            controller.note_link_latency(0.0)
            controller.observe_visual(
                detections, camera="overhead", width=320, height=240
            )
            if -0.100 <= float(position[0]) <= -0.070:
                return
        env.advance_idle(1.0)
    raise RuntimeError("Package did not enter the pre-apex inspection zone within 45 seconds.")


def refresh_intercept(env: Any, detector: Any, controller: ScriptedController) -> None:
    """Measure local package velocity and lead only the known grasp motion."""
    for index in range(2):
        frame = env.capture_rgb(camera="overhead", width=320, height=240)
        detections = list(detector.detect(frame))
        controller.note_link_latency(0.0)
        if index == 1:
            controller.set_local_motion_horizon(0.35)
        controller.observe_visual(
            detections, camera="overhead", width=320, height=240
        )
        if index == 0:
            env.advance_idle(0.5)


def inspect_package_once(env: Any, detector: Any, instruction: str) -> tuple[dict[str, Any], Any]:
    """Capture exactly one barcode pose and one damage pose before grasp."""
    overhead = env.capture_rgb(camera="overhead", width=320, height=240)
    overhead_detections = list(detector.detect(overhead))
    frames = env.capture_condition_views(width=320, height=240, frames_per_view=3)
    sequences = {
        name: images if isinstance(images, list) else [images]
        for name, images in frames.items()
    }
    sequence_detections = {
        name: [list(detector.detect(image)) for image in images]
        for name, images in sequences.items()
    }
    for damage_frame, frame_detections in zip(
        sequences.get("damage", []), sequence_detections.get("damage", [])
    ):
        cue = detect_package_damage_mark(damage_frame, frame_detections)
        if cue is not None:
            frame_detections.append(cue)
    views = {
        "overhead": overhead_detections,
        **{
            name: majority_fuse(batches)
            for name, batches in sequence_detections.items()
        },
    }
    scene = build_scene_graph(overhead_detections, 320, 240, task=instruction)
    scene["visual_observations"] = visual_observations_for_views(views)
    return scene, overhead


def save_demo(frames: list[Any], path: Path, fps: int) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp4":
        import cv2

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open MP4 encoder for {path}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        print(f"[demo] wrote {path} frames={len(frames)} fps={fps}")
        return
    if path.suffix.lower() != ".gif":
        raise ValueError("--record-output must end in .mp4 or .gif")
    images = [Image.fromarray(frame).convert("RGB") for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
    )
    print(f"[demo] wrote {path} frames={len(images)} fps={fps}")


def main(*, forced_architecture: str | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("a", "b"),
        default=forced_architecture or "b",
        help="A runs fully locally; B transmits the semantic payload.",
    )
    parser.add_argument(
        "--scene",
        choices=("warehouse_normal", "barcode_missing", "package_damaged"),
        default="warehouse_normal",
    )
    parser.add_argument("--instruction", default="Inspect the held package and sort it.")
    parser.add_argument("--planner", choices=("heuristic", "gpt"), default="heuristic")
    parser.add_argument("--channel", default="level1")
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--channel-seed", type=int, default=None)
    parser.add_argument("--transmission-attempts", type=int, default=3)
    parser.add_argument("--grasp-attempts", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--yolo-weights",
        default="outputs/train/warehouse_yolov8n_v9_single_wrist_damage_balanced/weights/best.pt",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--advance-world-during-network", action="store_true")
    parser.add_argument("--hold-window", action="store_true")
    parser.add_argument("--verbose-inspection", action="store_true")
    parser.add_argument(
        "--post-run-seconds",
        type=float,
        default=1.0,
        help="Simulation time to keep recording after placement (default: 1 second).",
    )
    parser.add_argument("--record-demo", action="store_true")
    parser.add_argument(
        "--record-camera", choices=("observer", "overhead"), default="observer"
    )
    parser.add_argument("--record-fps", type=int, default=10)
    parser.add_argument("--record-output", type=Path, default=None)
    args = parser.parse_args()
    if forced_architecture is not None:
        args.architecture = forced_architecture

    env = _build_env(args.scene, args.gui, args.realtime, "v3", True)
    detector = YoloDetector(
        model_path=args.yolo_weights,
        conf_threshold=0.05,
        device=args.device,
    )
    planner = get_planner("heuristic" if args.architecture == "a" else args.planner)
    controller = ScriptedController(env)
    recording_saved = False
    record_output = args.record_output or Path(
        "outputs/mujoco_rendered_demo"
    ) / f"compact-feeder-{datetime.now():%Y%m%d-%H%M%S}" / "episode_0000.mp4"
    try:
        # Reset places the box at the feeder's left entry; there is deliberately
        # no pre-observation feeder delay in this compact demo.
        env.reset(seed=args.seed, instruction=args.instruction)
        detector.detect(env.capture_rgb(camera="overhead", width=320, height=240))
        if args.record_demo:
            env.start_demo_recording(
                camera=args.record_camera,
                width=320,
                height=240,
                fps=args.record_fps,
            )
        wait_for_inspection_zone(env, detector, controller)

        scene, frame = inspect_package_once(env, detector, args.instruction)
        if args.verbose_inspection:
            print("inspection=" + json.dumps(scene["visual_observations"], sort_keys=True))
        payload = build_payload(
            CompressionLevel.SCENE_GRAPH, scene, args.instruction, frame=frame
        )
        total_channel_latency = 0.0
        dropped_payloads = 0
        attempts_used = 0
        if args.architecture == "b":
            channel = get_channel(
                args.channel,
                seed=args.seed if args.channel_seed is None else args.channel_seed,
                realtime=args.realtime,
            )
            sent = None
            for attempts_used in range(1, max(1, args.transmission_attempts) + 1):
                sent = channel.transmit(payload.num_bytes)
                total_channel_latency += sent.latency_seconds
                if args.advance_world_during_network:
                    env.advance_idle(sent.latency_seconds)
                if sent.delivered:
                    break
                dropped_payloads += 1
            if sent is None or not sent.delivered:
                raise RuntimeError(
                    "Pre-pickup inspection payload was dropped on every bounded "
                    "attempt; safe stop before grasp."
                )
        target = planner.plan(payload)
        if (
            target.command == "STOP"
            or target.destination is None
            or target.confidence < 0.5
        ):
            raise RuntimeError(
                f"Planner safe stop: command={target.command} "
                f"destination={target.destination} confidence={target.confidence:.3f}"
            )

        session = controller.start(target)
        # The package continued moving while the semantic payload crossed the
        # link. Re-enter the local visual pickup zone before issuing approach;
        # GPT never supplies or updates a joint target.
        wait_for_pick_zone(env, detector, controller)
        refresh_intercept(env, detector, controller)
        controller.advance(session, target)  # one approach
        grasp_attempts_used = 0
        while not env.holding_package and grasp_attempts_used < max(1, args.grasp_attempts):
            grasp_attempts_used += 1
            if grasp_attempts_used > 1:
                session.phase = "grasp"
                wait_for_pick_zone(env, detector, controller)
            refresh_intercept(env, detector, controller)
            controller.advance(session, target)
        if not env.holding_package:
            raise RuntimeError(
                "Local intercept failed after every bounded grasp attempt; "
                "no placement was executed."
            )
        controller.advance(session, target)  # direct carry
        controller.advance(session, target)  # direct release
        if args.post_run_seconds > 0:
            env.advance_idle(args.post_run_seconds)
        final_position = tuple(round(float(value), 4) for value in env.sample_position)
        print(
            f"architecture={args.architecture.upper()} scene={args.scene} "
            f"success={session.result.success} "
            f"destination={target.destination} steps={session.result.steps} "
            f"bytes={payload.num_bytes * attempts_used} "
            f"channel_latency={total_channel_latency:.3f}s "
            f"transmission_attempts={attempts_used} drops={dropped_payloads} "
            f"grasp_attempts={grasp_attempts_used} final_position={final_position}"
        )
        if args.record_demo:
            save_demo(env.stop_demo_recording(), record_output, args.record_fps)
            recording_saved = True
        if args.hold_window and args.gui:
            env.wait_for_viewer_close()
    finally:
        if args.record_demo and not recording_saved:
            save_demo(env.stop_demo_recording(), record_output, args.record_fps)
        env.close()


if __name__ == "__main__":
    main()
