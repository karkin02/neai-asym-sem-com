"""Architecture B trial runner: perception -> channel -> GPT-4o-mini -> execute.

``run_trial`` is fully dependency-injected (env, detector, planner, controller,
channel are passed in) so it is unit-testable with fakes and no ``mujoco``.
``main`` wires the real components and writes an A-compatible ``metrics.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from shared.metrics import TrialRecord, write_metrics
from shared.perception import build_scene_graph, crop_bbox

from .channel import ChannelSimulator, get_channel
from .controller import ScriptedController
from .payload import CompressionLevel, build_payload
from .planner import Planner, get_planner

DEFAULT_INSTRUCTION = "Pick up the sample and place it in the left tray."


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

    target = planner.plan(payload)
    execution = controller.execute(target)

    record.success = execution.success
    record.steps = execution.steps
    record.failure_reason = execution.failure_reason
    record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
    record.extra = {"action_target": target.as_dict()}
    return record


def _build_env(scene: str, gui: bool, realtime: bool) -> Any:
    from shared.environments import SO101MuJoCoEnvironment  # lazy: needs mujoco

    return SO101MuJoCoEnvironment(
        gui=gui,
        realtime=realtime,
        observation_images=False,
        kinematic_control=True,
        scenario=scene,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Architecture B — networked GPT-4o-mini control")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--scene", default="warehouse_normal")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--compression", choices=[c.value for c in CompressionLevel], default="scene_graph")
    parser.add_argument("--channel", choices=["clean", "degraded"], default="clean")
    parser.add_argument("--planner", choices=["gpt", "heuristic"], default="gpt")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--clip", action="store_true", help="Also compute a CLIP confidence for logging.")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/architecture_b_demo"))
    args = parser.parse_args(argv)

    from shared.perception import ClipGrounder, YoloDetector  # lazy: heavy vision stack

    env = _build_env(args.scene, args.gui, args.realtime)
    detector = YoloDetector(model_path=args.yolo_weights, conf_threshold=args.conf_threshold)
    grounder = ClipGrounder() if args.clip else None
    planner = get_planner(args.planner)
    controller = ScriptedController(env)

    run_dir = args.output / time.strftime("architecture_b-%Y%m%d-%H%M%S")
    config = {
        "scene": args.scene,
        "compression": args.compression,
        "channel": args.channel,
        "planner": args.planner,
        "clip": bool(args.clip),
    }

    records = []
    try:
        for i in range(args.episodes):
            channel = get_channel(args.channel, seed=args.seed + i, realtime=args.realtime)
            record = run_trial(
                env=env,
                detector=detector,
                planner=planner,
                controller=controller,
                channel=channel,
                compression_level=CompressionLevel(args.compression),
                instruction=args.instruction,
                seed=args.seed + i,
                episode_id=i,
                scene=args.scene,
                grounder=grounder,
            )
            records.append(record)
            print(f"[B] episode {i}: success={record.success} bytes={record.network_payload_bytes} "
                  f"latency={record.latency_seconds:.3f}s channel={record.channel_condition}")
    finally:
        if hasattr(env, "close"):
            env.close()

    path = write_metrics(run_dir, "B", records, config)
    print(f"[B] wrote {path}")


if __name__ == "__main__":
    main()
