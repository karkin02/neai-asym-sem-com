"""Architecture C trial runner: hybrid CLIP-gated routing.

Perception + CLIP grounding -> routing gate. High-confidence recognized
instructions run locally on SmolVLA (reusing A's inference via
``shared.smolvla_runtime``) at zero network cost; everything else escalates over
the same channel + GPT-4o-mini path as Architecture B.

``run_trial`` is dependency-injected (env, detector, grounder, local runtime,
planner, controller, channel) so it is unit-testable with fakes and no heavy
stack. ``main`` wires the real components.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from shared.metrics import TrialRecord, write_metrics
from shared.perception import build_scene_graph, crop_bbox
from shared.smolvla_runtime import SmolVlaRuntime, rollout_chunk

from architecture_b.channel import ChannelSimulator, get_channel
from architecture_b.controller import ScriptedController
from architecture_b.payload import CompressionLevel, build_payload
from architecture_b.planner import Planner, get_planner

from .router import RoutingConfig, decide_route

DEFAULT_INSTRUCTION = "Pick up the sample and place it in the left tray."


def run_trial(
    *,
    env: Any,
    detector: Any,
    grounder: Any,
    local_runtime: SmolVlaRuntime,
    planner: Planner,
    controller: ScriptedController,
    channel: ChannelSimulator,
    compression_level: CompressionLevel,
    instruction: str,
    seed: int,
    episode_id: int,
    scene: str,
    routing_config: RoutingConfig = RoutingConfig(),
    referring_expression: Optional[str] = None,
    frame_size: tuple[int, int] = (320, 240),
    max_local_steps: int = 50,
) -> TrialRecord:
    """Run one Architecture C trial and return an A-comparable record."""
    width, height = frame_size
    start = time.perf_counter()

    observation = env.reset(seed=seed, instruction=instruction)
    overhead = env.capture_rgb(camera="overhead", width=width, height=height)

    detections = detector.detect(overhead)
    scene_graph = build_scene_graph(detections, width, height, task=instruction)
    crops = [crop_bbox(overhead, d.bbox) for d in detections]
    clip_confidence = grounder.score(referring_expression or instruction, crops).confidence

    decision = decide_route(
        instruction, clip_confidence, [d.label for d in detections], routing_config
    )

    record = TrialRecord(
        architecture="C",
        episode_id=episode_id,
        seed=seed,
        instruction=instruction,
        scene=scene,
        success=False,
        clip_confidence=clip_confidence,
    )

    # ---- Local route (SmolVLA), if the gate allows AND the runtime is usable ----
    if decision.route == "local" and local_runtime.available():
        wrist = env.capture_rgb(camera="wrist", width=width, height=height)
        chunk = local_runtime.predict_chunk(
            overhead, wrist, observation.robot_state, instruction
        )
        success, steps, info = rollout_chunk(env, chunk, max_steps=max_local_steps)
        record.success = success
        record.steps = steps
        record.failure_reason = info.get("failure_reason") if not success else None
        record.escalated = False
        record.route = "local"
        record.network_payload_bytes = 0
        record.latency_seconds = time.perf_counter() - start
        record.extra = {"routing_reason": decision.reason}
        return record

    # ---- Escalation route (identical to Architecture B) ----
    reason = decision.reason
    if decision.route == "local" and not local_runtime.available():
        reason = f"{reason or 'gate=local'}; local runtime unavailable -> escalated"

    payload = build_payload(compression_level, scene_graph, instruction, frame=overhead)
    transmission = channel.transmit(payload.num_bytes)
    record.escalated = True
    record.route = "escalated"
    record.network_payload_bytes = payload.num_bytes
    record.channel_condition = transmission.condition
    record.compression_level = str(payload.level.value)
    record.extra = {"routing_reason": reason}

    if not transmission.delivered:
        record.failure_reason = "channel_drop"
        record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
        return record

    target = planner.plan(payload)
    execution = controller.execute(target)
    record.success = execution.success
    record.steps = execution.steps
    record.failure_reason = execution.failure_reason
    record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
    record.extra["action_target"] = target.as_dict()
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
    parser = argparse.ArgumentParser(description="Architecture C — hybrid CLIP-gated control")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--scene", default="warehouse_normal")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--clip-threshold", type=float, default=0.60)
    parser.add_argument("--compression", choices=[c.value for c in CompressionLevel], default="scene_graph")
    parser.add_argument("--channel", choices=["clean", "degraded"], default="clean")
    parser.add_argument("--planner", choices=["gpt", "heuristic"], default="gpt")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SmolVLA checkpoint for the local route.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    parser.add_argument("--yolo-weights", default="yolov8n.pt")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/architecture_c_demo"))
    args = parser.parse_args(argv)

    from shared.perception import ClipGrounder, YoloDetector  # lazy: heavy vision stack

    env = _build_env(args.scene, args.gui, args.realtime)
    detector = YoloDetector(model_path=args.yolo_weights, conf_threshold=args.conf_threshold)
    grounder = ClipGrounder()
    runtime = SmolVlaRuntime(checkpoint=args.checkpoint, device=args.device) if args.checkpoint else SmolVlaRuntime(device=args.device)
    planner = get_planner(args.planner)
    controller = ScriptedController(env)
    routing_config = RoutingConfig(clip_threshold=args.clip_threshold)

    if not runtime.available():
        print(f"[C] SmolVLA checkpoint not found at {runtime.checkpoint}; all trials will escalate.")

    run_dir = args.output / time.strftime("architecture_c-%Y%m%d-%H%M%S")
    config = {
        "scene": args.scene,
        "clip_threshold": args.clip_threshold,
        "compression": args.compression,
        "channel": args.channel,
        "planner": args.planner,
        "local_available": runtime.available(),
    }

    records = []
    try:
        for i in range(args.episodes):
            channel = get_channel(args.channel, seed=args.seed + i, realtime=args.realtime)
            record = run_trial(
                env=env,
                detector=detector,
                grounder=grounder,
                local_runtime=runtime,
                planner=planner,
                controller=controller,
                channel=channel,
                compression_level=CompressionLevel(args.compression),
                instruction=args.instruction,
                seed=args.seed + i,
                episode_id=i,
                scene=args.scene,
                routing_config=routing_config,
            )
            records.append(record)
            print(f"[C] episode {i}: route={record.route} escalated={record.escalated} "
                  f"clip={record.clip_confidence} success={record.success} "
                  f"bytes={record.network_payload_bytes}")
    finally:
        if hasattr(env, "close"):
            env.close()

    path = write_metrics(run_dir, "C", records, config)
    escalated = sum(1 for r in records if r.escalated)
    print(f"[C] escalation rate: {escalated}/{len(records)}")
    print(f"[C] wrote {path}")


if __name__ == "__main__":
    main()
