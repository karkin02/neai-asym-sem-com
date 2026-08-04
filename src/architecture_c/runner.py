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
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from shared.metrics import TrialRecord, write_metrics
from shared.perception import (
    build_scene_graph,
    crop_bbox,
    normalize_object_class,
    target_class_from_text,
)
from shared.instructions import instruction_for_scenario
from shared.smolvla_runtime import DEFAULT_VLM, SmolVlaRuntime, rollout_chunk
from architecture_a.action_safety import (
    SO101_ACTION_BOUND_TOLERANCE,
    SO101_ACTION_HIGH,
    SO101_ACTION_LOW,
    bound_action_chunk,
)

from architecture_b.channel import ChannelSimulator, get_channel
from architecture_b.controller import ScriptedController
from architecture_b.payload import (
    CompressionLevel,
    build_payload,
    visual_observations_for_views,
)
from architecture_b.planner import Planner, get_planner

from .router import RoutingConfig, decide_route

DEFAULT_INSTRUCTION = "Pick up the sample and place it in the left tray."


def checkpoint_for_scene(
    scene: str,
    explicit: Path | None,
    manifest_path: Path = Path("config/architecture_a_model.json"),
) -> Path | None:
    """Resolve A's approved scenario checkpoint without duplicating routing policy."""
    if explicit is not None or not manifest_path.exists():
        return explicit
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        scenario_entry = manifest.get("scenario_checkpoints", {}).get(scene, {})
        selected = scenario_entry.get("checkpoint") or manifest.get("checkpoint")
        return Path(selected) if selected else None
    except (OSError, ValueError, TypeError):
        return explicit


def _has_target_detection(instruction: str, detections: Sequence[Any]) -> bool:
    """Whether detections contain the object named by the warehouse instruction."""
    text = instruction.lower()
    labels = {str(d.label).lower() for d in detections}
    if "sample" in text or "package" in text or "box" in text:
        return bool(labels & {"sample", "package", "box"})
    return bool(labels)


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
    max_local_attempts: int = 2,
    demo_frames: list[Any] | None = None,
) -> TrialRecord:
    """Run one Architecture C trial and return an A-comparable record."""
    width, height = frame_size
    start = time.perf_counter()

    observation = env.reset(seed=seed, instruction=instruction)
    condition_gate_failed = False
    condition_detections = None
    if (
        scene in {"warehouse_normal", "barcode_missing", "package_damaged"}
        and callable(getattr(env, "capture_condition_views", None))
    ):
        from architecture_a.inspection import inspect_environment

        inspection, _, condition_detections = inspect_environment(
            env, detector, width=width, height=height
        )
        if inspection.should_escalate or inspection.instruction is None:
            condition_gate_failed = True
        else:
            instruction = inspection.instruction
            observation = env.reset(seed=seed, instruction=instruction)
    if demo_frames is not None:
        demo_frames.append(env.capture_rgb(camera="observer", width=480, height=360))
    overhead = env.capture_rgb(camera="overhead", width=width, height=height)

    overhead_detections = detector.detect(overhead)
    scene_graph = build_scene_graph(overhead_detections, width, height, task=instruction)
    if condition_detections is not None:
        scene_graph["visual_observations"] = visual_observations_for_views(
            condition_detections
        )
    detections = overhead_detections
    perception_frame = overhead
    perception_view = "overhead"
    wrist = None
    if not _has_target_detection(instruction, overhead_detections):
        wrist = env.capture_rgb(camera="wrist", width=width, height=height)
        wrist_detections = detector.detect(wrist)
        if _has_target_detection(instruction, wrist_detections):
            detections = wrist_detections
            perception_frame = wrist
            perception_view = "wrist_fallback"
    target_class = target_class_from_text(referring_expression or instruction)
    target_detections = [
        detection
        for detection in detections
        if target_class is not None
        and normalize_object_class(str(detection.label)) == target_class
    ]
    perception_valid = bool(target_detections) and not condition_gate_failed
    grounding_detections = target_detections if perception_valid else detections
    crops = [crop_bbox(perception_frame, d.bbox) for d in grounding_detections]
    # Generic YOLO weights may not recognize synthetic warehouse geometry.
    # Preserve a meaningful CLIP routing signal by grounding against the full
    # observation rather than passing an empty candidate list (confidence 0).
    if not crops:
        crops = [overhead]
    grounding = grounder.score(referring_expression or instruction, crops)
    clip_confidence = grounding.confidence
    clip_margin = getattr(grounding, "margin", None)

    decision = decide_route(
        instruction,
        clip_confidence,
        [d.label for d in detections],
        routing_config,
        clip_margin=clip_margin,
        perception_valid=perception_valid,
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
        local_runtime.reset(seed)
        success = False
        steps = 0
        info: dict[str, Any] = {}
        attempts = 0
        current_observation = observation
        for attempts in range(1, max(1, max_local_attempts) + 1):
            if attempts > 1:
                overhead = env.capture_rgb(camera="overhead", width=width, height=height)
                wrist = None
            if wrist is None:
                wrist = env.capture_rgb(camera="wrist", width=width, height=height)
            chunk = local_runtime.predict_chunk(
                overhead, wrist, current_observation.robot_state, instruction
            )
            bounded = bound_action_chunk(
                chunk,
                action_low=SO101_ACTION_LOW,
                action_high=SO101_ACTION_HIGH,
                tolerance=SO101_ACTION_BOUND_TOLERANCE,
            )
            bound_diagnostics = {
                "action_bound_clipped_rows": bounded.clipped_rows,
                "maximum_action_bound_overshoot": bounded.maximum_overshoot,
                "rejected_action_violation": bounded.violation,
            }
            if not bounded.accepted:
                info = {
                    "failure_reason": "action_chunk_out_of_bounds",
                    **bound_diagnostics,
                }
                if attempts < max(1, max_local_attempts):
                    continue
                break
            chunk = bounded.values
            preview = getattr(env, "preview_action_success", None)
            if callable(preview) and not preview(chunk[:max_local_steps]):
                info = {
                    "failure_reason": "preexecution_outcome_validation_failed",
                    **bound_diagnostics,
                }
                if attempts < max(1, max_local_attempts):
                    continue
                break
            chunk_success, chunk_steps, info, final_observation = rollout_chunk(
                env,
                chunk,
                max_steps=max_local_steps,
                on_step=(
                    lambda: demo_frames.append(
                        env.capture_rgb(camera="observer", width=480, height=360)
                    )
                    if demo_frames is not None
                    else None
                ),
            )
            info.update(bound_diagnostics)
            steps += chunk_steps
            success = chunk_success
            if success or final_observation is None:
                break
            current_observation = final_observation
        record.success = success
        record.steps = steps
        record.failure_reason = info.get("failure_reason") if not success else None
        if success:
            record.escalated = False
            record.route = "local"
            record.network_payload_bytes = 0
            record.latency_seconds = time.perf_counter() - start
            record.extra = {
                "routing_reason": decision.reason,
                "clip_margin": clip_margin,
                "perception_view": perception_view,
                "local_attempts": attempts,
                "action_bound_clipped_rows": info.get("action_bound_clipped_rows", 0),
                "maximum_action_bound_overshoot": info.get(
                    "maximum_action_bound_overshoot", 0.0
                ),
            }
            return record

        # Architecture C is hybrid: a locally accepted task that fails physical
        # validation must use B's recovery path instead of being reported as a
        # terminal local failure. Keep the local attempt in the metrics so the
        # recovery is observable and distinguishable from perception routing.
        local_failure_reason = info.get("failure_reason") or "maximum_steps_exceeded"
        reason = f"{decision.reason}; local execution failed: {local_failure_reason}"
        record.extra = {
            "local_attempted": True,
            "local_steps": steps,
            "local_attempts": attempts,
            "local_failure_reason": local_failure_reason,
            "action_bound_clipped_rows": info.get("action_bound_clipped_rows", 0),
            "maximum_action_bound_overshoot": info.get(
                "maximum_action_bound_overshoot", 0.0
            ),
            "rejected_action_violation": info.get("rejected_action_violation"),
        }
    else:
        reason = decision.reason

    # ---- Escalation route (identical to Architecture B) ----
    if decision.route == "local" and not local_runtime.available():
        reason = f"{reason or 'gate=local'}; local runtime unavailable -> escalated"

    payload = build_payload(compression_level, scene_graph, instruction, frame=overhead)
    transmission = channel.transmit(payload.num_bytes)
    record.escalated = True
    record.route = "escalated"
    record.network_payload_bytes = payload.num_bytes
    record.channel_condition = transmission.condition
    record.compression_level = str(payload.level.value)
    record.extra.update({
        "routing_reason": reason,
        "clip_margin": clip_margin,
        "perception_view": perception_view,
    })

    if not transmission.delivered:
        record.failure_reason = "channel_drop"
        record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
        return record

    # An obstacle escalation enters a stationary hold before becoming a final
    # STOP. Re-observe a bounded number of times so a transient perception
    # event is not treated as an instantaneous terminal decision. Absence of a
    # detection is not positive proof that a reported obstacle cleared, so the
    # local safety layer remains stopped unless a future certified adapter can
    # provide an explicit clear-path signal.
    obstacle_reported = "obstacle" in instruction.lower() or any(
        normalize_object_class(str(item.label)) == "obstacle" for item in detections
    )
    if obstacle_reported:
        recheck_labels: list[list[str]] = []
        for _ in range(3):
            hold_frame = env.capture_rgb(camera="overhead", width=width, height=height)
            hold_detections = detector.detect(hold_frame)
            recheck_labels.append([str(item.label) for item in hold_detections])
        record.success = True
        record.steps = 0
        record.failure_reason = None
        record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
        record.extra.update({
            "recovery_command": "STOP",
            "safety_hold": True,
            "obstacle_rechecks": len(recheck_labels),
            "obstacle_recheck_labels": recheck_labels,
            "joint_commands_executed": 0,
            "stop_reason": "reported_obstacle_not_positively_verified_clear",
        })
        return record

    target = planner.plan(payload)
    execution = controller.execute(target)
    record.success = execution.success
    record.steps = execution.steps
    record.failure_reason = execution.failure_reason
    record.latency_seconds = (time.perf_counter() - start) + transmission.latency_seconds
    record.extra["action_target"] = target.as_dict()
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
    parser = argparse.ArgumentParser(description="Architecture C — hybrid CLIP-gated control")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1010)
    parser.add_argument("--scene", default="warehouse_normal")
    parser.add_argument("--warehouse-layout", choices=("v1", "v2", "v3"), default="v3")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--clip-threshold", type=float, default=0.60)
    parser.add_argument(
        "--clip-margin-threshold",
        type=float,
        default=-0.023721705733121712,
        help="Minimum target-vs-alternative CLIP prototype margin.",
    )
    parser.add_argument("--compression", choices=[c.value for c in CompressionLevel], default="scene_graph")
    parser.add_argument("--channel", choices=["clean", "degraded"], default="clean")
    parser.add_argument("--planner", choices=["gpt", "heuristic"], default="gpt")
    parser.add_argument("--checkpoint", type=Path, default=None, help="SmolVLA checkpoint for the local route.")
    parser.add_argument(
        "--vlm",
        default=DEFAULT_VLM,
        help="Hugging Face VLM model ID or an absolute local model directory.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="SmolVLA, YOLO, and CLIP device (for ROCm PyTorch, use cuda).",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.05)
    parser.add_argument(
        "--local-attempts",
        type=int,
        default=3,
        help="Maximum fresh SmolVLA chunks before validated B recovery.",
    )
    parser.add_argument(
        "--yolo-weights",
        default="outputs/train/warehouse_yolov8n_v9_single_wrist_damage_balanced/weights/best.pt",
    )
    parser.add_argument(
        "--clip-prototypes",
        type=Path,
        default=Path("outputs/train/warehouse_clip_prototypes_v4_v3_medium.npz"),
        help="Frozen-CLIP warehouse prototype calibration file.",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument(
        "--record-demo",
        action="store_true",
        help="Record the original close angled observer view for every frame.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/architecture_c_demo"))
    args = parser.parse_args(argv)
    instruction = args.instruction or instruction_for_scenario(args.scene)

    from shared.perception import (  # lazy: heavy vision stack
        ClipGrounder,
        WarehouseClipGrounder,
        YoloDetector,
    )

    env = _build_env(args.scene, args.gui, args.realtime, args.warehouse_layout)
    detector = YoloDetector(
        model_path=args.yolo_weights,
        conf_threshold=args.conf_threshold,
        device=args.device,
    )
    grounder = (
        WarehouseClipGrounder(args.clip_prototypes, device=args.device)
        if args.clip_prototypes.exists()
        else ClipGrounder(device=args.device)
    )
    selected_checkpoint = checkpoint_for_scene(args.scene, args.checkpoint)
    runtime = (
        SmolVlaRuntime(checkpoint=selected_checkpoint, vlm=args.vlm, device=args.device)
        if selected_checkpoint
        else SmolVlaRuntime(vlm=args.vlm, device=args.device)
    )
    planner = get_planner(args.planner)
    controller = ScriptedController(env)
    routing_config = RoutingConfig(
        clip_threshold=args.clip_threshold,
        clip_margin_threshold=args.clip_margin_threshold,
    )

    if not runtime.available():
        print(f"[C] SmolVLA checkpoint not found at {runtime.checkpoint}; all trials will escalate.")

    run_dir = args.output / time.strftime("architecture_c-%Y%m%d-%H%M%S")
    config = {
        "scene": args.scene,
        "warehouse_layout": args.warehouse_layout,
        "clip_threshold": args.clip_threshold,
        "clip_margin_threshold": args.clip_margin_threshold,
        "compression": args.compression,
        "channel": args.channel,
        "planner": args.planner,
        "local_available": runtime.available(),
        "local_checkpoint": str(runtime.checkpoint),
        "local_attempts": args.local_attempts,
        "policy_device": args.device,
        "clip_prototypes": str(args.clip_prototypes) if args.clip_prototypes.exists() else None,
        "record_demo": bool(args.record_demo),
    }

    records = []
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        for i in range(args.episodes):
            demo_frames = [] if args.record_demo else None
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
                instruction=instruction,
                seed=args.seed + i,
                episode_id=i,
                scene=args.scene,
                routing_config=routing_config,
                max_local_attempts=args.local_attempts,
                demo_frames=demo_frames,
            )
            records.append(record)
            if demo_frames:
                from PIL import Image

                images = [Image.fromarray(frame) for frame in demo_frames]
                images[0].save(
                    run_dir / f"episode_{i:04d}.gif",
                    save_all=True,
                    append_images=images[1:],
                    duration=50,
                    loop=0,
                )
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
