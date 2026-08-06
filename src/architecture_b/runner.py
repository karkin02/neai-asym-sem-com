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
    max_frame_attempts: int = 3,
    advance_world_during_network: bool = False,
    initial_world_delay: float = 0.0,
    max_trial_seconds: float | None = None,
) -> TrialRecord:
    """Run one Architecture B trial and return an A-comparable record."""
    width, height = frame_size
    start = time.perf_counter()
    configure_world_motion = getattr(controller, "set_world_advances_during_network", None)
    if callable(configure_world_motion):
        configure_world_motion(advance_world_during_network)

    def measured_latency() -> float:
        return (time.perf_counter() - start) + (0.0 if channel.realtime else channel_latency)

    def needs_condition_reacquisition(observations: dict[str, Any]) -> bool:
        damage_confidence = float(observations.get("damage_mark_max_confidence", 0.0))
        damage_threshold = float(observations.get("damage_mark_confidence_threshold", 0.45))
        return (
            not observations.get("inspection_complete", False)
            or 0.0 < damage_confidence < damage_threshold
        )

    def allow_reacquisition_motion() -> None:
        # Move an ambiguous package away from the endpoint/turnaround before
        # repeating the wrist view. This is a fixed visual recovery action and
        # does not inspect simulator scenario metadata or the hidden condition.
        if bool(getattr(env, "_inbound_feeder", False)):
            advance = getattr(env, "advance_idle", None)
            if callable(advance):
                advance(5.0)

    def package_in_visual_pick_zone(items: Sequence[Any]) -> bool:
        projector = getattr(env, "camera_pixel_to_world", None)
        if not callable(projector):
            return True
        packages = [
            item for item in items
            if str(item.label) in {"package", "box", "sample", "parcel"}
            and item.bbox[2] - item.bbox[0] < 50
            and item.bbox[3] - item.bbox[1] < 50
        ]
        if not packages:
            return False
        item = max(packages, key=lambda value: float(value.confidence))
        position = projector(item.center, camera=camera, width=width, height=height)
        return abs(float(position[0])) <= 0.065

    env.reset(seed=seed, instruction=instruction)
    if initial_world_delay > 0 and callable(getattr(env, "advance_idle", None)):
        env.advance_idle(initial_world_delay)
    frame = env.capture_rgb(camera=camera, width=width, height=height)

    detections = detector.detect(frame)
    visual_observer = getattr(controller, "observe_visual", None)
    if callable(visual_observer):
        visual_observer(detections, camera=camera, width=width, height=height)
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
    observations = visual_observations_for_views(view_detections)
    if callable(capture_conditions) and needs_condition_reacquisition(observations):
        allow_reacquisition_motion()
        frame = env.capture_rgb(camera=camera, width=width, height=height)
        detections = detector.detect(frame)
        if callable(visual_observer):
            visual_observer(detections, camera=camera, width=width, height=height)
        scene_graph = build_scene_graph(detections, width, height, task=instruction)
        condition_frames = capture_conditions(width=width, height=height)
        view_detections = {
            camera: list(detections),
            **{view: list(detector.detect(image)) for view, image in condition_frames.items()},
        }
        if "damage" in condition_frames:
            damage_cue = detect_package_damage_mark(
                condition_frames["damage"], view_detections["damage"]
            )
            if damage_cue is not None:
                view_detections["damage"].append(damage_cue)
        observations = visual_observations_for_views(view_detections)
    if callable(capture_conditions) and needs_condition_reacquisition(observations):
        allow_reacquisition_motion()
        frame = env.capture_rgb(camera=camera, width=width, height=height)
        detections = detector.detect(frame)
        if callable(visual_observer):
            visual_observer(detections, camera=camera, width=width, height=height)
        scene_graph = build_scene_graph(detections, width, height, task=instruction)
        condition_frames = capture_conditions(width=width, height=height)
        view_detections = {
            camera: list(detections),
            **{view: list(detector.detect(image)) for view, image in condition_frames.items()},
        }
        if "damage" in condition_frames:
            damage_cue = detect_package_damage_mark(
                condition_frames["damage"], view_detections["damage"]
            )
            if damage_cue is not None:
                view_detections["damage"].append(damage_cue)
        observations = visual_observations_for_views(view_detections)
    scene_graph["visual_observations"] = observations
    if bool(getattr(env, "_inbound_feeder", False)):
        frame = env.capture_rgb(camera=camera, width=width, height=height)
        detections = detector.detect(frame)
        if callable(visual_observer):
            visual_observer(detections, camera=camera, width=width, height=height)
        tracked_scene = build_scene_graph(detections, width, height, task=instruction)
        tracked_scene["visual_observations"] = observations
        scene_graph = tracked_scene

    clip_confidence = None
    if grounder is not None:
        crops = [crop_bbox(frame, d.bbox) for d in detections]
        clip_confidence = grounder.score(referring_expression or instruction, crops).confidence

    payload = build_payload(compression_level, scene_graph, instruction, frame=frame)
    transmitted_bytes = payload.num_bytes
    dropped_frames = 0
    channel_latency = 0.0
    for frame_attempt in range(1, max(1, max_frame_attempts) + 1):
        transmission = channel.transmit(payload.num_bytes)
        if advance_world_during_network and callable(getattr(env, "advance_idle", None)):
            env.advance_idle(transmission.latency_seconds)
        channel_latency += transmission.latency_seconds
        note_latency = getattr(controller, "note_link_latency", None)
        if transmission.delivered and callable(note_latency):
            note_latency(transmission.latency_seconds)
        if transmission.delivered:
            break
        dropped_frames += 1
        if frame_attempt < max(1, max_frame_attempts):
            # A lost observation is skipped. Capture and encode a fresh frame;
            # no planner or robot command runs until one is delivered.
            frame = env.capture_rgb(camera=camera, width=width, height=height)
            detections = detector.detect(frame)
            scene_graph = build_scene_graph(detections, width, height, task=instruction)
            if callable(capture_conditions):
                condition_frames = capture_conditions(width=width, height=height)
                view_detections = {
                    camera: list(detections),
                    **{view: list(detector.detect(image)) for view, image in condition_frames.items()},
                }
                if "damage" in condition_frames:
                    damage_cue = detect_package_damage_mark(
                        condition_frames["damage"], view_detections["damage"]
                    )
                    if damage_cue is not None:
                        view_detections["damage"].append(damage_cue)
            else:
                wrist_frame = env.capture_rgb(camera="wrist", width=width, height=height)
                view_detections = {
                    camera: list(detections),
                    "wrist": list(detector.detect(wrist_frame)),
                }
            scene_graph["visual_observations"] = visual_observations_for_views(view_detections)
            payload = build_payload(compression_level, scene_graph, instruction, frame=frame)
            transmitted_bytes += payload.num_bytes

    record = TrialRecord(
        architecture="B",
        episode_id=episode_id,
        seed=seed,
        instruction=instruction,
        scene=scene,
        success=False,
        network_payload_bytes=transmitted_bytes,
        clip_confidence=clip_confidence,
        route="cloud",
        channel_condition=transmission.condition,
        compression_level=str(payload.level.value),
    )

    if not transmission.delivered:
        # Every frame in the bounded recapture window was lost.
        record.failure_reason = "channel_drop"
        record.extra = {"frame_attempts": frame_attempt, "dropped_frames": dropped_frames}
        record.latency_seconds = measured_latency()
        return record

    record.extra = {"frame_attempts": frame_attempt, "dropped_frames": dropped_frames}
    if max_trial_seconds is not None and measured_latency() >= max_trial_seconds:
        record.failure_reason = "trial_timeout"
        record.latency_seconds = measured_latency()
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
        record.latency_seconds = measured_latency()
        record.extra = {
            "frame_attempts": frame_attempt,
            "dropped_frames": dropped_frames,
            "recovery_command": "STOP",
            "safety_hold": True,
            "obstacle_rechecks": len(recheck_labels),
            "obstacle_recheck_labels": recheck_labels,
            "joint_commands_executed": 0,
            "stop_reason": "reported_obstacle_not_positively_verified_clear",
        }
        return record

    target = planner.plan(payload)
    initial_observations = scene_graph.get("visual_observations", {})
    barcode_latched = bool(
        initial_observations.get("barcode_inspection_complete")
        and initial_observations.get("barcode_detected")
    )
    missing_barcode_latched = bool(
        initial_observations.get("barcode_inspection_complete")
        and not initial_observations.get("barcode_detected")
        and target.destination == "inspection_tray"
    )
    damage_latched = bool(initial_observations.get("damage_mark_detected"))
    command = str(getattr(target, "command", "STOP")).upper()
    confidence = float(getattr(target, "confidence", 0.0))
    if command == "STOP" or confidence < 0.5 or target.destination is None:
        record.steps = 0
        record.failure_reason = "planner_safe_stop"
        record.latency_seconds = measured_latency()
        record.extra = {
            "frame_attempts": frame_attempt,
            "dropped_frames": dropped_frames,
            "action_target": target.as_dict(),
            "visual_observations": scene_graph["visual_observations"],
            "joint_commands_executed": 0,
            "stop_reason": "invalid_or_low_confidence_semantic_plan",
        }
        return record
    prepare = getattr(controller, "prepare", None)
    if callable(prepare):
        prepare(seed)

    checkpoint_log: list[dict[str, Any]] = []
    deadline_exceeded = False

    def replan_checkpoint(phase: str) -> ActionTarget | None:
        nonlocal channel_latency, dropped_frames, deadline_exceeded
        for attempt in range(1, max(1, max_frame_attempts) + 1):
            fresh_frame = env.capture_rgb(camera=camera, width=width, height=height)
            fresh_detections = detector.detect(fresh_frame)
            if callable(visual_observer):
                visual_observer(fresh_detections, camera=camera, width=width, height=height)
            if bool(getattr(env, "_inbound_feeder", False)):
                # Two causal overhead frames provide a visual velocity estimate
                # around a feeder endpoint without using modeled link delay.
                env.advance_idle(0.5)
                fresh_frame = env.capture_rgb(camera=camera, width=width, height=height)
                fresh_detections = detector.detect(fresh_frame)
                if callable(visual_observer):
                    visual_observer(
                        fresh_detections, camera=camera, width=width, height=height
                    )
            fresh_scene = build_scene_graph(fresh_detections, width, height, task=instruction)
            fresh_views = {camera: list(fresh_detections)}
            capture = getattr(env, "capture_condition_views", None)
            if callable(capture):
                condition_frames = capture(width=width, height=height)
                fresh_views.update(
                    {view: list(detector.detect(image)) for view, image in condition_frames.items()}
                )
                if "damage" in condition_frames:
                    from shared.perception import detect_package_damage_mark

                    cue = detect_package_damage_mark(condition_frames["damage"], fresh_views["damage"])
                    if cue is not None:
                        fresh_views["damage"].append(cue)
            observations = visual_observations_for_views(fresh_views)
            # A moving package can reverse after the overhead observation while
            # the wrist is travelling to its calibrated inspection pose.  An
            # incomplete close-view set is not evidence that the condition
            # changed: reacquire once from the package's current position.
            if (
                callable(capture)
                and needs_condition_reacquisition(observations)
            ):
                allow_reacquisition_motion()
                fresh_frame = env.capture_rgb(camera=camera, width=width, height=height)
                fresh_detections = detector.detect(fresh_frame)
                if callable(visual_observer):
                    visual_observer(
                        fresh_detections, camera=camera, width=width, height=height
                    )
                fresh_scene = build_scene_graph(
                    fresh_detections, width, height, task=instruction
                )
                condition_frames = capture(width=width, height=height)
                fresh_views = {
                    camera: list(fresh_detections),
                    **{view: list(detector.detect(image)) for view, image in condition_frames.items()},
                }
                if "damage" in condition_frames:
                    cue = detect_package_damage_mark(condition_frames["damage"], fresh_views["damage"])
                    if cue is not None:
                        fresh_views["damage"].append(cue)
                observations = visual_observations_for_views(fresh_views)
            fresh_scene["visual_observations"] = observations
            if bool(getattr(env, "_inbound_feeder", False)):
                fresh_frame = env.capture_rgb(camera=camera, width=width, height=height)
                fresh_detections = detector.detect(fresh_frame)
                if callable(visual_observer):
                    visual_observer(
                        fresh_detections, camera=camera, width=width, height=height
                    )
                tracked_scene = build_scene_graph(
                    fresh_detections, width, height, task=instruction
                )
                tracked_scene["visual_observations"] = observations
                fresh_scene = tracked_scene
            fresh_payload = build_payload(
                compression_level, fresh_scene, instruction, frame=fresh_frame
            )
            record.network_payload_bytes += fresh_payload.num_bytes
            sent = channel.transmit(fresh_payload.num_bytes)
            billed_delay = sent.latency_seconds
            if max_trial_seconds is not None:
                remaining = max(0.0, max_trial_seconds - measured_latency())
                if billed_delay >= remaining:
                    if advance_world_during_network and callable(getattr(env, "advance_idle", None)):
                        env.advance_idle(remaining)
                    channel_latency += remaining
                    deadline_exceeded = True
                    return None
            if advance_world_during_network and callable(getattr(env, "advance_idle", None)):
                env.advance_idle(billed_delay)
            channel_latency += billed_delay
            if sent.delivered and callable(note_latency):
                note_latency(sent.latency_seconds)
            if not sent.delivered:
                dropped_frames += 1
                continue
            if max_trial_seconds is not None and measured_latency() >= max_trial_seconds:
                deadline_exceeded = True
                return None
            candidate = planner.plan(fresh_payload)
            temporal_latch = False
            missing_barcode_temporal_latch = False
            damage_temporal_latch = False
            local_hold_latch = False
            if phase in {"carry", "release"} and bool(getattr(env, "holding_package", False)):
                if str(getattr(candidate, "command", "STOP")).upper() == "STOP":
                    candidate = target
                    local_hold_latch = True
            if (
                barcode_latched
                and target.destination == "conveyor"
                and (
                    str(getattr(candidate, "command", "STOP")).upper() == "STOP"
                    or candidate.destination != "conveyor"
                )
            ):
                # A certified positive barcode cannot disappear. A later
                # negative from a moving/non-certified view is not new proof.
                candidate = target
                temporal_latch = True
            if (
                missing_barcode_latched
                and target.destination == "inspection_tray"
                and (
                    str(getattr(candidate, "command", "STOP")).upper() == "STOP"
                    or candidate.destination != "inspection_tray"
                )
            ):
                candidate = target
                missing_barcode_temporal_latch = True
            if (
                damage_latched
                and target.destination == "rejection_tray"
                and candidate.destination != "rejection_tray"
            ):
                candidate = target
                damage_temporal_latch = True
            valid = (
                str(getattr(candidate, "command", "STOP")).upper() != "STOP"
                and float(getattr(candidate, "confidence", 0.0)) >= 0.5
                and candidate.destination is not None
            )
            if (
                valid
                and phase == "grasp"
                and bool(getattr(env, "_inbound_feeder", False))
                and callable(visual_observer)
            ):
                # Semantic routing may arrive late, but final interception is
                # local: align from a current camera frame without predicting
                # or exposing the network delay to the controller.
                current_frame = env.capture_rgb(camera=camera, width=width, height=height)
                current_detections = detector.detect(current_frame)
                for _ in range(16):
                    if package_in_visual_pick_zone(current_detections):
                        break
                    env.advance_idle(1.0)
                    current_frame = env.capture_rgb(
                        camera=camera, width=width, height=height
                    )
                    current_detections = detector.detect(current_frame)
                if callable(note_latency):
                    note_latency(0.0)
                visual_observer(
                    current_detections, camera=camera, width=width, height=height
                )
                env.advance_idle(0.5)
                current_frame = env.capture_rgb(camera=camera, width=width, height=height)
                current_detections = detector.detect(current_frame)
                set_motion_horizon = getattr(controller, "set_local_motion_horizon", None)
                if callable(set_motion_horizon):
                    set_motion_horizon(1.0)
                visual_observer(
                    current_detections, camera=camera, width=width, height=height
                )
            checkpoint_log.append({
                "phase": phase,
                "attempts": attempt,
                "command": candidate.command,
                "destination": candidate.destination,
                "accepted": valid,
                "temporal_barcode_latch": temporal_latch,
                "temporal_missing_barcode_latch": missing_barcode_temporal_latch,
                "temporal_damage_latch": damage_temporal_latch,
                "local_hold_latch": local_hold_latch,
            })
            return candidate if valid else None
        checkpoint_log.append({"phase": phase, "attempts": max_frame_attempts, "accepted": False})
        return None

    start_session = getattr(controller, "start", None)
    advance_session = getattr(controller, "advance", None)
    if callable(start_session) and callable(advance_session):
        session = start_session(target)
        first_execution_phase = True
        while session.phase not in {"complete", "failed"}:
            # The initial plan authorizes approach; every later phase requires
            # fresh visual evidence and may safely reroute the destination.
            if first_execution_phase:
                phase_target = target
            elif session.phase == "release":
                reviewed = replan_checkpoint("release")
                if reviewed is None:
                    phase_target = None
                else:
                    # After carry, semantic destination is locked. This review
                    # may stop release but can never reroute it.
                    checkpoint_log[-1]["requested_destination"] = reviewed.destination
                    checkpoint_log[-1]["destination"] = target.destination
                    checkpoint_log[-1]["destination_locked"] = True
                    phase_target = target
            else:
                phase_target = replan_checkpoint(session.phase)
            if phase_target is None:
                session.result.failure_reason = (
                    "trial_timeout" if deadline_exceeded else "replanning_safe_stop"
                )
                session.phase = "failed"
                break
            target = phase_target
            advance_session(session, target)
            first_execution_phase = False
        execution = session.result
    else:
        execution = controller.execute(target)

    if execution.success:
        audit = replan_checkpoint("post_release")
        checkpoint_log[-1]["audit_only"] = True
        checkpoint_log[-1]["motion_authority"] = False
        checkpoint_log[-1]["audit_received"] = audit is not None

    final_latency = measured_latency()
    final_timeout = bool(
        max_trial_seconds is not None and final_latency >= max_trial_seconds
    )
    record.success = bool(execution.success and not final_timeout)
    record.steps = execution.steps
    record.failure_reason = "trial_timeout" if final_timeout else execution.failure_reason
    record.latency_seconds = final_latency
    record.extra = {
        "frame_attempts": frame_attempt,
        "dropped_frames": dropped_frames,
        "action_target": target.as_dict(),
        "visual_observations": scene_graph["visual_observations"],
        "replanning_checkpoints": checkpoint_log,
    }
    return record


def _build_env(scene: str, gui: bool, realtime: bool, warehouse_layout: str,
               inbound_feeder: bool = False) -> Any:
    from shared.environments import SO101MuJoCoEnvironment  # lazy: needs mujoco

    return SO101MuJoCoEnvironment(
        gui=gui,
        realtime=realtime,
        observation_images=False,
        kinematic_control=True,
        scenario=scene,
        warehouse_layout=warehouse_layout,
        inbound_feeder=inbound_feeder,
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
    parser.add_argument("--channel", choices=["clean", "throttled", "restricted", "delayed", "degraded", "practical", "stressed", "extreme", "level1", "level2", "level3", "level4", "level5"], default="clean")
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
    parser.add_argument("--frame-attempts", type=int, default=3)
    parser.add_argument("--advance-world-during-network", action="store_true")
    parser.add_argument("--inbound-feeder", action="store_true",
                        help="Enable the bidirectional semicircular inbound feeder.")
    parser.add_argument("--feeder-observation-delay", type=float, default=12.0,
                        help="Seconds to advance an enabled feeder before initial inspection.")
    parser.add_argument("--max-trial-seconds", type=float, default=60.0,
                        help="Safe-stop deadline including modeled network delay.")
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

    env = _build_env(args.scene, args.gui, args.realtime, args.warehouse_layout,
                     args.inbound_feeder)
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

    # Load perception backends before episode timing. The measured trial still
    # includes normal per-frame inference, but excludes one-time model loading.
    cold_start_begin = time.perf_counter()
    env.reset(seed=args.seed, instruction=instruction)
    warm_frame = env.capture_rgb(camera="overhead", width=320, height=240)
    detector.detect(warm_frame)
    if grounder is not None:
        grounder.score(instruction, [warm_frame])
    cold_start_seconds = time.perf_counter() - cold_start_begin

    run_dir = args.output / time.strftime("architecture_b-%Y%m%d-%H%M%S")
    config = {
        "channel_benchmark_version": "v2-refined-thresholds",
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
        "cold_start_seconds_excluded": cold_start_seconds,
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
                max_frame_attempts=args.frame_attempts,
                advance_world_during_network=args.advance_world_during_network,
                initial_world_delay=(args.feeder_observation_delay if args.inbound_feeder else 0.0),
                max_trial_seconds=args.max_trial_seconds,
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
