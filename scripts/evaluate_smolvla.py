from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import truststore
from PIL import Image

from architecture_a.contracts import Action
from architecture_a.so101_env import SO101MuJoCoEnvironment
from architecture_a.vla_confidence import estimate_vla_confidence
from shared.handoff import FileHandoffTransport, build_escalation_request


DEFAULT_CHECKPOINT = Path(
    "outputs/train/smolvla_pickplace_50/checkpoints/last/pretrained_model"
)
DEFAULT_VLM = Path(".hf-cache/checkpoints/smolvlm2_500m_video_instruct")
TASK_TEMPLATE = "Pick up the red sample and place it in the {target}."
WAREHOUSE_TASKS = {
    "conveyor": "Pick up the package and place it on the conveyor.",
    "left_tray": "Put the package in the blue inspection tray.",
    "right_tray": "Put the package in the yellow rejection tray.",
}
ACTION_LOW = np.array((-1.8, -1.7, -2.0, -1.8, -2.8, 0.0))
ACTION_HIGH = np.array((1.8, 1.5, 1.8, 1.8, 2.8, 0.025))


def image_tensor(image: np.ndarray) -> torch.Tensor:
    return (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .contiguous()
        .to(dtype=torch.float32)
        .div_(255.0)
    )


def save_replay(frames: list[np.ndarray], path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame) for frame in frames]
    palette = images[0].convert(
        "P",
        palette=Image.Palette.ADAPTIVE,
        colors=256,
        dither=Image.Dither.NONE,
    )
    quantized = [
        image.quantize(palette=palette, dither=Image.Dither.NONE)
        for image in images[1:]
    ]
    palette.save(
        path,
        save_all=True,
        append_images=quantized,
        duration=round(1000 / fps),
        loop=0,
        disposal=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained SmolVLA checkpoint in the SO-101 MuJoCo task."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--vlm", type=Path, default=DEFAULT_VLM)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--replan-every",
        type=int,
        default=0,
        help="Clear the action queue and predict a fresh chunk every N steps; 0 is open-loop.",
    )
    parser.add_argument(
        "--action-smoothing",
        type=float,
        default=1.0,
        help="Weight for each new policy command in (0, 1]; 1 disables low-pass filtering.",
    )
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.0,
        help="Maximum arm-joint change per action; 0 disables the limit.",
    )
    parser.add_argument(
        "--max-gripper-delta",
        type=float,
        default=0.0,
        help="Maximum gripper change per action; 0 disables the limit.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--scene",
        choices=("pick_place", "warehouse_normal", "unexpected_obstacle"),
        default="pick_place",
        help="MuJoCo scene; warehouse_normal is OOD for the current checkpoint.",
    )
    parser.add_argument("--vla-confidence-threshold", type=float, default=0.25)
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Open the live MuJoCo viewer and play actions in real time.",
    )
    parser.add_argument(
        "--gui-replay",
        action="store_true",
        help="Replay the final episode, then hold its final pose until the viewer closes.",
    )
    args = parser.parse_args()
    if args.replan_every < 0:
        parser.error("--replan-every must be zero or greater.")
    if not 0.0 < args.action_smoothing <= 1.0:
        parser.error("--action-smoothing must be in (0, 1].")
    if args.max_joint_delta < 0.0 or args.max_gripper_delta < 0.0:
        parser.error("Action delta limits must be zero or greater.")
    if not 0.0 <= args.vla_confidence_threshold <= 1.0:
        parser.error("--vla-confidence-threshold must be between 0 and 1.")

    checkpoint = args.checkpoint.resolve()
    vlm = args.vlm.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if not vlm.is_dir():
        raise FileNotFoundError(f"Local SmolVLM directory does not exist: {vlm}")

    # Use Windows' certificate store if a library performs an incidental HTTPS check.
    truststore.inject_into_ssl()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(str(checkpoint))
    config.device = args.device
    config.vlm_model_name = str(vlm)
    policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=config)
    policy.to(args.device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "tokenizer_processor": {"tokenizer_name": str(vlm)},
        },
    )

    run_dir = args.output / time.strftime("smolvla-%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = SO101MuJoCoEnvironment(
        gui=args.gui,
        realtime=args.gui,
        observation_images=False,
        kinematic_control=True,
        scenario=args.scene,
    )
    episode_results: list[dict[str, object]] = []
    try:
        for episode_id in range(args.episodes):
            seed = args.seed + episode_id
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            probe = environment.reset(seed=seed, instruction="")
            target = str(probe.metadata["target_name"])
            instruction = (
                TASK_TEMPLATE.format(target=target)
                if args.scene == "pick_place"
                else WAREHOUSE_TASKS[target]
            )
            observation = environment.reset(seed=seed, instruction=instruction)
            initial_sample_position = np.asarray(
                observation.metadata["sample_position"], dtype=np.float32
            )
            policy.reset()
            frames = [
                environment.capture_rgb(
                    camera="overhead", width=320, height=240
                )
            ]
            success = False
            action_history: list[tuple[float, ...]] = []
            clipped_commands = 0
            invalid_action = False
            preexecution_stop_reason = None
            previous_command = np.asarray(
                observation.robot_state, dtype=np.float32
            )
            action_queue: list[np.ndarray] = []
            previous_chunk: np.ndarray | None = None
            vla_confidences = []
            replans = 0
            start = time.perf_counter()

            for step in range(1, args.max_steps + 1):
                if (
                    args.replan_every > 0
                    and step > 1
                    and (step - 1) % args.replan_every == 0
                ):
                    policy.reset()
                    action_queue.clear()
                    replans += 1
                if not action_queue:
                    overhead = environment.capture_rgb(
                        camera="overhead", width=320, height=240
                    )
                    wrist = environment.capture_rgb(
                        camera="wrist", width=320, height=240
                    )
                    batch = preprocessor(
                        {
                            "observation.images.overhead": image_tensor(overhead),
                            "observation.images.wrist": image_tensor(wrist),
                            "observation.state": torch.tensor(
                                observation.robot_state, dtype=torch.float32
                            ),
                            "task": instruction,
                        }
                    )
                    amp_context = (
                        torch.autocast(device_type="cuda")
                        if config.use_amp and args.device.startswith("cuda")
                        else nullcontext()
                    )
                    with torch.inference_mode(), amp_context:
                        chunk = postprocessor(
                            policy.predict_action_chunk(batch)
                        )
                    chunk_values = chunk.squeeze(0).cpu().numpy()
                    estimate = estimate_vla_confidence(
                        chunk_values,
                        action_low=ACTION_LOW,
                        action_high=ACTION_HIGH,
                        previous_chunk=previous_chunk,
                    )
                    vla_confidences.append(estimate)
                    previous_chunk = chunk_values.copy()
                    collision_index = environment.predict_obstacle_collision(
                        chunk_values
                    )
                    if not np.isfinite(chunk_values).all():
                        preexecution_stop_reason = "non_finite_action_chunk"
                    elif np.any(
                        (chunk_values < ACTION_LOW) | (chunk_values > ACTION_HIGH)
                    ):
                        preexecution_stop_reason = "action_chunk_out_of_bounds"
                    elif collision_index is not None:
                        preexecution_stop_reason = (
                            f"predicted_obstacle_collision_at_command_{collision_index}"
                        )
                    elif estimate.confidence < args.vla_confidence_threshold:
                        preexecution_stop_reason = "low_vla_action_consistency"
                    if preexecution_stop_reason is not None:
                        invalid_action = True
                        break
                    if args.action_smoothing < 1.0:
                        padded = np.pad(
                            chunk_values[:, :5],
                            ((1, 1), (0, 0)),
                            mode="edge",
                        )
                        side_weight = (1.0 - args.action_smoothing) / 2.0
                        chunk_values[:, :5] = (
                            side_weight * padded[:-2]
                            + args.action_smoothing * padded[1:-1]
                            + side_weight * padded[2:]
                        )
                    action_queue.extend(chunk_values)

                filtered = action_queue.pop(0)
                if not np.isfinite(filtered).all():
                    invalid_action = True
                    break
                clipped_commands += int(
                    np.any((filtered < ACTION_LOW) | (filtered > ACTION_HIGH))
                )
                delta = filtered - previous_command
                if args.max_joint_delta > 0.0:
                    delta[:5] = np.clip(
                        delta[:5],
                        -args.max_joint_delta,
                        args.max_joint_delta,
                    )
                if args.max_gripper_delta > 0.0:
                    delta[5] = np.clip(
                        delta[5],
                        -args.max_gripper_delta,
                        args.max_gripper_delta,
                    )
                executed = previous_command + delta
                previous_command = executed
                values = tuple(float(value) for value in executed)
                action_history.append(values)
                transition = environment.step(
                    Action("joint_position", values=values)
                )
                observation = transition.observation
                success = bool(transition.info["success"])
                frames.append(
                    environment.capture_rgb(
                        camera="overhead", width=320, height=240
                    )
                )
                if transition.terminated or transition.truncated:
                    break

            held_object = bool(observation.metadata["held_object"])
            final_sample = np.asarray(observation.metadata["sample_position"])
            if success:
                failure_reason = None
            elif invalid_action:
                failure_reason = preexecution_stop_reason or "invalid_policy_action"
            elif held_object:
                failure_reason = "object_not_released"
            elif (
                np.linalg.norm(
                    final_sample[:2] - initial_sample_position[:2]
                )
                < 0.03
            ):
                failure_reason = "grasp_failed"
            else:
                failure_reason = "placement_failed"

            minimum_vla_confidence = min(
                (item.confidence for item in vla_confidences), default=0.0
            )
            result = {
                "episode_id": episode_id,
                "seed": seed,
                "instruction": instruction,
                "scene": args.scene,
                "scene_distribution": (
                    "training_distribution"
                    if args.scene == "pick_place"
                    else "out_of_distribution_warehouse_layout"
                ),
                "success": success,
                "steps": step,
                "latency_seconds": round(time.perf_counter() - start, 3),
                "replans": replans,
                "clipped_commands": clipped_commands,
                "failure_reason": failure_reason,
                "preexecution_stop_reason": preexecution_stop_reason,
                "minimum_vla_confidence": round(minimum_vla_confidence, 6),
                "vla_confidence_method": "chunk_smoothness_and_temporal_consistency",
                "vla_confidence_samples": [
                    {
                        "confidence": round(item.confidence, 6),
                        "chunk_smoothness": round(item.chunk_smoothness, 6),
                        "temporal_consistency": round(item.temporal_consistency, 6),
                    }
                    for item in vla_confidences
                ],
                "escalation_recommended": (
                    not success
                    or minimum_vla_confidence < args.vla_confidence_threshold
                ),
                "final_sample_position": [
                    round(float(value), 5)
                    for value in observation.metadata["sample_position"]
                ],
                "target_position": [
                    round(float(value), 5)
                    for value in observation.metadata["target_position"]
                ],
                "action_min": (
                    [
                        round(float(value), 5)
                        for value in np.min(action_history, axis=0)
                    ]
                    if action_history
                    else None
                ),
                "action_max": (
                    [
                        round(float(value), 5)
                        for value in np.max(action_history, axis=0)
                    ]
                    if action_history
                    else None
                ),
            }
            if result["escalation_recommended"]:
                evidence_name = f"escalation_{episode_id:04d}.png"
                Image.fromarray(frames[-1]).save(run_dir / evidence_name)
                reasons = []
                if failure_reason is not None:
                    reasons.append(failure_reason)
                if minimum_vla_confidence < args.vla_confidence_threshold:
                    reasons.append("low_vla_action_consistency")
                escalation_payload = build_escalation_request(
                    handoff_stage=(
                        "pre_execution_validation"
                        if preexecution_stop_reason is not None
                        else "post_execution_validation"
                    ),
                    episode_id=episode_id,
                    instruction=instruction,
                    reasons=reasons,
                    evidence_image=evidence_name,
                    robot_state=observation.robot_state,
                    task_state={
                        "held_object": held_object,
                        "package_position": result["final_sample_position"],
                        "target_name": observation.metadata["target_name"],
                        "target_position": result["target_position"],
                        "problem": observation.metadata["problem"],
                    },
                    architecture_a_signals={
                        "minimum_vla_confidence": round(
                            minimum_vla_confidence, 6
                        ),
                        "required_vla_confidence": args.vla_confidence_threshold,
                        "task_success": success,
                        "clipped_commands": clipped_commands,
                    },
                )
                (run_dir / f"escalation_{episode_id:04d}.json").write_text(
                    json.dumps(escalation_payload, indent=2), encoding="utf-8"
                )
                FileHandoffTransport(run_dir / "handoff").write_request(
                    escalation_payload
                )
            episode_results.append(result)
            if args.gui_replay and episode_id == args.episodes - 1:
                replay_environment = SO101MuJoCoEnvironment(
                    gui=True,
                    realtime=True,
                    observation_images=False,
                    kinematic_control=True,
                    scenario=args.scene,
                )
                try:
                    replay_environment.reset(seed=seed, instruction=instruction)
                    for values in action_history:
                        replay_environment.step(
                            Action("joint_position", values=values)
                        )
                    print("Replay complete. Close the MuJoCo window to finish.")
                    replay_environment.hold_viewer_static()
                finally:
                    replay_environment.close()
            elif args.gui and episode_id == args.episodes - 1:
                environment.close()
            save_replay(frames, run_dir / f"episode_{episode_id:04d}.gif")
            print(json.dumps(result))
    finally:
        environment.close()

    summary = {
        "checkpoint": str(checkpoint),
        "scene": args.scene,
        "episodes": len(episode_results),
        "successes": sum(bool(item["success"]) for item in episode_results),
        "success_rate": (
            sum(bool(item["success"]) for item in episode_results)
            / len(episode_results)
        ),
        "replan_every": args.replan_every,
        "action_smoothing": args.action_smoothing,
        "gui": args.gui,
        "gui_replay": args.gui_replay,
        "max_joint_delta": args.max_joint_delta,
        "max_gripper_delta": args.max_gripper_delta,
        "escalations": sum(
            bool(item["escalation_recommended"])
            for item in episode_results
        ),
        "failure_counts": dict(
            Counter(
                str(item["failure_reason"])
                for item in episode_results
                if item["failure_reason"] is not None
            )
        ),
        "results": episode_results,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Evaluation artifacts: {run_dir.resolve()}")


if __name__ == "__main__":
    main()
