#!/usr/bin/env python3
"""
Main entry point for the LeRobot Simulation Environment.

Demonstrates the full pipeline: environment instantiation, data collection
via teleoperation or random policy, behaviour-cloning training, evaluation,
and optional live visualization.  Run directly with ``python run_sim.py``
or import individual components for custom workflows.

Usage examples::

    # Full automated pipeline (collect, train, evaluate)
    python run_sim.py --task pusht --mode train

    # Keyboard teleoperation with live rendering
    python run_sim.py --task reach --mode teleop

    # Vision-based teleoperation using real webcam + YOLO detection
    python run_sim.py --task pick_place --mode vision --target-label cup

    # Evaluate a previously trained policy
    python run_sim.py --task pick_place --mode eval
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict

import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

from lerobot_sim.datasets.dataset_manager import SimDatasetRecorder
from lerobot_sim.envs.configs import (
    PickPlaceSimConfig,
    PushTSimConfig,
    ReachSimConfig,
    SimEnvConfig,
)
from lerobot_sim.envs.factory import make_sim_env
from lerobot_sim.policies.policy import MLPPolicy, RandomPolicy
from lerobot_sim.policies.trainer import Trainer, TrainerConfig
from lerobot_sim.teleop.keyboard_teleop import KeyboardTeleop
from lerobot_sim.teleop.vision_teleop import VisionTeleop
from lerobot_sim.visualization.visualizer import SimVisualizer

load_dotenv()
VISUAL_ASSISTANT_DIR = os.path.join(os.path.dirname(__file__), "..", "visual_assistant")
sys.path.append(VISUAL_ASSISTANT_DIR)
from detector import Detector
from prompt_templates import build_action_prompt
from action_utils import parse_action_response, bbox_center_to_action, object_world_pos_for_task


# ======================================================================
# Configuration builders
# ======================================================================


def _build_env_config(task: str) -> SimEnvConfig:
    """Return the environment configuration for a given task name.

    Args:
        task: One of ``'pusht'``, ``'pick_place'``, or ``'reach'``.

    Returns:
        A concrete ``SimEnvConfig`` instance.

    Raises:
        ValueError: If the task name is not recognised.
    """
    registry = {
        "pusht": PushTSimConfig,
        "pick_place": PickPlaceSimConfig,
        "reach": ReachSimConfig,
    }
    if task not in registry:
        raise ValueError(f"Unknown task '{task}'. Choose from {list(registry)}")
    return registry[task]()


def _build_policy(cfg: SimEnvConfig, hidden_dim: int = 256) -> MLPPolicy:
    """Create an MLP policy sized for the given environment configuration.

    Args:
        cfg: Environment configuration with ``state_dim`` and ``action_dim``.
        hidden_dim: Width of hidden layers.

    Returns:
        An ``MLPPolicy`` instance.
    """
    state_dim = getattr(cfg, "state_dim", 2)
    action_dim = getattr(cfg, "action_dim", 2)
    return MLPPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim)


def _build_trainer_config(args: argparse.Namespace) -> TrainerConfig:
    """Construct a ``TrainerConfig`` from parsed CLI arguments.

    Args:
        args: Namespace from ``argparse``.

    Returns:
        A ``TrainerConfig`` instance.
    """
    return TrainerConfig(
        num_demo_episodes=args.demo_episodes,
        num_train_steps=args.train_steps,
        num_eval_episodes=args.eval_episodes,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def _find_latest_episode_dir(output_dir: str) -> str:
    """Return the most recently modified episode_* directory in output_dir."""
    candidates = glob.glob(os.path.join(output_dir, "episode_*"))
    if not candidates:
        raise FileNotFoundError(f"No episode_* folders found in {output_dir}")
    return max(candidates, key=os.path.getmtime)


def _plot_episode(episode_dir: str) -> str:
    """Load tabular.npz from an episode dir and save an action/agent_pos plot.

    Args:
        episode_dir: Path to a single episode's data folder.

    Returns:
        Path to the saved PNG.
    """
    data = np.load(os.path.join(episode_dir, "tabular.npz"))
    action = data["action"]
    agent_pos = data["agent_pos"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    for i in range(action.shape[1]):
        axes[0].plot(action[:, i], label=f"action[{i}]", linewidth=1, alpha=0.8)
    axes[0].set_title("Action trajectory")
    axes[0].set_xlabel("timestep")
    axes[0].legend(loc="upper right", fontsize=6, ncol=2)
    axes[0].grid(alpha=0.3)

    for i in range(agent_pos.shape[1]):
        axes[1].plot(agent_pos[:, i], label=f"agent_pos[{i}]", linewidth=1, alpha=0.8)
    axes[1].set_title("Agent position trajectory")
    axes[1].set_xlabel("timestep")
    axes[1].legend(loc="upper right", fontsize=6, ncol=2)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(episode_dir, "episode_check.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved episode plot to {out_path}")
    return out_path


# ======================================================================
# Mode runners
# ======================================================================


def _run_train(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Execute the full collect, train, evaluate pipeline and save best model.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    policy = _build_policy(env_cfg, hidden_dim=args.hidden_dim)
    trainer_cfg = _build_trainer_config(args)
    trainer = Trainer(env_cfg=env_cfg, policy=policy, config=trainer_cfg)
    results = trainer.run()
    print(f"\nFinal eval: {results['eval_metrics']}")
    print(f"Best model saved to: {args.output_dir}/best_policy.npz")


def _load_trained_policy(args: argparse.Namespace) -> MLPPolicy:
    """Load the best saved policy checkpoint from output_dir.

    Args:
        args: Parsed CLI arguments with ``output_dir``.

    Returns:
        An ``MLPPolicy`` with restored weights.

    Raises:
        FileNotFoundError: If no checkpoint exists.
    """
    best = os.path.join(args.output_dir, "best_policy.npz")
    if not os.path.exists(best):
        raise FileNotFoundError(f"No checkpoint at {best}. Run --mode train first.")
    print(f"Loading checkpoint: {best}")
    return MLPPolicy.load(best)


def _run_eval(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Load the best trained policy and run inference/evaluation.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    policy = _load_trained_policy(args)
    trainer_cfg = _build_trainer_config(args)
    trainer = Trainer(env_cfg=env_cfg, policy=policy, config=trainer_cfg)
    metrics = trainer.evaluate()
    print(f"\nInference results: {metrics}")


def _run_teleop_loop(env: Any, teleop: KeyboardTeleop, recorder: SimDatasetRecorder) -> None:
    """Run the keyboard teleoperation loop for one episode.

    Args:
        env: Gymnasium environment instance.
        teleop: Keyboard teleop interface.
        recorder: Dataset recorder.
    """
    obs, _ = env.reset()
    recorder.start_episode()
    done = False
    while not done:
        action = teleop.get_action()
        recorder.record_step(obs, action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    recorder.end_episode()


def _run_teleop(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Run keyboard teleoperation to collect demonstrations.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    from lerobot_sim.envs.factory import _env_class_for_config

    env = _env_class_for_config(env_cfg)(env_cfg)
    action_dim = getattr(env_cfg, "action_dim", 2)
    teleop = KeyboardTeleop(action_dim=action_dim)
    recorder = SimDatasetRecorder(output_dir=args.output_dir)
    print("Teleop mode: use W/A/S/D to move, Q to quit.")
    print("Recording one episode via terminal input …")
    _run_teleop_loop(env, teleop, recorder)
    recorder.save_metadata()
    print(f"Episode recorded to {args.output_dir}")
    episode_dir = _find_latest_episode_dir(args.output_dir)
    _plot_episode(episode_dir)


def _run_vision_teleop_loop(env: Any, teleop: VisionTeleop, cap: Any, recorder: SimDatasetRecorder) -> None:
    """Run the vision-based teleoperation loop for one episode.

    Args:
        env: Gymnasium environment instance.
        teleop: VisionTeleop interface.
        cap: OpenCV VideoCapture object for the webcam.
        recorder: Dataset recorder.
    """
    obs, _ = env.reset()
    recorder.start_episode()
    done = False
    while not done:
        ret, frame = cap.read()
        if not ret:
            continue
        teleop.set_frame(frame)
        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        action = teleop.get_action()
        recorder.record_step(obs, action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    recorder.end_episode()


def _run_vision_teleop(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Run real-webcam vision-based teleoperation to collect demonstrations.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    from lerobot_sim.envs.factory import _env_class_for_config

    env = _env_class_for_config(env_cfg)(env_cfg)
    action_dim = getattr(env_cfg, "action_dim", 2)

    detector = Detector(
        model_path=os.path.join(VISUAL_ASSISTANT_DIR, "yolov8n.pt"),
        conf_threshold=0.5,
    )
    teleop = VisionTeleop(detector=detector, action_dim=action_dim, target_label=args.target_label)

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    recorder = SimDatasetRecorder(output_dir=args.output_dir)
    print(f"Vision teleop mode: tracking '{args.target_label}' from webcam.")
    try:
        _run_vision_teleop_loop(env, teleop, cap, recorder)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    recorder.save_metadata()
    print(f"Episode recorded to {args.output_dir}")
    episode_dir = _find_latest_episode_dir(args.output_dir)
    _plot_episode(episode_dir)


def _run_llm_vision_loop(
    env: Any,
    detector: Detector,
    client: OpenAI,
    cap: Any,
    recorder: SimDatasetRecorder,
    instruction: str,
    action_dim: int,
    gpt_every_n: int = 10,
    object_world_pos: Any = None,
) -> None:
    """Run LLM-driven vision control for one episode.

    GPT-4o-mini is only queried every `gpt_every_n` frames; the last
    valid action is reused in between to keep the control loop fast.

    If ``object_world_pos`` is given, the sim's object is spawned there
    (reconstructed from the real camera) at reset instead of a random pose.
    """
    reset_options = {"object_world_pos": object_world_pos} if object_world_pos is not None else None
    obs, _ = env.reset(options=reset_options)
    recorder.start_episode()
    done = False
    step_idx = 0

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    last_action = np.zeros(action_dim, dtype=np.float32)

    while not done:
        ret, frame = cap.read()
        if not ret:
            continue

        cv2.imshow("Webcam Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        objects = detector.detect(frame)

        if step_idx % gpt_every_n == 0:
            print(f"[detector] step {step_idx} objects: {objects}")
            print(f"[instruction] {instruction}")

            prompt = build_action_prompt(objects, instruction)
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content
                parsed = parse_action_response(response_text)
                print(f"[llm] parsed action target: {parsed}")

                if parsed.get("target_position") is not None:
                    last_action = np.array(
                        bbox_center_to_action(
                            parsed["target_position"],
                            frame_width=frame_width,
                            frame_height=frame_height,
                            action_dim=action_dim,
                            gripper_cmd=parsed.get("gripper", "open"),
                        ),
                        dtype=np.float32,
                    )
                else:
                    print("[llm] no target found, holding last action")
            except Exception as e:
                print(f"[llm] error: {e}")

        recorder.record_step(obs, last_action)
        obs, _, terminated, truncated, _ = env.step(last_action)
        done = terminated or truncated
        step_idx += 1

    recorder.end_episode()


def _reconstruct_object_world_pos(
    detector: Detector, cap: Any, task: str, target_label: str, instruction: str
) -> Any:
    """Grab a webcam frame, detect objects, and map the chosen object's bbox
    into a world position for the given task.

    Selection prefers the object matching ``target_label``, then any label
    named in the instruction, then the most confident detection. Returns the
    world position (2-D for pusht, 3-D otherwise), or ``None`` if nothing was
    detected so the caller can fall back to random placement.
    """
    frame = None
    for _ in range(5):  # warm up so exposure/autofocus settle
        ret, f = cap.read()
        if ret and f is not None:
            frame = f
    if frame is None:
        return None
    detections = detector.detect(frame)
    if not detections:
        print("[reconstruct] no objects detected; using random object placement")
        return None
    instr = (instruction or "").lower()
    tl = (target_label or "").lower()
    match = next((d for d in detections if d["label"].lower() == tl), None)
    if match is None:
        match = next((d for d in detections if d["label"].lower() in instr), None)
    det = match or max(detections, key=lambda d: d.get("conf", 0.0))
    h, w = frame.shape[:2]
    pos = object_world_pos_for_task(task, det["bbox"], w, h)
    print(f"[reconstruct] '{det['label']}' bbox {det['bbox']} -> world {np.round(pos, 3)}")
    return pos


def _run_llm_vision(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    from lerobot_sim.envs.factory import _env_class_for_config

    env = _env_class_for_config(env_cfg)(env_cfg)
    action_dim = getattr(env_cfg, "action_dim", 2)

    detector = Detector(
        model_path=os.path.join(VISUAL_ASSISTANT_DIR, "yolov8n.pt"),
        conf_threshold=0.5,
    )
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam.")

    recorder = SimDatasetRecorder(output_dir=args.output_dir)
    print(f"LLM vision mode: instruction='{args.instruction}'")

    # Reconstruct the simulated object's position from the real camera BEFORE
    # the episode starts (the GPT-driven action loop below is unchanged).
    object_world_pos = _reconstruct_object_world_pos(
        detector, cap, args.task, args.target_label, args.instruction
    )

    try:
        _run_llm_vision_loop(
            env=env,
            detector=detector,
            client=client,
            cap=cap,
            recorder=recorder,
            instruction=args.instruction,
            action_dim=action_dim,
            object_world_pos=object_world_pos,
        )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    recorder.save_metadata()
    print(f"Episode recorded to {args.output_dir}")
    episode_dir = _find_latest_episode_dir(args.output_dir)
    _plot_episode(episode_dir)


def _run_visualize(env_cfg: SimEnvConfig, args: argparse.Namespace) -> None:
    """Run a random policy with live visualization.

    Args:
        env_cfg: Simulation environment configuration.
        args: Parsed CLI arguments.
    """
    from lerobot_sim.envs.factory import _env_class_for_config

    env = _env_class_for_config(env_cfg)(env_cfg)
    action_dim = getattr(env_cfg, "action_dim", 2)
    policy = RandomPolicy(action_dim=action_dim, seed=args.seed)
    viz = SimVisualizer(
        width=env_cfg.observation_width,
        height=env_cfg.observation_height,
        fps=env_cfg.fps,
    )
    _run_visualize_loop(env, policy, viz, env_cfg.episode_length)
    viz.close()


def _run_visualize_loop(env: Any, policy: Any, viz: SimVisualizer, max_steps: int) -> None:
    """Step through the env, rendering every frame to the visualizer.

    Args:
        env: Gymnasium environment.
        policy: Policy providing actions.
        viz: Visualizer instance.
        max_steps: Maximum steps before stopping.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    episode = 0
    for step in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        image = obs.get("pixels", env.render())
        alive = viz.render_frame(image, step=step, reward=total_reward, success=info.get("is_success", False))
        if not alive:
            break
        if terminated or truncated:
            episode += 1
            print(f"Episode {episode} done (reward={total_reward:.2f})")
            obs, _ = env.reset()
            total_reward = 0.0


# ======================================================================
# CLI
# ======================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(description="LeRobot Simulation Environment")
    parser.add_argument("--task", choices=["pusht", "pick_place", "reach"], default="pusht")
    parser.add_argument("--mode", choices=["train", "eval", "teleop", "visualize", "vision", "llm_vision"], default="train")
    parser.add_argument("--instruction", default="pick up the cup", help="Natural language instruction for LLM vision mode")
    parser.add_argument("--demo-episodes", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--output-dir", default="./sim_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-label", default="cup", help="YOLO class label to track for vision teleop")
    return parser.parse_args()


# ======================================================================
# Dispatch
# ======================================================================


_MODE_DISPATCH = {
    "train": _run_train,
    "eval": _run_eval,
    "teleop": _run_teleop,
    "visualize": _run_visualize,
    "vision": _run_vision_teleop,
    "llm_vision": _run_llm_vision,
}


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    args = _parse_args()

    env_cfg = _build_env_config(args.task)
    print(f"Task: {args.task} | Mode: {args.mode} | Seed: {args.seed}")
    print(f"Env config: {env_cfg.task}, fps={env_cfg.fps}, obs={env_cfg.obs_type}")
    print("-" * 60)

    runner = _MODE_DISPATCH[args.mode]
    runner(env_cfg, args)
