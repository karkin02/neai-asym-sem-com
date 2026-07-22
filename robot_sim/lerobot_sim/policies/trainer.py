"""
Policy training pipeline for behaviour cloning and evaluation.

Provides ``Trainer`` which orchestrates data collection from a simulation
environment (or from an existing dataset), trains a policy via behaviour
cloning, and evaluates the trained policy for a configurable number of
episodes.

Classes:
    TrainerConfig: Dataclass holding all training hyper-parameters.
    Trainer: Main training / evaluation loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lerobot_sim.datasets.dataset_manager import SimDatasetRecorder
from lerobot_sim.envs.configs import SimEnvConfig
from lerobot_sim.envs.factory import make_sim_env
from lerobot_sim.policies.policy import BasePolicy, MLPPolicy, RandomPolicy


@dataclass
class TrainerConfig:
    """Hyper-parameters for the behaviour-cloning trainer.

    Attributes:
        num_demo_episodes: Episodes to collect with the demonstration policy.
        num_train_steps: Gradient-descent steps to train the learner.
        num_eval_episodes: Episodes to run during evaluation.
        log_interval: Print a training log every *n* steps.
        output_dir: Directory for saving checkpoints and datasets.
        seed: Global random seed.
    """

    num_demo_episodes: int = 10
    num_train_steps: int = 200
    num_eval_episodes: int = 5
    log_interval: int = 20
    output_dir: str = "./training_output"
    seed: int = 42


class Trainer:
    """Orchestrates data collection, training, and evaluation.

    The training pipeline follows three phases:

    1. **Collect**: run a demonstration policy in the environment and
       record the resulting episodes.
    2. **Train**: fit a learner policy to the collected demonstrations
       via behaviour cloning (MSE loss on state→action pairs).
    3. **Evaluate**: run the trained policy and report mean reward and
       success rate.

    Attributes:
        env_cfg: Configuration of the simulation environment.
        policy: The learner policy to be trained.
        config: Training hyper-parameters.
    """

    def __init__(
        self,
        env_cfg: SimEnvConfig,
        policy: BasePolicy,
        config: TrainerConfig | None = None,
    ) -> None:
        """Initialise the trainer.

        Args:
            env_cfg: Simulation environment configuration.
            policy: The learner policy to train and evaluate.
            config: Optional ``TrainerConfig``; defaults are used when *None*.
        """
        self.env_cfg = env_cfg
        self.policy = policy
        self.config = config or TrainerConfig()
        self._demo_data: List[Dict[str, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Phase 1 – Data Collection
    # ------------------------------------------------------------------

    def _create_single_env(self) -> Any:
        """Instantiate a single (non-vectorised) environment.

        Returns:
            A Gymnasium ``Env`` instance.
        """
        from lerobot_sim.envs.factory import _env_class_for_config, _resolve_config

        cfg = _resolve_config(self.env_cfg) if isinstance(self.env_cfg, str) else self.env_cfg
        cls = _env_class_for_config(cfg)
        return cls(cfg)

    def _collect_one_episode(self, env: Any, demo_policy: BasePolicy, recorder: SimDatasetRecorder) -> None:
        """Roll out one episode, recording every timestep.

        Args:
            env: Gymnasium environment instance.
            demo_policy: Policy used to generate demonstrations.
            recorder: Active ``SimDatasetRecorder``.
        """
        obs, _ = env.reset()
        recorder.start_episode()
        done = False
        self._run_episode_loop(env, obs, demo_policy, recorder, done)
        recorder.end_episode()

    def _run_episode_loop(
        self,
        env: Any,
        obs: Dict,
        policy: BasePolicy,
        recorder: SimDatasetRecorder,
        done: bool,
    ) -> None:
        """Step through the environment until termination or truncation.

        Args:
            env: Gymnasium environment.
            obs: Initial observation dictionary.
            policy: Policy producing actions.
            recorder: Recorder capturing each step.
            done: Initial done flag (should be *False*).
        """
        while not done:
            action = policy.select_action(obs)
            recorder.record_step(obs, action)
            self._store_demo_transition(obs, action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    def _store_demo_transition(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> None:
        """Append a (state, action) pair to the in-memory demo buffer.

        Args:
            obs: Observation dictionary.
            action: Action array.
        """
        if "agent_pos" in obs:
            self._demo_data.append({"state": obs["agent_pos"].copy(), "action": action.copy()})

    def _select_expert_policy(self) -> BasePolicy:
        """Pick the scripted expert policy matching the environment.

        Returns:
            An expert ``BasePolicy`` for the current task.
        """
        from lerobot_sim.policies.policy import (
            PushTExpertPolicy,
            PickPlaceExpertPolicy,
            ReachExpertPolicy,
        )

        expert_map = {
            "PushT-Sim-v0": PushTExpertPolicy,
            "PickPlace-Sim-v0": PickPlaceExpertPolicy,
            "Reach-Sim-v0": ReachExpertPolicy,
        }
        task = getattr(self.env_cfg, "task", "")
        cls = expert_map.get(task, None)
        if cls is not None:
            return cls(seed=self.config.seed)
        return RandomPolicy(self.policy.action_dim, seed=self.config.seed)

    def collect_demonstrations(self) -> SimDatasetRecorder:
        """Collect demonstration episodes using a scripted expert policy.

        Returns:
            The ``SimDatasetRecorder`` with all episodes flushed to disk.
        """
        env = self._create_single_env()
        demo_policy = self._select_expert_policy()
        print(f"  Demo policy: {type(demo_policy).__name__}")
        recorder = SimDatasetRecorder(output_dir=self.config.output_dir)
        for ep in range(self.config.num_demo_episodes):
            self._collect_one_episode(env, demo_policy, recorder)
        recorder.save_metadata()
        return recorder

    # ------------------------------------------------------------------
    # Phase 2 – Training (Behaviour Cloning)
    # ------------------------------------------------------------------

    def _sample_batch(self, batch_size: int = 32) -> Dict[str, np.ndarray]:
        """Sample a mini-batch of (state, action) pairs from the demo buffer.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dictionary with ``'states'`` and ``'actions'`` arrays.
        """
        indices = np.random.randint(len(self._demo_data), size=batch_size)
        states = np.array([self._demo_data[i]["state"] for i in indices])
        actions = np.array([self._demo_data[i]["action"] for i in indices])
        return {"states": states, "actions": actions}

    def _log_training_step(self, step: int, loss: float) -> None:
        """Print a training progress message.

        Args:
            step: Current training step (zero-based).
            loss: Scalar loss value.
        """
        print(f"  [train] step {step:>5d} | loss = {loss:.6f}")

    def _save_best_checkpoint(self, loss: float) -> None:
        """Save the policy if current loss is the best so far.

        Args:
            loss: Current step's loss value.
        """
        if not hasattr(self, "_best_loss") or loss < self._best_loss:
            self._best_loss = loss
            save_path = str(Path(self.config.output_dir) / "best_policy.npz")
            self.policy.save(save_path)
            print(f"  [checkpoint] New best loss {loss:.6f} -> saved {save_path}")

    def train(self) -> List[float]:
        """Train the policy via behaviour cloning on collected demonstrations.

        Saves the best checkpoint (lowest loss) to ``output_dir/best_policy.npz``.

        Returns:
            List of per-step MSE loss values.

        Raises:
            RuntimeError: If no demonstration data has been collected.
        """
        if not self._demo_data:
            raise RuntimeError("No demo data. Call collect_demonstrations() first.")
        losses: List[float] = []
        print(f"Training for {self.config.num_train_steps} steps ...")
        for step in range(self.config.num_train_steps):
            loss = self._train_one_step(step)
            losses.append(loss)
            self._save_best_checkpoint(loss)
        self._save_final_checkpoint()
        print(f"Training complete. Final loss: {losses[-1]:.6f}")
        return losses

    def _save_final_checkpoint(self) -> None:
        """Save the policy at the end of training as last_policy.npz."""
        save_path = str(Path(self.config.output_dir) / "last_policy.npz")
        self.policy.save(save_path)
        print(f"  [checkpoint] Final model saved to {save_path}")

    def _train_one_step(self, step: int) -> float:
        """Execute a single training iteration over a mini-batch.

        Args:
            step: Current step index.

        Returns:
            Mean MSE loss over the batch.
        """
        batch = self._sample_batch(batch_size=32)
        loss = self._train_on_batch(batch, step)
        return loss

    def _train_on_batch(self, batch: Dict[str, np.ndarray], step: int) -> float:
        """Accumulate gradients over the batch, then apply once.

        Args:
            batch: Dictionary with 'states' and 'actions' arrays.
            step: Current step index for logging.

        Returns:
            Mean loss over the batch.
        """
        self.policy.begin_batch()
        total_loss = 0.0
        for i in range(batch["states"].shape[0]):
            state = batch["states"][i].flatten().astype(np.float64)
            action = batch["actions"][i].flatten().astype(np.float64)
            total_loss += self.policy.train_step(state, action)
        self.policy.finish_batch()
        avg_loss = total_loss / batch["states"].shape[0]
        if step % self.config.log_interval == 0:
            self._log_training_step(step, avg_loss)
        return avg_loss

    # ------------------------------------------------------------------
    # Phase 3 – Evaluation
    # ------------------------------------------------------------------

    def _eval_one_episode(self, env: Any) -> Dict[str, float]:
        """Evaluate the policy for one episode and return metrics.

        Args:
            env: Gymnasium environment instance.

        Returns:
            Dictionary with ``'reward'`` and ``'success'`` keys.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        success = False
        total_reward, success = self._run_eval_loop(env, obs, total_reward, success)
        return {"reward": total_reward, "success": float(success)}

    def _run_eval_loop(self, env: Any, obs: Dict, total_reward: float, success: bool) -> tuple[float, bool]:
        """Step through the environment, accumulating reward.

        Args:
            env: Gymnasium environment.
            obs: Initial observation.
            total_reward: Running reward accumulator.
            success: Running success flag.

        Returns:
            Tuple of (total_reward, success).
        """
        done = False
        while not done:
            action = self.policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            success = success or info.get("is_success", False)
            done = terminated or truncated
        return total_reward, success

    def _load_best_checkpoint(self) -> None:
        """Load the best checkpoint into self.policy if it exists."""
        best_path = str(Path(self.config.output_dir) / "best_policy.npz")
        if Path(best_path).exists() and isinstance(self.policy, MLPPolicy):
            self.policy = MLPPolicy.load(best_path)
            print(f"  Loaded best checkpoint from {best_path}")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the trained policy over multiple episodes.

        Returns:
            Dictionary with ``'mean_reward'`` and ``'success_rate'``.
        """
        env = self._create_single_env()
        results = [self._eval_one_episode(env) for _ in range(self.config.num_eval_episodes)]
        mean_reward = float(np.mean([r["reward"] for r in results]))
        success_rate = float(np.mean([r["success"] for r in results]))
        print(f"Eval ({self.config.num_eval_episodes} eps): reward={mean_reward:.3f}, success={success_rate:.1%}")
        return {"mean_reward": mean_reward, "success_rate": success_rate}

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """Execute the complete collect → train → evaluate pipeline.

        Returns:
            Dictionary with ``'losses'`` and ``'eval_metrics'``.
        """
        print("=" * 60)
        print("Phase 1: Collecting demonstrations …")
        self.collect_demonstrations()
        print("=" * 60)
        print("Phase 2: Training policy …")
        losses = self.train()
        print("=" * 60)
        print("Phase 3: Evaluating policy …")
        self._load_best_checkpoint()
        metrics = self.evaluate()
        return {"losses": losses, "eval_metrics": metrics}
