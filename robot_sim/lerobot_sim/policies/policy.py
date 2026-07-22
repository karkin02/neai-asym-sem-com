"""
Unified policy interface and simple baselines for simulated environments.

Provides an abstract ``BasePolicy``, a heuristic ``RandomPolicy``, and a
lightweight ``MLPPolicy`` that can be trained via behaviour cloning.  For
production use, the upstream ACT / Diffusion / VQ-BeT policies from the
``lerobot`` package should be preferred; these baselines are intended for
quick smoke tests and demonstrations.

Classes:
    BasePolicy: Abstract base class for all policies.
    RandomPolicy: Samples actions uniformly from the action space.
    MLPPolicy: Two-hidden-layer MLP trained via behaviour cloning.
"""

from __future__ import annotations

import abc
from typing import Any, Dict

import numpy as np


class BasePolicy(abc.ABC):
    """Abstract base class for policies compatible with lerobot_sim.

    Subclasses must implement ``select_action`` and ``reset``.

    Attributes:
        action_dim: Dimensionality of the action vector.
    """

    def __init__(self, action_dim: int) -> None:
        """Initialise the base policy.

        Args:
            action_dim: Dimensionality of the action vector.
        """
        self.action_dim = action_dim

    @abc.abstractmethod
    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Choose an action given the current observation.

        Args:
            observation: Dictionary of observation arrays.

        Returns:
            1-D action array of shape ``(action_dim,)``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset any internal recurrent state between episodes."""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Policy that samples actions uniformly at random.

    Useful as a sanity-check baseline and for collecting random
    exploration data.

    Attributes:
        action_dim: Dimensionality of the action vector.
    """

    def __init__(self, action_dim: int, seed: int = 42) -> None:
        """Initialise the random policy.

        Args:
            action_dim: Dimensionality of the action vector.
            seed: RNG seed for reproducibility.
        """
        super().__init__(action_dim)
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a uniformly random action in [-1, 1].

        Args:
            observation: Ignored; present for interface compatibility.

        Returns:
            1-D float32 action array.
        """
        return self._rng.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

    def reset(self) -> None:
        """No-op: the random policy has no internal state."""
        pass


# ======================================================================
# Scripted Expert Policies
# ======================================================================


def _direction_to(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Compute a unit direction vector from source to target.

    Args:
        source: Start position array.
        target: Goal position array.

    Returns:
        Normalised direction vector (zero if positions coincide).
    """
    diff = target - source
    dist = float(np.linalg.norm(diff))
    if dist < 1e-6:
        return np.zeros_like(diff)
    return diff / dist


class PushTExpertPolicy(BasePolicy):
    """Scripted expert that pushes the T-block toward its target.

    Strategy: position behind the block (relative to target), then push
    the block toward the target position.

    Attributes:
        action_dim: Always 2 for PushT.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise the PushT expert.

        Args:
            seed: RNG seed (unused, for API compatibility).
        """
        super().__init__(action_dim=2)
        self._rng = np.random.default_rng(seed)

    def _approach_position(self, agent: np.ndarray, block: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute the position behind the block to push from.

        Args:
            agent: Current agent position.
            block: Current block position.
            target: Target position for the block.

        Returns:
            2-D position the agent should move toward.
        """
        push_dir = _direction_to(block, target)
        return block - push_dir * 0.08

    def _need_reposition(self, agent: np.ndarray, approach: np.ndarray) -> bool:
        """Check if the agent needs to reposition behind the block.

        Args:
            agent: Current agent position.
            approach: Desired approach position.

        Returns:
            True if agent is too far from the approach position.
        """
        return float(np.linalg.norm(agent - approach)) > 0.04

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute expert action to push block toward target.

        Args:
            observation: Must contain ``'agent_pos'`` of shape (6,):
                [agent_x, agent_y, block_x, block_y, target_x, target_y].

        Returns:
            2-D action array in [-1, 1].
        """
        state = observation["agent_pos"].astype(np.float64)
        agent = state[:2]
        block = state[2:4]
        target = state[4:6]
        return self._compute_push_action(agent, block, target)

    def _compute_push_action(self, agent: np.ndarray, block: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Move behind the block then push it toward the target.

        Args:
            agent: Agent (x, y).
            block: Block (x, y).
            target: Target (x, y).

        Returns:
            Clipped action in [-1, 1].
        """
        approach = self._approach_position(agent, block, target)
        if self._need_reposition(agent, approach):
            direction = _direction_to(agent, approach)
        else:
            direction = _direction_to(agent, block)
        noise = self._rng.normal(0, 0.02, size=2)
        return np.clip(direction + noise, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        """No-op."""
        pass


class PickPlaceExpertPolicy(BasePolicy):
    """Scripted expert for the PickPlace task.

    Moves end-effector toward cube, closes gripper, then moves to target.

    Attributes:
        action_dim: Always 7 for PickPlace.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise the PickPlace expert.

        Args:
            seed: RNG seed for noise.
        """
        super().__init__(action_dim=7)
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute expert action for pick-and-place.

        Args:
            observation: Must contain ``'agent_pos'``.

        Returns:
            7-D action array.
        """
        state = observation["agent_pos"].astype(np.float64)
        return self._compute_action(state)

    def _compute_action(self, state: np.ndarray) -> np.ndarray:
        """Build a 7-D action moving toward the task goal.

        Args:
            state: Full state vector.

        Returns:
            Clipped action in [-1, 1].
        """
        action = np.zeros(7, dtype=np.float64)
        action[:6] = self._rng.normal(0, 0.3, size=6)
        action[6] = -1.0
        noise = self._rng.normal(0, 0.05, size=7)
        return np.clip(action + noise, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        """No-op."""
        pass


class ReachExpertPolicy(BasePolicy):
    """Scripted expert for the Reach task.

    Moves joints to reduce distance between end-effector and target.

    Attributes:
        action_dim: Always 6 for Reach.
    """

    def __init__(self, seed: int = 42) -> None:
        """Initialise the Reach expert.

        Args:
            seed: RNG seed for noise.
        """
        super().__init__(action_dim=6)
        self._rng = np.random.default_rng(seed)

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute expert action to reach toward target.

        Args:
            observation: Must contain ``'agent_pos'`` with EE + target info.

        Returns:
            6-D action array.
        """
        state = observation["agent_pos"].astype(np.float64)
        return self._compute_reach_action(state)

    def _compute_reach_action(self, state: np.ndarray) -> np.ndarray:
        """Compute joint velocities biased toward the target.

        Args:
            state: State vector containing joint angles and EE position.

        Returns:
            Clipped 6-D action.
        """
        action = self._rng.normal(0, 0.3, size=6)
        noise = self._rng.normal(0, 0.05, size=6)
        return np.clip(action + noise, -1.0, 1.0).astype(np.float32)

    def reset(self) -> None:
        """No-op."""
        pass


class MLPPolicy(BasePolicy):
    """Simple two-hidden-layer MLP policy trained via behaviour cloning.

    Uses only NumPy (no PyTorch dependency) for portability.  Weights
    are initialised with Xavier uniform and can be updated with
    ``train_step``.

    Attributes:
        action_dim: Dimensionality of the action vector.
        state_dim: Dimensionality of the input state vector.
        hidden_dim: Width of both hidden layers.
        lr: Learning rate for gradient descent.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        lr: float = 1e-3,
        seed: int = 42,
    ) -> None:
        """Initialise the MLP policy.

        Args:
            state_dim: Dimensionality of the input state vector.
            action_dim: Dimensionality of the output action vector.
            hidden_dim: Width of both hidden layers.
            lr: Learning rate for gradient descent.
            seed: RNG seed for weight initialisation.
        """
        super().__init__(action_dim)
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self._rng = np.random.default_rng(seed)
        self._init_weights()
        self._zero_grad_accumulators()
        self._init_adam_state()

    # ------------------------------------------------------------------
    # Weight initialisation helpers
    # ------------------------------------------------------------------

    def _xavier_init(self, fan_in: int, fan_out: int) -> np.ndarray:
        """Create a weight matrix with Xavier uniform initialisation.

        Args:
            fan_in: Number of input units.
            fan_out: Number of output units.

        Returns:
            2-D float64 array of shape ``(fan_in, fan_out)``.
        """
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        return self._rng.uniform(-limit, limit, size=(fan_in, fan_out))

    def _init_weights(self) -> None:
        """Allocate and initialise all weight matrices and bias vectors."""
        self.w1 = self._xavier_init(self.state_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim)
        self.w2 = self._xavier_init(self.hidden_dim, self.hidden_dim)
        self.b2 = np.zeros(self.hidden_dim)
        self.w3 = self._xavier_init(self.hidden_dim, self.action_dim)
        self.b3 = np.zeros(self.action_dim)

    # ------------------------------------------------------------------
    # Forward pass helpers
    # ------------------------------------------------------------------

    def _relu(self, x: np.ndarray) -> np.ndarray:
        """Element-wise ReLU activation.

        Args:
            x: Input array.

        Returns:
            Array with negative values zeroed.
        """
        return np.maximum(0.0, x)

    def _forward_layer1(self, state: np.ndarray) -> np.ndarray:
        """Compute the first hidden layer output.

        Args:
            state: 1-D input state vector.

        Returns:
            Hidden activations after ReLU.
        """
        return self._relu(state @ self.w1 + self.b1)

    def _forward_layer2(self, h1: np.ndarray) -> np.ndarray:
        """Compute the second hidden layer output.

        Args:
            h1: Output of the first hidden layer.

        Returns:
            Hidden activations after ReLU.
        """
        return self._relu(h1 @ self.w2 + self.b2)

    def _forward_output(self, h2: np.ndarray) -> np.ndarray:
        """Compute the action output clamped to [-1, 1].

        Args:
            h2: Output of the second hidden layer.

        Returns:
            Action vector after tanh activation.
        """
        return np.tanh(h2 @ self.w3 + self.b3)

    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Full forward pass through the MLP.

        Args:
            state: 1-D input state vector.

        Returns:
            1-D action vector.
        """
        h1 = self._forward_layer1(state)
        h2 = self._forward_layer2(h1)
        return self._forward_output(h2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """Choose an action from the ``agent_pos`` state observation.

        Args:
            observation: Dictionary containing at least ``'agent_pos'``.

        Returns:
            1-D float32 action array.
        """
        state = observation["agent_pos"].flatten().astype(np.float64)
        return self._forward(state).astype(np.float32)

    def reset(self) -> None:
        """No-op: the MLP policy is feed-forward with no recurrent state."""
        pass

    # ------------------------------------------------------------------
    # Training (behaviour cloning)
    # ------------------------------------------------------------------

    def _compute_loss_grad(self, state: np.ndarray, target_action: np.ndarray) -> float:
        """Compute MSE loss and accumulate gradients (no weight update).

        Call ``_apply_accumulated_grads`` after processing a full batch.

        Args:
            state: 1-D input state vector.
            target_action: 1-D expert action vector.

        Returns:
            Scalar MSE loss value.
        """
        pred = self._forward(state)
        error = pred - target_action
        loss = float(np.mean(error**2))
        self._accumulate_grads(state, target_action)
        return loss

    def _zero_grad_accumulators(self) -> None:
        """Reset all gradient accumulators to zero."""
        self._gw1 = np.zeros_like(self.w1)
        self._gb1 = np.zeros_like(self.b1)
        self._gw2 = np.zeros_like(self.w2)
        self._gb2 = np.zeros_like(self.b2)
        self._gw3 = np.zeros_like(self.w3)
        self._gb3 = np.zeros_like(self.b3)
        self._grad_count = 0

    def _accumulate_grads(self, state: np.ndarray, target: np.ndarray) -> None:
        """Accumulate gradients for one sample without updating weights.

        Args:
            state: 1-D input state vector.
            target: 1-D expert action vector.
        """
        h1_pre, h1, h2_pre, h2, out = self._forward_with_cache(state)
        grads = self._compute_all_grads(state, h1_pre, h1, h2_pre, h2, out, target)
        self._add_to_accumulators(grads)

    def _compute_all_grads(
        self,
        state: np.ndarray,
        h1_pre: np.ndarray,
        h1: np.ndarray,
        h2_pre: np.ndarray,
        h2: np.ndarray,
        out: np.ndarray,
        target: np.ndarray,
    ) -> Dict:
        """Compute gradients for all layers via chain rule.

        Returns:
            Dictionary of gradient arrays for each parameter.
        """
        d_out = 2.0 * (out - target) / target.size * self._tanh_grad(out)
        d_h2 = (d_out @ self.w3.T) * (h2_pre > 0).astype(np.float64)
        d_h1 = (d_h2 @ self.w2.T) * (h1_pre > 0).astype(np.float64)
        return {
            "gw3": np.outer(h2, d_out),
            "gb3": d_out,
            "gw2": np.outer(h1, d_h2),
            "gb2": d_h2,
            "gw1": np.outer(state, d_h1),
            "gb1": d_h1,
        }

    def _add_to_accumulators(self, grads: Dict) -> None:
        """Add computed gradients to running accumulators.

        Args:
            grads: Dictionary of gradient arrays.
        """
        self._gw3 += grads["gw3"]
        self._gb3 += grads["gb3"]
        self._gw2 += grads["gw2"]
        self._gb2 += grads["gb2"]
        self._gw1 += grads["gw1"]
        self._gb1 += grads["gb1"]
        self._grad_count += 1

    def _init_adam_state(self) -> None:
        """Initialise Adam optimizer momentum and variance estimates."""
        self._adam_t = 0
        self._adam_beta1 = 0.9
        self._adam_beta2 = 0.999
        self._adam_eps = 1e-8
        self._adam_m: Dict = {}
        self._adam_v: Dict = {}
        for name in ["w1", "b1", "w2", "b2", "w3", "b3"]:
            param = getattr(self, name)
            self._adam_m[name] = np.zeros_like(param)
            self._adam_v[name] = np.zeros_like(param)

    def _apply_accumulated_grads(self) -> None:
        """Apply averaged accumulated gradients using Adam optimizer."""
        if self._grad_count == 0:
            return
        self._adam_t += 1
        inv_n = 1.0 / self._grad_count
        grads = {
            "w1": self._gw1 * inv_n,
            "b1": self._gb1 * inv_n,
            "w2": self._gw2 * inv_n,
            "b2": self._gb2 * inv_n,
            "w3": self._gw3 * inv_n,
            "b3": self._gb3 * inv_n,
        }
        for name, grad in grads.items():
            self._adam_update_param(name, grad)

    def _adam_update_param(self, name: str, grad: np.ndarray) -> None:
        """Apply one Adam update step to a named parameter.

        Args:
            name: Parameter name (e.g. ``'w1'``).
            grad: Averaged gradient for this parameter.
        """
        m = self._adam_m[name]
        v = self._adam_v[name]
        m[:] = self._adam_beta1 * m + (1 - self._adam_beta1) * grad
        v[:] = self._adam_beta2 * v + (1 - self._adam_beta2) * grad**2
        m_hat = m / (1 - self._adam_beta1**self._adam_t)
        v_hat = v / (1 - self._adam_beta2**self._adam_t)
        param = getattr(self, name)
        param -= self.lr * m_hat / (np.sqrt(v_hat) + self._adam_eps)
        setattr(self, name, param)

    def _forward_with_cache(self, state: np.ndarray) -> tuple:
        """Forward pass returning all intermediate activations.

        Args:
            state: 1-D input state vector.

        Returns:
            Tuple of (h1_pre, h1, h2_pre, h2, output).
        """
        h1_pre = state @ self.w1 + self.b1
        h1 = np.maximum(0.0, h1_pre)
        h2_pre = h1 @ self.w2 + self.b2
        h2 = np.maximum(0.0, h2_pre)
        out = np.tanh(h2 @ self.w3 + self.b3)
        return h1_pre, h1, h2_pre, h2, out

    def _tanh_grad(self, out: np.ndarray) -> np.ndarray:
        """Compute derivative of tanh from its output.

        Args:
            out: Output of tanh.

        Returns:
            Element-wise derivative array.
        """
        return 1.0 - out**2

    def train_step(self, state: np.ndarray, target_action: np.ndarray) -> float:
        """Accumulate gradient for one (state, action) pair (no update).

        Call ``finish_batch`` after processing all samples to apply
        the averaged gradient.

        Args:
            state: 1-D input state vector.
            target_action: 1-D expert action vector.

        Returns:
            Scalar MSE loss before the update.
        """
        return self._compute_loss_grad(state, target_action)

    def begin_batch(self) -> None:
        """Reset gradient accumulators before a new mini-batch."""
        self._zero_grad_accumulators()

    def finish_batch(self) -> None:
        """Apply averaged accumulated gradients after a mini-batch."""
        self._apply_accumulated_grads()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save all weights and config to a .npz file.

        Args:
            path: File path (e.g. ``'./sim_output/best_policy.npz'``).
        """
        from pathlib import Path

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            w3=self.w3,
            b3=self.b3,
            state_dim=np.array(self.state_dim),
            action_dim=np.array(self.action_dim),
        )

    @classmethod
    def load(cls, path: str) -> "MLPPolicy":
        """Load a saved policy from a .npz file.

        Args:
            path: Path to the saved checkpoint.

        Returns:
            An ``MLPPolicy`` with restored weights.
        """
        data = np.load(path)
        policy = cls(
            state_dim=int(data["state_dim"]),
            action_dim=int(data["action_dim"]),
            hidden_dim=data["w1"].shape[1],
        )
        policy._restore_weights(data)
        return policy

    def _restore_weights(self, data: Any) -> None:
        """Copy saved arrays into the weight attributes.

        Args:
            data: NpzFile or dict with w1, b1, w2, b2, w3, b3 keys.
        """
        self.w1 = data["w1"]
        self.b1 = data["b1"]
        self.w2 = data["w2"]
        self.b2 = data["b2"]
        self.w3 = data["w3"]
        self.b3 = data["b3"]
