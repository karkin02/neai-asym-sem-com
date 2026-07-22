"""
Simulated 6-DOF robot arm (SO-100 style) with forward kinematics.

Provides joint-level control, end-effector computation via a simplified
planar-link chain, joint-limit enforcement, and gripper state tracking.
The arm is usable both standalone and as the physics back-end inside the
Gymnasium simulation environments.

Classes:
    SimRobotArm: The main robot arm simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np

from lerobot_sim.utils.constants import (
    SO100_JOINT_LOWER,
    SO100_JOINT_UPPER,
    SO100_LINK_LENGTHS,
    SO100_NUM_JOINTS,
)


@dataclass
class SimRobotArm:
    """A simulated 6-DOF robot arm with SO-100-style kinematics.

    The arm is modelled as a serial chain of revolute joints.  Each joint
    has configurable position limits, and a simplified forward-kinematics
    routine computes the end-effector position from the current joint
    angles and link lengths.

    Attributes:
        num_joints: Number of revolute joints.
        joint_positions: Current joint angles in radians.
        joint_lower: Per-joint lower limits (radians).
        joint_upper: Per-joint upper limits (radians).
        link_lengths: Physical length of each link (metres).
        gripper_open: Whether the gripper is currently open.
        ee_position: Cached 3-D end-effector position [x, y, z].
    """

    num_joints: int = SO100_NUM_JOINTS
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(SO100_NUM_JOINTS))
    joint_lower: Tuple[float, ...] = SO100_JOINT_LOWER
    joint_upper: Tuple[float, ...] = SO100_JOINT_UPPER
    link_lengths: Tuple[float, ...] = SO100_LINK_LENGTHS
    gripper_open: bool = True
    ee_position: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> np.ndarray:
        """Reset all joints to zero and recompute the end-effector position.

        Args:
            seed: Optional seed (reserved for future stochastic resets).

        Returns:
            A copy of the post-reset joint positions.
        """
        self.joint_positions = np.zeros(self.num_joints)
        self.gripper_open = True
        self.ee_position = self._forward_kinematics()
        return self.joint_positions.copy()

    def _clip_positions(self, raw: np.ndarray) -> np.ndarray:
        """Clip joint positions to their limits.

        Args:
            raw: Unclamped joint-position array.

        Returns:
            Clamped joint-position array.
        """
        lower = np.array(self.joint_lower[: self.num_joints])
        upper = np.array(self.joint_upper[: self.num_joints])
        return np.clip(raw, lower, upper)

    def _apply_joint_action(self, action: np.ndarray) -> None:
        """Add joint-velocity deltas and enforce limits.

        Args:
            action: Per-joint velocity commands (radians / step).
        """
        raw = self.joint_positions + action[: self.num_joints]
        self.joint_positions = self._clip_positions(raw)

    def _update_gripper(self, action: np.ndarray) -> None:
        """Update gripper state from the last element of *action* if present.

        Args:
            action: Full action vector; gripper command at index ``num_joints``.
        """
        if action.shape[0] > self.num_joints:
            self.gripper_open = action[self.num_joints] > 0.0

    def step(self, action: np.ndarray) -> np.ndarray:
        """Apply a joint-velocity action and return the new end-effector position.

        Args:
            action: Action vector whose first ``num_joints`` entries are per-joint
                velocity commands and an optional last entry controls the gripper.

        Returns:
            Updated 3-D end-effector position [x, y, z].
        """
        self._apply_joint_action(action)
        self._update_gripper(action)
        self.ee_position = self._forward_kinematics()
        return self.ee_position.copy()

    def get_state(self) -> np.ndarray:
        """Return the full proprioceptive state vector.

        Concatenates joint positions (6), end-effector xyz (3), and a
        scalar gripper flag (1) into a single flat vector of length 10.

        Returns:
            1-D NumPy array of shape ``(10,)``.
        """
        gripper_val = np.array([1.0 if self.gripper_open else 0.0])
        return np.concatenate([self.joint_positions, self.ee_position, gripper_val])

    # ------------------------------------------------------------------
    # Forward kinematics helpers
    # ------------------------------------------------------------------

    def _accumulate_angles(self) -> np.ndarray:
        """Compute cumulative joint angles for the serial chain.

        Returns:
            Array of cumulative sums of joint positions.
        """
        return np.cumsum(self.joint_positions)

    def _compute_xy(self, cum_angles: np.ndarray) -> Tuple[float, float]:
        """Sum link contributions in the XY plane.

        Args:
            cum_angles: Cumulative joint angle array.

        Returns:
            Tuple of (x, y) end-effector coordinates.
        """
        lengths = np.array(self.link_lengths[: self.num_joints])
        x = float(np.sum(lengths * np.cos(cum_angles)))
        y = float(np.sum(lengths * np.sin(cum_angles)))
        return x, y

    def _compute_z(self) -> float:
        """Compute a simplified z-height based on link geometry.

        Uses the sum of all link lengths as the base height, lowered by
        the absolute deviation of each joint from zero.

        Returns:
            Scalar z-coordinate of the end-effector.
        """
        base_z = float(np.sum(self.link_lengths[: self.num_joints]))
        deviation = float(np.sum(np.abs(self.joint_positions))) * 0.02
        return max(0.0, base_z - deviation)

    def _forward_kinematics(self) -> np.ndarray:
        """Compute the 3-D end-effector position via simplified FK.

        Returns:
            NumPy array of shape ``(3,)`` with [x, y, z].
        """
        cum_angles = self._accumulate_angles()
        x, y = self._compute_xy(cum_angles)
        z = self._compute_z()
        return np.array([x, y, z], dtype=np.float64)
