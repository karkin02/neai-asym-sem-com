"""
Gamepad teleoperation interface for demonstration collection.

Translates joystick / gamepad analog-stick inputs into continuous action
vectors.  Requires Pygame for real-time input capture.

Classes:
    GamepadTeleop: Maps gamepad analog sticks to robot actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GamepadTeleop:
    """Maps gamepad analog-stick input to robot actions.

    The left stick controls the first two action dimensions (typically
    end-effector x/y or base motion).  The right-stick y axis controls a
    third dimension if ``action_dim >= 3`` (e.g. z-height).  A trigger
    button toggles the gripper.

    Attributes:
        action_dim: Dimensionality of the output action vector.
        speed: Scale factor multiplied onto raw stick values.
        deadzone: Minimum stick deflection to register as motion.
        joystick: Pygame ``Joystick`` instance (initialised on first use).
    """

    action_dim: int = 7
    speed: float = 0.8
    deadzone: float = 0.1
    joystick: Optional[object] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def init_joystick(self) -> bool:
        """Initialise Pygame and the first connected joystick.

        Returns:
            *True* if a joystick was found and initialised; *False* otherwise.
        """
        try:
            import pygame
        except ImportError:
            return False
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            return False
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        return True

    def get_action(self) -> np.ndarray:
        """Read the current gamepad state and return an action vector.

        Returns:
            1-D float32 action array of shape ``(action_dim,)``.
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        if self.joystick is None:
            return action
        self._read_left_stick(action)
        self._read_right_stick(action)
        self._read_trigger(action)
        return action

    # ------------------------------------------------------------------
    # Stick-reading helpers
    # ------------------------------------------------------------------

    def _apply_deadzone(self, value: float) -> float:
        """Zero out values below the deadzone threshold.

        Args:
            value: Raw axis value in [-1, 1].

        Returns:
            Adjusted value (zero if below deadzone).
        """
        return value if abs(value) > self.deadzone else 0.0

    def _read_left_stick(self, action: np.ndarray) -> None:
        """Write left-stick axes into the first two action slots.

        Args:
            action: Mutable action array.
        """
        lx = self._apply_deadzone(self.joystick.get_axis(0)) * self.speed
        ly = self._apply_deadzone(self.joystick.get_axis(1)) * self.speed
        action[0] = lx
        if self.action_dim > 1:
            action[1] = ly

    def _read_right_stick(self, action: np.ndarray) -> None:
        """Write right-stick y-axis into the third action slot.

        Args:
            action: Mutable action array.
        """
        if self.action_dim > 2 and self.joystick.get_numaxes() > 3:
            ry = self._apply_deadzone(self.joystick.get_axis(3)) * self.speed
            action[2] = ry

    def _read_trigger(self, action: np.ndarray) -> None:
        """Map a trigger / button to the gripper action slot.

        Args:
            action: Mutable action array.
        """
        if self.action_dim > 6 and self.joystick.get_numbuttons() > 0:
            action[6] = 1.0 if self.joystick.get_button(0) else -1.0
