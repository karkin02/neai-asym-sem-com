"""
Keyboard teleoperation interface for demonstration collection.

Translates keyboard arrow-key presses into continuous action vectors for
a simulated robot environment.  Works with any ``gymnasium.Env`` that
accepts an ``(action_dim,)`` float action.  A terminal-based fallback is
provided when Pygame is not available.

Classes:
    KeyboardTeleop: Maps keyboard input to robot actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class KeyboardTeleop:
    """Maps keyboard input to robot actions for demonstration recording.

    In the default 2-D mode only (dx, dy) are controlled.  For higher-DOF
    environments the remaining action dimensions stay at zero unless
    additional key bindings are configured.

    When Pygame is available the teleop captures key-down / key-up events
    in real time.  Otherwise, it reads single-character commands from stdin
    (useful in headless / SSH scenarios).

    Attributes:
        action_dim: Dimensionality of the output action vector.
        speed: Magnitude of action per key press.
        key_state: Tracks which directional keys are currently held.
    """

    action_dim: int = 2
    speed: float = 0.5
    key_state: Dict[str, bool] = field(
        default_factory=lambda: {
            "up": False,
            "down": False,
            "left": False,
            "right": False,
        }
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self) -> np.ndarray:
        """Return the current action vector derived from held keys.

        Returns:
            1-D float32 array of shape ``(action_dim,)``.
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        dx, dy = self._keys_to_delta()
        action[0] = dx
        if self.action_dim > 1:
            action[1] = dy
        return action

    def process_pygame_events(self) -> bool:
        """Pump Pygame events and update ``key_state`` accordingly.

        Returns:
            *False* if a QUIT event was received; *True* otherwise.

        Raises:
            ImportError: If Pygame is not installed.
        """
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("Pygame required for real-time teleop: pip install pygame") from exc
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            self._handle_pygame_key_event(event, pygame)
        return True

    def process_terminal_input(self, char: str) -> bool:
        """Update key state from a single-character terminal command.

        Supported characters: ``w`` (up), ``s`` (down), ``a`` (left),
        ``d`` (right), ``q`` (quit).

        Args:
            char: Single character read from stdin.

        Returns:
            *False* if the quit character was received; *True* otherwise.
        """
        self._reset_key_state()
        mapping = {"w": "up", "s": "down", "a": "left", "d": "right"}
        if char == "q":
            return False
        if char in mapping:
            self.key_state[mapping[char]] = True
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _keys_to_delta(self) -> Tuple[float, float]:
        """Convert the current key state to (dx, dy) deltas.

        Returns:
            Tuple of (dx, dy) each in ``[-speed, speed]``.
        """
        dx = 0.0
        dy = 0.0
        if self.key_state["right"]:
            dx += self.speed
        if self.key_state["left"]:
            dx -= self.speed
        if self.key_state["up"]:
            dy += self.speed
        if self.key_state["down"]:
            dy -= self.speed
        return dx, dy

    def _handle_pygame_key_event(self, event: object, pygame_module: object) -> None:
        """Update key_state from a single Pygame KEYDOWN / KEYUP event.

        Args:
            event: Pygame event object.
            pygame_module: The ``pygame`` module (passed to avoid re-import).
        """
        pg = pygame_module
        key_map = {
            pg.K_UP: "up",
            pg.K_DOWN: "down",
            pg.K_LEFT: "left",
            pg.K_RIGHT: "right",
        }
        if hasattr(event, "key") and event.key in key_map:
            direction = key_map[event.key]
            self.key_state[direction] = event.type == pg.KEYDOWN

    def _reset_key_state(self) -> None:
        """Set all directional keys to *False*.

        Used before applying terminal input so that only the current
        command is active.
        """
        for key in self.key_state:
            self.key_state[key] = False
