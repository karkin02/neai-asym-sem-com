"""
Real-time visualizer and episode replay tool.

Provides a Pygame-based window that renders live simulation frames,
overlays telemetry (step count, reward, success), and can replay
previously recorded episodes from disk.

Classes:
    SimVisualizer: Live rendering and episode replay.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from lerobot_sim.utils.constants import COLOR_SUCCESS, COLOR_TEXT


@dataclass
class SimVisualizer:
    """Pygame-based visualizer for simulation environments.

    Can operate in two modes:

    1. **Live mode** — call ``render_frame`` each step with the current
       observation and the visualizer blits the pixel image plus an HUD
       overlay to a Pygame window.
    2. **Replay mode** — call ``replay_episode`` with a path to a
       recorded episode directory and the visualizer plays back the saved
       frames at the configured FPS.

    Attributes:
        width: Window width in pixels.
        height: Window height in pixels.
        fps: Target frames per second.
        window_title: Caption displayed in the title bar.
    """

    width: int = 384
    height: int = 384
    fps: int = 10
    window_title: str = "LeRobot Sim Visualizer"
    _screen: Optional[Any] = None
    _clock: Optional[Any] = None

    # ------------------------------------------------------------------
    # Initialisation / teardown
    # ------------------------------------------------------------------

    def init_display(self) -> None:
        """Create the Pygame window and clock.

        Raises:
            ImportError: If Pygame is not installed.
        """
        try:
            import pygame
        except ImportError as exc:
            raise ImportError("Pygame required: pip install pygame") from exc
        pygame.init()
        self._screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(self.window_title)
        self._clock = pygame.time.Clock()

    def close(self) -> None:
        """Destroy the Pygame window and quit Pygame."""
        try:
            import pygame

            pygame.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Live rendering
    # ------------------------------------------------------------------

    def _image_to_surface(self, image: np.ndarray) -> Any:
        """Convert an (H, W, 3) uint8 NumPy image to a Pygame surface.

        Args:
            image: RGB image array.

        Returns:
            A Pygame ``Surface`` object.
        """
        import pygame

        return pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))

    def _scale_surface(self, surface: Any) -> Any:
        """Scale a Pygame surface to the window dimensions.

        Args:
            surface: Input Pygame ``Surface``.

        Returns:
            Scaled Pygame ``Surface``.
        """
        import pygame

        return pygame.transform.scale(surface, (self.width, self.height))

    def _draw_hud_text(self, text: str, y_offset: int) -> None:
        """Draw a single line of HUD text at the given y-offset.

        Args:
            text: The string to render.
            y_offset: Vertical pixel position.
        """
        import pygame

        font = pygame.font.SysFont("monospace", 16)
        rendered = font.render(text, True, COLOR_TEXT)
        self._screen.blit(rendered, (8, y_offset))

    def _draw_hud(self, step: int, reward: float, success: bool) -> None:
        """Draw the full heads-up display overlay.

        Args:
            step: Current timestep.
            reward: Cumulative reward.
            success: Whether the task has been solved.
        """
        self._draw_hud_text(f"Step: {step}", 4)
        self._draw_hud_text(f"Reward: {reward:.3f}", 22)
        status = "SUCCESS" if success else "running"
        self._draw_hud_text(f"Status: {status}", 40)

    def render_frame(
        self,
        image: np.ndarray,
        step: int = 0,
        reward: float = 0.0,
        success: bool = False,
    ) -> bool:
        """Blit one frame to the window with HUD overlay.

        Args:
            image: (H, W, 3) uint8 RGB image.
            step: Current timestep (shown in HUD).
            reward: Cumulative reward (shown in HUD).
            success: Success flag (shown in HUD).

        Returns:
            True if still running, False if user closed the window.
        """
        if self._screen is None:
            self.init_display()
        surface = self._image_to_surface(image)
        scaled = self._scale_surface(surface)
        self._screen.blit(scaled, (0, 0))
        self._draw_hud(step, reward, success)
        return self._flip_display()

    def _pump_events(self) -> bool:
        """Process Pygame events and return False if user quit.

        Returns:
            True if the window should stay open, False on quit.
        """
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _flip_display(self) -> bool:
        """Update the Pygame display, pump events, and tick the clock.

        Returns:
            True if still running, False if user closed the window.
        """
        import pygame

        pygame.display.flip()
        alive = self._pump_events()
        if self._clock is not None:
            self._clock.tick(self.fps)
        return alive

    # ------------------------------------------------------------------
    # Episode replay
    # ------------------------------------------------------------------

    def _load_episode_images(self, episode_dir: Path) -> List[np.ndarray]:
        """Load all saved image frames from an episode directory.

        Looks for ``.npy`` files inside a ``pixels`` sub-folder, sorted
        by filename.

        Args:
            episode_dir: Path to the recorded episode.

        Returns:
            List of (H, W, 3) uint8 NumPy arrays.
        """
        img_dir = episode_dir / "pixels"
        if not img_dir.exists():
            return []
        files = sorted(img_dir.glob("*.npy"))
        return [np.load(f) for f in files]

    def _load_episode_tabular(self, episode_dir: Path) -> Dict[str, np.ndarray]:
        """Load the tabular data archive for an episode.

        Args:
            episode_dir: Path to the recorded episode.

        Returns:
            Dictionary of NumPy arrays keyed by feature name.
        """
        npz_path = episode_dir / "tabular.npz"
        if not npz_path.exists():
            return {}
        return dict(np.load(npz_path))

    def replay_episode(self, episode_dir: str | Path) -> None:
        """Replay a recorded episode in the visualizer window.

        Args:
            episode_dir: Path to the episode directory (e.g.
                ``'./recorded_data/episode_000000'``).
        """
        ep_path = Path(episode_dir)
        images = self._load_episode_images(ep_path)
        if not images:
            print(f"No images found in {ep_path}. Nothing to replay.")
            return
        self._play_image_sequence(images)
        print(f"Replay of {ep_path.name} complete ({len(images)} frames).")

    def _play_image_sequence(self, images: List[np.ndarray]) -> None:
        """Display a sequence of images at the configured FPS.

        Args:
            images: Ordered list of (H, W, 3) uint8 arrays.
        """
        for idx, img in enumerate(images):
            alive = self.render_frame(img, step=idx)
            if not alive:
                break
