from __future__ import annotations

import time
from typing import Any

from .contracts import Action, Observation, StepResult
from .mock import MockPickPlaceEnvironment

try:
    import pybullet as p
except ImportError:  # pragma: no cover - exercised only on missing optional runtime
    p = None


class PyBulletMockEnvironment:
    """PyBullet rendering for the logical mock pick-and-place task.

    Physics is intentionally visual-only here. Task state and scoring still come
    from MockPickPlaceEnvironment, keeping tests deterministic while providing a
    useful live view of Architecture A.
    """

    OBJECT_COLORS = {
        "red_sample": (0.85, 0.12, 0.10, 1.0),
        "blue_sample": (0.10, 0.30, 0.85, 1.0),
        "green_sample": (0.10, 0.65, 0.25, 1.0),
    }
    OBJECT_POSITIONS = {
        "red_sample": (-0.18, 0.0, 0.055),
        "blue_sample": (0.0, 0.0, 0.055),
        "green_sample": (0.18, 0.0, 0.055),
    }
    TARGET_POSITIONS = {
        "left_tray": (-0.22, 0.35, 0.035),
        "right_tray": (0.22, 0.35, 0.035),
    }
    CAMERA_EYE = (0.0, -1.15, 1.10)
    CAMERA_TARGET = (0.0, 0.14, 0.02)

    def __init__(self, *, gui: bool = True, realtime: bool = True) -> None:
        if p is None:
            raise RuntimeError(
                "PyBullet is not installed. Install the project dependencies first."
            )
        self._logic = MockPickPlaceEnvironment()
        self._realtime = realtime
        mode = p.GUI if gui else p.DIRECT
        self._client = p.connect(mode)
        if self._client < 0:
            raise RuntimeError("Unable to initialize PyBullet.")
        self._sample_body: int | None = None
        self._gripper_body: int | None = None
        self._object_name = ""
        self._target_name = ""
        self._configure_world()

    def reset(self, *, seed: int, instruction: str) -> Observation:
        observation = self._logic.reset(seed=seed, instruction=instruction)
        self._object_name = str(observation.metadata["visible_objects"][0])
        self._target_name = str(observation.metadata["visible_targets"][0])
        self._build_episode_scene()
        return observation

    def step(self, action: Action) -> StepResult:
        if action.name == "pick":
            self._animate_pick()
        elif action.name == "place":
            target = str(action.arguments.get("target", self._target_name))
            self._animate_place(target)
        return self._logic.step(action)

    def capture_rgb(self, *, width: int = 640, height: int = 480) -> Any:
        view = p.computeViewMatrix(
            cameraEyePosition=self.CAMERA_EYE,
            cameraTargetPosition=self.CAMERA_TARGET,
            cameraUpVector=(0.0, 0.0, 1.0),
        )
        projection = p.computeProjectionMatrixFOV(
            fov=46.0,
            aspect=width / height,
            nearVal=0.02,
            farVal=4.0,
        )
        _, _, rgba, _, _ = p.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        return rgba

    def close(self) -> None:
        if self._client >= 0:
            p.disconnect(self._client)
            self._client = -1

    def _configure_world(self) -> None:
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client)
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI,
            0,
            physicsClientId=self._client,
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=1.69,
            cameraYaw=0,
            cameraPitch=-42,
            cameraTargetPosition=self.CAMERA_TARGET,
            physicsClientId=self._client,
        )

    def _build_episode_scene(self) -> None:
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._client)
        self._create_box(
            half_extents=(0.46, 0.43, 0.025),
            position=(0.0, 0.12, -0.025),
            color=(0.62, 0.65, 0.68, 1.0),
            mass=0.0,
        )
        self._create_tray("left_tray", (0.20, 0.45, 0.90, 1.0))
        self._create_tray("right_tray", (0.95, 0.72, 0.18, 1.0))
        self._sample_body = self._create_box(
            half_extents=(0.035, 0.035, 0.035),
            position=self.OBJECT_POSITIONS[self._object_name],
            color=self.OBJECT_COLORS[self._object_name],
            mass=0.05,
        )
        self._gripper_body = self._create_box(
            half_extents=(0.055, 0.025, 0.018),
            position=(-0.34, -0.22, 0.18),
            color=(0.15, 0.16, 0.18, 1.0),
            mass=0.0,
        )
        p.stepSimulation(physicsClientId=self._client)

    def _create_tray(self, name: str, color: tuple[float, ...]) -> None:
        x, y, z = self.TARGET_POSITIONS[name]
        self._create_box(
            half_extents=(0.13, 0.11, 0.012),
            position=(x, y, z),
            color=color,
            mass=0.0,
        )
        for dx, dy, sx, sy in (
            (0.0, 0.11, 0.13, 0.01),
            (0.0, -0.11, 0.13, 0.01),
            (0.13, 0.0, 0.01, 0.11),
            (-0.13, 0.0, 0.01, 0.11),
        ):
            self._create_box(
                half_extents=(sx, sy, 0.03),
                position=(x + dx, y + dy, z + 0.03),
                color=color,
                mass=0.0,
            )

    def _create_box(
        self,
        *,
        half_extents: tuple[float, float, float],
        position: tuple[float, float, float],
        color: tuple[float, ...],
        mass: float,
    ) -> int:
        collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client,
        )
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._client,
        )
        return p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self._client,
        )

    def _animate_pick(self) -> None:
        if self._sample_body is None:
            return
        start = self.OBJECT_POSITIONS[self._object_name]
        self._animate(start, (start[0], start[1], 0.30), carry_sample=True)

    def _animate_place(self, target: str) -> None:
        if target not in self.TARGET_POSITIONS:
            return
        start = p.getBasePositionAndOrientation(
            self._gripper_body,
            physicsClientId=self._client,
        )[0]
        target_position = self.TARGET_POSITIONS[target]
        above = (target_position[0], target_position[1], 0.30)
        down = (target_position[0], target_position[1], 0.09)
        self._animate(start, above, carry_sample=True)
        self._animate(above, down, carry_sample=True)

    def _animate(
        self,
        start: tuple[float, ...],
        end: tuple[float, ...],
        *,
        carry_sample: bool,
        frames: int = 24,
    ) -> None:
        for frame in range(1, frames + 1):
            alpha = frame / frames
            position = tuple(
                float(start[index] + (end[index] - start[index]) * alpha)
                for index in range(3)
            )
            p.resetBasePositionAndOrientation(
                self._gripper_body,
                position,
                (0.0, 0.0, 0.0, 1.0),
                physicsClientId=self._client,
            )
            if carry_sample and self._sample_body is not None:
                sample_position = (position[0], position[1], position[2] - 0.055)
                p.resetBasePositionAndOrientation(
                    self._sample_body,
                    sample_position,
                    (0.0, 0.0, 0.0, 1.0),
                    physicsClientId=self._client,
                )
            p.stepSimulation(physicsClientId=self._client)
            if self._realtime:
                time.sleep(1.0 / 60.0)
