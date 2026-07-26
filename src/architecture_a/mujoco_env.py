from __future__ import annotations

import random
import time
from typing import Any

import mujoco

from .contracts import Action, Observation, StepResult


_SCENE_XML = """
<mujoco model="architecture_a_pick_place">
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <visual>
    <global offwidth="640" offheight="480"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <material name="table" rgba="0.52 0.55 0.58 1"/>
    <material name="left" rgba="0.12 0.38 0.88 1"/>
    <material name="right" rgba="0.95 0.64 0.08 1"/>
    <material name="sample" rgba="0.85 0.12 0.10 1"/>
    <material name="robot" rgba="0.12 0.13 0.15 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.2 1.6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="table" type="box" pos="0 0.12 -0.025"
          size="0.46 0.43 0.025" material="table"/>

    <body name="left_tray" pos="-0.22 0.35 0.025">
      <geom type="box" size="0.13 0.11 0.012" material="left"/>
      <geom type="box" pos="0 0.11 0.035" size="0.13 0.01 0.035" material="left"/>
      <geom type="box" pos="0 -0.11 0.035" size="0.13 0.01 0.035" material="left"/>
      <geom type="box" pos="0.13 0 0.035" size="0.01 0.11 0.035" material="left"/>
      <geom type="box" pos="-0.13 0 0.035" size="0.01 0.11 0.035" material="left"/>
    </body>
    <body name="right_tray" pos="0.22 0.35 0.025">
      <geom type="box" size="0.13 0.11 0.012" material="right"/>
      <geom type="box" pos="0 0.11 0.035" size="0.13 0.01 0.035" material="right"/>
      <geom type="box" pos="0 -0.11 0.035" size="0.13 0.01 0.035" material="right"/>
      <geom type="box" pos="0.13 0 0.035" size="0.01 0.11 0.035" material="right"/>
      <geom type="box" pos="-0.13 0 0.035" size="0.01 0.11 0.035" material="right"/>
    </body>

    <body name="sample" pos="0 0 0.055">
      <freejoint name="sample_free"/>
      <geom name="sample_geom" type="box" size="0.035 0.035 0.035"
            mass="0.05" friction="1.2 0.01 0.001" material="sample"/>
    </body>

    <body name="gripper" mocap="true" pos="-0.34 -0.22 0.22">
      <geom type="box" size="0.055 0.025 0.018" material="robot"
            contype="0" conaffinity="0"/>
      <geom type="box" pos="-0.045 0 -0.045" size="0.009 0.025 0.04"
            material="robot" contype="0" conaffinity="0"/>
      <geom type="box" pos="0.045 0 -0.045" size="0.009 0.025 0.04"
            material="robot" contype="0" conaffinity="0"/>
    </body>

    <camera name="evaluation" pos="0 -1.15 1.10"
            xyaxes="1 0 0 0 0.65 0.76" fovy="46"/>
  </worldbody>
  <equality>
    <weld name="grasp" body1="gripper" body2="sample"
          relpose="0 0 -0.085 1 0 0 0" active="false"
          solref="0.01 1"/>
  </equality>
</mujoco>
"""


class MuJoCoPickPlaceEnvironment:
    """MuJoCo task adapter used before the learned policy is connected."""

    OBJECTS = ("red_sample", "blue_sample", "green_sample")
    TARGETS = ("left_tray", "right_tray")
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
        "left_tray": (-0.22, 0.35, 0.085),
        "right_tray": (0.22, 0.35, 0.085),
    }
    PARK_POSITION = (-0.34, -0.22, 0.22)

    def __init__(
        self,
        *,
        gui: bool = False,
        realtime: bool = True,
        observation_images: bool = False,
    ) -> None:
        self._model = mujoco.MjModel.from_xml_string(_SCENE_XML)
        self._data = mujoco.MjData(self._model)
        self._rng = random.Random()
        self._realtime = realtime
        self._observation_images = observation_images
        self._viewer: Any | None = None
        self._renderer: mujoco.Renderer | None = None
        self._object = ""
        self._target = ""
        self._held_object: str | None = None
        self._steps = 0
        self._sample_joint = self._model.joint("sample_free").qposadr[0]
        self._sample_geom = self._model.geom("sample_geom").id
        self._grasp_equality = self._model.equality("grasp").id
        if gui:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self._model, self._data)

    def reset(self, *, seed: int, instruction: str) -> Observation:
        self._rng.seed(seed)
        self._object = self._rng.choice(self.OBJECTS)
        self._target = self._rng.choice(self.TARGETS)
        self._held_object = None
        self._steps = 0
        mujoco.mj_resetData(self._model, self._data)
        self._data.eq_active[self._grasp_equality] = 0
        self._set_sample_pose(self.OBJECT_POSITIONS[self._object])
        self._data.mocap_pos[0] = self.PARK_POSITION
        self._model.geom_rgba[self._sample_geom] = self.OBJECT_COLORS[self._object]
        self._advance(20)
        return self._observation()

    def step(self, action: Action) -> StepResult:
        self._steps += 1
        success = False
        failure: str | None = None

        if action.name == "pick":
            candidate = str(action.arguments.get("object", ""))
            if candidate != self._object or self._held_object is not None:
                failure = "wrong_object"
            else:
                object_position = self._data.xpos[self._model.body("sample").id].copy()
                above = (float(object_position[0]), float(object_position[1]), 0.24)
                grasp = (float(object_position[0]), float(object_position[1]), 0.14)
                self._move_gripper(above)
                self._move_gripper(grasp)
                self._data.eq_active[self._grasp_equality] = 1
                self._advance(20)
                self._held_object = candidate
                self._move_gripper(above)
        elif action.name == "place":
            destination = str(action.arguments.get("target", ""))
            if self._held_object != self._object or destination != self._target:
                failure = "wrong_target_or_empty_gripper"
            else:
                target = self.TARGET_POSITIONS[destination]
                above = (target[0], target[1], 0.24)
                self._move_gripper(above)
                self._move_gripper((target[0], target[1], 0.15))
                self._data.eq_active[self._grasp_equality] = 0
                self._held_object = None
                self._advance(80)
                success = self._sample_in_target(destination)
                if not success:
                    failure = "sample_outside_target"
                self._move_gripper(above)
        else:
            failure = "unsupported_action"

        terminated = success or failure is not None
        truncated = self._steps >= 4 and not terminated
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=terminated,
            truncated=truncated,
            info={"success": success, "failure_reason": failure},
        )

    def capture_rgb(self, *, width: int = 640, height: int = 480) -> Any:
        if (
            self._renderer is None
            or self._renderer.width != width
            or self._renderer.height != height
        ):
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self._model, height=height, width=width)
        self._renderer.update_scene(self._data, camera="evaluation")
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def _observation(self) -> Observation:
        robot_state = tuple(float(value) for value in self._data.mocap_pos[0])
        return Observation(
            image=self.capture_rgb() if self._observation_images else None,
            robot_state=robot_state,
            metadata={
                "visible_objects": [self._object],
                "visible_targets": [self._target],
                "held_object": self._held_object,
                "simulator": "mujoco",
            },
        )

    def _set_sample_pose(self, position: tuple[float, float, float]) -> None:
        address = self._sample_joint
        self._data.qpos[address : address + 3] = position
        self._data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        self._data.qvel[:] = 0.0

    def _move_gripper(
        self,
        target: tuple[float, float, float],
        *,
        frames: int = 35,
    ) -> None:
        start = self._data.mocap_pos[0].copy()
        for frame in range(1, frames + 1):
            alpha = frame / frames
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            self._data.mocap_pos[0] = start + (target - start) * smooth
            self._advance(1)

    def _advance(self, frames: int) -> None:
        for _ in range(frames):
            mujoco.mj_step(self._model, self._data)
            if self._viewer is not None:
                self._viewer.sync()
            if self._realtime:
                time.sleep(self._model.opt.timestep)

    def _sample_in_target(self, target: str) -> bool:
        sample_position = self._data.xpos[self._model.body("sample").id]
        target_position = self.TARGET_POSITIONS[target]
        return (
            abs(float(sample_position[0]) - target_position[0]) < 0.11
            and abs(float(sample_position[1]) - target_position[1]) < 0.09
            and float(sample_position[2]) < 0.14
        )
