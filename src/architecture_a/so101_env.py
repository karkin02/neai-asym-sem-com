from __future__ import annotations

import time
import random
from typing import Any

from shared.destinations import normalize_destination

import mujoco
import numpy as np

from .contracts import Action, Observation, StepResult


_SO101_XML = """
<mujoco model="so101_compatible_pick_place">
  <compiler angle="radian"/>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="1.2" armature="0.03"/>
    <geom friction="1.0 0.01 0.001" condim="3"/>
    <position kp="45" kv="5"/>
  </default>
  <visual>
    <global offwidth="640" offheight="480"/>
  </visual>
  <asset>
    <material name="table" rgba="0.52 0.55 0.58 1"/>
    <material name="arm" rgba="0.18 0.22 0.27 1"/>
    <material name="joint" rgba="0.08 0.09 0.10 1"/>
    <material name="sample" rgba="0.84 0.12 0.10 1"/>
    <material name="left" rgba="0.12 0.38 0.88 1"/>
    <material name="right" rgba="0.95 0.64 0.08 1"/>
    <material name="package" rgba="0.55 0.30 0.12 1"/>
    <material name="label" rgba="0.94 0.94 0.90 1"/>
    <material name="barcode" rgba="0.04 0.04 0.04 1"/>
    <material name="conveyor" rgba="0.12 0.14 0.16 1"/>
    <material name="roller" rgba="0.66 0.69 0.72 1"/>
    <material name="damage" rgba="0.90 0.05 0.05 1"/>
    <material name="outbound" rgba="0.12 0.62 0.30 1"/>
    <material name="obstacle" rgba="0.92 0.22 0.06 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.2 1.5" dir="0 0 -1" diffuse="0.85 0.85 0.85"/>
    <geom name="table" type="box" pos="0 0.12 -0.025"
          size="0.48 0.44 0.025" material="table"/>

    <body name="left_tray" pos="-0.17 0.16 0.025">
      <geom name="left_tray_geom" type="box" size="0.05 0.05 0.012" material="left"/>
    </body>
    <body name="right_tray" pos="0.17 0.16 0.025">
      <geom name="right_tray_geom" type="box" size="0.05 0.05 0.012" material="right"/>
    </body>
    <body name="sample" pos="0 0.10 0.045">
      <freejoint/>
      <geom name="sample_geom" type="box" size="0.03 0.03 0.03"
            mass="0.04" material="sample"/>
      <geom name="package_label" type="box" pos="0 0 0.0305"
            size="0.020 0.014 0.001" material="label"
            contype="0" conaffinity="0" rgba="1 1 1 0"/>
      <geom name="barcode_1" type="box" pos="-0.012 0 0.032"
            size="0.0015 0.011 0.0005" material="barcode"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="barcode_2" type="box" pos="-0.005 0 0.032"
            size="0.001 0.011 0.0005" material="barcode"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="barcode_3" type="box" pos="0.003 0 0.032"
            size="0.002 0.011 0.0005" material="barcode"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="barcode_4" type="box" pos="0.012 0 0.032"
            size="0.001 0.011 0.0005" material="barcode"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="damage_mark_1" type="box" pos="0 0 0.032"
            size="0.021 0.0025 0.001" euler="0 0 0.72" material="damage"
            contype="0" conaffinity="0" rgba="1 0 0 0"/>
      <geom name="damage_mark_2" type="box" pos="0 0 0.0325"
            size="0.021 0.0025 0.001" euler="0 0 -0.72" material="damage"
            contype="0" conaffinity="0" rgba="1 0 0 0"/>
      <geom name="damage_mark_front" type="box" pos="0 -0.031 0.004"
            size="0.020 0.001 0.003" euler="0 0.55 0" material="damage"
            contype="0" conaffinity="0" rgba="1 0 0 0"/>
      <geom name="damage_mark_side" type="box" pos="0.031 0 0.004"
            size="0.001 0.020 0.003" euler="-0.55 0 0" material="damage"
            contype="0" conaffinity="0" rgba="1 0 0 0"/>
    </body>

    <body name="warehouse_conveyor" pos="0 0.30 0.055">
      <geom name="conveyor_belt" type="box" size="0.30 0.055 0.018"
            material="conveyor" contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="conveyor_roller_left" type="cylinder" pos="-0.28 0 0"
            size="0.026 0.057" euler="1.5708 0 0" material="roller"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="conveyor_roller_right" type="cylinder" pos="0.28 0 0"
            size="0.026 0.057" euler="1.5708 0 0" material="roller"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="conveyor_leg_1" type="box" pos="-0.25 -0.038 -0.036"
            size="0.012 0.012 0.018" material="roller"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="conveyor_leg_2" type="box" pos="-0.25 0.038 -0.036"
            size="0.012 0.012 0.018" material="roller"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="conveyor_leg_3" type="box" pos="0.25 -0.038 -0.036"
            size="0.012 0.012 0.018" material="roller"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <geom name="conveyor_leg_4" type="box" pos="0.25 0.038 -0.036"
            size="0.012 0.012 0.018" material="roller"
            contype="0" conaffinity="0" rgba="0 0 0 0"/>
      <body name="belt_slat_1" pos="-0.24 0 0.020">
        <joint name="belt_slat_joint_1" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_1" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
      <body name="belt_slat_2" pos="-0.16 0 0.020">
        <joint name="belt_slat_joint_2" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_2" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
      <body name="belt_slat_3" pos="-0.08 0 0.020">
        <joint name="belt_slat_joint_3" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_3" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
      <body name="belt_slat_4" pos="0 0 0.020">
        <joint name="belt_slat_joint_4" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_4" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
      <body name="belt_slat_5" pos="0.08 0 0.020">
        <joint name="belt_slat_joint_5" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_5" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
      <body name="belt_slat_6" pos="0.16 0 0.020">
        <joint name="belt_slat_joint_6" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_6" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
      <body name="belt_slat_7" pos="0.24 0 0.020">
        <joint name="belt_slat_joint_7" type="slide" axis="1 0 0"/>
        <geom name="belt_slat_geom_7" type="box" size="0.025 0.048 0.003"
              contype="0" conaffinity="0" rgba="0.55 0.58 0.61 0"/>
      </body>
    </body>
    <body name="outbound_bin" pos="0.385 0.30 0.018">
      <geom name="outbound_bin_floor" type="box" pos="0 0 0"
            size="0.085 0.085 0.012" material="outbound" rgba="0.12 0.62 0.30 0"/>
      <geom name="outbound_bin_front" type="box" pos="0 -0.078 0.045"
            size="0.085 0.008 0.045" material="outbound" rgba="0.12 0.62 0.30 0"/>
      <geom name="outbound_bin_back" type="box" pos="0 0.078 0.045"
            size="0.085 0.008 0.045" material="outbound" rgba="0.12 0.62 0.30 0"/>
      <geom name="outbound_bin_end" type="box" pos="0.078 0 0.045"
            size="0.008 0.085 0.045" material="outbound" rgba="0.12 0.62 0.30 0"/>
    </body>
    <body name="unexpected_obstacle" pos="0 0.055 0">
      <joint name="obstacle_hinge" type="hinge" axis="1 0 0"
             range="-1.5708 0.05" damping="0.01"/>
      <geom name="obstacle_post" type="box" pos="0 -0.025 0.075"
            size="0.055 0.025 0.075" mass="0.30"
            material="obstacle" rgba="0.92 0.22 0.06 0"/>
      <geom name="obstacle_stripe" type="box" pos="0 -0.051 0.10"
            size="0.045 0.002 0.010" material="barcode"
            contype="0" conaffinity="0" rgba="0.04 0.04 0.04 0"/>
    </body>

    <body name="base" pos="0 -0.28 0">
      <geom type="cylinder" size="0.07 0.035" pos="0 0 0.035" material="joint"/>
      <body name="shoulder_pan_link" pos="0 0 0.07">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1"
               range="-1.8 1.8"/>
        <geom type="cylinder" size="0.045 0.045" material="joint"/>
        <body name="upper_arm" pos="0 0 0.035">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0"
                 range="-1.7 1.5"/>
          <geom type="capsule" fromto="0 0 0 0 0 0.21"
                size="0.032" mass="0.30" material="arm"/>
          <body name="elbow_link" pos="0 0 0.21">
            <joint name="elbow_flex" type="hinge" axis="0 1 0"
                   range="-2.0 1.8"/>
            <geom type="sphere" size="0.042" material="joint"/>
            <geom type="capsule" fromto="0 0 0 0 0 0.19"
                  size="0.028" mass="0.24" material="arm"/>
            <body name="wrist_flex_link" pos="0 0 0.19">
              <joint name="wrist_flex" type="hinge" axis="0 1 0"
                     range="-1.8 1.8"/>
              <geom type="sphere" size="0.036" material="joint"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.09"
                    size="0.024" mass="0.12" material="arm"/>
              <body name="wrist_roll_link" pos="0 0 0.09">
                <joint name="wrist_roll" type="hinge" axis="0 0 1"
                       range="-2.8 2.8"/>
                <geom type="cylinder" size="0.03 0.035" material="joint"/>
                <camera name="wrist" pos="0 0.09 0.10"
                        xyaxes="1 0 0 0 0.70 0.70" fovy="70"/>
                <!-- Dedicated close-inspection mount. Its fixed orientation is
                     calibrated for a 0.12 m package standoff; keeping this
                     separate preserves SmolVLA's original wrist observation. -->
                <camera name="wrist_inspection" pos="0 0.09 0.10"
                        quat="-0.14163245 -0.45569926 0.71569259 0.50996328"
                        fovy="70"/>
                <!-- Fixed wrist-mounted oblique camera. At the calibrated
                     damage pose it reproduces the former package-following
                     free-camera geometry without moving independently. -->
                <camera name="damage_inspection" pos="0.004246 -0.047924 0.003285"
                        quat="-0.292137 -0.386066 0.863981 -0.138370"
                        fovy="45"/>
                <body name="left_finger" pos="-0.027 0 0.03">
                  <joint name="gripper" type="slide" axis="1 0 0"
                         range="0 0.025"/>
                  <geom type="box" pos="0 0 0.035"
                        size="0.008 0.022 0.04" material="joint"/>
                </body>
                <body name="right_finger" pos="0.027 0 0.03">
                  <joint name="gripper_mirror" type="slide" axis="-1 0 0"
                         range="0 0.025"/>
                  <geom type="box" pos="0 0 0.035"
                        size="0.008 0.022 0.04" material="joint"/>
                </body>
                <site name="grasp_site" pos="0 0 0.10" size="0.012"
                      rgba="0.2 0.9 0.2 0.35"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <camera name="overhead" pos="0 -1.05 1.05"
            xyaxes="1 0 0 0 0.67 0.74" fovy="48"/>
    <!-- Original angled warehouse view, retained only for human demo execution frames. -->
    <camera name="observer" pos="0 -1.05 1.05"
            xyaxes="1 0 0 0 0.67 0.74" fovy="31"/>
  </worldbody>
  <contact>
    <pair geom1="sample_geom" geom2="conveyor_belt"
          friction="1.0 0.01 0.001 0.0001 0.0001"/>
  </contact>
  <equality>
    <joint joint1="gripper_mirror" joint2="gripper"
           polycoef="0 1 0 0 0"/>
  </equality>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" ctrlrange="-1.8 1.8"/>
    <position name="shoulder_lift" joint="shoulder_lift" ctrlrange="-1.7 1.5"/>
    <position name="elbow_flex" joint="elbow_flex" ctrlrange="-2.0 1.8"/>
    <position name="wrist_flex" joint="wrist_flex" ctrlrange="-1.8 1.8"/>
    <position name="wrist_roll" joint="wrist_roll" ctrlrange="-2.8 2.8"/>
    <position name="gripper" joint="gripper" ctrlrange="0 0.025" kp="120" kv="8"/>
  </actuator>
</mujoco>
"""


class SO101MuJoCoEnvironment:
    """Six-actuator SO-101-compatible simulation boundary.

    The joint naming and action layout match LeRobot's SO-101 convention. The
    primitive geometry is intentionally approximate and is not a mechanical
    digital twin of the production arm.
    """

    JOINT_NAMES = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    TARGET_POSITIONS = {
        "left_tray": np.array((-0.14, 0.25, 0.075)),
        "right_tray": np.array((0.14, 0.25, 0.075)),
    }
    WAREHOUSE_LAYOUTS = {
        "v1": {
            "left_tray": np.array((-0.17, 0.16, 0.075)),
            "right_tray": np.array((0.17, 0.16, 0.075)),
            "conveyor": np.array((0.0, 0.30, 0.075)),
        },
        "v2": {
            "left_tray": np.array((-0.15, 0.14, 0.075)),
            "right_tray": np.array((0.15, 0.14, 0.075)),
            "conveyor": np.array((0.0, 0.26, 0.075)),
        },
        "v3": {
            "left_tray": np.array((-0.15, 0.14, 0.075)),
            "right_tray": np.array((0.15, 0.14, 0.075)),
            "conveyor": np.array((0.0, 0.26, 0.075)),
        },
    }
    WAREHOUSE_TARGET_POSITIONS = {
        "left_tray": np.array((-0.17, 0.16, 0.075)),
        "right_tray": np.array((0.17, 0.16, 0.075)),
        "conveyor": np.array((0.0, 0.30, 0.075)),
    }
    CONVEYOR_POSITION = np.array((0.0, 0.30, 0.075))
    SCENARIOS = (
        "pick_place",
        "warehouse_normal",
        "barcode_missing",
        "package_damaged",
        "unexpected_obstacle",
        "obstacle_vla_violation",
        "damaged_vla_violation",
    )
    BARCODE_INSPECTION_POSE = np.array(
        (-1.4047251044, -0.5137837024, -1.2718737072,
         0.1710056685, 1.2997269143, 0.02)
    )
    DAMAGE_INSPECTION_POSE = np.array(
        (1.1573390786, 0.3129013740, 1.4962409985,
         -0.4586209168, -1.1732841271, 0.02)
    )

    def __init__(
        self,
        *,
        gui: bool = False,
        realtime: bool = True,
        observation_images: bool = True,
        kinematic_control: bool = False,
        scenario: str = "pick_place",
        warehouse_layout: str = "v1",
    ) -> None:
        if scenario not in self.SCENARIOS:
            raise ValueError(
                f"Unknown scenario {scenario!r}; expected one of {self.SCENARIOS}."
            )
        self._model = mujoco.MjModel.from_xml_string(_SO101_XML)
        self._data = mujoco.MjData(self._model)
        self._rng = random.Random()
        self._realtime = realtime
        self._observation_images = observation_images
        self._kinematic_control = kinematic_control
        self._scenario = scenario
        if warehouse_layout not in self.WAREHOUSE_LAYOUTS:
            raise ValueError(
                f"Unknown warehouse layout {warehouse_layout!r}; expected v1, v2, or v3."
            )
        self._warehouse_layout = warehouse_layout
        self.WAREHOUSE_TARGET_POSITIONS = {
            name: value.copy()
            for name, value in self.WAREHOUSE_LAYOUTS[warehouse_layout].items()
        }
        self.CONVEYOR_POSITION = self.WAREHOUSE_TARGET_POSITIONS["conveyor"].copy()
        self._viewer: Any | None = None
        self._renderers: dict[tuple[str, int, int], mujoco.Renderer] = {}
        self._steps = 0
        self._conveyor_phase = 0.0
        self._collision_stop = False
        self._obstacle_tipped = False
        self._held = False
        self._target_name = "left_tray"
        self._instruction = ""
        self._joint_qpos_addresses = tuple(
            int(self._model.joint(name).qposadr[0]) for name in self.JOINT_NAMES
        )
        self._joint_dof_addresses = tuple(
            int(self._model.joint(name).dofadr[0]) for name in self.JOINT_NAMES
        )
        self._sample_joint_address = int(self._model.body("sample").jntadr[0])
        self._sample_qpos_address = int(
            self._model.jnt_qposadr[self._sample_joint_address]
        )
        self._sample_dof_address = int(
            self._model.jnt_dofadr[self._sample_joint_address]
        )
        self._grasp_site_id = self._model.site("grasp_site").id
        arm_body_ids = {
            self._model.body(name).id
            for name in (
                "base",
                "shoulder_pan_link",
                "upper_arm",
                "elbow_link",
                "wrist_flex_link",
                "wrist_roll_link",
                "left_finger",
                "right_finger",
            )
        }
        self._arm_geom_ids = {
            geom_id
            for geom_id, body_id in enumerate(self._model.geom_bodyid)
            if int(body_id) in arm_body_ids
        }
        self._obstacle_geom_id = self._model.geom("obstacle_post").id
        arm_body_ids = {
            self._model.body(name).id
            for name in (
                "base",
                "shoulder_pan_link",
                "upper_arm",
                "elbow_link",
                "wrist_flex_link",
                "wrist_roll_link",
                "left_finger",
                "right_finger",
            )
        }
        self._arm_geom_ids = {
            geom_id
            for geom_id, body_id in enumerate(self._model.geom_bodyid)
            if int(body_id) in arm_body_ids
        }
        self._obstacle_hinge_qpos = int(
            self._model.joint("obstacle_hinge").qposadr[0]
        )
        self._obstacle_hinge_dof = int(
            self._model.joint("obstacle_hinge").dofadr[0]
        )
        self._control_ranges = self._model.actuator_ctrlrange.copy()
        self._warehouse_geom_names = (
            "package_label",
            "conveyor_belt",
            "conveyor_roller_left",
            "conveyor_roller_right",
            *(f"conveyor_leg_{index}" for index in range(1, 5)),
            *(f"belt_slat_geom_{index}" for index in range(1, 8)),
            "outbound_bin_floor",
            "outbound_bin_front",
            "outbound_bin_back",
            "outbound_bin_end",
            "obstacle_post",
            "obstacle_stripe",
        )
        self._conveyor_slat_bases = np.linspace(-0.24, 0.24, 7)
        self._conveyor_slat_qpos = tuple(
            int(self._model.joint(f"belt_slat_joint_{index}").qposadr[0])
            for index in range(1, 8)
        )
        self._barcode_geom_names = tuple(f"barcode_{index}" for index in range(1, 5))
        if gui:
            from mujoco import viewer

            self._viewer = viewer.launch_passive(self._model, self._data)

    def reset(self, *, seed: int, instruction: str) -> Observation:
        self._rng.seed(seed)
        self._instruction = instruction
        if self._scenario == "pick_place":
            self._target_name = self._rng.choice(tuple(self.TARGET_POSITIONS))
        elif self._scenario == "barcode_missing":
            self._target_name = "left_tray"
        elif self._scenario in ("package_damaged", "damaged_vla_violation"):
            self._target_name = "right_tray"
        else:
            self._target_name = "conveyor"
        mujoco.mj_resetData(self._model, self._data)
        self._apply_scenario_appearance()
        sample_x_limit = 0.08 if self._scenario != "pick_place" else 0.12
        sample_y_range = (
            (0.03, 0.11)
            if self._scenario != "pick_place" and self._warehouse_layout in {"v2", "v3"}
            else (0.04, 0.16)
        )
        sample_position = np.array(
            (
                self._rng.uniform(-sample_x_limit, sample_x_limit),
                self._rng.uniform(*sample_y_range),
                0.045,
            )
        )
        address = self._sample_qpos_address
        self._data.qpos[address : address + 3] = sample_position
        self._data.qpos[address + 3 : address + 7] = (1.0, 0.0, 0.0, 0.0)
        home = np.array((0.0, -0.45, 0.90, 0.55, 0.0, 0.018))
        self._data.ctrl[:] = home
        self._steps = 0
        self._conveyor_phase = 0.0
        self._collision_stop = False
        self._obstacle_tipped = False
        self._held = False
        self._advance(250)
        return self._observation()

    def step(self, action: Action) -> StepResult:
        if action.name != "joint_position":
            return StepResult(
                observation=self._observation(),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"success": False, "failure_reason": "expected_joint_position"},
            )
        if len(action.values) != 6:
            raise ValueError(f"SO-101 action must contain 6 values, got {len(action.values)}")

        requested = np.asarray(action.values, dtype=float)
        self._data.ctrl[:] = np.clip(
            requested,
            self._control_ranges[:, 0],
            self._control_ranges[:, 1],
        )
        released_object = False
        if requested[5] <= 0.006 and not self._held:
            distance = np.linalg.norm(
                self.end_effector_position - self.sample_position
            )
            self._held = bool(distance < 0.075)
        elif requested[5] >= 0.014:
            released_object = self._held
            self._held = False
        if self._kinematic_control:
            self._move_joints_kinematically(self._data.ctrl.copy(), frames=35)
            if released_object:
                self._advance(125)
        else:
            self._advance(35)
        self._steps += 1
        sample_position = self.sample_position
        target_position = self.target_positions[self._target_name]
        success = bool(
            not self._held
            and np.linalg.norm(sample_position[:2] - target_position[:2]) < 0.09
            and float(sample_position[2]) < 0.14
        )
        truncated = self._steps >= 200 and not success
        return StepResult(
            observation=self._observation(),
            reward=1.0 if success else 0.0,
            terminated=success,
            truncated=truncated,
            info={"success": success, "failure_reason": None},
        )

    def capture_rgb(
        self,
        *,
        camera: str = "overhead",
        width: int = 640,
        height: int = 480,
    ) -> Any:
        key = (camera, width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(
                self._model,
                height=height,
                width=width,
            )
        renderer = self._renderers[key]
        renderer.update_scene(self._data, camera=camera)
        return renderer.render()

    def capture_condition_views(
        self, *, width: int = 320, height: int = 240
    ) -> dict[str, np.ndarray]:
        """Capture non-destructive close views for barcode/damage inspection.

        Both views use the original policy wrist camera.  Only the arm pose is
        changed between inspections, matching how a physical robot uses one
        eye-in-hand camera. Simulator state is restored before returning.
        """
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        renderer = mujoco.Renderer(self._model, height=height, width=width)
        try:
            # Calibrated absolute arm poses. Package randomization in layout v3
            # remains inside both close views; these poses deliberately retain
            # the original wrist-camera intrinsics and mounting orientation.
            wrist_joints = self.BARCODE_INSPECTION_POSE
            for value, address in zip(wrist_joints, self._joint_qpos_addresses):
                self._data.qpos[address] = value
            self._data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
            renderer.update_scene(self._data, camera="wrist")
            barcode = renderer.render().copy()

            damage_joints = self.DAMAGE_INSPECTION_POSE
            for value, address in zip(damage_joints, self._joint_qpos_addresses):
                self._data.qpos[address] = value
            self._data.qvel[:] = 0.0
            mujoco.mj_forward(self._model, self._data)
            renderer.update_scene(self._data, camera="wrist")
            damage = renderer.render().copy()

        finally:
            renderer.close()
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            mujoco.mj_forward(self._model, self._data)
        return {"barcode": barcode, "damage": damage}

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def wait_for_viewer_close(self) -> None:
        """Keep the passive viewer responsive while holding the current pose."""
        if self._viewer is None:
            return
        while self._viewer.is_running():
            self._animate_conveyor()
            self._drive_conveyor_package()
            mujoco.mj_step(self._model, self._data)
            self._viewer.sync()
            time.sleep(self._model.opt.timestep)

    def hold_viewer_static(self) -> None:
        """Keep the current failed state visible without advancing physics."""
        if self._viewer is None:
            return
        while self._viewer.is_running():
            self._viewer.sync()
            time.sleep(self._model.opt.timestep)

    def _observation(self) -> Observation:
        state = tuple(
            float(self._data.qpos[address])
            for address in self._joint_qpos_addresses
        )
        return Observation(
            image=self.capture_rgb(camera="overhead")
            if self._observation_images
            else None,
            robot_state=state,
            metadata={
                "simulator": "mujoco",
                "embodiment": "so101_compatible",
                "joint_names": self.JOINT_NAMES,
                "sample_position": tuple(float(value) for value in self.sample_position),
                "end_effector_position": tuple(
                    float(value) for value in self.end_effector_position
                ),
                "target_name": self._target_name,
                "target_role": normalize_destination(self._target_name),
                "target_position": tuple(
                    float(value) for value in self.target_positions[self._target_name]
                ),
                "held_object": self._held,
                "scenario": self._scenario,
                "problem": (
                    "unexpected_obstacle"
                    if self._scenario
                    in ("unexpected_obstacle", "obstacle_vla_violation")
                    else "package_damaged"
                    if self._scenario in ("package_damaged", "damaged_vla_violation")
                    else self._scenario
                    if self._scenario == "barcode_missing"
                    else None
                ),
                "package": {
                    "id": "package_01",
                    "barcode": (
                        "missing" if self._scenario == "barcode_missing" else "present"
                    ),
                    "condition": (
                        "damaged"
                        if self._scenario in ("package_damaged", "damaged_vla_violation")
                        else "intact"
                    ),
                },
            },
        )

    def scene_graph(self) -> dict[str, object]:
        """Return the compact warehouse exception payload without camera data."""
        return {
            "task": "warehouse_pick",
            "zones": {
                "conveyor": {"role": "normal routing", "color": "dark"},
                "inspection_tray": {
                    "role": "manual inspection", "color": "blue", "physical_target": "left_tray"
                },
                "rejection_tray": {
                    "role": "damaged-package rejection", "color": "yellow", "physical_target": "right_tray"
                },
                "outbound_bin": {"role": "completed shipments", "color": "green"},
            },
            "objects": [
                {
                    "id": "package_01",
                    "type": "package",
                    "position": [round(float(value), 4) for value in self.sample_position],
                    "barcode": (
                        "missing" if self._scenario == "barcode_missing" else "present"
                    ),
                    "condition": (
                        "damaged"
                        if self._scenario in ("package_damaged", "damaged_vla_violation")
                        else "intact"
                    ),
                },
                {"id": "conveyor_01", "type": "conveyor", "state": "running"},
                {
                    "id": "outbound_bin_01",
                    "type": "outbound_bin",
                    "state": "ready",
                },
                *(
                    [
                        {
                            "id": "obstacle_01",
                            "type": "unexpected_obstacle",
                            "location": "robot_path",
                            "movable": "unknown",
                        }
                    ]
                    if self._scenario
                    in ("unexpected_obstacle", "obstacle_vla_violation")
                    else []
                ),
            ],
            "robot": {
                "holding": "package_01" if self._held else None,
                "blocked": self._scenario
                in ("unexpected_obstacle", "obstacle_vla_violation"),
            },
            "problem": (
                "unexpected_obstacle"
                if self._scenario
                in ("unexpected_obstacle", "obstacle_vla_violation")
                else "package_damaged"
                if self._scenario in ("package_damaged", "damaged_vla_violation")
                else self._scenario
                if self._scenario == "barcode_missing"
                else None
            ),
        }

    @property
    def target_positions(self) -> dict[str, np.ndarray]:
        if self._scenario == "pick_place":
            return self.TARGET_POSITIONS
        return self.WAREHOUSE_TARGET_POSITIONS

    def _apply_scenario_appearance(self) -> None:
        warehouse = self._scenario != "pick_place"
        warehouse_targets = self.WAREHOUSE_TARGET_POSITIONS
        scanner_visible_marks = warehouse and self._warehouse_layout in {"v2", "v3"}
        overhead_camera = self._model.camera("overhead")
        if warehouse and self._warehouse_layout == "v2":
            overhead_camera.pos = (0.0, 0.10, 1.10)
            overhead_camera.quat = (1.0, 0.0, 0.0, 0.0)
        elif warehouse and self._warehouse_layout == "v3":
            overhead_camera.pos = (0.0, 0.10, 1.10)
            overhead_camera.quat = (1.0, 0.0, 0.0, 0.0)
            overhead_camera.fovy = 31.0
        tray_positions = (
            (
                (*warehouse_targets["left_tray"][:2], 0.025),
                (*warehouse_targets["right_tray"][:2], 0.025),
            )
            if warehouse
            else ((-0.14, 0.25, 0.025), (0.14, 0.25, 0.025))
        )
        self._model.body("left_tray").pos = tray_positions[0]
        self._model.body("right_tray").pos = tray_positions[1]
        self._model.body("warehouse_conveyor").pos = (
            (*warehouse_targets["conveyor"][:2], 0.055)
            if warehouse
            else (0.0, 0.30, -1.0)
        )
        self._model.body("outbound_bin").pos = (
            (0.385, float(warehouse_targets["conveyor"][1]), 0.018)
            if warehouse
            else (0.385, 0.30, -1.0)
        )
        self._model.body("unexpected_obstacle").pos = (
            (0.0, 0.055, 0.0)
            if self._scenario in ("unexpected_obstacle", "obstacle_vla_violation")
            else (0.0, 0.055, -1.0)
        )
        tray_size = (0.07, 0.07, 0.012) if warehouse else (0.12, 0.10, 0.012)
        self._model.geom("left_tray_geom").size = tray_size
        self._model.geom("right_tray_geom").size = tray_size
        self._model.geom("sample_geom").rgba = (
            (0.55, 0.30, 0.12, 1.0) if warehouse else (0.84, 0.12, 0.10, 1.0)
        )
        # V2 packages use scanner-visible markings while v1 dimensions remain
        # unchanged for benchmark reproducibility.
        self._model.geom("package_label").size = (
            (0.026, 0.023, 0.001) if scanner_visible_marks else (0.020, 0.014, 0.001)
        )
        barcode_sizes = (
            ((0.0018, 0.019, 0.0007), (0.0013, 0.019, 0.0007),
             (0.0023, 0.019, 0.0007), (0.0014, 0.019, 0.0007))
            if scanner_visible_marks
            else ((0.0015, 0.011, 0.0005), (0.001, 0.011, 0.0005),
                  (0.002, 0.011, 0.0005), (0.001, 0.011, 0.0005))
        )
        for name, size in zip(self._barcode_geom_names, barcode_sizes):
            self._model.geom(name).size = size
        top_damage_size = (
            (0.027, 0.004, 0.0015)
            if scanner_visible_marks
            else (0.021, 0.0025, 0.001)
        )
        for name in ("damage_mark_1", "damage_mark_2"):
            self._model.geom(name).size = top_damage_size
        self._model.geom("damage_mark_front").size = (
            (0.026, 0.0015, 0.005)
            if scanner_visible_marks
            else (0.020, 0.001, 0.003)
        )
        self._model.geom("damage_mark_side").size = (
            (0.0015, 0.026, 0.005)
            if scanner_visible_marks
            else (0.001, 0.020, 0.003)
        )
        for name in self._warehouse_geom_names:
            geom = self._model.geom(name)
            geom.rgba[3] = 1.0 if warehouse else 0.0
        barcode_visible = self._scenario == "warehouse_normal"
        for name in self._barcode_geom_names:
            self._model.geom(name).rgba[3] = 1.0 if barcode_visible else 0.0
        for name in ("damage_mark_1", "damage_mark_2"):
            self._model.geom(name).rgba[3] = 0.0
        for name in ("damage_mark_front", "damage_mark_side"):
            self._model.geom(name).rgba[3] = (
            1.0
            if self._scenario in ("package_damaged", "damaged_vla_violation")
            else 0.0
            )

    def _advance(self, frames: int) -> None:
        for _ in range(frames):
            self._animate_conveyor()
            self._drive_conveyor_package()
            mujoco.mj_step(self._model, self._data)
            if self._held:
                self._carry_sample()
            if self._viewer is not None:
                self._viewer.sync()
            if self._realtime:
                time.sleep(self._model.opt.timestep)

    def advance_idle(self, seconds: float) -> None:
        """Advance warehouse machinery without issuing a robot action."""
        if seconds < 0.0:
            raise ValueError("seconds must be non-negative")
        self._advance(round(seconds / self._model.opt.timestep))

    def settle_collision(self, seconds: float = 0.5) -> None:
        """Apply the contact impulse and let the hinged obstacle fall physically."""
        if not self._collision_stop:
            return
        self._data.ctrl[:] = np.array((0.0, -0.45, 0.90, 0.55, 0.0, 0.018))
        frames = max(1, round(seconds / self._model.opt.timestep))
        for frame in range(frames):
            self._data.qfrc_applied[self._obstacle_hinge_dof] = (
                -0.45 if frame < frames // 5 else 0.0
            )
            self._animate_conveyor()
            mujoco.mj_step(self._model, self._data)
            if self._viewer is not None:
                self._viewer.sync()
            if self._realtime:
                time.sleep(self._model.opt.timestep)
        self._data.qfrc_applied[self._obstacle_hinge_dof] = 0.0
        self._obstacle_tipped = bool(
            self._data.qpos[self._obstacle_hinge_qpos] < np.deg2rad(-20.0)
        )

    @property
    def shipment_complete(self) -> bool:
        position = self.sample_position
        return bool(
            self._scenario in ("warehouse_normal", "damaged_vla_violation")
            and 0.295 < float(position[0]) < 0.47
            and abs(float(position[1]) - 0.30) < 0.08
            and float(position[2]) < 0.10
        )

    @property
    def collision_detected(self) -> bool:
        return self._collision_stop

    @property
    def obstacle_tipped(self) -> bool:
        return self._obstacle_tipped

    @property
    def obstacle_angle_degrees(self) -> float:
        return float(np.rad2deg(self._data.qpos[self._obstacle_hinge_qpos]))

    def _move_joints_kinematically(
        self,
        target: np.ndarray,
        *,
        frames: int,
    ) -> None:
        start = np.array(
            [self._data.qpos[address] for address in self._joint_qpos_addresses]
        )
        for frame in range(1, frames + 1):
            alpha = frame / frames
            smooth = alpha * alpha * (3.0 - 2.0 * alpha)
            position = start + (target - start) * smooth
            for index, address in enumerate(self._joint_qpos_addresses):
                self._data.qpos[address] = position[index]
            self._data.qvel[list(self._joint_dof_addresses)] = 0.0
            self._animate_conveyor()
            mujoco.mj_forward(self._model, self._data)
            if self._scenario == "obstacle_vla_violation" and self._arm_hit_obstacle():
                self._collision_stop = True
                break
            if self._held:
                self._carry_sample()
            if self._viewer is not None:
                self._viewer.sync()
            if self._realtime:
                time.sleep(self._model.opt.timestep)

    @property
    def sample_position(self) -> np.ndarray:
        return self._data.xpos[self._model.body("sample").id].copy()

    @property
    def end_effector_position(self) -> np.ndarray:
        return self._data.site_xpos[self._grasp_site_id].copy()

    def solve_ik(
        self,
        target_position: tuple[float, float, float] | np.ndarray,
        *,
        gripper: float,
        max_iterations: int = 120,
    ) -> tuple[float, ...]:
        """Solve position-only IK while preserving the live simulator state."""

        target = np.asarray(target_position, dtype=float)
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        q = np.array(
            [self._data.qpos[address] for address in self._joint_qpos_addresses[:5]]
        )
        jacobian = np.zeros((3, self._model.nv))
        damping = 2e-3

        try:
            for _ in range(max_iterations):
                for index, address in enumerate(self._joint_qpos_addresses[:5]):
                    self._data.qpos[address] = q[index]
                mujoco.mj_forward(self._model, self._data)
                error = target - self.end_effector_position
                if np.linalg.norm(error) < 0.004:
                    break
                jacobian.fill(0.0)
                mujoco.mj_jacSite(
                    self._model,
                    self._data,
                    jacobian,
                    None,
                    self._grasp_site_id,
                )
                reduced = jacobian[:, self._joint_dof_addresses[:5]]
                delta = reduced.T @ np.linalg.solve(
                    reduced @ reduced.T + damping * np.eye(3),
                    error,
                )
                q += np.clip(delta, -0.12, 0.12)
                for index, name in enumerate(self.JOINT_NAMES[:5]):
                    joint = self._model.joint(name)
                    q[index] = np.clip(q[index], joint.range[0], joint.range[1])
        finally:
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            mujoco.mj_forward(self._model, self._data)

        return tuple(float(value) for value in q) + (float(gripper),)

    def evaluate_ik_confidence(
        self,
        target_position: tuple[float, float, float] | np.ndarray,
    ) -> tuple[float, float]:
        """Score IK feasibility from the solved Cartesian residual."""
        target = np.asarray(target_position, dtype=float)
        solution = self.solve_ik(target, gripper=0.02)
        saved_qpos = self._data.qpos.copy()
        saved_qvel = self._data.qvel.copy()
        try:
            for index, address in enumerate(self._joint_qpos_addresses[:5]):
                self._data.qpos[address] = solution[index]
            mujoco.mj_forward(self._model, self._data)
            residual = float(np.linalg.norm(self.end_effector_position - target))
        finally:
            self._data.qpos[:] = saved_qpos
            self._data.qvel[:] = saved_qvel
            mujoco.mj_forward(self._model, self._data)
        confidence = float(np.exp(-residual / 0.025))
        return confidence, residual

    def predict_obstacle_collision(self, action_chunk: np.ndarray) -> int | None:
        """Return the first command predicted to contact the scenario obstacle."""
        commands = np.asarray(action_chunk, dtype=float)
        if commands.ndim != 2 or commands.shape[1] != len(self.JOINT_NAMES):
            raise ValueError("Expected an N x 6 absolute joint-position chunk.")
        if self._scenario not in ("unexpected_obstacle", "obstacle_vla_violation"):
            return None

        probe = mujoco.MjData(self._model)
        probe.qpos[:] = self._data.qpos
        probe.qvel[:] = self._data.qvel
        for command_index, command in enumerate(commands):
            for joint_index, address in enumerate(self._joint_qpos_addresses):
                probe.qpos[address] = command[joint_index]
            mujoco.mj_forward(self._model, probe)
            for contact_index in range(probe.ncon):
                contact = probe.contact[contact_index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if self._obstacle_geom_id in pair and pair & self._arm_geom_ids:
                    return command_index
        return None

    def preview_action_success(self, action_chunk: np.ndarray) -> bool:
        """Simulate an action chunk, then restore the episode exactly.

        This outcome gate prevents a locally generated chunk that cannot pass
        placement validation from disturbing the live scene before recovery.
        It is intentionally simulator-only; hardware adapters must provide
        their own certified forward model rather than claiming this capability.
        """
        commands = np.asarray(action_chunk, dtype=float)
        if commands.ndim != 2 or commands.shape[1] != len(self.JOINT_NAMES):
            raise ValueError("Expected an N x 6 absolute joint-position chunk.")
        saved = {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "ctrl": self._data.ctrl.copy(),
            "time": float(self._data.time),
            "steps": self._steps,
            "held": self._held,
            "collision_stop": self._collision_stop,
            "obstacle_tipped": self._obstacle_tipped,
            "conveyor_phase": self._conveyor_phase,
            "viewer": self._viewer,
            "realtime": self._realtime,
        }
        success = False
        try:
            self._viewer = None
            self._realtime = False
            for command in commands:
                result = self.step(
                    Action("joint_position", tuple(float(v) for v in command))
                )
                if result.info.get("success"):
                    success = True
                    break
                if result.terminated or result.truncated:
                    break
        finally:
            self._data.qpos[:] = saved["qpos"]
            self._data.qvel[:] = saved["qvel"]
            self._data.ctrl[:] = saved["ctrl"]
            self._data.time = saved["time"]
            self._steps = saved["steps"]
            self._held = saved["held"]
            self._collision_stop = saved["collision_stop"]
            self._obstacle_tipped = saved["obstacle_tipped"]
            self._conveyor_phase = saved["conveyor_phase"]
            self._viewer = saved["viewer"]
            self._realtime = saved["realtime"]
            mujoco.mj_forward(self._model, self._data)
        return success

    @property
    def joint_positions(self) -> tuple[float, ...]:
        return tuple(
            float(self._data.qpos[address])
            for address in self._joint_qpos_addresses
        )

    def _carry_sample(self) -> None:
        address = self._sample_qpos_address
        self._data.qpos[address : address + 3] = self.end_effector_position
        self._data.qvel[:6] = 0.0
        mujoco.mj_forward(self._model, self._data)

    def _animate_conveyor(self) -> None:
        if self._scenario == "pick_place":
            return
        self._conveyor_phase = (self._conveyor_phase + 0.00035) % 0.56
        for base, address in zip(self._conveyor_slat_bases, self._conveyor_slat_qpos):
            wrapped = ((base + self._conveyor_phase + 0.28) % 0.56) - 0.28
            self._data.qpos[address] = wrapped - base

    def _drive_conveyor_package(self) -> None:
        if self._scenario not in (
            "warehouse_normal",
            "obstacle_vla_violation",
            "damaged_vla_violation",
        ) or self._held:
            return
        position = self.sample_position
        on_belt = (
            abs(float(position[0])) < 0.29
            and abs(float(position[1]) - 0.30) < 0.065
            and 0.085 < float(position[2]) < 0.14
        )
        if on_belt:
            self._data.qvel[self._sample_dof_address] = 0.075

    def _arm_hit_obstacle(self) -> bool:
        for index in range(self._data.ncon):
            contact = self._data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._obstacle_geom_id in pair and pair & self._arm_geom_ids:
                return True
        return False


def main() -> None:
    environment = SO101MuJoCoEnvironment(gui=True, realtime=True)
    try:
        observation = environment.reset(seed=0, instruction="")
        poses = (
            (0.0, -0.45, 0.90, 0.55, 0.0, 0.018),
            (-0.55, -0.65, 1.10, 0.35, 0.4, 0.008),
            (0.55, -0.65, 1.10, 0.35, -0.4, 0.022),
        )
        for _ in range(3):
            for pose in poses:
                environment.step(Action("joint_position", values=pose))
        print({"joint_names": environment.JOINT_NAMES, "state": observation.robot_state})
    finally:
        environment.close()


if __name__ == "__main__":
    main()
