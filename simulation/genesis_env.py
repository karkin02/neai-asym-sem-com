import genesis as gs
import numpy as np
import matplotlib.pyplot as plt
import time
import os

robot_file = os.path.join(os.getcwd(), "simulation/hello_robot_stretch/assets/stretch.xml")

class GenesisEnv:

    def __init__(self, task="navigation", max_steps=100): # or "pick_place"

        # Create Simulation Scene
        self.scene = gs.Scene(
            show_viewer=False
        )

        # Add Ground
        self.scene.add_entity(
            gs.morphs.Plane()
        )

        # Add a robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=robot_file
            )
        )

        # Add table
        self.table = self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.8, 0.0, 0.35),
                size=(0.3, 0.3, 0.35)
            )
        )

        # Add object
        self.object = self.scene.add_entity(
            gs.morphs.Box(
                pos=(0.0, 0.0, 0.025),
                size=(0.05, 0.05, 0.05)
            )
        )

        # Add Camera
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(2.5, -2.5, 2),
            lookat=(0, 0, 0.5),
            fov=60
        )

        # Build Simulation
        self.scene.build()
        self.initial_qpos = self.robot.get_qpos().clone()
        self.initial_obj_pos = self.object.get_pos().clone()
        self.initial_obj_quat = self.object.get_quat().clone()
        
        self.task = task 
        self.goal_object = self.object
        self.goal_position = self.object.get_pos().clone()
        if self.task == "pick_place":
            self.place_position = self.table.get_pos().clone()
            self.place_position[2] += 0.35 + 0.025

        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0

        # Reset robot
        self.robot.set_qpos(self.initial_qpos)

        # Reset object
        self.object.set_pos(self.initial_obj_pos)
        self.object.set_quat(self.initial_obj_quat)

        # update goal position after reset
        self.goal_position = self.object.get_pos().clone()

        # Let physics settle
        for _ in range(20):
            self.scene.step()

        return self.get_rgb()

    def step(self, action):
        self.execute(action)
        self.scene.step()
        self.step_count += 1

        obs = self.get_rgb()
        reward = self.compute_reward()
        done = self.is_done()
        info = {}
        return obs, reward, done, info

    def get_rgb(self):
        output = self.camera.render()
        
        if isinstance(output, tuple):
            rgb = output[0]
        else:
            rgb = output

        rgb = rgb[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8)

        return rgb

    def execute(self, action):

        if action == "MoveAhead":

            self.robot.control_dofs_velocity(
                velocity=[5.0, 5.0],
                dofs_idx_local=[6, 7]
            )

            for _ in range(30):
                self.scene.step()

            self.robot.control_dofs_velocity(
                velocity=[0.0, 0.0],
                dofs_idx_local=[6, 7]
            )

        elif action == "TurnLeft":

            self.robot.control_dofs_velocity(
                velocity=[-3.0, 3.0],
                dofs_idx_local=[6, 7]
            )

            for _ in range(20):
                self.scene.step()

            self.robot.control_dofs_velocity(
                velocity=[0.0, 0.0],
                dofs_idx_local=[6, 7]
            )

        elif action == "TurnRight":

            self.robot.control_dofs_velocity(
                velocity=[3.0, -3.0],
                dofs_idx_local=[6, 7]
            )

            for _ in range(20):
                self.scene.step()

            self.robot.control_dofs_velocity(
                velocity=[0.0, 0.0],
                dofs_idx_local=[6, 7]
            )

        elif action == "PickUp":

            # Lift
            self.robot.control_dofs_position(
                position=[0.20],
                dofs_idx_local=[8]
            )

            for _ in range(20):
                self.scene.step()

            # Extend arm
            self.robot.control_dofs_position(
                position=[0.05, 0.05, 0.05, 0.05],
                dofs_idx_local=[9, 10, 11, 12]
            )

            for _ in range(20):
                self.scene.step()

            # Close gripper
            self.robot.control_dofs_position(
                position=[0.0, 0.0],
                dofs_idx_local=[15, 18]
            )

            for _ in range(20):
                self.scene.step()

        elif action == "PlaceOn":

            self.robot.control_dofs_position(
                position=[0.04, 0.04],
                dofs_idx_local=[15, 18]
            )

            for _ in range(20):
                self.scene.step()

        elif action == "Stop":

            self.robot.control_dofs_velocity(
                velocity=[0.0, 0.0],
                dofs_idx_local=[6, 7]
            )

    def navigation_success(self):

        robot_pos = self.robot.get_pos()

        distance = np.linalg.norm(
            robot_pos[:2] - self.goal_position[:2]
        )

        return distance < 0.3

    def pick_place_success(self):

        object_pos = self.goal_object.get_pos()

        distance = np.linalg.norm(
            object_pos - self.goal_position
        )

        return distance < 0.05

    def compute_reward(self):
        if self.task == "navigation":
            if self.navigation_success():
                return 10.0

        elif self.task == "pick_place":
            if self.pick_place_success():
                return 10.0

        return -0.01

    def is_done(self):
        if self.task == "navigation":
            if self.navigation_success():
                return True

        elif self.task == "pick_place":
            if self.pick_place_success():
                return True

        return self.step_count >= self.max_steps

    def metrics(self):

        return {
            "task": self.task,
            "steps": self.step_count,
            "success": self.compute_reward() == 10.0,
            "robot_position": self.robot.get_pos().tolist()
        }

    def close(self):
        pass


