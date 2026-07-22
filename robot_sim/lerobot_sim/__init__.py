"""
LeRobot Simulation Environment.

A full simulated robotics environment compatible with the HuggingFace LeRobot
ecosystem. Provides virtual robot arms, Gymnasium-compatible environments,
dataset integration with the HF Hub, policy training pipelines, keyboard/gamepad
teleoperation, and real-time visualization — all without physical hardware.

Modules:
    envs: Gymnasium-compatible simulation environments (PushT, PickPlace, Reach).
    policies: Policy wrappers for ACT, Diffusion, and simple baselines.
    datasets: Dataset loading, recording, and streaming via HF Hub.
    robots: Virtual robot arm kinematics and control.
    teleop: Keyboard and gamepad teleoperation for demonstration collection.
    visualization: Real-time rendering, episode replay, and data inspection.
    utils: Shared constants, type aliases, and helper utilities.
"""

__version__ = "0.1.0"
