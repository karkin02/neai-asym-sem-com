import json
import re

import numpy as np


def parse_action_response(response_text: str) -> dict:
    """Extracts and validates JSON from GPT-4o-mini's raw text response."""
    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response: {response_text}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        raise ValueError(f"Malformed JSON from LLM: {e}\nRaw: {response_text}")

    if "target_object" not in data or "target_position" not in data:
        raise ValueError(f"Missing required keys in LLM response: {data}")
    return data


def bbox_center_to_action(target_position, frame_width=640, frame_height=480,
                           action_dim=7, gripper_cmd="open") -> list:
    """Converts a pixel-space [x, y] target into the sim's action vector.

    Normalizes pixel coordinates to [-1, 1] range and places them in
    action[0]/action[1], matching the convention seen in recorded
    vision-teleop episodes. action[6] is used as the gripper trigger.
    """
    action = [0.0] * action_dim
    if target_position is None:
        return action

    x_center, y_center = target_position
    norm_x = (x_center / frame_width) * 2 - 1
    norm_y = (y_center / frame_height) * 2 - 1

    action[0] = round(float(norm_x), 4)
    action[1] = round(float(norm_y), 4)
    action[6] = 1.0 if gripper_cmd == "close" else 0.0
    return action


def bbox_to_world(bbox, frame_width, frame_height,
                  x_range=(-0.3, 0.3), y_range=(-0.3, 0.3), z=0.3, flip_y=True):
    """Heuristic map from a pixel bounding box to a 3D world position.

    This is a deliberately simple mapping (no camera calibration or depth
    estimation): it takes the bbox centre, normalises it to [0, 1] over the
    frame, then linearly maps into the simulator's world ranges. ``flip_y``
    matches each env's renderer (the arm envs flip y so "up" in the image is
    +y; pusht does not). ``z`` is a fixed nominal height — there is no depth
    reconstruction — and these tasks are x-y planar so z is cosmetic.

    Args:
        bbox: ``[x1, y1, x2, y2]`` pixel coordinates.
        frame_width: Frame width in pixels.
        frame_height: Frame height in pixels.
        x_range: ``(min, max)`` world x extent.
        y_range: ``(min, max)`` world y extent.
        z: Fixed world z (height).
        flip_y: If True, image-top maps to y_range max.

    Returns:
        ``np.array([x, y, z])`` in world coordinates.
    """
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    u = min(max(cx / max(frame_width, 1), 0.0), 1.0)
    v = min(max(cy / max(frame_height, 1), 0.0), 1.0)
    fy = (1.0 - v) if flip_y else v
    x = x_range[0] + u * (x_range[1] - x_range[0])
    y = y_range[0] + fy * (y_range[1] - y_range[0])
    return np.array([x, y, z], dtype=np.float64)


# Per-task world mapping: (x_range, y_range, nominal z, flip_y).
# z is None for the 2-D pusht task. Ranges mirror each env's workspace /
# renderer so a reconstructed object appears where it sits in the camera.
TASK_WORLD_MAP = {
    "pusht":      ((0.1, 0.9), (0.1, 0.9), None, False),
    "reach":      ((-0.3, 0.3), (-0.3, 0.3), 0.30, True),
    "pick_place": ((-0.3, 0.3), (-0.3, 0.3), 0.02, True),
}


def object_world_pos_for_task(task, bbox, frame_width, frame_height):
    """Map a detected bbox into a task-appropriate world position.

    Returns a 2D ``[x, y]`` for pusht (a planar task) and a 3D ``[x, y, z]``
    for the arm tasks. Raises KeyError for an unknown task.
    """
    x_range, y_range, z, flip_y = TASK_WORLD_MAP[task]
    pos = bbox_to_world(
        bbox, frame_width, frame_height,
        x_range=x_range, y_range=y_range,
        z=(0.0 if z is None else z), flip_y=flip_y,
    )
    return pos[:2] if z is None else pos