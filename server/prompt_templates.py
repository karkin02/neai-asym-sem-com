"""
Convert robot's structured JSON detections into natural language prompt
"""

def build_scene_prompt(payload: dict, task_goal: str) -> str:
    objects = payload.get("objects", [])

    # handle edge case where YOLO detected nothing
    if not objects:
        scene_desc = "No objects detected in the current view."
    else:
        # build phrases per detected object (bullet points)
        # e.g. "- chair (confidence 0.92m 1.2m away)"
        lines = []
        for obj in objects:
            dist_str = f",{obj['dist_m']}m away" if obj.get("dist_m") is not None else ""
            lines.append(
                f"- {obj['label']} (confidence {obj['conf']}{dist_str})"
            )
        scene_desc = "\n".join(lines)

    prompt = f"""
    You are controlling a robot in a simulated environment.

    Task goal: {task_goal}

    Current scene (detected objects):
    {scene_desc}

    Based on this scene, respond with a single concise action command
    (e.g., "move forward", "turn left", "pick up cup", "move toward chair").
    Only output the action, nothing else.
    """

    return prompt