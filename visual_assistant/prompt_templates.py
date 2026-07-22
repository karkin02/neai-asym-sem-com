"""
prompt_templates.py
Builds the natural language prompt sent to GPT-4o-mini,
combining detected scene objects with the user's question.
"""


def build_qa_prompt(objects: list, question: str) -> str:
    """ Question answering """
    if not objects:
        scene_desc = "No objects detected in the current view."
    else:
        lines = [f"- {o['label']} (confidence {o['conf']})" for o in objects]
        scene_desc = "\n".join(lines)

    prompt = f"""You are a visual assistant describing a real camera scene to a user.

    Detected objects in view:
    {scene_desc}

    User question: {question}

    Rules:
    - Answer clearly and concisely based only on what is visible.
    - If you cannot determine the answer from the detected objects, say so honestly.
    - Do not make up objects that were not detected.
    - Keep your answer to 1-3 sentences unless the user asks for more detail.
    """
    return prompt


def build_action_prompt(objects: list, instruction: str) -> str:
    """Builds a prompt asking GPT-4o-mini to return a structured action target."""
    if not objects:
        scene_desc = "No objects detected in the current view."
    else:
        lines = [f"- {o['label']} (confidence {o['conf']}, bbox {o['bbox']})" for o in objects]
        scene_desc = "\n".join(lines)

    prompt = f"""You are a robot vision-action planner. You receive detected objects
    from a real camera and a natural language instruction. Respond with a
    structured action target as STRICT JSON only, no extra text.

    Detected objects in view (bbox format is [x1, y1, x2, y2] in pixel coordinates):
    {scene_desc}

    Instruction: {instruction}

    Rules:
    - Pick the single object that best matches the instruction.
    - Compute target_position as the [x_center, y_center] of that object's bbox.
    - If no matching object is detected, set "target_object" to null and
    "target_position" to null.
    - Output ONLY valid JSON in this exact schema, nothing else:
    {{"target_object": "<label or null>", "target_position": [x_center, y_center] or null, "gripper": "open" or "close"}}
    """
    return prompt