TASK_DESCRIPTIONS = {
    "navigation":
        "Navigate to the target object and stop when the robot is close to it.",

    "pick_place":
        "Navigate to the target object, pick it up, carry it to the destination, place it there, and stop."
}
class VisionPlanner:

    def __init__(self, client, encoder, task):

        self.client = client
        self.encoder = encoder
        self.task = task

        self.actions = [
            "MoveAhead",
            "TurnLeft",
            "TurnRight",
            "PickUp",
            "PlaceOn",
            "Stop"
        ]


    def plan(self, image):

        description = TASK_DESCRIPTIONS[self.task]

        encoded = self.encoder.encode(image)

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=10,
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You are controlling a robot.
                    You receive an RGB image from a robot camera.

                    Task:
                    {description}

                    Available actions:
                    {", ".join(self.actions)}

                    Rules:
                    - Return exactly ONE action.
                    - Do not explain your reasoning.
                    - Do not output any text other than the action.
                    - Use PickUp only when the object is within reach.
                    - Use PlaceOn only when holding the object.
                    - Use Stop only after the task is complete.
                    """
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type":"text",
                            "text":
                            "Choose the next action."
                        },
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":
                                f"data:image/jpeg;base64,{encoded['image']}"
                            }
                        }
                    ]
                }
            ]
        )

        action = response.choices[0].message.content.strip()

        if action not in self.actions:
            action = "Stop"

        return action, {
            "bytes": encoded["bytes"],
            "tokens": response.usage.total_tokens
        }