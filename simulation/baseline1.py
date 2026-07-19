from openai import OpenAI
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
import os

load_dotenv()

class GPTVisionPlanner:

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"), 
            timeout=60.0, 
            max_retries=3
        )

        self.actions = [
            "MoveAhead",
            "TurnLeft",
            "TurnRight",
            "PickUp",
            "PlaceOn",
            "Stop"
        ]

    def encode_image(self, image):
        img = Image.fromarray(image)
        buffer = BytesIO()

        img.save(
            buffer,
            format="PNG"
        )

        png_bytes = buffer.getvalue()
        self.last_image_size = len(png_bytes)

        encoded = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")

        return encoded


    def plan(self, image):
        encoded_image = self.encode_image(image)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=30,
            messages=[
                {
                    "role": "system",
                    "content":
                    """
                    You are a robot navigation planner.

                    You receive an RGB image from a robot camera.

                    Select exactly one action:

                    MoveAhead
                    TurnLeft
                    TurnRight
                    PickUp
                    PlaceOn
                    Stop

                    Return only the action name.
                    """
                },
                {
                    "role": "user",
                    "content":[
                        {
                            "type":"text",
                            "text":
                            "What action should the robot take?"
                        },
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":
                                f"data:image/png;base64,{encoded_image}"
                            }
                        }
                    ]
                }
            ],
            temperature=0
        )

        action = response.choices[0].message.content.strip()

        if action not in self.actions:
            action = "Stop"

        return action, {"tokens": response.usage.total_tokens,
                        "bytes": self.last_image_size}

