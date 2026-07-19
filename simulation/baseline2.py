from openai import OpenAI
import os
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

class JPEGVisionPlanner:

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

    def encode_jpeg(self, image):
        # numpy array -> PIL
        img = Image.fromarray(image)
        buffer = BytesIO()

        # JPEG compression
        img.save(
            buffer,
            format="JPEG",
            quality=30
        )

        jpeg_bytes = buffer.getvalue()
        self.last_jpeg_size = len(jpeg_bytes)

        encoded = base64.b64encode(
            jpeg_bytes
        ).decode("utf-8")

        return encoded

    def plan(self, image):
        encoded_image = self.encode_jpeg(image)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=30,
            messages=[
                {
                    "role": "system",
                    "content":
                    """
                    You are a robot navigation planner.

                    Choose exactly one action:

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
                            "Select the next robot action."
                        },
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":
                                f"data:image/jpeg;base64,{encoded_image}"
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
                        "bytes": self.last_jpeg_size}

