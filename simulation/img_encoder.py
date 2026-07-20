import base64
from io import BytesIO
from PIL import Image


class RawEncoder:

    def encode(self, image):

        img = Image.fromarray(image)

        buffer = BytesIO()
        img.save(buffer, format="PNG")

        data = buffer.getvalue()

        return {
            "image": base64.b64encode(data).decode("utf-8"),
            "bytes": len(data)
        }


class JPEGEncoder:

    def __init__(self, quality=30):
        self.quality = quality

    def encode(self, image):

        img = Image.fromarray(image)

        buffer = BytesIO()
        img.save(
            buffer,
            format="JPEG",
            quality=self.quality
        )

        data = buffer.getvalue()

        return {
            "image": base64.b64encode(data).decode("utf-8"),
            "bytes": len(data)
        }