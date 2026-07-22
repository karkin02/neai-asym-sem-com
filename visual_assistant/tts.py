"""
tts.py
Text-to-speech output using OpenAI's TTS API.
Converts the assistant's text answer into spoken audio and plays it.
"""

import os
import tempfile
from openai import OpenAI

try:
    import simpleaudio as sa
    HAS_SIMPLEAUDIO = True
except ImportError:
    HAS_SIMPLEAUDIO = False


class TextToSpeech:
    def __init__(self, client: OpenAI, voice="alloy"):
        self.client = client
        self.voice = voice

    def speak(self, text: str):
        """
        Generates speech audio from text and plays it back.
        Falls back to printing if audio playback is unavailable.
        """
        response = self.client.audio.speech.create(
            model="tts-1",
            voice=self.voice,
            input=text
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(response.content)
            temp_path = f.name

        if HAS_SIMPLEAUDIO:
            try:
                import subprocess
                subprocess.run(["ffplay", "-nodisp", "-autoexit", temp_path],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                print(f"[TTS audio saved to {temp_path}, playback failed]")
        else:
            print(f"[TTS audio saved to {temp_path} - install ffmpeg/ffplay to auto-play]")

        os.unlink(temp_path) if os.path.exists(temp_path) else None