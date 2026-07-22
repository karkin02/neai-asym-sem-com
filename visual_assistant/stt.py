"""
stt.py
Speech-to-text input using OpenAI's Whisper API.
Records a short audio clip from the microphone and transcribes it.
"""

import tempfile
import os
from openai import OpenAI

try:
    import sounddevice as sd
    import soundfile as sf
    HAS_AUDIO_LIBS = True
except ImportError:
    HAS_AUDIO_LIBS = False


class SpeechToText:
    def __init__(self, client: OpenAI, duration=4, samplerate=16000):
        self.client = client
        self.duration = duration
        self.samplerate = samplerate

    def listen(self) -> str:
        """
        Records audio from the default microphone for self.duration seconds,
        then transcribes it using Whisper. Returns the transcribed text.
        """
        if not HAS_AUDIO_LIBS:
            raise RuntimeError(
                "sounddevice and soundfile are required for voice input. "
                "Install with: pip install sounddevice soundfile"
            )

        print(f"Listening for {self.duration} seconds...")
        audio = sd.rec(int(self.duration * self.samplerate),
                        samplerate=self.samplerate, channels=1)
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, self.samplerate)
            temp_path = f.name

        with open(temp_path, "rb") as audio_file:
            transcript = self.client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        os.unlink(temp_path)
        return transcript.text