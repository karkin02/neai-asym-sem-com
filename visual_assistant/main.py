"""
main.py
Visual Assistant - main loop.

Captures webcam frames continuously, detects objects with YOLO,
answers user questions about the scene using GPT-4o-mini, and
optionally speaks the answer aloud.

Run modes:
    python main.py            -> text input, text output
    python main.py --voice    -> voice input, spoken output
"""

import os
import sys
import threading
import queue
import cv2
from openai import OpenAI
from dotenv import load_dotenv

from detector import Detector
from prompt_templates import build_qa_prompt

load_dotenv()

USE_VOICE = "--voice" in sys.argv

question_queue = queue.Queue()
stop_flag = threading.Event()
answer_ready_queue = queue.Queue()
stop_flag = threading.Event()

tts = None
stt = None


def input_worker():
    while not stop_flag.is_set():
        if USE_VOICE:
            question = stt.listen()
            print(f"You said: {question}")
        else:
            question = input("Ask about the scene (or 'q' to quit): ")

        if question.strip().lower() == "q":
            stop_flag.set()
            question_queue.put("q")
            break
        
        question_queue.put(question)
        answer_ready_queue.get()    # block here until main thread signals done


def draw_detections(frame, objects):
    """Draws bounding boxes and confidence scores on the frame for debugging."""
    for obj in objects:
        x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
        label = f"{obj['label']} {obj['conf']:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame, label, (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
    return frame


def main():
    global tts, stt

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    detector = Detector(model_path="yolov8n.pt", conf_threshold=0.5)

    if USE_VOICE:
        from tts import TextToSpeech
        from stt import SpeechToText
        tts = TextToSpeech(client)
        stt = SpeechToText(client)

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot access webcam. Check camera index or permissions.")

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    for _ in range(5):
        cap.read()

    print("Visual Assistant ready.")
    print("Voice mode: ON" if USE_VOICE else "Voice mode: OFF (text input)")
    print("Camera feed is live. Press 'q' in the video window or type 'q' to quit.\n")

    input_thread = threading.Thread(target=input_worker, daemon=True)
    input_thread.start()

    latest_frame = None

    try:
        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, retrying...")
                continue

            latest_frame = frame
            objects = detector.detect(latest_frame)          # run every frame
            display_frame = draw_detections(frame.copy(), objects)

            cv2.imshow("Camera Feed", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_flag.set()
                break

            try:
                question = question_queue.get_nowait()
            except queue.Empty:
                continue

            if question.strip().lower() == "q":
                stop_flag.set()
                break

            prompt = build_qa_prompt(objects, question)

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                answer = response.choices[0].message.content
                print(f"Assistant: {answer}\n")

                if USE_VOICE:
                    tts.speak(answer)

            except Exception as e:
                print(f"Error getting response: {e}\n")
            finally:
                answer_ready_queue.put(True)

    finally:
        stop_flag.set()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()