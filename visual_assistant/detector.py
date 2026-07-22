"""
detector.py
Wraps YOLO object detection for use on real webcam frames.
"""

from ultralytics import YOLO


class Detector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Runs YOLO on a single BGR frame (as returned by OpenCV)
        and returns a list of detected objects as dicts:
        [{"label": "person", "conf": 0.91, "bbox": [x1, y1, x2, y2]}, ...]
        """
        results = self.model(frame, verbose=False)[0]
        objects = []

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            label = results.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            objects.append({
                "label": label,
                "conf": round(conf, 2),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            })

        return objects