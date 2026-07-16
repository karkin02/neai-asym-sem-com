from ultralytics import YOLO
import numpy as np
import time

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        # load pretrained YOLO model
        self.model = YOLO(model_path)
        # confidence score
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.darray, depth_map: np.darray = None) -> dict:
        """
        frame:      RGB image (H, W ,3), single camera frame from the sim
        depth_map:  optional depth image (H, W), same resolution as frame,
                    used to estimate how far away each object is
        """
        # time inference
        start = time.time()
        results = self.model(frame, conf = self.conf_threshold, verbose = False)[0]
        inference_time_ms = (time.time() - start) * 1000

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])                # class index (e.g. 0 = person)
            label = self.model.names[cls_id]        # human-readable label (e.g. "chair")
            conf = float(box.conf[0])               # model's confidence in this detection
            x1, y1, x2, y2 = box.xyxy[0].tolist()   # boulding box corners in pixels
            
            # look up distance at box's center pixel if depth map exist
            distance_m = None
            if depth_map is not None:
                cx, cy = int((x1 + x2 / 2)), int((y1 + y2) / 2)
                distance_m = round(float(depth_map[cy, cx]), 2)

            # store information for downstream
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                "distance_m": distance_m,
            })

        return {
            "detections": detections,
            "inference_time_ms": round(inference_time_ms, 2)
        }