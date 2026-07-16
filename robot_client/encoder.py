import json


class SemanticEncoder:
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold

    def encode(self, detection_result: dict, step: int) -> bytes:
        # drop any detections below confidence bar (to control payload size vs. information richness tradeoff)
        filtered = [
            d for d in detection_result["detections"]
            if d["confidence"] >= self.confidence_threshold
        ]

        # build minimal JSON structure for server side
        payload = {
            "step": step,
            "objects": [
                {
                    "label": d["label"],
                    "conf": d["confidence"],
                    "bbox": d["bbox"],
                    "dist_m": d["distance_m"],
                }
                for d in filtered
            ],
        }

        json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return json_bytes
    
    @staticmethod
    def payload_size_bytes(json_bytes: bytes) -> int:
        # bytes transmitted per step
        return len(json_bytes)