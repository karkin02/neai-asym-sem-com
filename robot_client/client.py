"""
Robot-side loop:
- capture frame -> detect -> encode -> throttle -> send to server -> get action
"""

import asyncio
import httpx
from robot_client.detector import ObjectDetector
from robot_client.encoder import SemanticEncoder
from network_sim.throttle import NetworkThrottle

class RobotClient:
    def __init__(self, server_url: str, bandwidth_kbps: float = None,
                 latency_ms: float = 50, conf_threshold: float = 0.5):
        self.server_url = server_url
        self.detector = ObjectDetector(conf_threshold=conf_threshold)
        self.encoder = SemanticEncoder(confidence_threshold=conf_threshold)
        self.throttle = NetworkThrottle(bandwidth_kbps=bandwidth_kbps, latency_ms=latency_ms)

    async def step(self, frame, depth_map, task_goal: str, step_num: int) -> dict:
        # run YOLO detection on the current camera frame
        detection_result = self.detector.detect(frame, depth_map)

        # compress detections into JSON payload
        payload_bytes = self.encoder.encode(detection_result, step_num)
        payload_size = self.encoder.payload_size_bytes(payload_bytes)

        # simulate sending payload over a constrained network link
        transmission_time_ms = await self.throttle.send(payload_bytes)

        # POST payload to cloud server, wait action back
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.server_url}/plan",
                content=payload_bytes,
                headers={
                    "X_Task_Goal": task_goal,
                    "Content-Type": "application/json",
                }
            )
            server_result = response.json()

        return {
            "step": step_num,
            "payload_bytes": payload_size,                              # bandwidth metric   
            "trasmission_time_ms": round(transmission_time_ms, 2),      # network delay
            "yolo_inference_ms": detection_result["inference_time_ms"], # robot compute cost
            "action": server_result["action"],                          # what the LLM decided to do
            "llm_latency_ms": server_result["latency_ms"],              # cloud compute + API
            "token_count": server_result["token_count"],
        }