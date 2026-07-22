#!/usr/bin/env python3
"""
Flask web UI for the asymmetric semantic-communication pipeline.

Wires the whole story into one browser dashboard:

    webcam  ->  YOLO detection  ->  compact JSON payload  ->  GPT-4o-mini
            ->  parsed action    ->  robot sim step        ->  live render

The design mirrors ``robot_sim/run_sim.py``'s ``llm_vision`` mode but exposes
it through a browser instead of OpenCV windows.  Three background threads keep
both video feeds smooth even while the (slow) LLM call is in flight:

    * capture thread  -- reads the webcam, runs YOLO, annotates the frame
    * llm thread      -- periodically plans an action from the latest payload
    * sim thread      -- steps the environment at a fixed rate and renders it

Run:  python web_ui/app.py   then open http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template, request

# ----------------------------------------------------------------------
# Make the sibling packages importable (mirrors run_sim.py's sys.path hack)
# ----------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
_ROBOT_SIM_DIR = os.path.join(_ROOT, "robot_sim")
_VISUAL_ASSISTANT_DIR = os.path.join(_ROOT, "visual_assistant")
for _p in (_ROBOT_SIM_DIR, _VISUAL_ASSISTANT_DIR):
    if _p not in sys.path:
        sys.path.append(_p)

from lerobot_sim.envs.configs import (  # noqa: E402
    PickPlaceSimConfig,
    PushTSimConfig,
    ReachSimConfig,
)
from lerobot_sim.envs.factory import _env_class_for_config  # noqa: E402
from lerobot_sim.policies.policy import PushTExpertPolicy  # noqa: E402

from detector import Detector  # noqa: E402  (from visual_assistant/)
from prompt_templates import build_action_prompt  # noqa: E402
from action_utils import parse_action_response, object_world_pos_for_task  # noqa: E402

load_dotenv(os.path.join(_ROOT, ".env"))

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
_TASK_CONFIGS = {
    "pusht": PushTSimConfig,
    "pick_place": PickPlaceSimConfig,
    "reach": ReachSimConfig,
}
_CAPTURE_FPS = 20          # webcam + detection loop target
_SIM_FPS = 20              # environment stepping target
_LLM_PERIOD_S = 2.0        # seconds between GPT-4o-mini plans
_YOLO_MODEL = os.path.join(_VISUAL_ASSISTANT_DIR, "yolov8n.pt")


def _placeholder(text: str, width: int = 640, height: int = 480) -> np.ndarray:
    """Return a dark BGR frame with centred text (used before streams start)."""
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.putText(frame, text, (20, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (200, 200, 200), 2, cv2.LINE_AA)
    return frame


class PipelineController:
    """Owns the webcam, detector, LLM client and sim, and the shared state.

    All mutable state read by the Flask routes is guarded by ``self._lock``.
    Frames are stored as BGR uint8 arrays ready for ``cv2.imencode``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._running = False
        self._threads: List[threading.Thread] = []

        # Configurable at runtime
        self._task = "pusht"
        self._instruction = "move toward the cup"

        # Heavy resources (created lazily on first start)
        self._detector: Optional[Detector] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._env: Any = None
        self._env_action_dim = 2
        self._is_arm_task = False
        self._control_mode = "idle"
        self._openai_client: Any = None
        self._llm_available = False

        # Shared outputs (read by routes)
        self._webcam_frame = _placeholder("Press Start")
        self._sim_frame = _placeholder("Sim idle", 512, 512)
        self._detections: List[Dict[str, Any]] = []
        self._last_action = np.zeros(7, dtype=np.float32)      # from the LLM/vision
        self._applied_action = np.zeros(7, dtype=np.float32)   # what actually drove the sim
        self._llm_raw: str = ""
        self._llm_parsed: Dict[str, Any] = {}
        self._llm_status = "idle"
        self._reward = 0.0
        self._success = False
        self._step = 0
        self._frame_wh = (640, 480)
        self._last_obs: Dict[str, np.ndarray] = {}
        self._pusht_expert: Any = None

        # Vision-derived goal: the pixel-space centre of the selected object.
        # The real world defines *where* the robot should go.
        self._det_target_px: Optional[tuple] = None   # smoothed pixel centre of tracked object
        self._llm_object_label: Optional[str] = None  # object the LLM chose (if enabled)
        self._tracked_label: str = ""                 # object currently being followed
        self._target_label: str = ""                  # shown in the UI
        self._object_world_pos = None                 # live twin position (for the UI)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, task: str, instruction: str) -> Dict[str, Any]:
        """Start (or restart) the pipeline for a given task and instruction."""
        with self._lock:
            if self._running:
                self._instruction = instruction or self._instruction
                return {"ok": True, "msg": "already running; instruction updated"}
            self._task = task if task in _TASK_CONFIGS else "pusht"
            self._instruction = instruction or self._instruction

        try:
            self._init_resources()
        except Exception as exc:  # surface init failures to the UI
            return {"ok": False, "msg": f"init failed: {exc}"}

        with self._lock:
            self._running = True
        self._threads = [
            threading.Thread(target=self._capture_loop, daemon=True),
            threading.Thread(target=self._sim_loop, daemon=True),
            threading.Thread(target=self._llm_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()
        return {"ok": True, "msg": f"started task={self._task}"}

    def stop(self) -> Dict[str, Any]:
        """Stop all loops and release the webcam."""
        with self._lock:
            self._running = False
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._webcam_frame = _placeholder("Stopped")
        return {"ok": True, "msg": "stopped"}

    def reset_episode(self) -> Dict[str, Any]:
        """Reset the sim environment without stopping the pipeline, re-spawning
        the object from the latest camera detection."""
        if self._env is not None:
            # Use the capture thread's latest detection (avoid a second reader
            # racing the webcam) rather than grabbing a fresh frame here.
            self._reset_with_vision(live_capture=False)
            with self._lock:
                self._step = 0
                self._reward = 0.0
                self._success = False
        return {"ok": True, "msg": "episode reset"}

    def set_instruction(self, instruction: str) -> Dict[str, Any]:
        with self._lock:
            self._instruction = instruction
        return {"ok": True, "msg": "instruction updated"}

    # ------------------------------------------------------------------
    # Resource init
    # ------------------------------------------------------------------
    def _init_resources(self) -> None:
        """Instantiate detector, webcam, env and (optionally) the LLM client."""
        if self._detector is None:
            self._detector = Detector(model_path=_YOLO_MODEL, conf_threshold=0.5)

        # Environment for the selected task (reset happens after the webcam is
        # open, so the object can be reconstructed from the camera at reset).
        cfg = _TASK_CONFIGS[self._task]()
        self._env = _env_class_for_config(cfg)(cfg)
        self._env_action_dim = getattr(cfg, "action_dim", 2)
        self._is_arm_task = self._task in ("reach", "pick_place")
        self._pusht_expert = PushTExpertPolicy() if self._task == "pusht" else None
        self._det_target_px = None
        self._llm_object_label = None
        self._tracked_label = ""
        self._control_mode = {
            "pusht": "digital twin · push",
            "reach": "digital twin · reach",
            "pick_place": "digital twin · pick & place",
        }.get(self._task, self._task)
        with self._lock:
            self._sim_frame = _placeholder("Sim starting",
                                           cfg.observation_width, cfg.observation_height)

        # Webcam (Windows: prefer MSMF, fall back to DSHOW; then any backend)
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot access webcam (device 0).")
        self._cap = cap
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self._frame_wh = (w, h)

        # Reconstruct the object's position from the real camera and spawn it
        # there at reset (grab a live frame — the capture thread isn't up yet).
        self._reset_with_vision(live_capture=True)

        # OpenAI client is optional: without a key the pipeline still runs,
        # it just won't plan actions from the LLM.
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI

                self._openai_client = OpenAI(api_key=api_key)
                self._llm_available = True
            except Exception:
                self._llm_available = False
        else:
            self._llm_available = False

    # ------------------------------------------------------------------
    # Thread: webcam capture + YOLO detection
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        period = 1.0 / _CAPTURE_FPS
        while self._is_running():
            t0 = time.time()
            ret, frame = self._cap.read() if self._cap else (False, None)
            if not ret or frame is None:
                time.sleep(period)
                continue

            detections = self._detector.detect(frame)
            annotated = self._annotate(frame, detections)
            selected = self._select_target(detections)

            with self._lock:
                self._webcam_frame = annotated
                self._detections = detections
                if selected is not None:
                    label, center = selected
                    self._tracked_label = label
                    self._target_label = label
                    prev = self._det_target_px
                    if prev is None:
                        self._det_target_px = center
                    else:  # exponential smoothing kills per-frame YOLO jitter
                        self._det_target_px = (
                            prev[0] * 0.8 + center[0] * 0.2,
                            prev[1] * 0.8 + center[1] * 0.2,
                        )

            self._sleep_remaining(t0, period)

    def _select_target(self, detections: List[Dict[str, Any]]):
        """Choose the object to follow, with hysteresis so the goal does not
        flip between objects each frame.

        Priority: the LLM's chosen object, then a label named in the
        instruction, then the object already being tracked (if still visible),
        then the most confident detection. Returns ``(label, (cx, cy))`` in
        pixels, or ``None`` if nothing was detected.
        """
        if not detections:
            return None
        with self._lock:
            instr = self._instruction.lower()
            llm_label = self._llm_object_label
            tracked = self._tracked_label

        def best_with(label: Optional[str]):
            if not label:
                return None
            cand = [d for d in detections if d["label"].lower() == label.lower()]
            return max(cand, key=lambda d: d.get("conf", 0.0)) if cand else None

        det = best_with(llm_label)
        if det is None:
            cand = [d for d in detections if d["label"].lower() in instr]
            det = max(cand, key=lambda d: d.get("conf", 0.0)) if cand else None
        if det is None:
            det = best_with(tracked)
        if det is None:
            det = max(detections, key=lambda d: d.get("conf", 0.0))
        x1, y1, x2, y2 = det["bbox"]
        return det["label"], ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _annotate(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """Draw bounding boxes + labels on a copy of the BGR frame."""
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det["bbox"])
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
            tag = f"{det['label']} {det['conf']:.2f}"
            cv2.putText(out, tag, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 220, 0), 1, cv2.LINE_AA)
        return out

    # ------------------------------------------------------------------
    # Thread: periodic LLM planning
    # ------------------------------------------------------------------
    def _llm_loop(self) -> None:
        while self._is_running():
            if not self._llm_available:
                with self._lock:
                    self._llm_status = "no OPENAI_API_KEY - LLM disabled"
                time.sleep(_LLM_PERIOD_S)
                continue

            with self._lock:
                objects = list(self._detections)
                instruction = self._instruction

            try:
                prompt = build_action_prompt(objects, instruction)
                resp = self._openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = resp.choices[0].message.content or ""
                parsed = parse_action_response(raw)

                # The LLM's chosen target pixel refines which object the robot
                # pursues; the sim goal is derived from it (see _sync_target).
                obj = parsed.get("target_object")
                with self._lock:
                    self._llm_raw = raw
                    self._llm_parsed = parsed
                    self._llm_object_label = obj if isinstance(obj, str) else None
                    self._llm_status = "ok"
            except Exception as exc:
                with self._lock:
                    self._llm_status = f"error: {exc}"
            time.sleep(_LLM_PERIOD_S)

    # ------------------------------------------------------------------
    # Closed-loop arm control (reach / pick_place)
    # ------------------------------------------------------------------
    @staticmethod
    def _ik_delta(arm: Any, goal: np.ndarray, max_step: float = 0.06) -> np.ndarray:
        """Per-step joint delta that reduces ``||ee - goal||``.

        Builds a finite-difference Jacobian ``J = d(ee)/d(q)`` and steps along
        the descent direction ``J^T (goal - ee)`` (Jacobian-transpose IK). The
        step shrinks as the arm nears the goal so it decelerates instead of
        overshooting. The arm's joints are restored, so the env is not mutated.
        """
        q = arm.joint_positions.copy()
        base_ee = arm.ee_position
        error = np.asarray(goal, dtype=np.float64) - base_ee
        eps = 1e-4
        jac = np.zeros((3, arm.num_joints))
        for i in range(arm.num_joints):
            dq = q.copy()
            dq[i] += eps
            arm.joint_positions = dq
            jac[:, i] = (arm._forward_kinematics() - base_ee) / eps
        arm.joint_positions = q  # restore — no net mutation
        grad = jac.T @ error
        norm = float(np.linalg.norm(grad))
        if norm < 1e-8:
            return np.zeros(arm.num_joints)
        step = min(max_step, float(np.linalg.norm(error)))
        return (grad / norm) * step

    def _arm_control_action(self) -> np.ndarray:
        """Closed-loop joint command driving the arm through its task.

        * ``reach``      -> drive the end-effector onto the target (x-y plane).
        * ``pick_place`` -> approach the cube (gripper open), close to grasp it,
          then carry it onto the target (gripper closed).

        Goals are aimed at the current end-effector height (``ee[2]``) so the
        controller only has to solve the reachable x-y reach, not the arm's
        unreachable z. The env reads ``action[6] > 0`` as *open*, so closing
        uses a negative gripper command.
        """
        env = self._env
        arm = env._arm
        ee = arm.ee_position
        action = np.zeros(self._env_action_dim, dtype=np.float32)

        def _xy_goal(pos: np.ndarray) -> np.ndarray:
            return np.array([pos[0], pos[1], ee[2]], dtype=np.float64)

        if self._task == "reach":
            goal = _xy_goal(env._target_pos)
        else:  # pick_place
            cube = env._cube_pos
            if getattr(env, "_cube_grasped", False):
                goal, gripper = _xy_goal(env._target_pos), -1.0   # carry to target
            elif float(np.linalg.norm((ee - cube)[:2])) < 0.05:
                goal, gripper = _xy_goal(cube), -1.0              # close to grasp
            else:
                goal, gripper = _xy_goal(cube), 1.0               # open, approach
            if self._env_action_dim > 6:
                action[6] = gripper

        delta = self._ik_delta(arm, goal)
        n = min(arm.num_joints, self._env_action_dim)
        action[:n] = delta[:n]
        return action

    # ------------------------------------------------------------------
    # Vision -> sim scene (reconstruct the object at reset)
    # ------------------------------------------------------------------
    def _reconstruct_object_world_pos(self, live_capture: bool):
        """Map the selected detected object to a world position for this task.

        When ``live_capture`` is True (start-up, before the capture thread
        runs) a frame is grabbed directly; otherwise the capture thread's
        latest smoothed centre is used (avoids two readers racing the webcam).
        Returns the world position, or ``None`` if nothing was detected.
        """
        center = None
        if live_capture and self._cap is not None:
            frame = None
            for _ in range(5):  # warm up so exposure/autofocus settle
                ret, f = self._cap.read()
                if ret and f is not None:
                    frame = f
            if frame is not None:
                selected = self._select_target(self._detector.detect(frame))
                if selected is not None:
                    label, center = selected
                    with self._lock:
                        self._det_target_px = center
                        self._tracked_label = label
                        self._target_label = label
        if center is None:
            with self._lock:
                center = self._det_target_px
        if center is None:
            return None
        fw, fh = self._frame_wh
        # bbox_to_world uses the box centre; pass a degenerate box at the centre.
        return object_world_pos_for_task(
            self._task, [center[0], center[1], center[0], center[1]], fw, fh
        )

    def _reset_with_vision(self, live_capture: bool) -> None:
        """Reset the env, spawning the object at the camera-reconstructed pose
        (falls back to the env's random placement if nothing was detected)."""
        pos = self._reconstruct_object_world_pos(live_capture)
        if self._task == "pusht":
            # The twin mirrors the push *target* (pinning the block would fight
            # the push), so spawn the block normally and seed the target.
            self._last_obs, _ = self._env.reset()
            if pos is not None:
                self._env._target_pos = np.clip(
                    np.asarray(pos, dtype=np.float32)[:2], 0.0, 1.0
                )
        else:
            options = {"object_world_pos": pos} if pos is not None else None
            self._last_obs, _ = self._env.reset(options=options)

    def _twin_sync(self) -> None:
        """Continuously mirror the tracked object into the sim (digital twin).

        Maps the smoothed detection centre to a world position and writes it to
        the task's live entity, with a deadband so a still object stays put:
          reach      -> the target (reach onto it)
          pick_place -> the cube, but frozen once grasped (physics carries it)
          pusht      -> the push target (block is pushed onto it)
        """
        with self._lock:
            px = self._det_target_px
        if px is None:
            return
        fw, fh = self._frame_wh
        pos = object_world_pos_for_task(self._task, [px[0], px[1], px[0], px[1]], fw, fh)
        env = self._env
        if self._task == "reach":
            new = np.asarray(pos, dtype=np.float64)
            if float(np.linalg.norm((new - env._target_pos)[:2])) > 0.01:
                env._target_pos = new
        elif self._task == "pick_place":
            if not getattr(env, "_cube_grasped", False):
                new = np.array([pos[0], pos[1], 0.02], dtype=np.float64)
                if float(np.linalg.norm((new - env._cube_pos)[:2])) > 0.01:
                    env._cube_pos = new
        else:  # pusht -> mirror the push target
            new = np.clip(np.asarray(pos, dtype=np.float32)[:2], 0.0, 1.0)
            if float(np.linalg.norm(new - env._target_pos)) > 0.02:
                env._target_pos = new
        with self._lock:
            self._object_world_pos = [round(float(c), 3) for c in np.asarray(pos).ravel()[:3]]

    def _compute_action(self) -> np.ndarray:
        """Action driving the robot toward the (vision-defined) target."""
        if self._task == "pusht":
            obs = self._last_obs or {}
            if self._pusht_expert is not None and "agent_pos" in obs:
                return np.asarray(self._pusht_expert.select_action(obs), dtype=np.float32)
            return np.zeros(self._env_action_dim, dtype=np.float32)
        return self._arm_control_action()

    # ------------------------------------------------------------------
    # Thread: environment stepping + render
    # ------------------------------------------------------------------
    def _sim_loop(self) -> None:
        period = 1.0 / _SIM_FPS
        while self._is_running():
            t0 = time.time()
            # Digital twin: continuously mirror the real object into the sim,
            # then the controller drives the robot toward the live entity.
            self._twin_sync()
            action = self._compute_action()
            try:
                obs, reward, terminated, truncated, info = self._env.step(action)
                self._last_obs = obs
                frame_rgb = obs.get("pixels")
                if frame_rgb is None:
                    frame_rgb = self._env.render()
                sim_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                with self._lock:
                    self._sim_frame = sim_bgr
                    self._applied_action = np.asarray(action, dtype=np.float32)
                    self._reward = float(reward)
                    self._success = bool(info.get("is_success", False))
                    self._step += 1
                # No auto-reset: the scene persists so the robot settles on the
                # target and stays (following it if the real object moves). Use
                # the "Reset episode" button to re-randomise the scene.
            except Exception:
                pass
            self._sleep_remaining(t0, period)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    @staticmethod
    def _sleep_remaining(t0: float, period: float) -> None:
        elapsed = time.time() - t0
        if elapsed < period:
            time.sleep(period - elapsed)

    def webcam_jpeg(self) -> bytes:
        with self._lock:
            frame = self._webcam_frame
        return _encode_jpeg(frame)

    def sim_jpeg(self) -> bytes:
        with self._lock:
            frame = self._sim_frame
        return _encode_jpeg(frame)

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the current pipeline state."""
        with self._lock:
            payload = self._detections
            payload_json = json.dumps(payload)
            fw, fh = self._frame_wh
            raw_bytes = fw * fh * 3
            payload_bytes = len(payload_json.encode("utf-8"))
            return {
                "running": self._running,
                "task": self._task,
                "instruction": self._instruction,
                "detections": payload,
                "payload_bytes": payload_bytes,
                "raw_frame_bytes": raw_bytes,
                "compression_ratio": round(raw_bytes / payload_bytes, 1) if payload_bytes else None,
                "llm_status": self._llm_status,
                "llm_available": self._llm_available,
                "llm_raw": self._llm_raw,
                "llm_parsed": self._llm_parsed,
                "control_mode": self._control_mode,
                "target_label": self._target_label,
                "object_world_pos": self._object_world_pos,
                "action": [round(float(a), 4) for a in self._applied_action[: self._env_action_dim]],
                "reward": round(self._reward, 4),
                "success": self._success,
                "step": self._step,
            }


def _encode_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else b""


# ----------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=os.path.join(_HERE, "templates"),
    static_folder=os.path.join(_HERE, "static"),
)
controller = PipelineController()


def _mjpeg(get_frame) -> Response:
    """Return an MJPEG streaming response that repeatedly calls ``get_frame``."""
    def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        while True:
            jpeg = get_frame()
            if jpeg:
                yield boundary + jpeg + b"\r\n"
            time.sleep(1.0 / 25.0)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index() -> str:
    return render_template("index.html", tasks=list(_TASK_CONFIGS))


@app.route("/webcam_feed")
def webcam_feed() -> Response:
    return _mjpeg(controller.webcam_jpeg)


@app.route("/sim_feed")
def sim_feed() -> Response:
    return _mjpeg(controller.sim_jpeg)


@app.route("/state")
def state() -> Response:
    return jsonify(controller.snapshot())


@app.route("/control", methods=["POST"])
def control() -> Response:
    data = request.get_json(force=True) or {}
    action = data.get("action")
    if action == "start":
        result = controller.start(data.get("task", "pusht"), data.get("instruction", ""))
    elif action == "stop":
        result = controller.stop()
    elif action == "reset":
        result = controller.reset_episode()
    elif action == "instruction":
        result = controller.set_instruction(data.get("instruction", ""))
    else:
        result = {"ok": False, "msg": f"unknown action '{action}'"}
    return jsonify(result)


if __name__ == "__main__":
    # threaded=True so MJPEG streams + control routes are served concurrently.
    app.run(host="127.0.0.1", port=5000, threaded=True, debug=False)
