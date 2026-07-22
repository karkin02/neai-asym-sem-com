# neai-asym-sem-com

**Asymmetric Semantic Communication** — a framework that compresses an AI
agent's visual observations into compact JSON payloads using YOLOv8, enabling
efficient cloud-based LLM planning under constrained network conditions.

Instead of shipping raw pixels to the cloud, a lightweight **edge** device
extracts *meaning* (detected objects + positions) and transmits only a small
JSON payload. A heavy **cloud** LLM reasons over that payload and returns a
plan or answer. The asymmetry is deliberate: a cheap, bandwidth-constrained
edge uplink paired with rich cloud compute on the downlink.

```
        EDGE (local, lightweight)                CLOUD (heavy)
  ┌────────────────────────────────┐      ┌────────────────────────┐
  │  webcam ──► YOLOv8 detection    │      │   GPT-4o-mini           │
  │            │                    │ JSON │   • scene Q&A           │
  │            └─► compact JSON  ───┼─────►│   • action planning     │
  │               payload (~bytes)  │ uplink                        │
  └────────────────────────────────┘      └───────────┬────────────┘
                                                       │ action / answer
                                       ┌───────────────▼────────────┐
                                       │  robot sim  /  speech out   │
                                       └─────────────────────────────┘
```

---

## Repository layout

| Path | What it is |
|------|-----------|
| [visual_assistant/](visual_assistant/) | The **edge + assistant** app: webcam → YOLOv8 → compact JSON → GPT-4o-mini scene Q&A, with optional voice I/O (Whisper STT + OpenAI TTS). |
| [robot_sim/](robot_sim/) | A hardware-free **robot-learning** package (`lerobot-sim`) modeled on HuggingFace LeRobot: Gymnasium sim environments, a NumPy behaviour-cloning policy, teleoperation, and vision/LLM-driven control. Has its own [README](robot_sim/README.md) and [TUTORIAL](robot_sim/TUTORIAL.md). |
| [web_ui/](web_ui/) | A **unified Flask dashboard** that runs the whole pipeline live in the browser (webcam → detection → JSON → LLM → robot sim). See [web_ui/README.md](web_ui/README.md). |
| [requirements.txt](requirements.txt) | Core dependencies for the `visual_assistant` side. |
| `.env` | Holds `OPENAI_API_KEY` (git-ignored — see [Security](#security)). |

---

## Components

### 1. Visual Assistant — [visual_assistant/](visual_assistant/)

A continuous webcam loop that answers natural-language questions about the live
scene. It is the clearest demonstration of the "semantic communication" idea:
only detected objects (not pixels) are described to the cloud LLM.

- [detector.py](visual_assistant/detector.py) — `Detector` wraps Ultralytics
  YOLOv8 (`yolov8n.pt`); `detect(frame)` returns `[{label, conf, bbox}, ...]`.
- [prompt_templates.py](visual_assistant/prompt_templates.py) — `build_qa_prompt`
  (scene Q&A) and `build_action_prompt` (structured robot-action JSON).
- [action_utils.py](visual_assistant/action_utils.py) — `parse_action_response`
  and `bbox_center_to_action` (pixel target → normalized action vector).
- [stt.py](visual_assistant/stt.py) / [tts.py](visual_assistant/tts.py) —
  OpenAI Whisper speech-to-text and OpenAI TTS.
- [main.py](visual_assistant/main.py) — the run loop.

### 2. Robot Simulation — [robot_sim/](robot_sim/)

A self-contained `lerobot-sim` package (Apache-2.0) using **NumPy + Gymnasium**
only — no PyTorch. It simulates robot-arm tasks, collects demonstrations,
trains a behaviour-cloning MLP, and evaluates/visualizes it.

- Tasks: `pusht` (2-D), `pick_place` (6-DOF), `reach` (6-DOF).
- [run_sim.py](robot_sim/run_sim.py) — CLI entry point dispatching on `--mode`:
  `train`, `eval`, `teleop`, `visualize`, `vision`, `llm_vision`.
- The `vision` / `llm_vision` modes bridge into `visual_assistant/` — YOLO
  detections drive the sim directly (geometry) or via a GPT-4o-mini plan.
- A companion [lerobot-policy-deployment-app/](robot_sim/lerobot-policy-deployment-app/)
  deploys the trained policy onto an Arduino microcontroller.

See the [robot_sim README](robot_sim/README.md) and [TUTORIAL](robot_sim/TUTORIAL.md)
for full details.

### 3. Web UI — [web_ui/](web_ui/)

A Flask + HTML/JS dashboard that unifies everything into one browser view with
four panels: **edge camera**, **semantic payload** (with live compression ratio
vs. the raw frame), **cloud LLM plan**, and the **robot sim** executing the
action. See [web_ui/README.md](web_ui/README.md).

---

## Setup

Requires **Python 3.10+** and a webcam (device `0`) for the vision modes.

```bash
# 1. (recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 2. install dependencies
pip install -r requirements.txt   # visual_assistant + core
pip install -r web_ui/requirements.txt   # adds Flask + sim deps for the UI
pip install -e robot_sim          # the lerobot-sim package (for robot_sim/web_ui)
```

Create a `.env` in the repo root with your OpenAI key:

```
OPENAI_API_KEY="sk-..."
```

The `yolov8n.pt` YOLOv8-nano weights are already bundled in
[visual_assistant/](visual_assistant/).

---

## Quick start

### Visual assistant (scene Q&A)

```bash
cd visual_assistant
python main.py            # text in, text out
python main.py --voice    # voice in, spoken out
```
Type a question about what the camera sees; type `q` (or press `q` in the video
window) to quit.

### Robot simulation

```bash
cd robot_sim

# full collect → train → evaluate pipeline
python run_sim.py --task pusht --mode train

# watch a random policy with live rendering
python run_sim.py --task reach --mode visualize

# webcam + YOLO object tracking drives the arm
python run_sim.py --task pick_place --mode vision --target-label cup
# can be any target label

# natural-language instruction → GPT-4o-mini → arm
python run_sim.py --task pick_place --mode llm_vision --instruction "pick up the cup"
```

### Unified web UI

```bash
python web_ui/app.py
```
Open <http://127.0.0.1:5000> and click **Start**. Pick a task, type an
instruction, and watch the full edge→cloud→sim pipeline live.

---

## How it fits together

1. **Perceive (edge).** A webcam frame is captured; YOLOv8 runs locally and
   emits a small list of `{label, conf, bbox}` dicts — the *semantic payload*.
2. **Transmit.** Only that JSON goes "uplink" to the cloud (orders of magnitude
   smaller than the raw frame — the web UI shows the exact ratio).
3. **Reason (cloud).** GPT-4o-mini either answers a question about the scene
   (`build_qa_prompt`) or returns a structured action target
   (`build_action_prompt` → `parse_action_response`).
4. **Act.** For robotics, the action target is converted to the sim's action
   vector (`bbox_center_to_action`) and stepped through a Gymnasium env; for the
   assistant, the answer is printed and optionally spoken.

---

## Security

The root `.env` stores a live `OPENAI_API_KEY` in plaintext. It is git-ignored,
but treat any key that has been shared or committed as compromised and rotate it.
Never commit `.env`.

---

## License

`robot_sim/` is Apache-2.0 (see [robot_sim/LICENSE](robot_sim/LICENSE)). This is
an SUTD Term 3 NEAI course project.
