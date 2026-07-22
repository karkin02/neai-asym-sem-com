# Web UI — Asymmetric Semantic Communication

A single-page Flask dashboard that runs the whole pipeline live in the browser:

```
webcam → YOLOv8 (edge) → compact JSON payload → GPT-4o-mini (cloud)
       → parsed action → robot sim step → live render
```

The four panels map onto the "semantic communication" thesis:

1. **Edge** — webcam feed with YOLO bounding boxes.
2. **Semantic payload (uplink)** — the exact JSON sent to the cloud, plus the
   compression ratio vs. the raw frame (raw pixels stay local; only meaning is
   transmitted).
3. **Cloud** — the GPT-4o-mini action plan (parsed + raw).
4. **Robot sim** — the selected environment executing the planned action, with
   reward / success / step metrics.

## Setup

From the repo root:

```bash
pip install -r web_ui/requirements.txt
```

Make sure `OPENAI_API_KEY` is set in the root `.env` (already loaded by the app).
Without a key the UI still runs — the sim holds a zero action and the cloud panel
shows `LLM disabled`.

## Run

```bash
python web_ui/app.py
```

Then open <http://127.0.0.1:5000> and click **Start**.

- **Task** — choose `pusht`, `pick_place`, or `reach`.
- **Instruction** — natural-language goal handed to the LLM (e.g. "move toward
  the cup"). Edit and hit **Send instruction** to change it live.
- **Reset episode** restarts the sim without touching the camera.
- **Stop** releases the webcam.

## How the real world drives the sim

The detected object in the camera **defines the sim's target**, and the robot
acts on it according to the task (shown as the `control:` / `target =` caption
under the sim):

1. YOLO detects objects each frame; one is selected to follow. Selection is
   stable with hysteresis — it prefers the object the LLM chose, then the
   instruction's label, then whatever it's already tracking — so the goal does
   not flip between objects.
2. That object's screen position is smoothed (EMA + deadband) and mapped into
   the sim as the **target**, so the target holds still under detection jitter
   and only moves when the real object actually moves.
3. The robot pursues that target per its task:
   - **`reach`** — the end-effector moves onto the target and stays.
   - **`pick_place`** — the arm grasps the cube and carries it onto the target.
   - **`pusht`** — a scripted pusher expert pushes the block onto the target.

The scene does **not** auto-reset, so the robot settles on the target and holds;
move the real object and the target (and robot) follow. Use **Reset episode** to
re-randomise. The LLM still runs and populates the panels — it chooses *which*
object to track, while the detector supplies the precise, stable position.

### Planar (x-y) tasks

The bundled `SimRobotArm` uses a simplified forward-kinematics model whose
end-effector can't reach the object's nominal z-plane, and the renderer is 2-D
(it plots x, y only). So the arm envs are treated as **x-y planar** tasks:
grasp and success are measured in the x-y plane (`reach._compute_reward`,
`pick_place._update_grasp` / `_compute_reward` in `robot_sim/lerobot_sim/`).
The on-screen rule is intuitive — **when the dots overlap, the task succeeds.**
This makes `reach` and `pick_place` reliably solvable by the controller (and
by trained policies) instead of being physically impossible.

## Notes

- The backend uses three background threads (capture+YOLO, LLM planning, sim
  stepping) so the video feeds stay smooth while the ~1 s LLM call is in flight.
- The LLM is queried every ~2 s; the last action is reused in between.
- Webcam uses device `0` (MSMF → DSHOW → default backend fallback on Windows).
