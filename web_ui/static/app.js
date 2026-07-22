"use strict";

const $ = (id) => document.getElementById(id);

async function post(action, extra = {}) {
  const res = await fetch("/control", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...extra }),
  });
  return res.json();
}

function fmtBytes(n) {
  if (n == null) return "–";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

function setStatus(msg) {
  $("status").textContent = msg;
}

// Colors mirror robot_sim/lerobot_sim/utils/constants.py.
const C_ROBOT = "rgb(66,133,244)";
const C_TARGET = "rgb(219,68,55)";
const C_OBJECT = "rgb(244,180,0)";
const LEGENDS = {
  pusht: [[C_ROBOT, "Robot (pusher)"], [C_TARGET, "Target zone"], [C_OBJECT, "T-block"]],
  pick_place: [[C_ROBOT, "Arm (end-effector)"], [C_TARGET, "Target"], [C_OBJECT, "Cube"]],
  reach: [[C_ROBOT, "Arm (end-effector)"], [C_TARGET, "Target"]],
};
let _legendTask = null;

function renderLegend(task) {
  if (task === _legendTask) return; // only rebuild when the task changes
  _legendTask = task;
  const items = LEGENDS[task] || [];
  $("legend").innerHTML = items
    .map(
      ([color, label]) =>
        `<span class="legend-item"><span class="swatch" style="background:${color}"></span>${label}</span>`
    )
    .join("");
}

$("btn-start").addEventListener("click", async () => {
  setStatus("starting…");
  const r = await post("start", {
    task: $("task").value,
    instruction: $("instruction").value,
  });
  setStatus(r.msg || (r.ok ? "started" : "error"));
});

$("btn-stop").addEventListener("click", async () => {
  const r = await post("stop");
  setStatus(r.msg || "stopped");
});

$("btn-reset").addEventListener("click", async () => {
  const r = await post("reset");
  setStatus(r.msg || "reset");
});

$("btn-instruction").addEventListener("click", async () => {
  const r = await post("instruction", { instruction: $("instruction").value });
  setStatus(r.msg || "instruction sent");
});

async function poll() {
  try {
    const s = await (await fetch("/state")).json();

    $("raw-bytes").textContent = fmtBytes(s.raw_frame_bytes);
    $("payload-bytes").textContent = fmtBytes(s.payload_bytes);
    $("ratio").textContent = s.compression_ratio ? `${s.compression_ratio}×` : "–";
    $("payload").textContent = JSON.stringify(s.detections, null, 2);

    $("llm-status").textContent = s.llm_status;
    $("llm-parsed").textContent = JSON.stringify(s.llm_parsed, null, 2);
    $("llm-raw").textContent = s.llm_raw || "–";

    renderLegend(s.task);
    $("control-mode").textContent =
      "control: " + (s.control_mode || "–") +
      (s.target_label ? "  ·  target = " + s.target_label : "") +
      (s.object_world_pos ? "  ·  @ " + JSON.stringify(s.object_world_pos) : "");
    $("action").textContent = JSON.stringify(s.action);
    $("reward").textContent = s.reward;
    $("success").textContent = s.success ? "yes" : "no";
    $("success").style.color = s.success ? "var(--green)" : "var(--muted)";
    $("step").textContent = s.step;

    if (!$("status").textContent.includes("error")) {
      setStatus(s.running ? `running · ${s.task}` : "idle");
    }
  } catch (e) {
    /* server not ready; keep polling */
  }
}

setInterval(poll, 500);
poll();
