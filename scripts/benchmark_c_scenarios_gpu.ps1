param([int]$WarmEpisodes = 20)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$python = Join-Path $repo ".venv312-rocm\Scripts\python.exe"
$checkpoint = Join-Path $repo "outputs\train\warehouse_v3_150\checkpoints\002000\pretrained_model"
$vlm = Join-Path $repo ".hf-cache\checkpoints\smolvlm2_500m_video_instruct"
$yolo = Join-Path $repo "outputs\train\warehouse_yolov8n_v9_single_wrist_damage_balanced\weights\best.pt"

$env:PYTHONPATH = Join-Path $repo "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:YOLO_CONFIG_DIR = Join-Path $repo "runtime\ultralytics"
New-Item -ItemType Directory -Force -Path $env:YOLO_CONFIG_DIR | Out-Null

$cases = @(
    @{ Scene = "warehouse_normal"; Seed = 40000 },
    @{ Scene = "barcode_missing"; Seed = 50000 },
    @{ Scene = "package_damaged"; Seed = 60000 }
)
$episodes = $WarmEpisodes + 1
foreach ($case in $cases) {
    & $python -m architecture_c.runner `
        --checkpoint $checkpoint --vlm $vlm --yolo-weights $yolo `
        --episodes $episodes --seed $case.Seed `
        --scene $case.Scene --warehouse-layout v3 `
        --planner heuristic --device cuda `
        --output ("outputs\architecture_c_scenario_gpu\" + $case.Scene)
}

@'
import json
import math
from pathlib import Path
from statistics import mean, median

root = Path("outputs/architecture_c_scenario_gpu")
total_success = total_escalated = total_bytes = total_warm = 0
for scene in ("warehouse_normal", "barcode_missing", "package_damaged"):
    metrics = max((root / scene).rglob("metrics.json"), key=lambda p: p.stat().st_mtime)
    rows = json.loads(metrics.read_text(encoding="utf-8-sig"))["results"][1:]
    latency = sorted(float(row["latency_seconds"]) for row in rows)
    success = sum(bool(row.get("success")) for row in rows)
    escalated = sum(bool(row.get("escalated")) for row in rows)
    payload = sum(int(row.get("network_payload_bytes") or 0) for row in rows)
    clips = [float(row["clip_confidence"]) for row in rows if row.get("clip_confidence") is not None]
    p95 = latency[max(0, min(len(latency)-1, math.ceil(0.95*len(latency))-1))]
    print(
        f"{scene}: success={success}/{len(rows)} escalated={escalated}/{len(rows)} "
        f"mean={mean(latency):.3f}s median={median(latency):.3f}s p95={p95:.3f}s "
        f"clip={min(clips):.4f}-{max(clips):.4f} bytes={payload} metrics={metrics}"
    )
    total_success += success
    total_escalated += escalated
    total_bytes += payload
    total_warm += len(rows)
print(
    f"TOTAL: success={total_success}/{total_warm} "
    f"escalated={total_escalated}/{total_warm} bytes={total_bytes}"
)
'@ | & $python -
