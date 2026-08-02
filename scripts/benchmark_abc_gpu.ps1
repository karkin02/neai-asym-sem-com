param(
    [int]$Episodes = 3,
    [int]$Seed = 40000
)

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

foreach ($required in @($python, $checkpoint, $vlm, $yolo)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required benchmark asset is missing: $required"
    }
}

& $python scripts\evaluate_smolvla.py `
    --checkpoint $checkpoint --vlm $vlm `
    --episodes $Episodes --seed $Seed `
    --scene warehouse_normal --warehouse-layout v3 `
    --device cuda --yolo-weights $yolo `
    --record-camera observer `
    --output outputs\architecture_a_standardized_gpu_latency

& $python -m architecture_b.runner `
    --checkpoint $checkpoint --vlm $vlm `
    --episodes $Episodes --seed $Seed `
    --scene warehouse_normal --warehouse-layout v3 `
    --planner heuristic --executor architecture_a `
    --device cuda --yolo-weights $yolo `
    --output outputs\architecture_b_standardized_gpu_latency

& $python -m architecture_c.runner `
    --checkpoint $checkpoint --vlm $vlm `
    --episodes $Episodes --seed $Seed `
    --scene warehouse_normal --warehouse-layout v3 `
    --planner heuristic --device cuda --yolo-weights $yolo `
    --output outputs\architecture_c_standardized_gpu_latency

@'
import json
from pathlib import Path
from statistics import mean, median, pstdev

for architecture in ("a", "b", "c"):
    root = Path(f"outputs/architecture_{architecture}_standardized_gpu_latency")
    metrics = max(root.rglob("metrics.json"), key=lambda path: path.stat().st_mtime)
    results = json.loads(metrics.read_text(encoding="utf-8-sig"))["results"]
    warm = results[1:]
    successful = [row for row in warm if row.get("success")]
    warm_latencies = [float(row["latency_seconds"]) for row in warm]
    ordered = sorted(warm_latencies)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
    print(
        f"{architecture.upper()}: warm_mean={mean(warm_latencies):.3f}s "
        f"median={median(warm_latencies):.3f}s "
        f"stdev={pstdev(warm_latencies):.3f}s "
        f"p95={ordered[p95_index]:.3f}s "
        f"warm_success={len(successful)}/{len(warm)} "
        f"latencies={[round(value, 3) for value in warm_latencies]} "
        f"metrics={metrics}"
    )
'@ | & $python -
