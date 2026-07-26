$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv312-rocm\Scripts\python.exe"

if (-not (Test-Path $python)) {
    throw "ROCm environment not found at $python"
}

$before = @'
import json
import torch
from importlib.metadata import PackageNotFoundError, version

try:
    gfx11 = version("amd-torch-device-gfx11")
except PackageNotFoundError:
    gfx11 = None

print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "gpu_available": torch.cuda.is_available(),
    "gfx11": gfx11,
    "gfx1103": version("amd-torch-device-gfx1103"),
}))
'@ | & $python -

$beforeState = $before | ConvertFrom-Json
if (-not $beforeState.hip -or -not $beforeState.gpu_available) {
    throw "The existing PyTorch installation is not a working ROCm build."
}
if ($beforeState.torch -notlike "2.11.*+rocm7.14.0") {
    throw "LeRobot 0.6 requires Torch <2.12. Run setup_rocm_windows.ps1 to install ROCm Torch 2.11 first."
}
if (
    $beforeState.gfx11 -or
    $beforeState.gfx1103 -notlike "2.11.*+rocm7.14.0"
) {
    throw "A stale gfx11 package remains or gfx1103 does not match Torch 2.11. Run setup_rocm_windows.ps1 first."
}

$env:PYTHONUTF8 = "1"
& $python -m pip install -e "$repoRoot[lerobot]"
if ($LASTEXITCODE -ne 0) {
    throw "LeRobot installation failed."
}

$verify = @'
import json
import torch
import lerobot
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

x = torch.randn(512, 512, device="cuda")
y = x @ x
torch.cuda.synchronize()

print(json.dumps({
    "torch": torch.__version__,
    "hip": torch.version.hip,
    "gpu": torch.cuda.get_device_name(0),
    "matrix_ok": bool(torch.isfinite(y).all().item()),
    "lerobot": getattr(lerobot, "__version__", "0.6.0"),
    "smolvla_policy": SmolVLAPolicy.__name__,
}))
'@ | & $python -

$afterState = $verify | ConvertFrom-Json
if (
    $afterState.torch -ne $beforeState.torch -or
    $afterState.hip -ne $beforeState.hip
) {
    throw "pip replaced the ROCm PyTorch build during LeRobot installation."
}

& $python -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The ROCm LeRobot environment has broken package requirements."
}

Write-Host "LeRobot ROCm environment is ready."
Write-Host "Torch: $($afterState.torch)"
Write-Host "HIP: $($afterState.hip)"
Write-Host "GPU: $($afterState.gpu)"
Write-Host "SmolVLA: $($afterState.smolvla_policy)"
