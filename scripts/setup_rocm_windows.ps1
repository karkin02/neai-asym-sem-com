param(
    [string]$Python312 = ""
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$venvPath = Join-Path $repoRoot ".venv312-rocm"
$python312 = $Python312
if (-not $python312) {
    $python312 = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $python312) {
        throw "Python 3.12 was not found. Install it or pass -Python312 <path>."
    }
}
$venvPython = Join-Path $venvPath "Scripts\python.exe"

if (-not (Test-Path $python312)) {
    throw "Python 3.12 was not found at $python312"
}

if (-not (Test-Path $venvPython)) {
    & $python312 -m venv $venvPath
}

& $venvPython -m ensurepip --upgrade
& $venvPython -m pip install --upgrade pip

# Torch 2.12 installs this aggregate wheel; Torch 2.11 gfx1103 does not use it.
& $venvPython -m pip uninstall -y "amd-torch-device-gfx11"
if ($LASTEXITCODE -ne 0) {
    throw "Could not remove the stale Torch 2.12 gfx11 package."
}

& $venvPython -m pip install `
    --force-reinstall `
    --index-url "https://repo.amd.com/rocm/whl-multi-arch/" `
    "torch[device-gfx1103]==2.11.0+rocm7.14.0" `
    "torchvision[device-gfx1103]==0.26.0+rocm7.14.0" `
    "torchaudio==2.11.0+rocm7.14.0"
if ($LASTEXITCODE -ne 0) {
    throw "Could not install the ROCm 2.11 gfx1103 stack."
}

# ROCm's wheel index can select versions newer than LeRobot 0.6 permits.
& $venvPython -m pip install `
    "numpy==2.2.6" `
    "fsspec[http]==2026.2.0" `
    "setuptools==80.10.2"
if ($LASTEXITCODE -ne 0) {
    throw "Could not normalize dependencies for LeRobot 0.6."
}

$verifyCode = @'
import torch

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("GPU available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ROCm installed, but PyTorch cannot access the GPU.")

print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda")
y = x @ x
torch.cuda.synchronize()
print("GPU matrix multiply:", tuple(y.shape), y.device, torch.isfinite(y).all().item())
'@

$verifyCode | & $venvPython -
if ($LASTEXITCODE -ne 0) {
    throw "ROCm GPU verification failed."
}

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) {
    throw "The combined ROCm and LeRobot package requirements are inconsistent."
}
