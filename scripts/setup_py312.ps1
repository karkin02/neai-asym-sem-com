param(
    [string]$Python312 = ""
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$uvCache = Join-Path $projectRoot ".uv-cache"
$venv = Join-Path $projectRoot ".venv312"
$python = Join-Path $venv "Scripts\python.exe"

$env:UV_CACHE_DIR = $uvCache

if (-not $Python312) {
    $Python312 = (& py -3.12 -c "import sys; print(sys.executable)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $Python312) {
        throw "Python 3.12 was not found. Install it or pass -Python312 <path>."
    }
}

Push-Location $projectRoot
try {
    if (-not (Test-Path -LiteralPath $Python312)) {
        throw "Python 3.12 was not found at: $Python312"
    }

    if (-not (Test-Path -LiteralPath $python)) {
        & $Python312 -m venv $venv
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create the Python 3.12 virtual environment."
        }
    }

    uv pip install --python $python -e ".[lerobot]"
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    & $python --version
    if ($LASTEXITCODE -ne 0) {
        throw "The Python 3.12 environment could not be started."
    }

    $env:PYTHONPATH = "src"
    & $python -m unittest discover -s tests -v
    if ($LASTEXITCODE -ne 0) {
        throw "The Python 3.12 test suite failed."
    }
} finally {
    Pop-Location
}
