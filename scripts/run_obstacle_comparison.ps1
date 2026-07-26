param(
    [int]$Seed = 1010,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv312-rocm\Scripts\python.exe"
$scenarioRunner = Join-Path $PSScriptRoot "run_warehouse_scenario.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python environment not found at $python"
}

Push-Location $projectRoot
try {
    $headlessArgument = @()
    if ($Headless) {
        $headlessArgument = @("--headless")
    }

    Write-Host ""
    Write-Host "PHASE 1/2: Unsafe VLA-only execution" -ForegroundColor Red
    Write-Host "Architecture B is bypassed. Close MuJoCo after inspecting the collision."
    & $python $scenarioRunner --scenario obstacle_vla_violation --seed $Seed @headlessArgument
    if ($LASTEXITCODE -ne 0) {
        throw "Unsafe obstacle scenario failed to run."
    }

    Write-Host ""
    Write-Host "PHASE 2/2: LLM-reviewed safe stop" -ForegroundColor Green
    Write-Host "The same proposal is rejected. Close MuJoCo to finish."
    & $python $scenarioRunner --scenario unexpected_obstacle --seed $Seed @headlessArgument
    if ($LASTEXITCODE -ne 0) {
        throw "Protected obstacle scenario failed to run."
    }

    Write-Host ""
    Write-Host "Comparison complete: collision versus zero-command stop."
} finally {
    Pop-Location
}
