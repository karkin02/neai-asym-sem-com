param(
    [int]$Seed = 1010,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv312-rocm\Scripts\python.exe"
$runner = Join-Path $PSScriptRoot "run_warehouse_scenario.py"
$headlessArgument = @()
if ($Headless) {
    $headlessArgument = @("--headless")
}

Push-Location $projectRoot
try {
    Write-Host "PHASE 1/2: Damaged package shipped without policy review" -ForegroundColor Red
    & $python $runner --scenario damaged_vla_violation --seed $Seed @headlessArgument
    if ($LASTEXITCODE -ne 0) { throw "Unsafe damaged-package phase failed." }

    Write-Host "PHASE 2/2: LLM policy review reroutes damaged package" -ForegroundColor Green
    & $python $runner --scenario package_damaged --seed $Seed @headlessArgument
    if ($LASTEXITCODE -ne 0) { throw "Protected damaged-package phase failed." }

    Write-Host "Comparison complete: outbound policy violation versus rejection routing."
} finally {
    Pop-Location
}
