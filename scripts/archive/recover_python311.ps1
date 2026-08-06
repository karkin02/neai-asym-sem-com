$ErrorActionPreference = "Stop"

$python = "C:\Users\yangc\AppData\Local\Programs\Python\Python311\python.exe"
$outputDir = Join-Path (Split-Path -Parent $PSScriptRoot) "outputs\python311_recovery"
$snapshot = Join-Path $outputDir "packages_before_recovery.txt"
$requirements = Join-Path $outputDir "missing_dependencies.txt"
$checkReport = Join-Path $outputDir "pip_check_after_recovery.txt"
$failureReport = Join-Path $outputDir "failed_dependency_installs.txt"

if (-not (Test-Path $python)) {
    throw "Python 3.11 was not found at $python"
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$allPackages = @(& $python -m pip list --format=freeze)
$allPackages |
    Sort-Object |
    Set-Content -Encoding ASCII $snapshot

$initialCheck = @(& $python -m pip check 2>&1)
$missingDependencies = foreach ($line in $initialCheck) {
    if ($line -match ' requires ([^,]+), which is not installed\.$') {
        $matches[1]
    }
}
$missingDependencies = @(
    $missingDependencies |
        Where-Object { $_ -notin @("torch", "torchvision") } |
        Sort-Object -Unique
)
$missingDependencies |
    Set-Content -Encoding ASCII $requirements

Write-Host "Package snapshot: $snapshot"
Write-Host "Missing dependency list: $requirements"
Write-Host "Restoring missing dependencies one at a time..."

& $python -m pip install --upgrade pip wheel "setuptools<81"

$failed = @()
foreach ($dependency in $missingDependencies) {
    Write-Host "Installing $dependency..."
    & $python -m pip install $dependency
    if ($LASTEXITCODE -ne 0) {
        $failed += $dependency
    }
}

if (& $python -m pip show torch-directml 2>$null) {
    Write-Host "Restoring the Torch and TorchVision versions required by torch-directml..."
    & $python -m pip install --force-reinstall "torch-directml==0.2.5.dev240914"
    if ($LASTEXITCODE -ne 0) {
        $failed += "torch-directml==0.2.5.dev240914"
    }
}

$failed |
    Sort-Object -Unique |
    Set-Content -Encoding ASCII $failureReport
& $python -m pip check 2>&1 |
    Tee-Object -FilePath $checkReport

Write-Host "Failed installs: $failureReport"
Write-Host "Consistency report: $checkReport"
