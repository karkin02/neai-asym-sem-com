param(
    [int]$SceneSeed = 1010,
    [int]$ChannelSeed = 1010,
    [string]$Device = "cpu",
    [string]$OutputRoot = "outputs/compact_feeder_gpt"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:YOLO_CONFIG_DIR = Join-Path (Get-Location) ".ultralytics"
$python = ".\.venv312-rocm\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python was not found at $python"
}
if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set."
}

$levels = @("level1", "level2", "level3")
$cases = @(
    @("warehouse_normal", "Inspect once, then send a normal package to the downstream conveyor."),
    @("barcode_missing", "Inspect once, then send a missing-barcode package to the left inspection tray."),
    @("package_damaged", "Inspect once, then send a damaged package to the right rejection tray.")
)

$results = @()
foreach ($level in $levels) {
    foreach ($case in $cases) {
        $scene = $case[0]
        $instruction = $case[1]
        Write-Host "[compact-gpt] level=$level scene=$scene"
        $savedErrorActionPreference = $ErrorActionPreference
        try {
            # Windows PowerShell wraps native stderr as NativeCommandError when
            # ErrorActionPreference is Stop. Capture it as trial output instead
            # so a failed episode is recorded and the matrix can continue.
            $ErrorActionPreference = "Continue"
            $lines = & $python scripts/run_compact_feeder_inspection.py `
                --scene $scene `
                --instruction $instruction `
                --planner gpt `
                --channel $level `
                --seed $SceneSeed `
                --channel-seed $ChannelSeed `
                --transmission-attempts 3 `
                --device $Device `
                --advance-world-during-network 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedErrorActionPreference
        }
        $lines | ForEach-Object { Write-Host $_ }
        $summary = [string]($lines | Where-Object { $_ -like "scene=*" } | Select-Object -Last 1)
        $results += [pscustomobject]@{
            level = $level
            scene = $scene
            success = ($exitCode -eq 0)
            summary = $summary
            error = if ($exitCode -eq 0) { $null } else { [string]($lines | Select-Object -Last 1) }
        }
    }
}

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runDirectory = Join-Path $OutputRoot "compact-gpt-$timestamp"
New-Item -ItemType Directory -Path $runDirectory -Force | Out-Null
$jsonPath = Join-Path $runDirectory "comparison.json"
$results | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$passed = @($results | Where-Object success).Count
Write-Host "[compact-gpt] completed $passed/$($results.Count)"
Write-Host "[compact-gpt] wrote $jsonPath"
