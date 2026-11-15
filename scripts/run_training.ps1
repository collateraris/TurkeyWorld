# Explicit foreground launcher. Use a persistent terminal/session for disconnects.
$ErrorActionPreference = 'Stop'
$trainingArgs = @($args)
if ($trainingArgs.Count -eq 0 -or $trainingArgs[0] -eq '--help') {
    Write-Output 'Usage: .\scripts\run_training.ps1 --module package.module [target arguments]'
    Write-Output 'Windows workers request system availability only; screen locking remains enabled.'
    Write-Output 'On Ubuntu use: bash scripts/run_training.sh --module package.module [arguments]'
    exit 0
}
if ($trainingArgs[0] -ne '--module' -and $trainingArgs[0] -ne '--script') {
    throw 'Expected --module or --script; use --help.'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = $env:TURKEYWORLD_PYTHON
if (-not $pythonPath) {
    $pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath = (Get-Command python -ErrorAction Stop).Source }
}
Push-Location -LiteralPath $projectRoot
try {
    & $pythonPath -m turkeyworld.bootstrap @trainingArgs
    exit $LASTEXITCODE
} finally { Pop-Location }
