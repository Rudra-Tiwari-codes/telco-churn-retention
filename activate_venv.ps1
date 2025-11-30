# PowerShell script to activate virtual environment
# Usage: .\activate_venv.ps1

$venvPath = Join-Path $PSScriptRoot "venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Error: Virtual environment not found at '$venvPath'" -ForegroundColor Red
    Write-Host "Please run .\setup_venv.ps1 first to create the virtual environment." -ForegroundColor Yellow
    exit 1
}

$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated!" -ForegroundColor Green
} else {
    Write-Host "Error: Activation script not found at '$activateScript'" -ForegroundColor Red
    exit 1
}

