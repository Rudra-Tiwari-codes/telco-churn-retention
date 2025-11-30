# PowerShell script to create virtual environment
# Usage: .\setup_venv.ps1

Write-Host "Creating virtual environment..." -ForegroundColor Cyan

# Check if Python is available
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python not found. Please install Python 3.11 or higher." -ForegroundColor Red
    exit 1
}

# Check Python version
$version = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
$major, $minor = $version -split '\.'
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 11)) {
    Write-Host "Error: Python 3.11 or higher is required. Found: $version" -ForegroundColor Red
    exit 1
}

# Create virtual environment
if (Test-Path "venv") {
    Write-Host "Virtual environment already exists at 'venv'" -ForegroundColor Yellow
    $response = Read-Host "Do you want to recreate it? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force venv
        Write-Host "Removed existing virtual environment" -ForegroundColor Yellow
    } else {
        Write-Host "Keeping existing virtual environment" -ForegroundColor Green
        exit 0
    }
}

python -m venv venv

if ($LASTEXITCODE -eq 0) {
    Write-Host "Virtual environment created successfully!" -ForegroundColor Green
    Write-Host "To activate, run: .\activate_venv.ps1" -ForegroundColor Cyan
} else {
    Write-Host "Error: Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

