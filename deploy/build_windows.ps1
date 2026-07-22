# Build the Windows .exe (onedir) + zip for Alpin Architect.
# Run from the repo root in PowerShell:  powershell -ExecutionPolicy Bypass -File deploy\build_windows.ps1
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot   # deploy\ — the spec resolves the repo root as its parent

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
pyinstaller skiresort.spec --clean --noconfirm

Compress-Archive -Path "dist\AlpinArchitect\*" -DestinationPath "dist\AlpinArchitect-win.zip" -Force

Write-Host "Built: deploy\dist\AlpinArchitect\AlpinArchitect.exe and deploy\dist\AlpinArchitect-win.zip"
