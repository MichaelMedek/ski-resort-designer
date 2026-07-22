# Build the Windows .exe (onedir) + zip for Alpin Architect.
# Run from the repo root in PowerShell:  powershell -ExecutionPolicy Bypass -File deploy\build_windows.ps1
$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot   # deploy\ — the spec resolves the repo root as its parent

Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

# Use pyinstaller on PATH if present, else invoke it as a module (robust in fresh venvs / CI).
if (Get-Command pyinstaller -ErrorAction SilentlyContinue) {
  pyinstaller skiresort.spec --clean --noconfirm
} else {
  python -m PyInstaller skiresort.spec --clean --noconfirm
}

Compress-Archive -Path "dist\AlpinArchitect\*" -DestinationPath "dist\AlpinArchitect-win.zip" -Force

Write-Host "Built: deploy\dist\AlpinArchitect\AlpinArchitect.exe and deploy\dist\AlpinArchitect-win.zip"
