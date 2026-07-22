# Packaging Alpin Architect as a desktop app

Double-clickable app via PyInstaller + pywebview. DEM (285 MB) downloads on first run. Build per-OS (no cross-compile), Python 3.11.

## Build on macOS

```bash
python3.11 -m venv .venv-skiresort && source .venv-skiresort/bin/activate   # venv (once)
pip install -r requirements.txt -r deploy/requirements-build.txt            # deps + build tools
rm -rf deploy/build deploy/dist                                             # clean
bash deploy/build_mac.sh                                                    # → dist/AlpinArchitect.dmg
open deploy/dist/AlpinArchitect.app                                         # test
```

Tip: prefix `SKIRESORT_DATA_ROOT=$PWD` to reuse your existing DEM (skips the download).

## Build on Windows

```powershell
py -3.11 -m venv .venv-skiresort; .venv-skiresort\Scripts\activate          # venv (once)
pip install -r requirements.txt -r deploy\requirements-build.txt            # deps + build tools
Remove-Item -Recurse -Force deploy\build, deploy\dist -EA SilentlyContinue  # clean
powershell -ExecutionPolicy Bypass -File deploy\build_windows.ps1           # → dist\AlpinArchitect-win.zip
deploy\dist\AlpinArchitect\AlpinArchitect.exe                               # test
```

## Ship

No single file runs on both OSes. You build the program, upload a wrapper that contains it, the user runs the program inside. Send each person the wrapper for their OS (all-Mac audience → just the `.dmg`):

| OS | Build | Upload | User runs |
|----|-------|--------|-----------|
| macOS | `AlpinArchitect.app` | `AlpinArchitect.dmg` | `AlpinArchitect.app` |
| Windows | folder + `AlpinArchitect.exe` | `AlpinArchitect-win.zip` | `AlpinArchitect.exe` |

## Files

- Build output (upload from here): `deploy/dist/` (git-ignored); `deploy/build/` is scratch.
- Runtime data (DEM, backups): macOS `~/Library/Application Support/AlpinArchitect/`, Windows `%LOCALAPPDATA%\AlpinArchitect\`.

## First launch (unsigned, one-time)

- macOS: right-click → Open → Open. If "damaged": `xattr -dr com.apple.quarantine /Applications/AlpinArchitect.app`
- Windows: double-click `.exe` → "Windows protected your PC" → More info → Run anyway.
