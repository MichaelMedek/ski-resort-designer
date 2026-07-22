# Packaging Alpin Architect as a desktop app

Double-clickable app via PyInstaller + pywebview. The 285 MB DEM downloads on first run (not bundled).
Both builds need Python 3.11 (PyInstaller freezes the interpreter + deps per-OS) — build the `.app`
on a Mac, the `.exe` on Windows. No cross-compile.

## Build on macOS

```bash
python3.11 -m venv .venv-skiresort            # create venv (once)
source .venv-skiresort/bin/activate           # activate it (every new shell)
pip install -r requirements.txt               # app deps (streamlit, rasterio, …)
pip install -r deploy/requirements-build.txt  # build tools (pyinstaller, pywebview)
rm -rf deploy/build deploy/dist               # clean
bash deploy/build_mac.sh                      # → deploy/dist/AlpinArchitect.app + .dmg
open deploy/dist/AlpinArchitect.app           # test
```

Tip: `SKIRESORT_DATA_ROOT=$PWD bash deploy/build_mac.sh` reuses your existing DEM (skips download).

## Build on Windows

```powershell
py -3.11 -m venv .venv-skiresort                      # create venv (once)
.venv-skiresort\Scripts\activate                      # activate it (every new shell)
pip install -r requirements.txt                       # app deps (streamlit, rasterio, …)
pip install -r deploy\requirements-build.txt          # build tools (pyinstaller, pywebview)
Remove-Item -Recurse -Force deploy\build, deploy\dist -ErrorAction SilentlyContinue  # clean
powershell -ExecutionPolicy Bypass -File deploy\build_windows.ps1  # → deploy\dist\AlpinArchitect-win.zip
deploy\dist\AlpinArchitect\AlpinArchitect.exe         # test
```

Share `deploy\dist\AlpinArchitect-win.zip` (users unzip, run the `.exe`).

## Where files are saved

- **Built app** (upload the `.dmg`): `deploy/dist/` — git-ignored.
- **Runtime data** (DEM, backups): macOS `~/Library/Application Support/AlpinArchitect/`, Windows `%LOCALAPPDATA%\AlpinArchitect\`.

## First launch (unsigned — clicked once)

- **macOS**: right-click the app → **Open** → **Open**. If macOS says the app is *"damaged and can't be opened"*, clear the quarantine flag once, then reopen: `xattr -dr com.apple.quarantine /Applications/AlpinArchitect.app`
- **Windows**: double-click the `.exe`. On the *"Windows protected your PC"* popup, click **More info** → **Run anyway**.
