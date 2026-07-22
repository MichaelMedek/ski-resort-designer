# Alpin Architect — desktop app

Desktop builds (macOS `.dmg`, Windows `.zip`) are produced **only** by the GitHub Actions release
pipeline (`.github/workflows/release.yml`): bump the version in `version.txt`, merge to `main`, and
it builds both installers and publishes a GitHub Release `v<version>`. No manual release.

## Test locally (macOS)

```bash
python3.11 -m venv .venv-skiresort && source .venv-skiresort/bin/activate
pip install -r requirements.txt -r deploy/requirements-build.txt
SKIRESORT_DATA_ROOT=$PWD bash deploy/build_mac.sh   # reuses your DEM
open deploy/dist/AlpinArchitect.app
```

## For users

Download from the repo's **Releases** page, then:

- **macOS** — `AlpinArchitect.dmg` → open, drag to Applications, run. First launch (unsigned): **System Settings → Privacy & Security → Open Anyway**, or run once: `xattr -dr com.apple.quarantine /Applications/AlpinArchitect.app`
- **Windows** — `AlpinArchitect-win.zip` → unzip, run `AlpinArchitect.exe`. On the SmartScreen popup: More info → Run anyway.

The 285 MB terrain data downloads automatically on first run.
