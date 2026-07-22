# Ski Resort Planner

Design ski resorts on real terrain with an addictive, game-like interface.

![Full Resort Overview](docs/images/7-FullResort.png)

## Download & Run

Get the app as a normal desktop program.

1. Open the [**Releases**](https://github.com/MichaelMedek/ski-resort-designer/releases/latest) page and download the file for your computer:
   - **Mac** → `AlpinArchitect.dmg`
   - **Windows** → `AlpinArchitect-win.zip`
2. Open it:
   - **Mac** → double-click the `.dmg`, drag **Alpin Architect** into **Applications**, then open it.
   - **Windows** → unzip the file, open the folder, double-click **`AlpinArchitect.exe`**.
3. First open shows a one-time safety prompt (the app isn't code-signed):
   - **Mac** → right-click the app → **Open** → **Open**.
   - **Windows** → **"Windows protected your PC"** → **More info** → **Run anyway**.

On first launch it downloads ~285 MB of terrain data automatically (once); after that it starts fast.

## Use the app from your browser

No install needed: open **https://ski-resort-designer.streamlit.app/**. If it's been idle it may be
asleep — click the **wake-up button** and wait a minute or two, then use it normally. Handy on a tablet
or any device that can't run the desktop app; just expect it to be a bit slower than the native version.

---

## Terrain Data

The app uses pre-cropped Alps DEM (Digital Elevation Model) data covering the European Alps at 60m resolution. The data is automatically downloaded from Hugging Face on first run.

For other regions, download the full 2.3GB EuroDEM from https://www.mapsforeurope.org/datasets/euro-dem and use `scripts/crop_dem_to_alps.py` as a template.

---

## Documentation

Detailed documentation  about how to use the app UI and the underlying algorithms is available in the `docs/` folder:

| Document | Contents |
|----------|----------|
| [User Guide](docs/DETAILS_UI.md) | How to use the application |
| [Technical Reference](docs/DETAILS.md) | Architecture and algorithms |

---

## Run from source (developers)

```bash
python3.11 -m venv .venv-skiresort && source .venv-skiresort/bin/activate
pip install -r requirements.txt
streamlit run skiresort_planner/app.py                      # opens at http://localhost:8501
```

Terrain data downloads on first run (see above); `Ctrl+C` stops the server. Debug logging:

```bash
SKIRESORT_LOG_LEVEL=DEBUG streamlit run skiresort_planner/app.py 2>&1 | tee output/debug_$(date +%Y%m%d_%H%M%S).log
```

Packaging the desktop app (installers are built automatically by CI on release) — see [deploy/README.md](deploy/README.md).
