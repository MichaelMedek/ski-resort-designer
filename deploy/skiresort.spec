# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for Alpin Architect (shared macOS + Windows).

onedir (not onefile): a macOS .app is a onedir bundle anyway, and onefile is fragile for the
GDAL/PROJ/GEOS native stack (env vars point into a temp dir that changes each launch). Build with:
    pyinstaller deploy/skiresort.spec --clean --noconfirm
"""

import os

from PyInstaller.utils.hooks import (
    collect_all,
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

# Repo root (spec runs with CWD = deploy/, so the package source is one level up).
REPO_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

datas: list = []
binaries: list = []
hiddenimports: list = []

# Risky packages: native extensions + data dirs + lazily-imported submodules. collect_all grabs
# binaries + datas + hidden submodules in one shot (GDAL/PROJ data, GEOS libs, streamlit static).
for pkg in ["streamlit", "rasterio", "pyproj", "pydeck", "shapely", "scipy", "plotly"]:
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# Custom Streamlit components register their frontend by path relative to __file__ — collect the
# frontend trees or the deck.gl map and the viewport-height JS read break.
datas += collect_data_files("streamlit_deckgl")
datas += collect_data_files("streamlit_js_eval")

# The app package itself (installed editable, so its source lives at the repo root). collect_submodules
# gets it into the PYZ for imports; but Streamlit reads the entrypoint script (app.py) FROM DISK, so
# also ship the source tree as data at skiresort_planner/ (launcher points bootstrap at that path).
hiddenimports += collect_submodules("skiresort_planner")
datas += [(os.path.join(REPO_ROOT, "skiresort_planner"), "skiresort_planner")]

# Packages whose version/metadata is read at runtime via importlib.metadata (streamlit scans many).
for pkg in [
    "streamlit", "pydeck", "rasterio", "pyproj", "numpy", "shapely", "scipy",
    "networkx", "plotly", "pandas", "streamlit_deckgl", "streamlit_js_eval",
    "python-statemachine", "click", "packaging", "altair", "tornado", "rich",
    "gitpython", "watchdog", "protobuf", "pyarrow", "pillow", "blinker",
    "cachetools", "tenacity", "toml", "platformdirs",
]:
    try:
        datas += copy_metadata(pkg, recursive=True)
    except Exception:
        pass  # optional/absent metadata is fine — not every package is installed

hiddenimports += [
    "streamlit.web.bootstrap",
    "streamlit.web.cli",
    "streamlit.components.v1",
    "streamlit.runtime.scriptrunner.magic_funcs",
]

a = Analysis(
    ["launcher.py"],
    pathex=[REPO_ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],  # rasterio/pyproj self-locate their own proj.db; a shared PROJ_LIB clashes
    excludes=["tkinter", "pytest", "hypothesis", "mypy"],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AlpinArchitect",
    console=False,  # set True to see server logs while debugging
    icon="icon.icns" if os.path.exists("icon.icns") else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    name="AlpinArchitect",
)

app = BUNDLE(
    coll,
    name="AlpinArchitect.app",
    icon="icon.icns" if os.path.exists("icon.icns") else None,
    bundle_identifier="com.alpinarchitect.desktop",
    info_plist={"NSHighResolutionCapable": True, "LSMinimumSystemVersion": "12.0"},
)
