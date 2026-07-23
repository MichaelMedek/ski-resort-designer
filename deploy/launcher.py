"""Frozen desktop entrypoint: run the Streamlit server as a child process (it needs the main thread
for signal handlers) and show it in a pywebview window. Frozen-only; dev uses `streamlit run`.
GDAL/PROJ dirs are left unset on purpose — rasterio and pyproj self-locate their own proj.db.
"""

import os
import socket
import subprocess
import sys
import time

_SERVER_FLAG = "SKIRESORT_RUN_SERVER"  # set on the child → it runs the server instead of the window
_PORT_ENV = "SKIRESORT_SERVER_PORT"


def _free_port() -> int:
    """Let the OS assign a free localhost port."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


def _wait_ready(port: int, timeout: float = 60.0) -> None:
    """Block until the server accepts connections, or raise on timeout."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError("Streamlit server did not start in time")


def _script_path() -> str:
    """Absolute path to skiresort_planner/app.py, frozen (in _MEIPASS) or in dev."""
    base = getattr(sys, "_MEIPASS", None)
    if base is not None:
        return os.path.join(base, "skiresort_planner", "app.py")
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "skiresort_planner", "app.py")


def _run_server() -> None:
    """Server-mode entry — runs in the child's main thread so Streamlit's signal setup works."""
    from streamlit.web import bootstrap

    port = int(os.environ[_PORT_ENV])
    flags = {
        "server.address": "127.0.0.1",
        "server.port": port,
        "server.headless": True,
        "global.developmentMode": False,
        "browser.gatherUsageStats": False,
        "server.fileWatcherType": "none",
        "client.toolbarMode": "minimal",
    }
    bootstrap.load_config_options(flag_options=flags)
    bootstrap.run(_script_path(), is_hello=False, args=[], flag_options=flags)


def _spawn_server(port: int) -> subprocess.Popen:
    """Re-exec this executable in server mode (frozen: the app binary; dev: python + this script)."""
    env = dict(os.environ)
    env[_SERVER_FLAG] = "1"
    env[_PORT_ENV] = str(port)
    cmd = [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, os.path.abspath(__file__)]
    return subprocess.Popen(cmd, env=env)


def main() -> None:
    """Spawn the server process, then open the pywebview window (blocks until closed)."""
    port = _free_port()
    proc = _spawn_server(port)
    try:
        _wait_ready(port)
        import webview

        # Enable file downloads (default off) so Streamlit's Save/Export buttons trigger a native save
        # dialog instead of rendering the JSON/GPX inline in the window. Must be set before start().
        webview.settings["ALLOW_DOWNLOADS"] = True
        webview.create_window("Alpin Architect", f"http://127.0.0.1:{port}", width=1400, height=900)
        webview.start()  # blocks until the window is closed
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    if os.environ.get(_SERVER_FLAG):
        _run_server()
    else:
        main()
