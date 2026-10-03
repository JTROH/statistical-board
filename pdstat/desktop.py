"""The combined app in its own desktop window (no browser tab, no fixed port).

    python3 run_desktop.py                    # from source
    open "dist/Statistical Workbench.app"     # after packaging/build_mac_app.sh

Runs the same FastAPI app as pdstat.webapp on a free loopback port, in a
background thread, and shows it in a native window (pywebview -> WKWebView on
macOS). Closing the window stops the server.

The two halves read and write relative paths (reports/, uploads/,
sample_data/, .env). From source those stay relative to wherever you launch.
A Finder-launched .app starts in "/", which is read-only, so the packaged app
switches to a data folder in ~/Documents first and seeds it with the sample
data on first launch.
"""

from __future__ import annotations

import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

APP_NAME = "Statistical Workbench"
DEFAULT_DATA_DIR = Path.home() / "Documents" / APP_NAME


def is_frozen() -> bool:
    """True inside the PyInstaller-built .app."""
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Where bundled data files (sample_data/, presets/) live."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def prepare_data_dir(data_dir: Path, seed_from: Path | None = None) -> Path:
    """Create data_dir, copy the sample datasets in once, and chdir into it.

    Must run before pdstat.webapp is imported: stat_board.config reads .env
    from the working directory at import time.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    seed = (seed_from or bundle_root()) / "sample_data"
    target = data_dir / "sample_data"
    if seed.is_dir() and not target.exists():
        shutil.copytree(seed, target)
    os.chdir(data_dir)
    return data_dir


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="pdstat.desktop", description=f"Run the {APP_NAME} in a desktop window.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help=f"folder for reports, uploads and .env (default when packaged: {DEFAULT_DATA_DIR})")
    parser.add_argument("--debug", action="store_true", help="enable the web inspector")
    args = parser.parse_args(argv)

    if args.data_dir or is_frozen():
        prepare_data_dir((args.data_dir or DEFAULT_DATA_DIR).expanduser())

    try:
        import uvicorn
        import webview

        from pdstat.webapp import app
    except ImportError as exc:
        print(f"Missing dependencies ({exc}). Run:  pip install -r requirements.txt pywebview", file=sys.stderr)
        return 1

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not server.started:
        if not thread.is_alive() or time.monotonic() > deadline:
            print("The local server did not start.", file=sys.stderr)
            return 1
        time.sleep(0.05)

    # The design memo is saved via a blob download; WKWebView drops it unless allowed.
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window(APP_NAME, f"http://127.0.0.1:{port}/", width=1280, height=860, min_size=(900, 600))
    webview.start(debug=args.debug)

    server.should_exit = True
    thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
