"""One local app for both halves: design (/design/) and analysis (/analyse/).

    python3 run_app.py --open        # or: python3 -m pdstat.webapp

This owns no routes of its own beyond the tab shell. It mounts the two existing
FastAPI apps unchanged, so each keeps working as a separate server too. The
front ends call their APIs with relative URLs ("api/..."), which is what lets
the same page work at "/" on its own port and at "/design/" here.

Loopback only, like the two apps it wraps.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from doe_advisor.server import app as design_app
from stat_board.webapp import app as board_app

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_PORT = 8700

app = FastAPI(title="Statistical Workbench")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# Relative API URLs only resolve under the mount if the page URL ends in "/".
@app.get("/design", include_in_schema=False)
def _design_slash() -> RedirectResponse:
    return RedirectResponse("/design/")


@app.get("/analyse", include_in_schema=False)
def _analyse_slash() -> RedirectResponse:
    return RedirectResponse("/analyse/")


app.mount("/design", design_app)
app.mount("/analyse", board_app)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import threading
    import webbrowser

    import uvicorn

    from stat_board import config

    parser = argparse.ArgumentParser(prog="pdstat.webapp", description="Run the design + analysis app locally.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--open", action="store_true", help="open a browser tab once the server is up")
    args = parser.parse_args(argv)

    url = f"http://127.0.0.1:{args.port}"
    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    print(f"Statistical Workbench is running at {url}   (Ctrl+C to stop)")
    if not config.credentials_present():
        print("Note: no Anthropic credentials found. Design, memos and board dry runs work; "
              "live board runs need ANTHROPIC_API_KEY.")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
