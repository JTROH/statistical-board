#!/usr/bin/env python3
"""Launch doe-advisor locally.

    python3 run.py --open

Binds to 127.0.0.1 only. Nothing you enter leaves your machine, except the
optional narration call to the Anthropic API when ANTHROPIC_API_KEY is set —
and that sends only the design facts, never your data.
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the doe-advisor web app locally.")
    parser.add_argument("--port", type=int, default=8711, help="port to listen on (default: 8711)")
    parser.add_argument("--open", action="store_true", help="open a browser tab once the server is up")
    args = parser.parse_args(argv)

    try:
        import uvicorn
    except ImportError:
        print("Missing dependencies. Run:  pip install -r requirements.txt", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}"
    if args.open:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    print(f"doe-advisor is running at {url}   (Ctrl+C to stop)")
    uvicorn.run("doe_advisor.server:app", host="127.0.0.1", port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
