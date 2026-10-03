#!/usr/bin/env python3
"""Launch the combined design + analysis app locally.

    python3 run_app.py --open

Binds to 127.0.0.1 only. Design is at /design/, analysis at /analyse/.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    try:
        from pdstat.webapp import main as run
    except ImportError as exc:
        print(f"Missing dependencies ({exc}). Run:  pip install -r requirements.txt", file=sys.stderr)
        return 1
    return run(argv)


if __name__ == "__main__":
    raise SystemExit(main())
