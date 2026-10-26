#!/usr/bin/env python3
"""Launch the combined design + analysis app in its own desktop window.

    python3 run_desktop.py

Needs pywebview (pip install pywebview). To build a double-clickable .app,
run packaging/build_mac_app.sh.
"""

from __future__ import annotations

from pdstat.desktop import main

if __name__ == "__main__":
    raise SystemExit(main())
