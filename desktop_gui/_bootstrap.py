"""Makes ``stat_board`` importable from a script run outside the package.

Shared by ``statistical_analysis.py`` and ``recommend.py`` so neither has to
assume the other already fixed up ``sys.path`` (and both are no-ops under
pytest, where the repo root is already on the path).
"""

import os
import sys
from pathlib import Path


def ensure_stat_board_importable() -> bool:
    """Insert the repo root onto ``sys.path`` if ``stat_board`` isn't already
    importable. Idempotent. Returns whether ``stat_board.engine`` is importable
    afterward. Honors ``STAT_BOARD_HOME`` if set."""
    try:
        import stat_board.engine  # noqa: F401
        return True
    except ModuleNotFoundError:
        pass

    candidates = []
    if os.environ.get("STAT_BOARD_HOME"):
        candidates.append(Path(os.environ["STAT_BOARD_HOME"]))
    candidates += list(Path(__file__).resolve().parents)  # repo root (bundled)

    for cand in candidates:
        if cand and (Path(cand) / "stat_board" / "engine").is_dir():
            sys.path.insert(0, str(cand))
            try:
                import stat_board.engine  # noqa: F401
                return True
            except ModuleNotFoundError:
                continue
    return False
