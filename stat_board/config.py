"""Runtime configuration for the standalone statistical board, read from the
environment with sensible defaults. (The Claude Code skill version needs none of
this — it runs on your Claude Code session. This is for `python -m stat_board`.)
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_env_file() -> None:
    """Best-effort `.env` loader so the web UI (which may launch outside a shell
    with exports) can still find ANTHROPIC_API_KEY and STAT_* settings. Real
    environment variables always win; lines are simple KEY=VALUE."""
    project_root = Path(__file__).resolve().parent.parent
    path = next((p for p in (Path(".env"), project_root / ".env") if p.is_file()), None)
    if path is None:
        return
    try:
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


_load_env_file()


def credentials_present() -> bool:
    """True if the Anthropic SDK will be able to authenticate."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    cfg_dir = Path(os.environ.get("ANTHROPIC_CONFIG_DIR",
                                  Path.home() / ".config" / "anthropic"))
    creds = cfg_dir / "credentials"
    return creds.is_dir() and any(creds.glob("*.json"))


# The model every role runs on. The board uses no server-side web tools, so any
# modern Claude model works; Opus 4.8 is the default for the sharpest reasoning.
MODEL: str = os.environ.get("STAT_MODEL", "claude-opus-4-8")

# Effort for output_config: low | medium | high | max.
EFFORT: str = os.environ.get("STAT_EFFORT", "high")

# Max judge-led rounds before the report is forced to finalize. Statistics tend
# to converge fast, so the default is intentionally low.
MAX_ROUNDS: int = int(os.environ.get("STAT_MAX_ROUNDS", "2"))

# Per-call output ceilings (we always stream, so large values don't risk a timeout).
MAX_TOKENS: int = int(os.environ.get("STAT_MAX_TOKENS", "16000"))
JUDGE_MAX_TOKENS: int = int(os.environ.get("STAT_JUDGE_MAX_TOKENS", "24000"))

# Cap on how many engine calls one analyst/verifier turn may make (safety valve).
MAX_TOOL_CALLS: int = int(os.environ.get("STAT_MAX_TOOL_CALLS", "24"))
