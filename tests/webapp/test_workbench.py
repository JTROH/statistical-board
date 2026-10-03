"""The combined app mounts both halves, and each half's page only uses
relative API URLs -- an absolute "/api/..." would silently hit the wrong app
once mounted under /design/ or /analyse/."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from pdstat.webapp import app

client = TestClient(app)
ROOT = Path(__file__).resolve().parents[2]


def test_shell_page_loads_both_tabs():
    resp = client.get("/")
    assert resp.status_code == 200
    assert 'src="/design/"' in resp.text and 'src="/analyse/"' in resp.text


def test_design_half_is_mounted():
    assert client.get("/design/").status_code == 200
    assert "model_orders" in client.get("/design/api/capabilities").json()


def test_analysis_half_is_mounted():
    assert client.get("/analyse/").status_code == 200
    assert "credentials" in client.get("/analyse/api/health").json()


def test_bare_mount_paths_redirect_to_the_trailing_slash():
    for path in ("/design", "/analyse"):
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert resp.headers["location"].endswith(path + "/")


def test_front_ends_use_only_relative_api_urls():
    for rel in ("doe_advisor/static/app.js", "doe_advisor/static/index.html", "stat_board/static/index.html"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        assert not re.search(r"""["'`]/api/""", text), rel
