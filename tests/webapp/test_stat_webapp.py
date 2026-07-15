"""Smoke tests only: the app builds and its read-only GET routes respond. SSE
streaming and POST /api/runs (which kicks off a real orchestrator run) are
exercised end-to-end by tests/stat_board/test_orchestrator.py instead -- see
that file's --dry-run coverage."""

from __future__ import annotations

from fastapi.testclient import TestClient

from stat_board.webapp import app

client = TestClient(app)


def test_index_serves_the_single_page_app():
    resp = client.get("/")
    assert resp.status_code == 200


def test_health_reports_model_and_credentials_status():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "credentials" in resp.json()


def test_reports_endpoint_lists_the_reports_directory():
    resp = client.get("/api/reports")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_reports_endpoint_does_not_double_count_the_transcript_pdf(tmp_path, monkeypatch):
    from stat_board import webapp

    monkeypatch.setattr(webapp, "REPORTS_DIR", tmp_path)
    (tmp_path / "foo.pdf").write_bytes(b"%PDF-fake")
    (tmp_path / "foo.transcript.json").write_text("{}")
    (tmp_path / "foo.transcript.md").write_text("# Transcript\n")
    (tmp_path / "foo.transcript.pdf").write_bytes(b"%PDF-fake")

    resp = client.get("/api/reports")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1  # not 2 -- the transcript.pdf isn't its own report row
    assert items[0]["name"] == "foo.pdf"
    assert items[0]["transcript"] == "foo.transcript.json"
    assert items[0]["transcript_pdf"] == "foo.transcript.pdf"
    assert items[0]["transcript_md"] == "foo.transcript.md"


def test_columns_endpoint_inspects_a_real_dataset(long_csv):
    resp = client.get("/api/columns", params={"data": str(long_csv)})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body["columns"]) == {"group", "value"}


def test_columns_endpoint_404s_on_a_missing_file(tmp_path):
    resp = client.get("/api/columns", params={"data": str(tmp_path / "nope.csv")})
    assert resp.status_code == 400


def test_start_run_requires_a_question():
    resp = client.post("/api/runs", json={"question": "", "data": "sample_data/long.csv"})
    assert resp.status_code == 400


def test_start_run_requires_the_data_file_to_exist(tmp_path):
    resp = client.post("/api/runs", json={"question": "Q?", "data": str(tmp_path / "nope.csv")})
    assert resp.status_code == 400
