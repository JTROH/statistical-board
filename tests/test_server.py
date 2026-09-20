"""HTTP surface.

The contract these tests protect: a bad form comes back as a readable message
with a 400, never as a stack trace or a silently wrong design.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from doe_advisor.server import app

GOOD = {
    "factors": [
        {"name": "pH", "low": 6.8, "high": 7.2, "units": ""},
        {"name": "DO", "low": 30, "high": 60, "units": "%"},
        {"name": "temp", "low": 35, "high": 37, "units": "C"},
        {"name": "feed", "low": 2, "high": 6, "units": "% v/v per day"},
    ],
    "responses": [{"name": "titer", "units": "g/L", "target_effect": 0.5, "noise_sd": 0.25}],
    "model_order": "main",
    "max_runs": 24,
    "n_center_points": 3,
    "expected_run_losses": 2,
}


@pytest.fixture
def client():
    return TestClient(app)


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------


def test_capabilities(client):
    data = client.get("/api/capabilities").json()
    assert data["version"]
    assert isinstance(data["llm"], bool)
    assert [m["value"] for m in data["model_orders"]] == ["main", "interaction", "quadratic"]
    assert sum(data["axis_weights"].values()) == pytest.approx(1.0)


def test_presets_endpoint(client):
    presets = client.get("/api/presets").json()["presets"]
    assert any(p["id"] == "upstream_mammalian" for p in presets)


def test_static_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "doe-advisor" in response.text


# --------------------------------------------------------------------------
# Design
# --------------------------------------------------------------------------


def test_design_returns_options_and_charts(client):
    data = client.post("/api/design", json=GOOD).json()
    assert 1 <= len(data["options"]) <= 3
    assert data["n_candidates"] > len(data["options"])
    for key in ("options", "fds", "layout"):
        assert data["charts"][key].startswith("data:image/png;base64,")


def test_design_marks_exactly_one_recommendation(client):
    options = client.post("/api/design", json=GOOD).json()["options"]
    assert sum("recommended" in o["roles"] for o in options) == 1


def test_design_reports_rejected_candidates_with_reasons(client):
    data = client.post("/api/design", json={**GOOD, "model_order": "quadratic", "max_runs": 20}).json()
    assert data["rejected"]
    assert all(r["reason"] for r in data["rejected"])


def test_design_serialises_every_field_the_front_end_uses(client):
    option = client.post("/api/design", json=GOOD).json()["options"][0]
    for field in (
        "name", "roles", "n_runs", "n_center_points", "n_model_terms", "residual_df",
        "score", "sub_scores", "weights", "score_explanation", "power", "worst_alias",
        "alias_statements", "i_value", "g_efficiency", "d_efficiency",
        "robustness_applicable", "robustness", "exceeds_declared_range", "detail",
    ):
        assert field in option, f"missing {field}"


def test_alias_statements_use_the_scientists_own_factor_names(client):
    """Regression: the server once passed an empty name list into the alias
    report, which happened to work only because labels were resolved earlier."""
    seven = {
        **GOOD,
        "factors": [{"name": n, "low": 0, "high": 1} for n in
                    ["pH", "DO", "temp", "feed", "agitation", "seed", "osmo"]],
        "max_runs": 20,
    }
    options = client.post("/api/design", json=seven).json()["options"]
    confounded = [o for o in options if o["worst_alias"] > 0.9]
    if confounded:
        joined = " ".join(confounded[0]["alias_statements"])
        assert any(name in joined for name in ("pH", "temp", "agitation", "feed"))


def test_score_explanation_is_returned_for_auditability(client):
    option = client.post("/api/design", json=GOOD).json()["options"][0]
    assert option["score_explanation"][-1].startswith("total =")


def test_impossible_spec_returns_no_options_not_an_error(client):
    """Nothing viable is a legitimate answer, not a failure."""
    response = client.post("/api/design", json={**GOOD, "model_order": "quadratic", "max_runs": 8})
    assert response.status_code == 200
    assert response.json()["options"] == []


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "fragment"),
    [
        ({"factors": [{"name": "pH", "low": 6, "high": 7}]}, "at least two factors"),
        ({"factors": [{"name": "pH", "low": 7, "high": 7}, {"name": "DO", "low": 1, "high": 2}]}, "not a factor"),
        ({**GOOD, "max_runs": 2}, "at least"),
        ({**GOOD, "model_order": "cubic"}, "Unknown model type"),
    ],
)
def test_bad_forms_return_readable_messages(client, payload, fragment):
    response = client.post("/api/design", json=payload)
    assert response.status_code == 400
    assert fragment in response.json()["error"]


def test_intake_extract_without_a_key_is_a_readable_400(client, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    response = client.post("/api/intake/extract", json={"text": "screen pH and DO"})
    assert response.status_code == 400
    assert "ANTHROPIC_API_KEY" in response.json()["error"]


# --------------------------------------------------------------------------
# Memo
# --------------------------------------------------------------------------


def test_memo_returns_a_pdf(client):
    response = client.post("/api/memo", json=GOOD)
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content[:5] == b"%PDF-"
    assert len(response.content) > 10_000


def test_memo_markdown_endpoint(client):
    data = client.post("/api/memo/markdown", json=GOOD).json()
    assert "## Recommendation" in data["markdown"]
    assert "## Run sheet" in data["markdown"]
    assert data["narration_source"] in ("template", "claude")


def test_memo_rejects_a_bad_form(client):
    response = client.post("/api/memo", json={"factors": []})
    assert response.status_code == 400


def test_capabilities_list_response_goals(client):
    goals = client.get("/api/capabilities").json()["response_goals"]
    assert {g["value"] for g in goals} == {"screen", "maximize", "minimize", "target"}


def test_design_returns_the_power_basis_and_unit_level_figures(client):
    data = client.post("/api/design", json=GOOD).json()
    basis = data["power_basis"]
    assert basis["name"] == "titer"
    assert basis["standardised_effect"] == 2.0
    assert basis["goal"] == "screen"
    for option in data["options"]:
        assert option["detectable_effect_units"] > 0
        assert 0 <= option["power_if_noise_1_5x"] <= option["power"]
    assert basis["runs_for_80_power"] > 0
    assert data["charts"]["power"].startswith("data:image/png;base64,")


def test_no_power_chart_without_a_target_effect(client):
    form = {**GOOD, "responses": [{"name": "titer", "units": "g/L"}]}
    data = client.post("/api/design", json=form).json()
    assert "power" not in data["charts"]
    assert data["power_basis"]["runs_for_80_power"] is None
