"""Tests for stat_board.config's credentials/.env-loading logic."""

from __future__ import annotations

import os

import pytest

import stat_board.config as config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_CONFIG_DIR", raising=False)


def test_credentials_present_false_with_nothing_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(config.Path, "home", lambda: tmp_path)
    assert config.credentials_present() is False


def test_credentials_present_true_with_api_key_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert config.credentials_present() is True


def test_credentials_present_true_with_auth_token_env_var(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "test-token")
    assert config.credentials_present() is True


def test_credentials_present_true_with_a_credentials_file_on_disk(monkeypatch, tmp_path):
    creds_dir = tmp_path / "credentials"
    creds_dir.mkdir()
    (creds_dir / "default.json").write_text("{}")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    assert config.credentials_present() is True


def test_credentials_present_false_with_an_empty_credentials_dir(monkeypatch, tmp_path):
    (tmp_path / "credentials").mkdir()
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    assert config.credentials_present() is False


def test_load_env_file_sets_unset_variables_from_cwd_dotenv(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("MY_TEST_VAR=from-dotenv\n# a comment\nMY_OTHER='quoted'\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MY_TEST_VAR", raising=False)
    monkeypatch.delenv("MY_OTHER", raising=False)
    config._load_env_file()
    try:
        assert os.environ["MY_TEST_VAR"] == "from-dotenv"
        assert os.environ["MY_OTHER"] == "quoted"
    finally:
        # _load_env_file sets os.environ directly, bypassing monkeypatch's
        # tracking -- clean up by hand so these don't leak into other tests.
        monkeypatch.delenv("MY_TEST_VAR", raising=False)
        monkeypatch.delenv("MY_OTHER", raising=False)


def test_load_env_file_never_overrides_a_real_env_var(monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("MY_TEST_VAR=from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MY_TEST_VAR", "from-shell")
    config._load_env_file()
    assert os.environ["MY_TEST_VAR"] == "from-shell"
