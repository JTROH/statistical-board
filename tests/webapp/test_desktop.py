"""The desktop launcher's setup steps. The window itself (pywebview) is not
exercised here -- it needs a GUI session."""

from __future__ import annotations

import os
import socket

from pdstat import desktop


def test_prepare_data_dir_seeds_sample_data_once_and_chdirs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seed = tmp_path / "bundle"
    (seed / "sample_data").mkdir(parents=True)
    (seed / "sample_data" / "a.csv").write_text("x\n1\n")
    data_dir = tmp_path / "Documents" / "Workbench"

    desktop.prepare_data_dir(data_dir, seed_from=seed)
    assert os.getcwd() == str(data_dir.resolve())
    assert (data_dir / "sample_data" / "a.csv").read_text() == "x\n1\n"

    # A second launch must not overwrite what the user changed.
    (data_dir / "sample_data" / "a.csv").write_text("edited")
    desktop.prepare_data_dir(data_dir, seed_from=seed)
    assert (data_dir / "sample_data" / "a.csv").read_text() == "edited"


def test_prepare_data_dir_without_seed_still_creates_the_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "empty"
    desktop.prepare_data_dir(data_dir, seed_from=tmp_path / "missing")
    assert data_dir.is_dir() and not (data_dir / "sample_data").exists()


def test_free_port_is_bindable_on_loopback():
    port = desktop.free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", port))


def test_bundle_root_from_source_is_the_repo():
    assert (desktop.bundle_root() / "sample_data").is_dir()
    assert not desktop.is_frozen()
