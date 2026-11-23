"""Suite-wide fixtures."""

import pytest


@pytest.fixture(autouse=True)
def _no_real_flywheel_credentials(monkeypatch, tmp_path):
    """Keep every test off the live Flywheel.

    A machine that has run `apin autoresearch register` holds a real token in the user's
    config directory (or in the environment). Without this, any test that scores a trial
    would post it to the shared record. Tests that need a token set their own.
    """
    monkeypatch.delenv("APIN_FLYWHEEL_TOKEN", raising=False)
    monkeypatch.setenv("APIN_FLYWHEEL_TOKEN_FILE", str(tmp_path / "no_flywheel_token"))
