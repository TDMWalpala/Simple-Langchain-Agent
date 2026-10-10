"""Tests for :mod:`simple_agent.config` — env validation and loading."""

from __future__ import annotations

import os

import pytest

from simple_agent.config import REQUIRED_KEYS, load_config, missing_keys, require_keys


def test_required_keys_cover_every_api_the_project_uses() -> None:
    assert REQUIRED_KEYS == ("GROQ_API_KEY", "TAVILY_API_KEY", "WEATHER_API_KEY")


def test_missing_keys_returns_empty_when_all_set(fake_env: dict[str, str]) -> None:
    assert missing_keys() == []


@pytest.mark.parametrize("key", REQUIRED_KEYS)
def test_missing_keys_detects_each_unset_key(
    fake_env: dict[str, str], monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    monkeypatch.delenv(key)
    assert missing_keys() == [key]


def test_missing_keys_treats_empty_string_as_missing(
    fake_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert missing_keys() == ["GROQ_API_KEY"]


def test_require_keys_passes_when_all_set(fake_env: dict[str, str]) -> None:
    require_keys(REQUIRED_KEYS)  # should not raise


def test_require_keys_lists_every_missing_key(
    monkeypatch: pytest.MonkeyPatch, no_dotenv: None
) -> None:
    for key in REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError) as excinfo:
        require_keys(REQUIRED_KEYS)
    message = str(excinfo.value)
    for key in REQUIRED_KEYS:
        assert key in message
    assert ".env.example" in message  # points the user at the fix


def test_load_config_sets_ssl_cert_file(
    fake_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    load_config()
    assert os.environ["SSL_CERT_FILE"]  # set for stdlib TLS users


def test_load_config_does_not_override_existing_ssl_cert_file(
    fake_env: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SSL_CERT_FILE", "/custom/ca.pem")
    load_config()
    assert os.environ["SSL_CERT_FILE"] == "/custom/ca.pem"


def test_load_config_raises_without_keys(
    monkeypatch: pytest.MonkeyPatch, no_dotenv: None
) -> None:
    for key in REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        load_config()
