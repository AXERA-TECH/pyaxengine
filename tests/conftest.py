"""Pytest configuration for axengine tests."""

import sys
from unittest.mock import MagicMock, patch

import pytest

sys.modules["axengine._axe_capi"] = MagicMock()
sys.modules["axengine._axclrt_capi"] = MagicMock()

with patch("ctypes.util.find_library", return_value="libax_engine.so"):
    pass


def pytest_configure(config):
    config.addinivalue_line("markers", "hardware: tests that require AX hardware")
    config.addinivalue_line("markers", "unit: unit tests that don't require hardware")


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    monkeypatch.setattr("axengine._providers.providers", ["AxEngineExecutionProvider"])
