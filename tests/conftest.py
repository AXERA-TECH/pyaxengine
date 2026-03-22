"""Pytest configuration for axengine tests."""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "hardware: tests that require AX hardware"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests that don't require hardware"
    )
