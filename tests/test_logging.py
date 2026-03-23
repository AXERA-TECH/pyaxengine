import logging

import pytest


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    monkeypatch.setattr('axengine._providers.providers', ['AxEngineExecutionProvider'])


@pytest.mark.unit
class TestLogging:

    def test_get_logger_returns_logger(self):
        from axengine._logging import get_logger
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_handler(self):
        from axengine._logging import get_logger
        logger = get_logger("test_handler")
        assert len(logger.handlers) > 0

    def test_logger_default_level(self, monkeypatch):
        from axengine._logging import get_logger
        monkeypatch.delenv('AXENGINE_LOG_LEVEL', raising=False)
        logger = get_logger("test_default")
        assert logger.level == logging.INFO

    def test_logger_custom_level(self, monkeypatch):
        from axengine._logging import get_logger
        monkeypatch.setenv('AXENGINE_LOG_LEVEL', 'DEBUG')
        logger = get_logger("test_debug")
        assert logger.level == logging.DEBUG

    def test_logger_name(self):
        from axengine._logging import get_logger
        logger = get_logger("my.module")
        assert logger.name == "my.module"
