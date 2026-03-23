import pytest


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    monkeypatch.setattr('axengine._providers.providers', ['AxEngineExecutionProvider'])


@pytest.mark.unit
class TestProviders:

    def test_get_all_providers(self):
        from axengine._providers import get_all_providers, axengine_provider_name, axclrt_provider_name
        providers = get_all_providers()
        assert isinstance(providers, list)
        assert axengine_provider_name in providers
        assert axclrt_provider_name in providers
        assert len(providers) == 2

    def test_get_available_providers(self):
        from axengine._providers import get_available_providers
        providers = get_available_providers()
        assert isinstance(providers, list)

    def test_provider_names(self):
        from axengine._providers import axengine_provider_name, axclrt_provider_name
        assert axengine_provider_name == "AxEngineExecutionProvider"
        assert axclrt_provider_name == "AXCLRTExecutionProvider"
