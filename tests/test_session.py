import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    monkeypatch.setattr('axengine._providers.providers', ['AxEngineExecutionProvider'])


@pytest.mark.unit
class TestInferenceSession:

    def test_init_unavailable_provider(self):
        from axengine import InferenceSession
        with patch('axengine._session.get_available_providers', return_value=['AxEngineExecutionProvider']):
            with pytest.raises(ValueError, match="not available"):
                InferenceSession('model.axmodel', providers='InvalidProvider')

    def test_init_invalid_provider_type(self):
        from axengine import InferenceSession
        with patch('axengine._session.get_available_providers', return_value=['AxEngineExecutionProvider']):
            with pytest.raises(TypeError, match="Invalid provider type"):
                InferenceSession('model.axmodel', providers=[123])

    def test_init_invalid_tuple_length(self):
        from axengine import InferenceSession
        with patch('axengine._session.get_available_providers', return_value=['AxEngineExecutionProvider']):
            with pytest.raises(ValueError, match="tuple with 2 elements"):
                InferenceSession('model.axmodel', providers=[('Provider',)])

    def test_init_invalid_tuple_name_type(self):
        from axengine import InferenceSession
        with patch('axengine._session.get_available_providers', return_value=['AxEngineExecutionProvider']):
            with pytest.raises(TypeError):
                InferenceSession('model.axmodel', providers=[(123, {})])

    def test_init_invalid_tuple_dict_type(self):
        from axengine import InferenceSession
        with patch('axengine._session.get_available_providers', return_value=['AxEngineExecutionProvider']):
            with pytest.raises(TypeError):
                InferenceSession('model.axmodel', providers=[('Provider', 'not_dict')])
