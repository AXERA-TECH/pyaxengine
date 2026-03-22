"""
Characterization tests for PyAXEngine - capturing current behavior.

These tests document the current state of the API, including any bugs.
DO NOT fix bugs here - just record what currently happens.
"""
import sys
import pytest


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    """Mock providers to allow import without hardware."""
    monkeypatch.setattr('axengine._providers.providers', ['AxEngineExecutionProvider'])
    

def test_axengine_imports():
    """Test that basic axengine imports work."""
    import axengine
    from axengine import InferenceSession, NodeArg, SessionOptions
    assert axengine is not None
    assert InferenceSession is not None
    assert NodeArg is not None
    assert SessionOptions is not None


def test_node_arg_creation():
    """Test NodeArg can be instantiated."""
    from axengine import NodeArg
    node = NodeArg(name="test", dtype="float32", shape=(1, 3, 224, 224))
    assert node.name == "test"
    assert node.dtype == "float32"
    assert node.shape == (1, 3, 224, 224)


def test_session_options_creation():
    """Test SessionOptions can be instantiated."""
    from axengine import SessionOptions
    opts = SessionOptions()
    assert opts is not None


def test_inference_session_signature():
    """Test InferenceSession has expected __init__ signature."""
    import inspect
    from axengine import InferenceSession
    sig = inspect.signature(InferenceSession.__init__)
    params = list(sig.parameters.keys())
    assert 'self' in params
    assert 'path_or_bytes' in params
    assert 'sess_options' in params
    assert 'providers' in params


def test_node_arg_attributes():
    """Test NodeArg has expected attributes."""
    from axengine import NodeArg
    node = NodeArg(name="input", dtype="uint8", shape=(1, 224, 224, 3))
    assert hasattr(node, 'name')
    assert hasattr(node, 'dtype')
    assert hasattr(node, 'shape')
