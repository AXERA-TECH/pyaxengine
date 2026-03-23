import pytest
import numpy as np


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    monkeypatch.setattr("axengine._providers.providers", ["AxEngineExecutionProvider"])


@pytest.mark.unit
class TestSessionOptions:
    def test_creation(self):
        from axengine import SessionOptions

        opts = SessionOptions()
        assert opts is not None

    def test_is_class(self):
        from axengine import SessionOptions

        assert isinstance(SessionOptions, type)


@pytest.mark.unit
class TestSession:
    def test_initialization(self):
        from axengine._base_session import Session
        from axengine._node import NodeArg

        class MockSession(Session):
            def run(self, output_names, input_feed, run_options=None):
                return []

        sess = MockSession()
        assert sess._shape_count == 0
        assert sess._inputs == []
        assert sess._outputs == []

    def test_validate_input_success(self):
        from axengine._base_session import Session
        from axengine._node import NodeArg

        class MockSession(Session):
            def run(self, output_names, input_feed, run_options=None):
                return []

        sess = MockSession()
        sess._inputs = [[NodeArg("input1", "float32", (1, 3))]]
        feed = {"input1": np.array([1, 2, 3])}
        sess._validate_input(feed)

    def test_validate_input_missing(self):
        from axengine._base_session import Session
        from axengine._node import NodeArg

        class MockSession(Session):
            def run(self, output_names, input_feed, run_options=None):
                return []

        sess = MockSession()
        sess._inputs = [[NodeArg("input1", "float32", (1, 3))]]
        feed = {}
        with pytest.raises(ValueError, match="Required inputs"):
            sess._validate_input(feed)

    def test_validate_output_success(self):
        from axengine._base_session import Session
        from axengine._node import NodeArg

        class MockSession(Session):
            def run(self, output_names, input_feed, run_options=None):
                return []

        sess = MockSession()
        sess._outputs = [[NodeArg("output1", "float32", (1, 10))]]
        sess._validate_output(["output1"])

    def test_validate_output_invalid(self):
        from axengine._base_session import Session
        from axengine._node import NodeArg

        class MockSession(Session):
            def run(self, output_names, input_feed, run_options=None):
                return []

        sess = MockSession()
        sess._outputs = [[NodeArg("output1", "float32", (1, 10))]]
        with pytest.raises(ValueError, match="not in model outputs"):
            sess._validate_output(["invalid"])

    def test_validate_output_none(self):
        from axengine._base_session import Session
        from axengine._node import NodeArg

        class MockSession(Session):
            def run(self, output_names, input_feed, run_options=None):
                return []

        sess = MockSession()
        sess._outputs = [[NodeArg("output1", "float32", (1, 10))]]
        sess._validate_output(None)
