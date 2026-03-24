import pytest


@pytest.fixture(autouse=True)
def mock_providers(monkeypatch):
    monkeypatch.setattr("axengine._providers.providers", ["AxEngineExecutionProvider"])


@pytest.mark.unit
class TestNodeArg:
    def test_creation(self):
        from axengine import NodeArg

        node = NodeArg(name="input", dtype="float32", shape=(1, 3, 224, 224))
        assert node.name == "input"
        assert node.dtype == "float32"
        assert node.shape == (1, 3, 224, 224)

    def test_different_dtypes(self):
        from axengine import NodeArg

        dtypes = ["uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "bfloat16"]
        for dtype in dtypes:
            node = NodeArg(name="test", dtype=dtype, shape=(1,))
            assert node.dtype == dtype

    def test_different_shapes(self):
        from axengine import NodeArg

        shapes = [(1,), (1, 3), (1, 3, 224), (1, 3, 224, 224), (2, 4, 8, 16, 32)]
        for shape in shapes:
            node = NodeArg(name="test", dtype="float32", shape=shape)
            assert node.shape == shape

    def test_empty_name(self):
        from axengine import NodeArg

        node = NodeArg(name="", dtype="float32", shape=(1,))
        assert node.name == ""

    def test_attributes_mutable(self):
        from axengine import NodeArg

        node = NodeArg(name="input", dtype="float32", shape=(1, 3, 224, 224))
        node.name = "output"
        node.dtype = "int8"
        node.shape = (1, 10)
        assert node.name == "output"
        assert node.dtype == "int8"
        assert node.shape == (1, 10)
