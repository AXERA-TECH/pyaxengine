import pytest
import numpy as np


@pytest.mark.hardware
class TestHardwareIntegration:

    def test_axengine_session_creation(self):
        from axengine import InferenceSession
        sess = InferenceSession('model.axmodel', providers='AxEngineExecutionProvider')
        assert sess is not None
        assert sess.get_providers() == 'AxEngineExecutionProvider'

    def test_axclrt_session_creation(self):
        from axengine import InferenceSession
        sess = InferenceSession('model.axmodel', providers='AXCLRTExecutionProvider')
        assert sess is not None
        assert sess.get_providers() == 'AXCLRTExecutionProvider'

    def test_inference_run(self):
        from axengine import InferenceSession
        sess = InferenceSession('model.axmodel')
        inputs = sess.get_inputs()
        input_data = {inputs[0].name: np.random.randn(*inputs[0].shape).astype(np.float32)}
        outputs = sess.run(None, input_data)
        assert len(outputs) > 0

    def test_context_manager(self):
        from axengine import InferenceSession
        with InferenceSession('model.axmodel') as sess:
            inputs = sess.get_inputs()
            assert len(inputs) > 0
