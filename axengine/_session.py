from ._node import NodeArg
from ._types import VNPUType

import numpy as np

class BaseInferenceSession:
    def __init__(self, *args, **kwargs) -> None:
        self._shape_count = 0
        self._inputs = []
        self._outputs = []

    def __del__(self):
        self._final()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._final()

    def _init(self, *args, **kwargs):
        return

    def _final(self):
        return

    def _get_version(self) -> str:
        return ''

    def _get_vnpu_type(self) -> VNPUType:
        return VNPUType(0)

    def _get_model_tool_version(self) -> str:
        return ''

    def _load(self) -> 0:
        return 0

    def _get_shape_count(self) -> int:
        return 0

    def _unload(self):
        return

    def get_inputs(self, shape_group=0) -> list[NodeArg]:
        if shape_group > self._shape_count:
            raise ValueError(f"Shape group '{shape_group}' is out of range, total {self._shape_count}.")
        selected_info = self._inputs[shape_group]
        return selected_info

    def get_outputs(self, shape_group=0) -> list[NodeArg]:
        if shape_group > self._shape_count:
            raise ValueError(f"Shape group '{shape_group}' is out of range, total {self._shape_count}.")
        selected_info = self._outputs[shape_group]
        return selected_info

    # copy from onnxruntime
    def _validate_input(self, feed_input_names):
        missing_input_names = []
        for i in self.get_inputs():
            if i.name not in feed_input_names:
                missing_input_names.append(i.name)
        if missing_input_names:
            raise ValueError(
                f"Required inputs ({missing_input_names}) are missing from input feed ({feed_input_names})."
            )

    def _validate_output(self, output_names):
        if output_names is not None:
            for name in output_names:
                if name not in [o.name for o in self.get_outputs()]:
                    raise ValueError(f"Output name '{name}' is not registered.")

    def run(self, output_names, input_feed, run_options=None) -> list[np.ndarray]:
        return []
