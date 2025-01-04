# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

from abc import ABC, abstractmethod

import numpy as np

from ._node import NodeArg


class SessionOptions:
    pass


class Session(ABC):
    def __init__(self) -> None:
        self._shape_count = 0
        self._inputs = []
        self._outputs = []

    def _validate_input(self, feed_input_names: dict[str, np.ndarray]):
        missing_input_names = []
        for i in self.get_inputs():
            if i.name not in feed_input_names:
                missing_input_names.append(i.name)
        if missing_input_names:
            raise ValueError(
                f"Required inputs ({missing_input_names}) are missing from input feed ({feed_input_names}).")

    def _validate_output(self, output_names: list[str]):
        if output_names is not None:
            for name in output_names:
                if name not in [o.name for o in self.get_outputs()]:
                    raise ValueError(f"Output name '{name}' is not in model outputs name list.")

    def get_inputs(self, shape_group: int = 0) -> list[NodeArg]:
        if shape_group > self._shape_count:
            raise ValueError(f"Shape group '{shape_group}' is out of range, total {self._shape_count}.")
        selected_info = self._inputs[shape_group]
        return selected_info

    def get_outputs(self, shape_group: int = 0) -> list[NodeArg]:
        if shape_group > self._shape_count:
            raise ValueError(f"Shape group '{shape_group}' is out of range, total {self._shape_count}.")
        selected_info = self._outputs[shape_group]
        return selected_info

    @abstractmethod
    def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
            run_options=None
    ) -> list[np.ndarray]:
        pass
