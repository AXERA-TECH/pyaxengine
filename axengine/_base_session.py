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
    """Configuration options for session initialization.

    Stores session-level configuration parameters used when creating
    and initializing a session instance.
    """

    pass  # Placeholder for future session configuration options


class Session(ABC):
    """Base class for inference sessions.

    Provides common interface for running model inference on Axera NPU devices.
    Supports multiple shape groups for dynamic input/output configurations.
    """

    def __init__(self) -> None:
        self._shape_count = 0
        self._inputs: list[list[NodeArg]] = []
        self._outputs: list[list[NodeArg]] = []

    def _validate_input(self, feed_input_names: dict[str, np.ndarray]) -> None:
        missing_input_names = []
        for i in self.get_inputs():
            if i.name not in feed_input_names:
                missing_input_names.append(i.name)
        if missing_input_names:
            raise ValueError(
                f"Required inputs ({missing_input_names}) are missing from input feed ({feed_input_names})."
            )

    def _validate_output(self, output_names: list[str] | None) -> None:
        if output_names is not None:
            for name in output_names:
                if name not in [o.name for o in self.get_outputs()]:
                    raise ValueError(f"Output name '{name}' is not in model outputs name list.")

    def get_inputs(self, shape_group: int = 0) -> list[NodeArg]:
        """Get input node information for the specified shape group.

        Args:
            shape_group: Index of the shape group (default: 0).

        Returns:
            List of input NodeArg objects for the shape group.

        Raises:
            ValueError: If shape_group is out of range.
        """
        if shape_group > self._shape_count:
            raise ValueError(f"Shape group '{shape_group}' is out of range, total {self._shape_count}.")
        selected_info = self._inputs[shape_group]
        return selected_info

    def get_outputs(self, shape_group: int = 0) -> list[NodeArg]:
        """Get output node information for the specified shape group.

        Args:
            shape_group: Index of the shape group (default: 0).

        Returns:
            List of output NodeArg objects for the shape group.

        Raises:
            ValueError: If shape_group is out of range.
        """
        if shape_group > self._shape_count:
            raise ValueError(f"Shape group '{shape_group}' is out of range, total {self._shape_count}.")
        selected_info = self._outputs[shape_group]
        return selected_info

    @abstractmethod
    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray], run_options: object | None = None
    ) -> list[np.ndarray]:
        """Run inference on the model.

        Args:
            output_names: Names of outputs to retrieve, or None for all outputs.
            input_feed: Dictionary mapping input names to numpy arrays.
            run_options: Optional runtime configuration.

        Returns:
            List of output numpy arrays.
        """
        pass
