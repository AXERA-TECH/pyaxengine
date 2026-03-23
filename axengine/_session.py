# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import os
from typing import Any, Sequence

import numpy as np

from ._base_session import SessionOptions
from ._logging import get_logger
from ._node import NodeArg
from ._providers import axclrt_provider_name, axengine_provider_name
from ._providers import get_available_providers

logger = get_logger(__name__)


class InferenceSession:
    """Inference session for running ONNX models on NPU.

    This class provides a high-level interface for loading and running ONNX models
    on Axera NPU devices. It supports multiple execution providers and is compatible
    with ONNXRuntime API.

    Attributes:
        _sess: Internal session object for the selected provider.
        _provider: Name of the execution provider being used.
        _provider_options: Configuration options for the provider.
    """

    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
        provider_options: Sequence[dict[Any, Any]] | None = None,
        **kwargs,
    ) -> None:
        """Initialize an InferenceSession.

        Args:
            path_or_bytes: Path to the ONNX model file or model bytes.
            sess_options: Session configuration options. Defaults to None.
            providers: Execution provider(s) to use. Can be a string for single provider
                or list of strings/tuples for multiple providers. Defaults to None (uses first available).
            provider_options: Provider-specific configuration options. Defaults to None.
            **kwargs: Additional arguments passed to the provider session.

        Raises:
            ValueError: If selected provider is not available or no valid provider found.
            TypeError: If provider format is invalid.
            RuntimeError: If session creation fails.
        """
        self._sess = None
        self._sess_options = sess_options
        self._provider = None
        self._provider_options = None
        self._available_providers = get_available_providers()

        # the providers should be available at least one, checked in __init__.py
        if providers is None:
            # using first available provider as default
            _provider_name = self._available_providers[0]
            self._provider = _provider_name
        else:
            # if only one provider is specified
            if isinstance(providers, str):
                if providers not in self._available_providers:
                    raise ValueError(f"Selected provider: '{providers}' is not available.")
                self._provider = providers
            # if multiple providers are specified, using the first one as default
            elif isinstance(providers, list):
                _unavailable_provider = []
                for p in providers:
                    if not (isinstance(p, str) or isinstance(p, tuple)):
                        raise TypeError(f"Invalid provider type: {type(p)}. Must be str or tuple.")
                    if isinstance(p, str):
                        if p not in self._available_providers:
                            _unavailable_provider.append(p)
                        elif self._provider is None:
                            self._provider = p
                    elif isinstance(p, tuple):
                        if len(p) != 2:
                            raise ValueError(f"Invalid provider type: {p}. Must be tuple with 2 elements.")
                        if not isinstance(p[0], str):
                            raise TypeError(f"Invalid provider type: {type(p[0])}. Must be str.")
                        if not isinstance(p[1], dict):
                            raise TypeError(f"Invalid provider type: {type(p[1])}. Must be dict.")
                        if p[0] not in self._available_providers:
                            _unavailable_provider.append(p[0])
                        elif self._provider is None:
                            self._provider = p[0]
                            # Provider options dict is validated above (line 91-92).
                            # Provider-specific validation happens in session constructors.
                            self._provider_options = p[1]
                if _unavailable_provider:
                    if self._provider is None:
                        raise ValueError(f"Selected provider(s): {_unavailable_provider} is(are) not available.")
                    else:
                        logger.warning(f"Selected provider(s): {_unavailable_provider} is(are) not available.")

        logger.info(f"Using provider: {self._provider}")

        if self._provider == axclrt_provider_name:
            from ._axclrt import AXCLRTSession

            self._sess = AXCLRTSession(path_or_bytes, sess_options, provider_options, **kwargs)
        if self._provider == axengine_provider_name:
            from ._axe import AXEngineSession

            self._sess = AXEngineSession(path_or_bytes, sess_options, provider_options, **kwargs)
        if self._sess is None:
            raise RuntimeError(f"Create session failed with provider: {self._provider}")

    def __enter__(self):
        """Enter context manager."""
        if self._sess is not None:
            self._sess.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit context manager."""
        if self._sess is not None:
            return self._sess.__exit__(exc_type, exc_value, traceback)
        return False

    def get_session_options(self):
        """Get session options.

        Returns:
            SessionOptions: The session configuration options.
        """
        return self._sess_options

    def get_providers(self):
        """Get the execution provider name.

        Returns:
            str: Name of the registered execution provider.
        """
        return self._provider

    def get_inputs(self, shape_group: int = 0) -> list[NodeArg]:
        """Get model input node information.

        Args:
            shape_group: Shape group index. Defaults to 0.

        Returns:
            list[NodeArg]: List of input node arguments.
        """
        return self._sess.get_inputs(shape_group)

    def get_outputs(self, shape_group: int = 0) -> list[NodeArg]:
        """Get model output node information.

        Args:
            shape_group: Shape group index. Defaults to 0.

        Returns:
            list[NodeArg]: List of output node arguments.
        """
        return self._sess.get_outputs(shape_group)

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray], run_options=None, shape_group: int = 0
    ) -> list[np.ndarray]:
        """Run inference on input data.

        Args:
            output_names: Names of outputs to return. If None, returns all outputs.
            input_feed: Dictionary mapping input names to numpy arrays.
            run_options: Runtime options for execution. Defaults to None.
            shape_group: Shape group index. Defaults to 0.

        Returns:
            list[np.ndarray]: List of output arrays in the order specified by output_names.
        """
        return self._sess.run(output_names, input_feed, run_options, shape_group)
