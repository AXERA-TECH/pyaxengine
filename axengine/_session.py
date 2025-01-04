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
from ._node import NodeArg
from ._providers import axclrt_provider_name, axengine_provider_name
from ._providers import get_available_providers


class InferenceSession:
    def __init__(
            self,
            path_or_bytes: str | bytes | os.PathLike,
            sess_options: SessionOptions | None = None,
            providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
            provider_options: Sequence[dict[Any, Any]] | None = None, **kwargs,
    ) -> None:
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
                    assert isinstance(p, str) or isinstance(p, tuple), \
                        f"Invalid provider type: {type(p)}. Must be str or tuple."
                    if isinstance(p, str):
                        if p not in self._available_providers:
                            _unavailable_provider.append(p)
                        elif self._provider is None:
                            self._provider = p
                    if isinstance(p, tuple):
                        assert len(p) == 2, f"Invalid provider type: {p}. Must be tuple with 2 elements."
                        assert isinstance(p[0], str), f"Invalid provider type: {type(p[0])}. Must be str."
                        assert isinstance(p[1], dict), f"Invalid provider type: {type(p[1])}. Must be dict."
                        if p[0] not in self._available_providers:
                            _unavailable_provider.append(p[0])
                        elif self._provider is None:
                            self._provider = p[0]
                            # FIXME: check provider options
                            self._provider_options = p[1]
                if _unavailable_provider:
                    if self._provider is None:
                        raise ValueError(f"Selected provider(s): {_unavailable_provider} is(are) not available.")
                    else:
                        print(f"[WARNING] Selected provider(s): {_unavailable_provider} is(are) not available.")

        # FIXME: can we remove this check?
        if self._provider is None:
            raise ValueError(f"No available provider found in {providers}.")
        print(f"[INFO] Using provider: {self._provider}")

        if self._provider == axclrt_provider_name:
            from ._axclrt import AXCLRTSession
            self._sess = AXCLRTSession(path_or_bytes, sess_options, provider_options, **kwargs)
        if self._provider == axengine_provider_name:
            from ._axe import AXEngineSession
            self._sess = AXEngineSession(path_or_bytes, sess_options, provider_options, **kwargs)
        if self._sess is None:
            raise RuntimeError(f"Create session failed with provider: {self._provider}")

    # add to support 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # not suppress exceptions
        return False

    def get_session_options(self):
        """
        Return the session options. See :class:`axengine.SessionOptions`.
        """
        return self._sess_options

    def get_providers(self):
        """
        Return list of registered execution providers.
        """
        return self._provider

    def get_inputs(self, shape_group: int = 0) -> list[NodeArg]:
        return self._sess.get_inputs(shape_group)

    def get_outputs(self, shape_group: int = 0) -> list[NodeArg]:
        return self._sess.get_outputs(shape_group)

    def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, np.ndarray],
            run_options=None
    ) -> list[np.ndarray]:
        return self._sess.run(output_names, input_feed, run_options)
