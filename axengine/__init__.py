# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

# thanks to community contributors list below:
#   zylo117: https://github.com/zylo117, first implementation of the axclrt backend

from ._providers import axengine_provider_name, axclrt_provider_name
from ._providers import get_all_providers, get_available_providers

# check if axclrt is installed, or is a supported chip(e.g. AX650, AX620E etc.)
_available_providers = get_available_providers()
if not _available_providers:
    raise ImportError(
        f"No providers found. Please make sure you have installed one of the following: {get_all_providers()}")
print("[INFO] Available providers: ", _available_providers)

from ._node import NodeArg
from ._session import SessionOptions, InferenceSession
