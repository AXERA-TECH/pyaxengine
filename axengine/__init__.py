# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

# thanks to community contributors list below:
#   zylo117: https://github.com/zylo117, first implementation of the axclrt backend

from ._logging import get_logger
from ._node import NodeArg as NodeArg
from ._providers import (
    axclrt_provider_name as axclrt_provider_name,
)
from ._providers import (
    axengine_provider_name as axengine_provider_name,
)
from ._providers import (
    get_all_providers as get_all_providers,
)
from ._providers import (
    get_available_providers,
)
from ._session import InferenceSession as InferenceSession
from ._session import SessionOptions as SessionOptions

logger = get_logger(__name__)

_available_providers = get_available_providers()

if not _available_providers:
    raise ImportError(
        "No execution providers available. Please ensure either ax_engine or axcl_rt library is installed."
    )

logger.info("Available providers: %s", _available_providers)
