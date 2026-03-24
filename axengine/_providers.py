# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes.util as cutil
import os

providers = []
axengine_provider_name = 'AxEngineExecutionProvider'
axclrt_provider_name = 'AXCLRTExecutionProvider'

_axengine_lib_name = 'ax_engine'
_axclrt_lib_name = 'axcl_rt'

_AXCL_HOST_DEVICE = '/dev/axcl_host'

# check if axcl_rt is installed, so if available, it's the default provider
_has_axclrt_lib = cutil.find_library(_axclrt_lib_name) is not None
# check if ax_engine is installed
_has_axengine_lib = cutil.find_library(_axengine_lib_name) is not None

if _has_axclrt_lib and _has_axengine_lib:
    if os.path.exists(_AXCL_HOST_DEVICE):
        providers = [axclrt_provider_name, axengine_provider_name]
    else:
        providers = [axengine_provider_name, axclrt_provider_name]
elif _has_axclrt_lib:
    providers = [axclrt_provider_name]
elif _has_axengine_lib:
    providers = [axengine_provider_name]


def get_all_providers():
    return [axengine_provider_name, axclrt_provider_name]


def get_available_providers():
    return providers
