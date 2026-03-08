# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes.util as cutil
from pathlib import Path

providers = []
axengine_provider_name = 'AxEngineExecutionProvider'
axclrt_provider_name = 'AXCLRTExecutionProvider'

_axengine_lib_name = 'ax_engine'
_axclrt_lib_name = 'axcl_rt'
_axclrt_lib_dir = Path('/usr/lib/axcl')


def _lib_exists(lib_name, search_dir=None):
    if cutil.find_library(lib_name) is not None:
        return True

    if search_dir is None:
        return False

    return any(search_dir.glob(f'lib{lib_name}.so*'))

# check if axcl_rt is installed, so if available, it's the default provider
if _lib_exists(_axclrt_lib_name, _axclrt_lib_dir):
    providers.append(axclrt_provider_name)

# check if ax_engine is installed
if _lib_exists(_axengine_lib_name):
    providers.append(axengine_provider_name)


def get_all_providers():
    return [axengine_provider_name, axclrt_provider_name]


def get_available_providers():
    return providers
