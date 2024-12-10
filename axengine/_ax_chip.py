# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

from . import _types
from ._ax_capi import E as _lib

__all__: ["T"]


def function_exists(lib, func_name):
    try:
        getattr(lib, func_name)
        return True
    except AttributeError:
        return False


def check_chip_type(clib):
    if not function_exists(clib, "AX_ENGINE_SetAffinity"):
        return _types.ChipType.M57H
    elif not function_exists(clib, "AX_ENGINE_GetTotalOps"):
        return _types.ChipType.MC50
    else:
        return _types.ChipType.MC20E


T = check_chip_type(_lib)
