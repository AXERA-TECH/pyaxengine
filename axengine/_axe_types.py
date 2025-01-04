# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

from enum import Enum


class VNPUType(Enum):
    DISABLED = 0
    ENABLED = 1
    BIG_LITTLE = 2
    LITTLE_BIG = 3


class ModelType(Enum):
    HALF = 0  # for MC20E, which means chip is AX630C(x), or AX620Q(x)
    FULL = 1  # for MC20E
    SINGLE = 0  # for MC50, which means chip is AX650A or AX650N, and M57H
    DUAL = 1  # for MC50
    TRIPLE = 2  # for MC50


class ChipType(Enum):
    MC20E = 0
    MC50 = 1
    M57H = 2
