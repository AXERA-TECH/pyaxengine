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
    SINGLE = 0
    DUAL = 1
    TRIPLE = 2
