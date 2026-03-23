# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

from typing import Tuple


class NodeArg(object):
    """Represents a node argument with type and shape information.

    Attributes:
        name: The name of the argument.
        dtype: The data type of the argument (e.g., 'float32', 'int64').
        shape: The shape of the argument as a tuple of integers.
    """

    def __init__(self, name: str, dtype: str, shape: Tuple[int, ...]) -> None:
        self.name: str = name
        self.dtype: str = dtype
        self.shape: Tuple[int, ...] = shape
