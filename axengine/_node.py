# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#


class NodeArg(object):
    def __init__(self, name: str, dtype: str, shape: tuple[int, ...]) -> None:
        self.name: str = name
        self.dtype: str = dtype
        self.shape: tuple[int, ...] = shape
