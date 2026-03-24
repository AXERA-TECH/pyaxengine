# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# AXCL dtype helpers — kept separate from _utils so AxEngine (board) path does not
# import axcl_rt at module load time.

import ml_dtypes as mldt
import numpy as np

from ._axclrt_capi import axclrt_cffi, axclrt_lib


def _transform_dtype_axclrt(dtype):
    if dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_UINT8):
        return np.dtype(np.uint8)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_INT8):
        return np.dtype(np.int8)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_UINT16):
        return np.dtype(np.uint16)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_INT16):
        return np.dtype(np.int16)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_UINT32):
        return np.dtype(np.uint32)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_INT32):
        return np.dtype(np.int32)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_FP32):
        return np.dtype(np.float32)
    elif dtype == axclrt_cffi.cast("axclrtEngineDataType", axclrt_lib.AXCL_DATA_TYPE_BF16):
        return np.dtype(mldt.bfloat16)
    else:
        raise ValueError(f"Unsupported data type '{dtype}'.")
