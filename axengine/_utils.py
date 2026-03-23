import ml_dtypes as mldt
import numpy as np

from ._axclrt_capi import axclrt_cffi, axclrt_lib
from ._axe_capi import engine_cffi, engine_lib


def _transform_dtype(dtype):
    if dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_UINT8):
        return np.dtype(np.uint8)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_SINT8):
        return np.dtype(np.int8)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_UINT16):
        return np.dtype(np.uint16)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_SINT16):
        return np.dtype(np.int16)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_UINT32):
        return np.dtype(np.uint32)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_SINT32):
        return np.dtype(np.int32)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_FLOAT32):
        return np.dtype(np.float32)
    elif dtype == engine_cffi.cast("AX_ENGINE_DATA_TYPE_T", engine_lib.AX_ENGINE_DT_BFLOAT16):
        return np.dtype(mldt.bfloat16)
    else:
        raise ValueError(f"Unsupported data type '{dtype}'.")


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
