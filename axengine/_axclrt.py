# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#
# first implementation of AXCLRTSession contributed by zylo117

import atexit
import os
import time
from typing import Any, Sequence

import ml_dtypes as mldt
import numpy as np

from ._axclrt_capi import axclrt_cffi, axclrt_lib
from ._axclrt_types import VNPUType, ModelType
from ._base_session import Session, SessionOptions
from ._node import NodeArg

__all__: ["AXCLRTSession"]

_is_axclrt_initialized = False
_is_axclrt_engine_initialized = False


def _transform_dtype(dtype):
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

def _initialize_axclrt():
    global _is_axclrt_initialized
    ret = axclrt_lib.axclInit([])
    if ret != 0:
        raise RuntimeError(f"Failed to initialize axcl runtime. {ret}.")
    _is_axclrt_initialized = True


def _finalize_axclrt():
    global _is_axclrt_initialized, _is_axclrt_engine_initialized
    if _is_axclrt_engine_initialized:
        axclrt_lib.axclrtEngineFinalize()
        _is_axclrt_engine_initialized = False
    if _is_axclrt_initialized:
        axclrt_lib.axclFinalize()
        _is_axclrt_initialized = False


_initialize_axclrt()
atexit.register(_finalize_axclrt)


def _get_vnpu_type() -> VNPUType:
    vnpu_type = axclrt_cffi.new("axclrtEngineVNpuKind *")
    ret = axclrt_lib.axclrtEngineGetVNpuKind(vnpu_type)
    if ret != 0:
        raise RuntimeError("Failed to get VNPU attribute.")
    return VNPUType(vnpu_type[0])


def _get_version():
    major, minor, patch = axclrt_cffi.new('int32_t *'), axclrt_cffi.new('int32_t *'), axclrt_cffi.new(
        'int32_t *')
    axclrt_lib.axclrtGetVersion(major, minor, patch)
    return f'{major[0]}.{minor[0]}.{patch[0]}'


class AXCLRTSession(Session):
    def __init__(
            self,
            path_or_bytes: str | bytes | os.PathLike,
            sess_options: SessionOptions | None = None,
            provider_options: dict[Any, Any] | None = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self._device_index = 0

        if provider_options is not None and "device_id" in provider_options[0]:
            self._device_index = provider_options[0].get("device_id", 0)

        lst = axclrt_cffi.new("axclrtDeviceList *")
        ret = axclrt_lib.axclrtGetDeviceList(lst)
        if ret != 0 or lst.num == 0:
            raise RuntimeError(f"Get AXCL device failed 0x{ret:08x}, find total {lst.num} device.")

        if self._device_index >= lst.num:
            raise RuntimeError(f"Device index {self._device_index} is out of range, total {lst.num} device.")

        self._device_id = lst.devices[self._device_index]
        ret = axclrt_lib.axclrtSetDevice(self._device_id)
        if ret != 0 or lst.num == 0:
            raise RuntimeError(f"Set AXCL device failed 0x{ret:08x}.")

        global _is_axclrt_engine_initialized
        vnpu_type = axclrt_cffi.cast(
            "axclrtEngineVNpuKind", VNPUType.DISABLED.value
        )
        # try to initialize NPU as disabled
        ret = axclrt_lib.axclrtEngineInit(vnpu_type)
        # if failed, try to get vnpu type
        if 0 != ret:
            vnpu = axclrt_cffi.new("axclrtEngineVNpuKind *")
            ret = axclrt_lib.axclrtEngineGetVNpuKind(vnpu)
            # if failed, that means the NPU is not available
            if ret != 0:
                raise RuntimeError(f"axclrtEngineInit as {vnpu.value} failed 0x{ret:08x}.")
            # if success, that means the NPU is already initialized as vnpu.value
            #   so the initialization is failed.
            # this means the other users maybe uninitialized the NPU suddenly
            #   and the app would be terminated unexpectedly at that moment.
            # but we can't do anything to fix this issue, just print a warning message.
            #   it because the api looks like onnxruntime, so there no window avoid this.
            # such as the life.
            else:
                print(f"[WARNING] Failed to initialize NPU as {vnpu_type}, NPU is already initialized as {vnpu.value}.")
        # initialize NPU successfully, mark the flag to ensure the engine will be finalized
        else:
            _is_axclrt_engine_initialized = True

        self.soc_name = axclrt_cffi.string(axclrt_lib.axclrtGetSocName()).decode()
        print(f"[INFO] SOC Name: {self.soc_name}")

        # model handle, context, info, io
        self._model_id = axclrt_cffi.new("uint64_t *")
        self._context_id = axclrt_cffi.new("uint64_t *")

        # get vnpu type
        self._vnpu_type = _get_vnpu_type()
        print(f"[INFO] VNPU type: {self._vnpu_type}")

        # load model
        ret = self._load(path_or_bytes)
        if 0 != ret:
            raise RuntimeError("Failed to load model.")
        print(f"[INFO] Compiler version: {self._get_model_tool_version()}")

        # get model info
        self._info = self._get_info()
        self._shape_count = self._get_shape_count()
        self._inputs = self._get_inputs()
        self._outputs = self._get_outputs()

        # prepare io
        self._io = self._prepare_io()

    def __del__(self):
        self._unload()

    def _load(self, path_or_bytes):
        # model buffer, almost copied from onnx runtime
        if isinstance(path_or_bytes, (str, os.PathLike)):
            _model_path = axclrt_cffi.new("char[]", path_or_bytes.encode('utf-8'))
            ret = axclrt_lib.axclrtEngineLoadFromFile(_model_path, self._model_id)
            if ret != 0:
                raise RuntimeError("axclrtEngineLoadFromFile failed.")
        elif isinstance(path_or_bytes, bytes):
            _model_buffer = axclrt_cffi.new("char[]", path_or_bytes)
            _model_buffer_size = len(path_or_bytes)

            dev_mem_ptr = axclrt_cffi.new('void **', axclrt_cffi.NULL)
            ret = axclrt_lib.axclrtMalloc(dev_mem_ptr, _model_buffer_size, axclrt_lib.AXCL_MEM_MALLOC_NORMAL_ONLY)
            if ret != 0:
                raise RuntimeError("axclrtMalloc failed.")

            ret = axclrt_lib.axclrtMemcpy(dev_mem_ptr[0], _model_buffer, _model_buffer_size, axclrt_lib.AXCL_MEMCPY_HOST_TO_DEVICE)
            if ret != 0:
                axclrt_lib.axclrtFree(dev_mem_ptr[0])
                raise RuntimeError("axclrtMemcpy failed.")

            ret = axclrt_lib.axclrtEngineLoadFromMem(dev_mem_ptr[0], _model_buffer_size, self._model_id)
            axclrt_lib.axclrtFree(dev_mem_ptr[0])
            if ret != 0:
                raise RuntimeError("axclrtEngineLoadFromMem failed.")
        else:
            raise TypeError(f"Unable to load model from type '{type(path_or_bytes)}'")

        ret = axclrt_lib.axclrtEngineCreateContext(self._model_id[0], self._context_id)
        if ret != 0:
            raise RuntimeError("axclrtEngineCreateContext failed")
        return ret

    def _unload(self):
        if self._io is not None:
            dev_size = axclrt_cffi.new("uint64_t *")
            dev_prt = axclrt_cffi.new("void **")
            for i in range(axclrt_lib.axclrtEngineGetNumInputs(self._info[0])):
                axclrt_lib.axclrtEngineGetInputBufferByIndex(self._io, i, dev_prt, dev_size)
                axclrt_lib.axclrtFree(dev_prt[0])
            for i in range(axclrt_lib.axclrtEngineGetNumOutputs(self._info[0])):
                axclrt_lib.axclrtEngineGetOutputBufferByIndex(self._io, i, dev_prt, dev_size)
                axclrt_lib.axclrtFree(dev_prt[0])
            axclrt_lib.axclrtEngineDestroyIO(self._io)
            self._io = None
        if self._model_id[0] is not None:
            axclrt_lib.axclrtEngineUnload(self._model_id[0])
            self._model_id[0] = 0

    def _get_model_tool_version(self):
        model_tool_version = axclrt_lib.axclrtEngineGetModelCompilerVersion(self._model_id[0])
        return axclrt_cffi.string(model_tool_version).decode()

    def _get_info(self):
        io_info = axclrt_cffi.new("axclrtEngineIOInfo *")
        ret = axclrt_lib.axclrtEngineGetIOInfo(self._model_id[0], io_info)
        if ret != 0:
            raise RuntimeError("axclrtEngineGetIOInfo failed.")
        return io_info

    def _get_shape_count(self):
        count = axclrt_cffi.new("int32_t *")
        ret = axclrt_lib.axclrtEngineGetShapeGroupsCount(self._info[0], count)
        if ret != 0:
            axclrt_lib.axclrtEngineUnload(self._model_id[0])
            raise RuntimeError("axclrtEngineGetShapeGroupsCount failed.")
        return count[0]

    def _get_inputs(self):
        inputs = []
        for group in range(self._shape_count):
            one_group_io = []
            for index in range(axclrt_lib.axclrtEngineGetNumInputs(self._info[0])):
                cffi_name = axclrt_lib.axclrtEngineGetInputNameByIndex(self._info[0], index)
                name = axclrt_cffi.string(cffi_name).decode("utf-8")

                cffi_dtype = axclrt_cffi.new("axclrtEngineDataType *")
                ret = axclrt_lib.axclrtEngineGetInputDataType(self._info[0], index, cffi_dtype)
                if ret != 0:
                    raise RuntimeError("axclrtEngineGetInputDataType failed.")
                dtype = _transform_dtype(cffi_dtype[0])

                cffi_dims = axclrt_cffi.new("axclrtEngineIODims *")
                ret = axclrt_lib.axclrtEngineGetInputDims(self._info[0], group, index, cffi_dims)
                if ret != 0:
                    raise RuntimeError("axclrtEngineGetInputDims failed.")
                shape = [cffi_dims.dims[i] for i in range(cffi_dims.dimCount)]

                meta = NodeArg(name, dtype, shape)
                one_group_io.append(meta)
            inputs.append(one_group_io)
        return inputs

    def _get_outputs(self):
        outputs = []
        for group in range(self._shape_count):
            one_group_io = []
            for index in range(axclrt_lib.axclrtEngineGetNumOutputs(self._info[0])):
                name = axclrt_lib.axclrtEngineGetOutputNameByIndex(self._info[0], index)

                cffi_dtype = axclrt_cffi.new("axclrtEngineDataType *")
                ret = axclrt_lib.axclrtEngineGetOutputDataType(self._info[0], index, cffi_dtype)
                if ret != 0:
                    raise RuntimeError("axclrtEngineGetOutputDataType failed.")
                dtype = _transform_dtype(cffi_dtype[0])

                cffi_dims = axclrt_cffi.new("axclrtEngineIODims *")
                ret = axclrt_lib.axclrtEngineGetOutputDims(self._info[0], group, index, cffi_dims)
                if ret != 0:
                    raise RuntimeError("axclrtEngineGetOutputDims failed.")
                shape = [cffi_dims.dims[i] for i in range(cffi_dims.dimCount)]

                meta = NodeArg(name, dtype, shape)
                one_group_io.append(meta)
            outputs.append(one_group_io)
        return outputs

    def _prepare_io(self):
        _io = axclrt_cffi.new("axclrtEngineIO *")
        ret = axclrt_lib.axclrtEngineCreateIO(self._info[0], _io)
        if ret != 0:
            raise RuntimeError(f"axclrtEngineCreateIO failed 0x{ret:08x}.")
        for i in range(axclrt_lib.axclrtEngineGetNumInputs(self._info[0])):
            max_size = 0
            for group in range(self._shape_count):
                size = axclrt_lib.axclrtEngineGetInputSizeByIndex(self._info[0], group, i)
                max_size = max(max_size, size)
            dev_ptr = axclrt_cffi.new("void **")
            ret = axclrt_lib.axclrtMalloc(dev_ptr, max_size, axclrt_lib.AXCL_MEM_MALLOC_NORMAL_ONLY)
            if 0 != ret or dev_ptr[0] == axclrt_cffi.NULL:
                raise RuntimeError(f"axclrtMalloc failed 0x{ret:08x} for input {i}.")
            ret = axclrt_lib.axclrtEngineSetInputBufferByIndex(_io[0], i, dev_ptr[0], max_size)
            if 0 != ret:
                raise RuntimeError(f"axclrtEngineSetInputBufferByIndex failed 0x{ret:08x} for input {i}.")
        for i in range(axclrt_lib.axclrtEngineGetNumOutputs(self._info[0])):
            max_size = 0
            for group in range(self._shape_count):
                size = axclrt_lib.axclrtEngineGetOutputSizeByIndex(self._info[0], group, i)
                max_size = max(max_size, size)
            dev_ptr = axclrt_cffi.new("void **")
            ret = axclrt_lib.axclrtMalloc(dev_ptr, max_size, axclrt_lib.AXCL_MEM_MALLOC_NORMAL_ONLY)
            if 0 != ret or dev_ptr[0] == axclrt_cffi.NULL:
                raise RuntimeError(f"axclrtMalloc failed 0x{ret:08x} for output {i}.")
            ret = axclrt_lib.axclrtEngineSetOutputBufferByIndex(_io[0], i, dev_ptr[0], max_size)
            if 0 != ret:
                raise RuntimeError(f"axclrtEngineSetOutputBufferByIndex failed 0x{ret:08x} for output {i}.")
        return _io[0]

    def run(
            self,
            output_names: list[str],
            input_feed: dict[str, np.ndarray],
            run_options=None
    ):
        self._validate_input(input_feed)
        self._validate_output(output_names)

        if None is output_names:
            output_names = [o.name for o in self.get_outputs()]

        # fill model io
        dev_prt = axclrt_cffi.new("void **")
        dev_size = axclrt_cffi.new("uint64_t *")
        for key, npy in input_feed.items():
            for i, one in enumerate(self.get_inputs()):
                if one.name == key:
                    assert (
                            list(one.shape) == list(npy.shape) and one.dtype == npy.dtype
                    ), f"model inputs({key}) expect shape {one.shape} and dtype {one.dtype}, howerver gets input with shape {npy.shape} and dtype {npy.dtype}"

                    if not (
                            not npy.flags.c_contiguous
                            and npy.flags.f_contiguous
                            and npy.flags.contiguous
                    ):
                        npy = np.ascontiguousarray(npy)
                    npy_ptr = axclrt_cffi.cast("void *", npy.ctypes.data)
                    ret = axclrt_lib.axclrtEngineGetInputBufferByIndex(self._io, i, dev_prt, dev_size)
                    if 0 != ret:
                        raise RuntimeError(f"axclrtEngineGetInputBufferByIndex failed for input {i}.")
                    ret = axclrt_lib.axclrtMemcpy(dev_prt[0], npy_ptr, npy.nbytes, axclrt_lib.AXCL_MEMCPY_HOST_TO_DEVICE)
                    if 0 != ret:
                        raise RuntimeError(f"axclrtMemcpy failed for input {i}.")

        # execute model
        ret = axclrt_lib.axclrtEngineExecute(self._model_id[0], self._context_id[0], 0, self._io)

        # get output
        outputs = []
        if 0 == ret:
            for i in range(len(self.get_outputs())):
                ret = axclrt_lib.axclrtEngineGetOutputBufferByIndex(self._io, i, dev_prt, dev_size)
                if 0 != ret:
                    raise RuntimeError(f"axclrtEngineGetOutputBufferByIndex failed for output {i}.")
                npy = np.zeros(self.get_outputs()[i].shape, dtype=self.get_outputs()[i].dtype)
                npy_ptr = axclrt_cffi.cast("void *", npy.ctypes.data)
                ret = axclrt_lib.axclrtMemcpy(npy_ptr, dev_prt[0], npy.nbytes, axclrt_lib.AXCL_MEMCPY_DEVICE_TO_HOST)
                if 0 != ret:
                    raise RuntimeError(f"axclrtMemcpy failed for output {i}.")
                name = self.get_outputs()[i].name
                if name in output_names:
                    outputs.append(npy)
            return outputs
        else:
            raise RuntimeError(f"axclrtEngineExecute failed 0x{ret:08x}")
