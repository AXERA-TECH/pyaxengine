# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import atexit
import os
from typing import Any, Sequence

import ml_dtypes as mldt
import numpy as np

from ._axe_capi import sys_lib, engine_cffi, engine_lib
from ._axe_types import VNPUType, ModelType, ChipType
from ._base_session import Session, SessionOptions
from ._node import NodeArg

__all__: ["AXEngineSession"]

_is_sys_initialized = False
_is_engine_initialized = False


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


def _check_cffi_func_exists(lib, func_name):
    try:
        getattr(lib, func_name)
        return True
    except AttributeError:
        return False


def _get_chip_type():
    if not _check_cffi_func_exists(engine_lib, "AX_ENGINE_SetAffinity"):
        return ChipType.M57H
    elif not _check_cffi_func_exists(engine_lib, "AX_ENGINE_GetTotalOps"):
        return ChipType.MC50
    else:
        return ChipType.MC20E


def _get_version():
    engine_version = engine_lib.AX_ENGINE_GetVersion()
    return engine_cffi.string(engine_version).decode("utf-8")


def _get_vnpu_type() -> VNPUType:
    vnpu_type = engine_cffi.new("AX_ENGINE_NPU_ATTR_T *")
    ret = engine_lib.AX_ENGINE_GetVNPUAttr(vnpu_type)
    if 0 != ret:
        raise RuntimeError("Failed to get VNPU attribute.")
    return VNPUType(vnpu_type.eHardMode)


def _initialize_engine():
    global _is_sys_initialized, _is_engine_initialized

    ret = sys_lib.AX_SYS_Init()
    if ret != 0:
        raise RuntimeError("Failed to initialize ax sys.")
    _is_sys_initialized = True

    # disabled mode by default
    vnpu_type = engine_cffi.new("AX_ENGINE_NPU_ATTR_T *")
    ret = engine_lib.AX_ENGINE_GetVNPUAttr(vnpu_type)
    if 0 != ret:
        # this means the NPU was not initialized
        vnpu_type.eHardMode = engine_cffi.cast(
            "AX_ENGINE_NPU_MODE_T", VNPUType.DISABLED.value
        )
    ret = engine_lib.AX_ENGINE_Init(vnpu_type)
    if ret != 0:
        raise RuntimeError("Failed to initialize ax sys engine.")
    _is_engine_initialized = True

    print(f"[INFO] Chip type: {_get_chip_type()}")
    print(f"[INFO] VNPU type: {_get_vnpu_type()}")
    print(f"[INFO] Engine version: {_get_version()}")


def _finalize_engine():
    global _is_sys_initialized, _is_engine_initialized

    if _is_engine_initialized:
        engine_lib.AX_ENGINE_Deinit()
    if _is_sys_initialized:
        sys_lib.AX_SYS_Deinit()


_initialize_engine()
atexit.register(_finalize_engine)


class AXEngineSession(Session):
    def __init__(
            self,
            path_or_bytes: str | bytes | os.PathLike,
            sess_options: SessionOptions | None = None,
            provider_options: dict[Any, Any] | None = None,
            **kwargs,
    ) -> None:
        super().__init__()

        self._chip_type = _get_chip_type()
        self._vnpu_type = _get_vnpu_type()

        # handle, context, info, io
        self._handle = engine_cffi.new("uint64_t **")
        self._context = engine_cffi.new("uint64_t **")
        self._io = engine_cffi.new("AX_ENGINE_IO_T *")

        # model buffer, almost copied from onnx runtime
        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model_name = os.path.splitext(os.path.basename(path_or_bytes))[0]
            with open(path_or_bytes, "rb") as f:
                data = f.read()
            self._model_buffer = engine_cffi.new("char[]", data)
            self._model_buffer_size = len(data)
        elif isinstance(path_or_bytes, bytes):
            self._model_buffer = engine_cffi.new("char[]", path_or_bytes)
            self._model_buffer_size = len(path_or_bytes)
        else:
            raise TypeError(f"Unable to load model from type '{type(path_or_bytes)}'")

        # get model type
        self._model_type = self._get_model_type()
        if self._chip_type is ChipType.MC20E:
            if self._model_type is ModelType.FULL:
                print(f"[INFO] Model type: {self._model_type.value} (full core)")
            if self._model_type is ModelType.HALF:
                print(f"[INFO] Model type: {self._model_type.value} (half core)")
        if self._chip_type is ChipType.MC50:
            if self._model_type is ModelType.SINGLE:
                print(f"[INFO] Model type: {self._model_type.value} (single core)")
            if self._model_type is ModelType.DUAL:
                print(f"[INFO] Model type: {self._model_type.value} (dual core)")
            if self._model_type is ModelType.TRIPLE:
                print(f"[INFO] Model type: {self._model_type.value} (triple core)")
        if self._chip_type is ChipType.M57H:
            print(f"[INFO] Model type: {self._model_type.value} (single core)")

        # check model type
        if self._chip_type is ChipType.MC50:
            # all types (single or dual or triple) of model are allowed in vnpu mode disabled
            # only single core model is allowed in vnpu mode enabled
            # only triple core model is NOT allowed in vnpu mode big-little or little-big
            if self._vnpu_type is VNPUType.ENABLED:
                if self._model_type is not ModelType.SINGLE:
                    raise ValueError(
                        f"Model type '{self._model_type}' is not allowed when vnpu is inited as {self._vnpu_type}."
                    )
            if (
                    self._vnpu_type is VNPUType.BIG_LITTLE
                    or self._vnpu_type is VNPUType.LITTLE_BIG
            ):
                if self._model_type is ModelType.TRIPLE:
                    raise ValueError(
                        f"Model type '{self._model_type}' is not allowed when vnpu is inited as {self._vnpu_type}."
                    )
        if self._chip_type is ChipType.MC20E:
            # all types of full or half core model are allowed in vnpu mode disabled
            # only half core model is allowed in vnpu mode enabled
            if self._vnpu_type is VNPUType.ENABLED:
                if self._model_type is ModelType.FULL:
                    raise ValueError(
                        f"Model type '{self._model_type}' is not allowed when vnpu is inited as {self._vnpu_type}."
                    )
        # if self._chip_type is ChipType.M57H:
        # there only one type of model will be compiled, so no need to check

        # load model
        ret = self._load()
        if 0 != ret:
            raise RuntimeError("Failed to load model.")
        print(f"[INFO] Compiler version: {self._get_model_tool_version()}")

        # get shape group count
        try:
            self._shape_count = self._get_shape_count()
        except AttributeError as e:
            print(f"[WARNING] {e}")
            self._shape_count = 1

        # get model shape
        self._info = self._get_info()
        self._inputs = self._get_inputs()
        self._outputs = self._get_outputs()

        # fill model io
        self._align = 128
        self._cmm_token = engine_cffi.new("AX_S8[]", b"PyEngine")
        self._io[0].nInputSize = len(self.get_inputs())
        self._io[0].nOutputSize = len(self.get_outputs())
        self._io[0].pInputs = engine_cffi.new(
            "AX_ENGINE_IO_BUFFER_T[{}]".format(self._io[0].nInputSize)
        )
        self._io[0].pOutputs = engine_cffi.new(
            "AX_ENGINE_IO_BUFFER_T[{}]".format(self._io[0].nOutputSize)
        )
        for i in range(len(self.get_inputs())):
            max_buf = 0
            for j in range(self._shape_count):
                max_buf = max(max_buf, self._info[j][0].pInputs[i].nSize)
            self._io[0].pInputs[i].nSize = max_buf
            phy = engine_cffi.new("AX_U64*")
            vir = engine_cffi.new("AX_VOID**")
            ret = sys_lib.AX_SYS_MemAllocCached(
                phy, vir, self._io[0].pInputs[i].nSize, self._align, self._cmm_token
            )
            if 0 != ret:
                raise RuntimeError("Failed to allocate memory for input.")
            self._io[0].pInputs[i].phyAddr = phy[0]
            self._io[0].pInputs[i].pVirAddr = vir[0]
        for i in range(len(self.get_outputs())):
            max_buf = 0
            for j in range(self._shape_count):
                max_buf = max(max_buf, self._info[j][0].pOutputs[i].nSize)
            self._io[0].pOutputs[i].nSize = max_buf
            phy = engine_cffi.new("AX_U64*")
            vir = engine_cffi.new("AX_VOID**")
            ret = sys_lib.AX_SYS_MemAllocCached(
                phy, vir, self._io[0].pOutputs[i].nSize, self._align, self._cmm_token
            )
            if 0 != ret:
                raise RuntimeError("Failed to allocate memory for output.")
            self._io[0].pOutputs[i].phyAddr = phy[0]
            self._io[0].pOutputs[i].pVirAddr = vir[0]

    def __del__(self):
        self._unload()

    def _get_model_type(self) -> ModelType:
        model_type = engine_cffi.new("AX_ENGINE_MODEL_TYPE_T *")
        ret = engine_lib.AX_ENGINE_GetModelType(
            self._model_buffer, self._model_buffer_size, model_type
        )
        if 0 != ret:
            raise RuntimeError("Failed to get model type.")
        return ModelType(model_type[0])

    def _get_model_tool_version(self):
        model_tool_version = engine_lib.AX_ENGINE_GetModelToolsVersion(
            self._handle[0]
        )
        return engine_cffi.string(model_tool_version).decode("utf-8")

    def _load(self):
        extra = engine_cffi.new("AX_ENGINE_HANDLE_EXTRA_T *")
        extra_name = engine_cffi.new("char[]", self._model_name.encode("utf-8"))
        extra.pName = extra_name

        # for onnx runtime do not support one model multiple context running in multi-thread as far as I know, so
        # the engine handle and context will create only once
        ret = engine_lib.AX_ENGINE_CreateHandleV2(
            self._handle, self._model_buffer, self._model_buffer_size, extra
        )
        if 0 == ret:
            ret = engine_lib.AX_ENGINE_CreateContextV2(
                self._handle[0], self._context
            )
        return ret

    def _get_info(self):
        total_info = []
        if 1 == self._shape_count:
            info = engine_cffi.new("AX_ENGINE_IO_INFO_T **")
            ret = engine_lib.AX_ENGINE_GetIOInfo(self._handle[0], info)
            if 0 != ret:
                raise RuntimeError("Failed to get model shape.")
            total_info.append(info)
        else:
            for i in range(self._shape_count):
                info = engine_cffi.new("AX_ENGINE_IO_INFO_T **")
                ret = engine_lib.AX_ENGINE_GetGroupIOInfo(
                    self._handle[0], i, info
                )
                if 0 != ret:
                    raise RuntimeError(f"Failed to get model the {i}th shape.")
                total_info.append(info)
        return total_info

    def _get_shape_count(self):
        count = engine_cffi.new("AX_U32 *")
        ret = engine_lib.AX_ENGINE_GetGroupIOInfoCount(self._handle[0], count)
        if 0 != ret:
            raise RuntimeError("Failed to get model shape group.")
        return count[0]

    def _unload(self):
        if self._handle[0] is not None:
            engine_lib.AX_ENGINE_DestroyHandle(self._handle[0])
        self._handle[0] = engine_cffi.NULL

    def _get_io(self, io_type: str):
        io_info = []
        for group in range(self._shape_count):
            one_group_io = []
            for index in range(getattr(self._info[group][0], f'n{io_type}Size')):
                current_io = getattr(self._info[group][0], f'p{io_type}s')[index]
                name = engine_cffi.string(current_io.pName).decode("utf-8")
                shape = [current_io.pShape[i] for i in range(current_io.nShapeSize)]
                dtype = _transform_dtype(current_io.eDataType)
                meta = NodeArg(name, dtype, shape)
                one_group_io.append(meta)
            io_info.append(one_group_io)
        return io_info

    def _get_inputs(self):
        return self._get_io('Input')

    def _get_outputs(self):
        return self._get_io('Output')

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
        for key, npy in input_feed.items():
            for i, one in enumerate(self.get_inputs()):
                if one.name == key:
                    assert (
                            list(one.shape) == list(npy.shape) and one.dtype == npy.dtype
                    ), f"model inputs({key}) expect shape {one.shape} and dtype {one.dtype}, however gets input with shape {npy.shape} and dtype {npy.dtype}"

                    if not (
                            not npy.flags.c_contiguous
                            and npy.flags.f_contiguous
                            and npy.flags.contiguous
                    ):
                        npy = np.ascontiguousarray(npy)
                    npy_ptr = engine_cffi.cast("void *", npy.ctypes.data)

                    engine_cffi.memmove(
                        self._io[0].pInputs[i].pVirAddr, npy_ptr, npy.nbytes
                    )
                    sys_lib.AX_SYS_MflushCache(
                        self._io[0].pInputs[i].phyAddr,
                        self._io[0].pInputs[i].pVirAddr,
                        self._io[0].pInputs[i].nSize,
                    )
                    break

        # execute model
        ret = engine_lib.AX_ENGINE_RunSyncV2(
            self._handle[0], self._context[0], self._io
        )

        # flush output
        outputs = []
        if 0 == ret:
            for i in range(len(self.get_outputs())):
                sys_lib.AX_SYS_MinvalidateCache(
                    self._io[0].pOutputs[i].phyAddr,
                    self._io[0].pOutputs[i].pVirAddr,
                    self._io[0].pOutputs[i].nSize,
                )
                npy = np.frombuffer(
                    engine_cffi.buffer(
                        self._io[0].pOutputs[i].pVirAddr, self._io[0].pOutputs[i].nSize
                    ),
                    dtype=self.get_outputs()[i].dtype,
                ).reshape(self.get_outputs()[i].shape)
                name = self.get_outputs()[i].name
                if name in output_names:
                    outputs.append(npy)
            return outputs
        else:
            raise RuntimeError("Failed to run model.")
