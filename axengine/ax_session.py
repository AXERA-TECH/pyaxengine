# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#
from ._session import BaseInferenceSession
from ._types import VNPUType, ModelType, ChipType
from ._types import _transform_dtype
from ._node import NodeArg

import os
import numpy as np

__all__: ["InferenceSession"]


class InferenceSession(BaseInferenceSession):
    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike,
    ) -> None:
        from . import _ax_capi as _capi
        from . import _ax_chip as _chip

        super().__init__()

        # load shared library
        self._sys_lib = _capi.S
        self._sys_ffi = _capi.M
        self._engine_lib = _capi.E
        self._engine_ffi = _capi.N

        # chip type
        self._chip_type = _chip.T
        print(f"[INFO] Chip type: {self._chip_type}")

        # handle, context, info, io
        self._handle = self._engine_ffi.new("uint64_t **")
        self._context = self._engine_ffi.new("uint64_t **")
        self._io = self._engine_ffi.new("AX_ENGINE_IO_T *")

        # init ax sys & engine
        ret = self._init()
        if 0 != ret:
            raise RuntimeError("Failed to initialize engine.")
        print(f"[INFO] Engine version: {self._get_version()}")

        # get vnpu type
        self._vnpu_type = self._get_vnpu_type()
        print(f"[INFO] VNPU type: {self._vnpu_type}")

        # model buffer, almost copied from onnx runtime
        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model_name = os.path.splitext(os.path.basename(path_or_bytes))[0]
            with open(path_or_bytes, "rb") as f:
                data = f.read()
            self._model_buffer = self._engine_ffi.new("char[]", data)
            self._model_buffer_size = len(data)
        elif isinstance(path_or_bytes, bytes):
            self._model_buffer = self._engine_ffi.new("char[]", path_or_bytes)
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
        self._shape_count = self._get_shape_count()

        # get model shape
        self._info = self._get_info()
        self._inputs = self._get_inputs()
        self._outputs = self._get_outputs()

        # fill model io
        self._align = 128
        self._cmm_token = self._engine_ffi.new("AX_S8[]", b"PyEngine")
        self._io[0].nInputSize = len(self.get_inputs())
        self._io[0].nOutputSize = len(self.get_outputs())
        self._io[0].pInputs = self._engine_ffi.new(
            "AX_ENGINE_IO_BUFFER_T[{}]".format(self._io[0].nInputSize)
        )
        self._io[0].pOutputs = self._engine_ffi.new(
            "AX_ENGINE_IO_BUFFER_T[{}]".format(self._io[0].nOutputSize)
        )
        for i in range(len(self.get_inputs())):
            max_buf = 0
            for j in range(self._shape_count):
                max_buf = max(max_buf, self._info[j][0].pInputs[i].nSize)
            self._io[0].pInputs[i].nSize = max_buf
            phy = self._engine_ffi.new("AX_U64*")
            vir = self._engine_ffi.new("AX_VOID**")
            ret = self._sys_lib.AX_SYS_MemAllocCached(
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
            phy = self._engine_ffi.new("AX_U64*")
            vir = self._engine_ffi.new("AX_VOID**")
            ret = self._sys_lib.AX_SYS_MemAllocCached(
                phy, vir, self._io[0].pOutputs[i].nSize, self._align, self._cmm_token
            )
            if 0 != ret:
                raise RuntimeError("Failed to allocate memory for output.")
            self._io[0].pOutputs[i].phyAddr = phy[0]
            self._io[0].pOutputs[i].pVirAddr = vir[0]

    def _init(self, vnpu=VNPUType.DISABLED):  # vnpu type, the default is disabled
        ret = self._sys_lib.AX_SYS_Init()
        if 0 != ret:
            raise RuntimeError("Failed to initialize system.")

        # get vnpu type first, check if npu was initialized
        vnpu_type = self._engine_ffi.new("AX_ENGINE_NPU_ATTR_T *")
        ret = self._engine_lib.AX_ENGINE_GetVNPUAttr(vnpu_type)
        if 0 != ret:
            # this means the NPU was not initialized
            vnpu_type.eHardMode = self._engine_ffi.cast(
                "AX_ENGINE_NPU_MODE_T", vnpu.value
            )

        return self._engine_lib.AX_ENGINE_Init(vnpu_type)

    def _final(self):
        if self._handle[0] is not None:
            self._unload()
        self._engine_lib.AX_ENGINE_Deinit()
        return self._sys_lib.AX_SYS_Deinit()

    def _get_version(self):
        engine_version = self._engine_lib.AX_ENGINE_GetVersion()
        return self._engine_ffi.string(engine_version).decode("utf-8")

    def _get_vnpu_type(self) -> VNPUType:
        vnpu_type = self._engine_ffi.new("AX_ENGINE_NPU_ATTR_T *")
        ret = self._engine_lib.AX_ENGINE_GetVNPUAttr(vnpu_type)
        if 0 != ret:
            raise RuntimeError("Failed to get VNPU attribute.")
        return VNPUType(vnpu_type.eHardMode)

    def _get_model_type(self) -> ModelType:
        model_type = self._engine_ffi.new("AX_ENGINE_MODEL_TYPE_T *")
        ret = self._engine_lib.AX_ENGINE_GetModelType(
            self._model_buffer, self._model_buffer_size, model_type
        )
        if 0 != ret:
            raise RuntimeError("Failed to get model type.")
        return ModelType(model_type[0])

    def _get_model_tool_version(self):
        model_tool_version = self._engine_lib.AX_ENGINE_GetModelToolsVersion(
            self._handle[0]
        )
        return self._engine_ffi.string(model_tool_version).decode("utf-8")

    def _load(self):
        extra = self._engine_ffi.new("AX_ENGINE_HANDLE_EXTRA_T *")
        extra_name = self._engine_ffi.new("char[]", self._model_name.encode("utf-8"))
        extra.pName = extra_name

        # for onnx runtime do not support one model multiple context running in multi-thread as far as I know, so
        # the engine handle and context will create only once
        ret = self._engine_lib.AX_ENGINE_CreateHandleV2(
            self._handle, self._model_buffer, self._model_buffer_size, extra
        )
        if 0 == ret:
            ret = self._engine_lib.AX_ENGINE_CreateContextV2(
                self._handle[0], self._context
            )
        return ret

    def _get_info(self):
        total_info = []
        if 1 == self._shape_count:
            info = self._engine_ffi.new("AX_ENGINE_IO_INFO_T **")
            ret = self._engine_lib.AX_ENGINE_GetIOInfo(self._handle[0], info)
            if 0 != ret:
                raise RuntimeError("Failed to get model shape.")
            total_info.append(info)
        else:
            for i in range(self._shape_count):
                info = self._engine_ffi.new("AX_ENGINE_IO_INFO_T **")
                ret = self._engine_lib.AX_ENGINE_GetGroupIOInfo(
                    self._handle[0], i, info
                )
                if 0 != ret:
                    raise RuntimeError(f"Failed to get model the {i}th shape.")
                total_info.append(info)
        return total_info

    def _get_shape_count(self):
        count = self._engine_ffi.new("AX_U32 *")
        ret = self._engine_lib.AX_ENGINE_GetGroupIOInfoCount(self._handle[0], count)
        if 0 != ret:
            raise RuntimeError("Failed to get model shape group.")
        return count[0]

    def _unload(self):
        return self._engine_lib.AX_ENGINE_DestroyHandle(self._handle[0])

    def _get_inputs(self):
        inputs = []
        for group in range(self._shape_count):
            one_group_input = []
            for index in range(self._info[group][0].nInputSize):
                current_input = self._info[group][0].pInputs[index]
                name = self._engine_ffi.string(current_input.pName).decode("utf-8")
                shape = []
                for i in range(current_input.nShapeSize):
                    shape.append(current_input.pShape[i])
                dtype = _transform_dtype(
                    self._engine_ffi, self._engine_lib, current_input.eDataType
                )
                meta = NodeArg(name, dtype, shape)
                one_group_input.append(meta)
            inputs.append(one_group_input)
        return inputs

    def _get_outputs(self):
        outputs = []
        for group in range(self._shape_count):
            one_group_output = []
            for index in range(self._info[group][0].nOutputSize):
                current_output = self._info[group][0].pOutputs[index]
                name = self._engine_ffi.string(current_output.pName).decode("utf-8")
                shape = []
                for i in range(current_output.nShapeSize):
                    shape.append(current_output.pShape[i])
                dtype = _transform_dtype(
                    self._engine_ffi, self._engine_lib, current_output.eDataType
                )
                meta = NodeArg(name, dtype, shape)
                one_group_output.append(meta)
            outputs.append(one_group_output)
        return outputs

    def run(self, output_names, input_feed, run_options=None):
        self._validate_input(list(input_feed.keys()))
        self._validate_output(output_names)

        if None is output_names:
            output_names = [o.name for o in self.get_outputs()]

        # fill model io
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
                    npy_ptr = self._engine_ffi.cast("void *", npy.ctypes.data)
                    self._engine_ffi.memmove(
                        self._io[0].pInputs[i].pVirAddr, npy_ptr, npy.nbytes
                    )
                    self._sys_lib.AX_SYS_MflushCache(
                        self._io[0].pInputs[i].phyAddr,
                        self._io[0].pInputs[i].pVirAddr,
                        self._io[0].pInputs[i].nSize,
                    )
                    break

        # execute model
        ret = self._engine_lib.AX_ENGINE_RunSyncV2(
            self._handle[0], self._context[0], self._io
        )

        # flush output
        outputs = []
        if 0 == ret:
            for i in range(len(self.get_outputs())):
                self._sys_lib.AX_SYS_MinvalidateCache(
                    self._io[0].pOutputs[i].phyAddr,
                    self._io[0].pOutputs[i].pVirAddr,
                    self._io[0].pOutputs[i].nSize,
                )
                npy = np.frombuffer(
                    self._engine_ffi.buffer(
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
