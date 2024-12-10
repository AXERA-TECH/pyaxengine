# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#
# created by zylo117
from ._session import BaseInferenceSession
from ._types import VNPUType
from ._node import NodeArg

import os
import numpy as np
import time

__all__: ["InferenceSession"]


class InferenceSession(BaseInferenceSession):
    def __init__(
        self,
        path_or_bytes: str | bytes | os.PathLike,
        device_id: int = 0
    ) -> None:
        from . import _axcl_capi as _capi

        super().__init__()

        # load shared library
        self._rt_lib = _capi.R
        self._rt_ffi = _capi.O

        self.soc_name = self._rt_ffi.string(self._rt_lib.axclrtGetSocName()).decode()
        print(f"[INFO] SOC Name: {self.soc_name}")

        # init axcl
        ret = self._init(device_id)
        if 0 != ret:
            raise RuntimeError("Failed to initialize axclrt.")
        print(f"[INFO] Runtime version: {self._get_version()}")

        # handle, context, info, io
        self._handle = self._rt_ffi.new("uint64_t *")
        self._context = self._rt_ffi.new("uint64_t *")
        self.io_info = self._rt_ffi.new("axclrtEngineIOInfo *")
        self.group_count = self._rt_ffi.new("int32_t *")

        # get vnpu type
        self._vnpu_type = self._get_vnpu_type()
        print(f"[INFO] VNPU type: {self._vnpu_type}")

        # model buffer, almost copied from onnx runtime
        if isinstance(path_or_bytes, (str, os.PathLike)):
            self._model_name = os.path.splitext(os.path.basename(path_or_bytes))[0]
            with open(path_or_bytes, "rb") as f:
                data = f.read()
            self._model_buffer = self._rt_ffi.new("char[]", data)
            self._model_buffer_size = len(data)
        elif isinstance(path_or_bytes, bytes):
            self._model_buffer = self._rt_ffi.new("char[]", path_or_bytes)
            self._model_buffer_size = len(path_or_bytes)
        else:
            raise TypeError(f"Unable to load model from type '{type(path_or_bytes)}'")

        # load model
        ret = self._load()
        if 0 != ret:
            raise RuntimeError("Failed to load model.")
        print(f"[INFO] Compiler version: {self._get_model_tool_version()}")

        # get shape group count
        self._shape_count = self._get_shape_count()
        self.ios = [self._rt_ffi.new("axclrtEngineIO *") for _ in range(self._shape_count)]
        self.io_datas = [self._rt_ffi.new("AXCL_IO_DATA_T *") for _ in range(self._shape_count)]
        self.io_buf_in = None
        self.io_buf_out = None

        self.mgroup_input_tensors = [[] for _ in range(self._shape_count)]
        self.mgroup_output_tensors = [[] for _ in range(self._shape_count)]
        self._inputs = []
        self._outputs = []

        self._sub_init()

        self._auto_sync_before_inference = True
        self._auto_sync_after_inference = True

    def _sub_init(self):
        for grp_id in range(self._shape_count):
            input_node_args = []
            output_node_args = []

            print(f'grp_id: {grp_id}')

            io = self.ios[grp_id]
            ret = self._rt_lib.axclrtEngineCreateIO(self.io_info[0], io)
            if 0 != ret:
                self._rt_lib.axclrtEngineUnload(self._handle[0])
                raise RuntimeError(f"Create io failed 0x{ret:08x}")

            io_data = self.io_datas[grp_id]
            ret = self._prepare_io(grp_id, self.io_info[0], io[0], io_data,
                                   (self._rt_lib.AX_ENGINE_ABST_DEFAULT,
                                    self._rt_lib.AX_ENGINE_ABST_DEFAULT))
            if ret != 0:
                self._free_io(io_data)
                self._rt_lib.axclrtEngineDestroyIO(io[0])
                self._rt_lib.axclrtEngineUnload(self._handle[0])
                raise RuntimeError("prepare_io failed.")

            print(f'input size: {io_data.nInputSize}')
            for i in range(io_data.nInputSize):
                tensor = self._rt_ffi.new("ax_runner_tensor_t *")
                tensor.nIdx = i
                tensor.sName = io_data.pInputs[i].Name
                tensor.nSize = io_data.pInputs[i].nSize
                for j in range(io_data.pInputs[i].dims.dimCount):
                    tensor.vShape[j] = io_data.pInputs[i].dims.dims[j]
                tensor.vShapeSize = io_data.pInputs[i].dims.dimCount
                tensor.phyAddr = self._rt_ffi.cast('unsigned long long', io_data.pInputs[i].pBuf)
                tensor.pVirAddr = io_data.pInputs[i].pVirAddr
                self.mgroup_input_tensors[grp_id].append(tensor)
                print(f'\tname: {self._rt_ffi.string(io_data.pInputs[i].Name).decode()}')
                print(f'\t\tshape: {' x '.join([str(io_data.pInputs[i].dims.dims[j]) for j in range(io_data.pInputs[i].dims.dimCount)])}')
                input_node_args.append(
                    NodeArg(self._rt_ffi.string(io_data.pInputs[i].Name).decode(), 'uint8',
                            [io_data.pInputs[i].dims.dims[j] for j in range(io_data.pInputs[i].dims.dimCount)]))

            print(f'output size: {io_data.nOutputSize}')
            for i in range(io_data.nOutputSize):
                tensor = self._rt_ffi.new("ax_runner_tensor_t *")
                tensor.nIdx = i
                tensor.sName = io_data.pOutputs[i].Name
                tensor.nSize = io_data.pOutputs[i].nSize
                for j in range(io_data.pOutputs[i].dims.dimCount):
                    tensor.vShape[j] = io_data.pOutputs[i].dims.dims[j]
                tensor.vShapeSize = io_data.pOutputs[i].dims.dimCount
                tensor.phyAddr = self._rt_ffi.cast('unsigned long long', io_data.pOutputs[i].pBuf)
                tensor.pVirAddr = io_data.pOutputs[i].pVirAddr
                self.mgroup_output_tensors[grp_id].append(tensor)
                print(f'\tname: {self._rt_ffi.string(io_data.pOutputs[i].Name).decode()}')
                print(f'\t\tshape: {' x '.join([str(io_data.pOutputs[i].dims.dims[j]) for j in range(io_data.pOutputs[i].dims.dimCount)])}')
                output_node_args.append(
                    NodeArg(self._rt_ffi.string(io_data.pOutputs[i].Name).decode(), 'float32',
                            [io_data.pOutputs[i].dims.dims[j] for j in range(io_data.pOutputs[i].dims.dimCount)]))

            self._inputs.append(input_node_args)
            self._outputs.append(output_node_args)

    def _prepare_io(self, grp_id, io_info, io, io_data, strategy):
        self._rt_lib.memset(io_data, 0, self._rt_ffi.sizeof('AXCL_IO_DATA_T'))

        inputNum = self._rt_lib.axclrtEngineGetNumInputs(io_info)
        outputNum = self._rt_lib.axclrtEngineGetNumOutputs(io_info)
        io_data.nInputSize = inputNum
        io_data.nOutputSize = outputNum
        self.io_buf_in = self._rt_ffi.new('AXCL_IO_BUF_T[]', inputNum)
        self.io_buf_out = self._rt_ffi.new('AXCL_IO_BUF_T[]', outputNum)
        io_data.pInputs = self.io_buf_in
        io_data.pOutputs = self.io_buf_out

        # alloc inputs
        for i in range(inputNum):
            bufSize = self._rt_lib.axclrtEngineGetInputSizeByIndex(io_info, grp_id, i)
            devPtr = self._rt_ffi.new('void **', self._rt_ffi.NULL)
            ret = 0
            if strategy[0] == self._rt_lib.AX_ENGINE_ABST_DEFAULT:
                ret = self._rt_lib.axclrtMalloc(devPtr, bufSize, self._rt_lib.AXCL_MEM_MALLOC_HUGE_FIRST)
            else:
                ret = self._rt_lib.axclrtMallocCached(devPtr, bufSize, self._rt_lib.AXCL_MEM_MALLOC_HUGE_FIRST)

            if ret != 0:
                self._free_io_index(io_data.pInputs, i)
                raise RuntimeError(f"Malloc input(index: {i}, size: {bufSize}) failed! 0x{ret:08x}")

            tmp = self._rt_ffi.new('char[]', bufSize)
            self._rt_lib.axclrtMemcpy(devPtr[0], tmp, bufSize, self._rt_lib.AXCL_MEMCPY_HOST_TO_DEVICE)

            dims = self._rt_ffi.new('axclrtEngineIODims *')
            ret = self._rt_lib.axclrtEngineGetInputDims(io_info, grp_id, i, dims)
            if ret != 0:
                self._free_io_index(io_data.pInputs, i)
                raise RuntimeError(f"Get input dims(index: {i}) failed! 0x{ret:08x}")

            io_data.pInputs[i].nIndex = i
            io_data.pInputs[i].nSize = bufSize
            io_data.pInputs[i].pBuf = devPtr[0]
            io_data.pInputs[i].dims = dims[0]
            io_data.pInputs[i].Name = self._rt_lib.axclrtEngineGetInputNameByIndex(io_info, i)
            io_data.pInputs[i].pVirAddr = self._rt_lib.malloc(bufSize)
            self._rt_lib.memset(io_data.pInputs[i].pVirAddr, 0, bufSize)
            ret = self._rt_lib.axclrtEngineSetInputBufferByIndex(io, i, devPtr[0], bufSize)
            if ret != 0:
                self._free_io_index(io_data.pInputs, i)
                raise RuntimeError(f"Set input buffer(index: {i}, size: {bufSize}) failed! 0x{ret:08x}")

        # alloc outputs
        for i in range(outputNum):
            bufSize = self._rt_lib.axclrtEngineGetOutputSizeByIndex(io_info, grp_id, i)
            devPtr = self._rt_ffi.new('void **', self._rt_ffi.NULL)
            ret = 0
            if strategy[0] == self._rt_lib.AX_ENGINE_ABST_DEFAULT:
                ret = self._rt_lib.axclrtMalloc(devPtr, bufSize, self._rt_lib.AXCL_MEM_MALLOC_HUGE_FIRST)
            else:
                ret = self._rt_lib.axclrtMallocCached(devPtr, bufSize, self._rt_lib.AXCL_MEM_MALLOC_HUGE_FIRST)

            if ret != 0:
                self._free_io_index(io_data.pOutputs, i)
                raise RuntimeError(f"Malloc output(index: {i}, size: {bufSize}) failed! 0x{ret:08x}")

            tmp = self._rt_ffi.new('char[]', bufSize)
            self._rt_lib.axclrtMemcpy(devPtr[0], tmp, bufSize, self._rt_lib.AXCL_MEMCPY_HOST_TO_DEVICE)

            dims = self._rt_ffi.new('axclrtEngineIODims *')
            ret = self._rt_lib.axclrtEngineGetOutputDims(io_info, grp_id, i, dims)
            if ret != 0:
                self._free_io_index(io_data.pOutputs, i)
                raise RuntimeError(f"Get output dims(index: {i}) failed! 0x{ret:08x}")

            io_data.pOutputs[i].nIndex = i
            io_data.pOutputs[i].nSize = bufSize
            io_data.pOutputs[i].pBuf = devPtr[0]
            io_data.pOutputs[i].dims = dims[0]
            io_data.pOutputs[i].Name = self._rt_lib.axclrtEngineGetOutputNameByIndex(io_info, i)
            io_data.pOutputs[i].pVirAddr = self._rt_lib.malloc(bufSize)
            self._rt_lib.memset(io_data.pOutputs[i].pVirAddr, 0, bufSize)
            ret = self._rt_lib.axclrtEngineSetOutputBufferByIndex(io, i, devPtr[0], bufSize)
            if ret != 0:
                self._free_io_index(io_data.pOutputs, i)
                raise RuntimeError(f"Set output buffer(index: {i}, size: {bufSize}) failed! 0x{ret:08x}")
        return 0

    def _free_io_index(self, pBuf, index):
        for i in range(index):
            self._rt_lib.axclrtFree(pBuf[i].pBuf)

    def _free_io(self, io_data):
        for j in range(io_data.nInputSize):
            self._rt_lib.axclrtFree(io_data.pInputs[j].pBuf)
            self._rt_lib.free(io_data.pInputs[j].pVirAddr)
        for j in range(io_data.nOutputSize):
            self._rt_lib.axclrtFree(io_data.pOutputs[j].pBuf)
            self._rt_lib.free(io_data.pOutputs[j].pVirAddr)

        # 不知道如何在ffi中直接调用
        # delete[] io_data->pInputs;
        # delete[] io_data->pOutputs;

    def _init(self, device_id=0, vnpu=VNPUType.DISABLED):  # vnpu type, the default is disabled
        ret = self._rt_lib.axclInit([])
        if ret != 0:
            raise RuntimeError("Failed to initialize runtime.")

        lst = self._rt_ffi.new("axclrtDeviceList *")
        ret = self._rt_lib.axclrtGetDeviceList(lst)
        if ret != 0 or lst.num == 0:
            raise RuntimeError(f"Get AXCL device failed 0x{ret:08x}, find total {lst.num} device.")

        ret = self._rt_lib.axclrtSetDevice(lst.devices[device_id])
        if ret != 0 or lst.num == 0:
            raise RuntimeError(f"Set AXCL device failed 0x{ret:08x}.")

        ret = self._rt_lib.axclrtEngineInit(vnpu.value)
        if ret != 0 or lst.num == 0:
            raise RuntimeError(f"axclrtEngineInit failed 0x{ret:08x}.")

        return 0

    def _final(self):
        if self._handle[0] is not None:
            self._unload()
        self._rt_lib.axclFinalize()
        return

    def _get_version(self):
        major, minor, patch = self._rt_ffi.new('int32_t *'), self._rt_ffi.new('int32_t *'), self._rt_ffi.new('int32_t *')
        self._rt_lib.axclrtGetVersion(major, minor, patch)
        return f'{major[0]}.{minor[0]}.{patch[0]}'

    def _get_vnpu_type(self) -> VNPUType:
        vnpu_type = self._rt_ffi.new("axclrtEngineVNpuKind *")
        ret = self._rt_lib.axclrtEngineGetVNpuKind(vnpu_type)
        if ret != 0:
            raise RuntimeError("Failed to get VNPU attribute.")
        return VNPUType(vnpu_type[0])

    def _get_model_tool_version(self):
        model_tool_version = self._rt_lib.axclrtEngineGetModelCompilerVersion(self._handle[0])
        return self._rt_ffi.string(model_tool_version).decode()

    def _load(self):
        devMem = self._rt_ffi.new('void **', self._rt_ffi.NULL)
        self._rt_lib.axclrtMalloc(devMem, self._model_buffer_size, self._rt_lib.AXCL_MEM_MALLOC_NORMAL_ONLY)
        self._rt_lib.axclrtMemcpy(devMem[0], self._model_buffer, self._model_buffer_size, self._rt_lib.AXCL_MEMCPY_HOST_TO_DEVICE)

        ret = self._rt_lib.axclrtEngineLoadFromMem(devMem[0], self._model_buffer_size, self._handle)
        if ret != 0:
            raise RuntimeError("axclrtEngineLoadFromMem failed")

        self._rt_lib.axclrtFree(devMem[0])

        ret = self._rt_lib.axclrtEngineCreateContext(self._handle[0], self._context)
        if ret != 0:
            raise RuntimeError("axclrtEngineCreateContext failed")

        ret = self._rt_lib.axclrtEngineGetIOInfo(self._handle[0], self.io_info)
        if ret != 0:
            raise RuntimeError("axclrtEngineGetIOInfo failed")

        return self.group_count[0]

    def _get_shape_count(self):
        ret = self._rt_lib.axclrtEngineGetShapeGroupsCount(self.io_info[0], self.group_count)
        if ret != 0:
            self._rt_lib.axclrtEngineUnload(self._handle[0])
            raise RuntimeError("axclrtEngineGetShapeGroupsCount failed")

        return self.group_count[0]

    def _unload(self):
        for grp_id in range(len(self.mgroup_input_tensors)):
            self._free_io(self.io_datas[grp_id])
            self._rt_lib.axclrtEngineDestroyIO(self.ios[grp_id][0])

        self._rt_lib.axclrtEngineUnload(self._handle[0])
        self._handle[0] = 0

        return

    def run(self, output_names, input_feed, run_options=None):
        self._validate_input(list(input_feed.keys()))
        self._validate_output(output_names)

        if None is output_names:
            output_names = [o.name for o in self.get_outputs()]

        grp_id = 0

        # fill model io
        for key, npy in input_feed.items():
            for i, one in enumerate(self.get_inputs()):
                if one.name == key:
                    npy_ptr = self._rt_ffi.cast("void *", npy.ctypes.data)
                    self._rt_lib.memcpy(self.mgroup_input_tensors[grp_id][i].pVirAddr, npy_ptr, npy.nbytes)
                    break

        # execute model
        t1 = time.time()
        if self._auto_sync_before_inference:
            for input_tensor in self.mgroup_input_tensors[grp_id]:
                self._rt_lib.axclrtMemcpy(self._rt_ffi.cast('void *', input_tensor.phyAddr), input_tensor.pVirAddr,
                                          input_tensor.nSize, self._rt_lib.AXCL_MEMCPY_HOST_TO_DEVICE)
        t2 = time.time()
        cost_host_to_device = t2 - t1

        t1 = time.time()
        ret = self._rt_lib.axclrtEngineExecute(self._handle[0], self._context[0], grp_id, self.ios[grp_id][0])
        if ret != 0:
            raise RuntimeError(f"axclrtEngineExecute failed 0x{ret:08x}")
        t2 = time.time()
        cost_inference = t2 - t1

        t1 = time.time()
        if self._auto_sync_after_inference:
            for output_tensor in self.mgroup_output_tensors[grp_id]:
                self._rt_lib.axclrtMemcpy(output_tensor.pVirAddr, self._rt_ffi.cast('void *', output_tensor.phyAddr),
                                          output_tensor.nSize, self._rt_lib.AXCL_MEMCPY_DEVICE_TO_HOST)
        t2 = time.time()
        cost_device_to_host = t2 - t1

        # flush output
        outputs = [np.frombuffer(self._rt_ffi.buffer(output_tensor.pVirAddr, output_tensor.nSize),
                                 dtype=self.get_outputs()[0].dtype).reshape(self.get_outputs()[i].shape)
                   for i, output_tensor in enumerate(self.mgroup_output_tensors[grp_id])
                   if self.get_outputs()[i].name in output_names]

        print(f'[INFO] cost time in host to device: {cost_host_to_device * 1009:.3f}ms, '
              f'inference: {cost_inference * 1009:.3f}ms, '
              f'device to host: {cost_device_to_host * 1009:.3f}ms')

        return outputs
