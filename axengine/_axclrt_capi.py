# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes.util

from cffi import FFI

__all__: ["axclrt_cffi", "axclrt_lib"]

axclrt_cffi = FFI()

# axcl_base.h
axclrt_cffi.cdef(
    """
    #define AXCL_MAX_DEVICE_COUNT 256
    typedef int32_t axclError;
    typedef void *axclrtContext;
"""
)

# axcl_rt_type.h
axclrt_cffi.cdef(
    """
    typedef struct axclrtDeviceList {
        uint32_t num;
        int32_t devices[AXCL_MAX_DEVICE_COUNT];
    } axclrtDeviceList;
    
    typedef enum axclrtMemMallocPolicy {
        AXCL_MEM_MALLOC_HUGE_FIRST,
        AXCL_MEM_MALLOC_HUGE_ONLY,
        AXCL_MEM_MALLOC_NORMAL_ONLY
    } axclrtMemMallocPolicy;
    
    typedef enum axclrtMemcpyKind {
        AXCL_MEMCPY_HOST_TO_HOST,
        AXCL_MEMCPY_HOST_TO_DEVICE,     //!< host vir -> device phy
        AXCL_MEMCPY_DEVICE_TO_HOST,     //!< host vir <- device phy
        AXCL_MEMCPY_DEVICE_TO_DEVICE,
        AXCL_MEMCPY_HOST_PHY_TO_DEVICE, //!< host phy -> device phy
        AXCL_MEMCPY_DEVICE_TO_HOST_PHY, //!< host phy <- device phy
    } axclrtMemcpyKind;
"""
)

# axcl_rt_engine_type.h
axclrt_cffi.cdef(
    """
    #define AXCLRT_ENGINE_MAX_DIM_CNT 32
    typedef void* axclrtEngineIOInfo;
    typedef void* axclrtEngineIO;

    typedef enum axclrtEngineVNpuKind {
        AXCL_VNPU_DISABLE = 0,
        AXCL_VNPU_ENABLE = 1,
        AXCL_VNPU_BIG_LITTLE = 2,
        AXCL_VNPU_LITTLE_BIG = 3,
    } axclrtEngineVNpuKind;
    
    typedef enum axclrtEngineDataType {
        AXCL_DATA_TYPE_NONE = 0,
        AXCL_DATA_TYPE_INT4 = 1,
        AXCL_DATA_TYPE_UINT4 = 2,
        AXCL_DATA_TYPE_INT8 = 3,
        AXCL_DATA_TYPE_UINT8 = 4,
        AXCL_DATA_TYPE_INT16 = 5,
        AXCL_DATA_TYPE_UINT16 = 6,
        AXCL_DATA_TYPE_INT32 = 7,
        AXCL_DATA_TYPE_UINT32 = 8,
        AXCL_DATA_TYPE_INT64 = 9,
        AXCL_DATA_TYPE_UINT64 = 10,
        AXCL_DATA_TYPE_FP4 = 11,
        AXCL_DATA_TYPE_FP8 = 12,
        AXCL_DATA_TYPE_FP16 = 13,
        AXCL_DATA_TYPE_BF16 = 14,
        AXCL_DATA_TYPE_FP32 = 15,
        AXCL_DATA_TYPE_FP64 = 16,
    } axclrtEngineDataType;
    
    typedef enum axclrtEngineDataLayout {
        AXCL_DATA_LAYOUT_NONE = 0,
        AXCL_DATA_LAYOUT_NHWC = 0,
        AXCL_DATA_LAYOUT_NCHW = 1,
    } axclrtEngineDataLayout;
    
    typedef struct axclrtEngineIODims {
        int32_t dimCount;
        int32_t dims[AXCLRT_ENGINE_MAX_DIM_CNT];
    } axclrtEngineIODims;
"""
)

# axcl.h
axclrt_cffi.cdef(
    """
    axclError axclInit(const char *config);
    axclError axclFinalize();
"""
)

# axcl_rt.h
axclrt_cffi.cdef(
    """
    axclError axclrtGetVersion(int32_t *major, int32_t *minor, int32_t *patch);
    const char *axclrtGetSocName();
"""
)

# axcl_rt_device.h
axclrt_cffi.cdef(
    """
    axclError axclrtGetDeviceList(axclrtDeviceList *deviceList);
    axclError axclrtSetDevice(int32_t deviceId);
    axclError axclrtResetDevice(int32_t deviceId);
"""
)

# axcl_rt_context.h
axclrt_cffi.cdef(
    """
    axclError axclrtCreateContext(axclrtContext *context, int32_t deviceId);
    axclError axclrtDestroyContext(axclrtContext context);
    axclError axclrtSetCurrentContext(axclrtContext context);
    axclError axclrtGetCurrentContext(axclrtContext *context);
    axclError axclrtGetDefaultContext(axclrtContext *context, int32_t deviceId);
"""
)

# axcl_rt_engine.h
axclrt_cffi.cdef(
    """
    axclError axclrtEngineInit(axclrtEngineVNpuKind npuKind);
    axclError axclrtEngineGetVNpuKind(axclrtEngineVNpuKind *npuKind);
    axclError axclrtEngineFinalize();

    axclError axclrtEngineLoadFromFile(const char *modelPath, uint64_t *modelId);
    axclError axclrtEngineLoadFromMem(const void *model, uint64_t modelSize, uint64_t *modelId);
    const char* axclrtEngineGetModelCompilerVersion(uint64_t modelId);
    axclError axclrtEngineUnload(uint64_t modelId);

    axclError axclrtEngineGetIOInfo(uint64_t modelId, axclrtEngineIOInfo *ioInfo);
    axclError axclrtEngineGetShapeGroupsCount(axclrtEngineIOInfo ioInfo, int32_t *count);

    uint32_t axclrtEngineGetNumInputs(axclrtEngineIOInfo ioInfo);
    uint32_t axclrtEngineGetNumOutputs(axclrtEngineIOInfo ioInfo);

    uint64_t axclrtEngineGetInputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index);
    uint64_t axclrtEngineGetOutputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index);

    axclError axclrtEngineGetInputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims);
    axclError axclrtEngineGetOutputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims);

    const char *axclrtEngineGetInputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index);
    const char *axclrtEngineGetOutputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index);

    int32_t axclrtEngineGetInputDataType(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataType *type);
    int32_t axclrtEngineGetOutputDataType(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataType *type);

    int32_t axclrtEngineGetInputDataLayout(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataLayout *layout);
    int32_t axclrtEngineGetOutputDataLayout(axclrtEngineIOInfo ioInfo, uint32_t index, axclrtEngineDataLayout *layout);

    axclError axclrtEngineCreateIO(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io);
    axclError axclrtEngineDestroyIO(axclrtEngineIO io);

    axclError axclrtEngineSetInputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size);
    axclError axclrtEngineSetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size);
    axclError axclrtEngineGetInputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size);
    axclError axclrtEngineGetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, void **dataBuffer, uint64_t *size);

    axclError axclrtEngineCreateContext(uint64_t modelId, uint64_t *contextId);

    axclError axclrtEngineExecute(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io);
"""
)

# axcl_rt_memory.h
axclrt_cffi.cdef(
    """
    axclError axclrtMalloc(void **devPtr, size_t size, axclrtMemMallocPolicy policy);
    axclError axclrtMallocCached(void **devPtr, size_t size, axclrtMemMallocPolicy policy);
    axclError axclrtMemcpy(void *dstPtr, const void *srcPtr, size_t count, axclrtMemcpyKind kind);
    axclError axclrtFree(void *devPtr);
    axclError axclrtMemFlush(void *devPtr, size_t size);
"""
)

rt_name = "axcl_rt"
rt_path = ctypes.util.find_library(rt_name)
assert (
        rt_path is not None
), f"Failed to find library {rt_name}. Please ensure it is installed and in the library path."

axclrt_lib = axclrt_cffi.dlopen(rt_path)
assert axclrt_lib is not None, f"Failed to load library {rt_path}. Please ensure it is installed and in the library path."
