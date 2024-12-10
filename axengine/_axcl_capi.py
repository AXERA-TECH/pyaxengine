# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#
# modified by zylo117

import ctypes.util
import platform

from cffi import FFI

__all__: ["R", "O"]

O = FFI()

# axcl_base.h
O.cdef(
    """
    #define AXCL_MAX_DEVICE_COUNT 256
    typedef int32_t axclError;
"""
)

# axcl_rt_type.h
O.cdef(
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
O.cdef(
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
    
    typedef struct axclrtEngineIODims {
        int32_t dimCount;
        int32_t dims[AXCLRT_ENGINE_MAX_DIM_CNT];
    } axclrtEngineIODims;
"""
)

# ax_model_runner_axcl.cpp
O.cdef(
    """
    typedef enum
    {
        AX_ENGINE_ABST_DEFAULT = 0,
        AX_ENGINE_ABST_CACHED = 1,
    } AX_ENGINE_ALLOC_BUFFER_STRATEGY_T;

    typedef struct
    {
        int nIndex;
        int nSize;
        void *pBuf;
        void *pVirAddr;
    
        const char *Name;
    
        axclrtEngineIODims dims;
    } AXCL_IO_BUF_T;
    
    typedef struct
    {
        uint32_t nInputSize;
        uint32_t nOutputSize;
        AXCL_IO_BUF_T *pInputs;
        AXCL_IO_BUF_T *pOutputs;
    } AXCL_IO_DATA_T;
"""
)

# ax_model_runner.hpp
O.cdef(
    """
    typedef struct
    {
        const char * sName;
        unsigned int nIdx;
        unsigned int vShape[AXCLRT_ENGINE_MAX_DIM_CNT];
        unsigned int vShapeSize;
        int nSize;
        unsigned long long phyAddr;
        void *pVirAddr;
    } ax_runner_tensor_t;
"""
)

# stdlib.h/string.h
O.cdef(
    """
    void free (void *__ptr);
    void *malloc(size_t size);
    void *memset (void *__s, int __c, size_t __n);
    void *memcpy (void * __dest, const void * __src, size_t __n);
"""
)



# axcl.h
O.cdef(
    """
    axclError axclInit(const char *config);
    axclError axclFinalize();
"""
)

# axcl_rt.h
O.cdef(
    """
    axclError axclrtGetVersion(int32_t *major, int32_t *minor, int32_t *patch);
    const char *axclrtGetSocName();
"""
)

# axcl_rt_device.h
O.cdef(
    """
    axclError axclrtGetDeviceList(axclrtDeviceList *deviceList);
    axclError axclrtSetDevice(int32_t deviceId);
"""
)

# axcl_rt_engine.h
O.cdef(
    """
    axclError axclrtEngineInit(axclrtEngineVNpuKind npuKind);
    axclError axclrtEngineLoadFromMem(const void *model, uint64_t modelSize, uint64_t *modelId);
    axclError axclrtEngineCreateContext(uint64_t modelId, uint64_t *contextId);
    axclError axclrtEngineGetVNpuKind(axclrtEngineVNpuKind *npuKind);
    const char* axclrtEngineGetModelCompilerVersion(uint64_t modelId);
    axclError axclrtEngineGetIOInfo(uint64_t modelId, axclrtEngineIOInfo *ioInfo);
    axclError axclrtEngineGetShapeGroupsCount(axclrtEngineIOInfo ioInfo, int32_t *count);
    axclError axclrtEngineCreateIO(axclrtEngineIOInfo ioInfo, axclrtEngineIO *io);
    uint32_t axclrtEngineGetNumInputs(axclrtEngineIOInfo ioInfo);
    uint32_t axclrtEngineGetNumOutputs(axclrtEngineIOInfo ioInfo);
    uint64_t axclrtEngineGetInputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index);
    axclError axclrtEngineGetInputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims);
    const char *axclrtEngineGetInputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index);
    axclError axclrtEngineSetInputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size);
    uint64_t axclrtEngineGetOutputSizeByIndex(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index);
    axclError axclrtEngineGetOutputDims(axclrtEngineIOInfo ioInfo, uint32_t group, uint32_t index, axclrtEngineIODims *dims);
    const char *axclrtEngineGetOutputNameByIndex(axclrtEngineIOInfo ioInfo, uint32_t index);
    axclError axclrtEngineSetOutputBufferByIndex(axclrtEngineIO io, uint32_t index, const void *dataBuffer, uint64_t size);
    axclError axclrtEngineExecute(uint64_t modelId, uint64_t contextId, uint32_t group, axclrtEngineIO io);
    axclError axclrtEngineDestroyIO(axclrtEngineIO io);
    axclError axclrtEngineUnload(uint64_t modelId);
"""
)

# axcl_rt_memory.h
O.cdef(
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

R = O.dlopen(rt_path)
assert R is not None, f"Failed to load library {rt_path}. Please ensure it is installed and in the library path."
