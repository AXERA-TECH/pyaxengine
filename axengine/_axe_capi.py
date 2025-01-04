# Copyright (c) 2019-2024 Axera Semiconductor Co., Ltd. All Rights Reserved.
#
# This source file is the property of Axera Semiconductor Co., Ltd. and
# may not be copied or distributed in any isomorphic form without the prior
# written consent of Axera Semiconductor Co., Ltd.
#

import ctypes.util
import platform

from cffi import FFI

__all__: ["sys_lib", "sys_cffi", "engine_lib", "engine_cffi"]

sys_cffi = FFI()

# ax_base_type.h
sys_cffi.cdef(
    """
    typedef int                         AX_S32;
    typedef unsigned int                AX_U32;
    typedef unsigned long long int      AX_U64;
    typedef signed char                 AX_S8;
    typedef void                        AX_VOID;
"""
)

# ax_sys_api.h
sys_cffi.cdef(
    """
    AX_S32 AX_SYS_Init(AX_VOID);
    AX_S32 AX_SYS_Deinit(AX_VOID);
    AX_S32 AX_SYS_MemAllocCached(AX_U64 *phyaddr, AX_VOID **pviraddr, AX_U32 size, AX_U32 align, const AX_S8 *token);
    AX_S32 AX_SYS_MemFree(AX_U64 phyaddr, AX_VOID *pviraddr);
    AX_S32 AX_SYS_MflushCache(AX_U64 phyaddr, AX_VOID *pviraddr, AX_U32 size);
    AX_S32 AX_SYS_MinvalidateCache(AX_U64 phyaddr, AX_VOID *pviraddr, AX_U32 size);
"""
)

sys_name = "ax_sys"
sys_path = ctypes.util.find_library(sys_name)
assert (
    sys_path is not None
), f"Failed to find library {sys_name}. Please ensure it is installed and in the library path."

sys_lib = sys_cffi.dlopen(sys_path)
assert sys_lib is not None, f"Failed to load library {sys_path}. Please ensure it is installed and in the library path."

engine_cffi = FFI()

# ax_base_type.h
engine_cffi.cdef(
    """
    typedef unsigned long long int      AX_U64;
    typedef unsigned int                AX_U32;
    typedef unsigned char               AX_U8;
    typedef int                         AX_S32;
    typedef signed char                 AX_S8;
    typedef char                        AX_CHAR;
    typedef void                        AX_VOID;
    
    typedef enum {
        AX_FALSE = 0,
        AX_TRUE  = 1,
    } AX_BOOL;
"""
)

# ax_engine_type.h, base type
engine_cffi.cdef(
    """
    typedef AX_U32                      AX_ENGINE_NPU_SET_T;
"""
)

# ax_engine_type.h, enum
engine_cffi.cdef(
    """
    typedef enum _AX_ENGINE_TENSOR_LAYOUT_E
    {
        AX_ENGINE_TENSOR_LAYOUT_UNKNOWN = 0,
        AX_ENGINE_TENSOR_LAYOUT_NHWC    = 1,
        AX_ENGINE_TENSOR_LAYOUT_NCHW    = 2,
    } AX_ENGINE_TENSOR_LAYOUT_T;

    typedef enum
    {
        AX_ENGINE_MT_PHYSICAL           = 0,
        AX_ENGINE_MT_VIRTUAL            = 1,
        AX_ENGINE_MT_OCM                = 2,
    } AX_ENGINE_MEMORY_TYPE_T;

    typedef enum
    {
        AX_ENGINE_DT_UNKNOWN            = 0,
        AX_ENGINE_DT_UINT8              = 1,
        AX_ENGINE_DT_UINT16             = 2,
        AX_ENGINE_DT_FLOAT32            = 3,
        AX_ENGINE_DT_SINT16             = 4,
        AX_ENGINE_DT_SINT8              = 5,
        AX_ENGINE_DT_SINT32             = 6,
        AX_ENGINE_DT_UINT32             = 7,
        AX_ENGINE_DT_FLOAT64            = 8,
        AX_ENGINE_DT_BFLOAT16           = 9,
        AX_ENGINE_DT_UINT10_PACKED      = 100,
        AX_ENGINE_DT_UINT12_PACKED      = 101,
        AX_ENGINE_DT_UINT14_PACKED      = 102,
        AX_ENGINE_DT_UINT16_PACKED      = 103,
    } AX_ENGINE_DATA_TYPE_T;

    typedef enum
    {
        AX_ENGINE_CS_FEATUREMAP         = 0,
        AX_ENGINE_CS_RAW8               = 12,
        AX_ENGINE_CS_RAW10              = 1,
        AX_ENGINE_CS_RAW12              = 2,
        AX_ENGINE_CS_RAW14              = 11,
        AX_ENGINE_CS_RAW16              = 3,
        AX_ENGINE_CS_NV12               = 4,
        AX_ENGINE_CS_NV21               = 5,
        AX_ENGINE_CS_RGB                = 6,
        AX_ENGINE_CS_BGR                = 7,
        AX_ENGINE_CS_RGBA               = 8,
        AX_ENGINE_CS_GRAY               = 9,
        AX_ENGINE_CS_YUV444             = 10,
    } AX_ENGINE_COLOR_SPACE_T;
"""
)

# ax_engine_type.h, architecturally agnostic struct
engine_cffi.cdef(
    """
    typedef enum {
        AX_ENGINE_VIRTUAL_NPU_DISABLE   = 0,
    } AX_ENGINE_NPU_MODE_T;

    typedef enum {
        AX_ENGINE_MODEL_TYPE0           = 0,
    } AX_ENGINE_MODEL_TYPE_T;

    typedef struct {
        AX_ENGINE_NPU_MODE_T            eHardMode;
        AX_U32                          reserve[8];
    } AX_ENGINE_NPU_ATTR_T;

    typedef struct _AX_ENGINE_IO_META_EX_T
    {
        AX_ENGINE_COLOR_SPACE_T         eColorSpace;
        AX_U64                          u64Reserved[18];
    } AX_ENGINE_IO_META_EX_T;
    
    typedef struct {
        AX_ENGINE_NPU_SET_T             nNpuSet;
        AX_S8*                          pName;
        AX_U32                          reserve[8];
    } AX_ENGINE_HANDLE_EXTRA_T;
    
    typedef struct _AX_ENGINE_CMM_INFO_T
    {
        AX_U32                          nCMMSize;
    } AX_ENGINE_CMM_INFO_T;

    typedef struct _AX_ENGINE_IO_SETTING_T
    {
        AX_U32                          nWbtIndex;
        AX_U64                          u64Reserved[7];
    }AX_ENGINE_IO_SETTING_T;
"""
)

# check architecture, 32bit or 64bit
arch = platform.architecture()[0]

# ax_engine_type.h, struct
if arch == "64bit":
    engine_cffi.cdef(
        """
        typedef struct _AX_ENGINE_IO_META_T
        {
            AX_CHAR*                    pName;
            AX_S32*                     pShape;
            AX_U8                       nShapeSize;
            AX_ENGINE_TENSOR_LAYOUT_T   eLayout;
            AX_ENGINE_MEMORY_TYPE_T     eMemoryType;
            AX_ENGINE_DATA_TYPE_T       eDataType;
            AX_ENGINE_IO_META_EX_T*     pExtraMeta;
            AX_U32                      nSize;
            AX_U32                      nQuantizationValue;
            AX_S32*                     pStride;
            AX_U64                      u64Reserved[9];
        } AX_ENGINE_IO_META_T;

        typedef struct _AX_ENGINE_IO_INFO_T
        {
            AX_ENGINE_IO_META_T*        pInputs;
            AX_U32                      nInputSize;
            AX_ENGINE_IO_META_T*        pOutputs;
            AX_U32                      nOutputSize;
            AX_U32                      nMaxBatchSize;
            AX_BOOL                     bDynamicBatchSize;
            AX_U64                      u64Reserved[11];
        } AX_ENGINE_IO_INFO_T;

        typedef struct _AX_ENGINE_IO_BUFFER_T
        {
            AX_U64                      phyAddr;
            AX_VOID*                    pVirAddr;
            AX_U32                      nSize;
            AX_S32*                     pStride;
            AX_U8                       nStrideSize;
            AX_U64                      u64Reserved[11];
        } AX_ENGINE_IO_BUFFER_T;

        typedef struct _AX_ENGINE_IO_T
        {
            AX_ENGINE_IO_BUFFER_T*      pInputs;
            AX_U32                      nInputSize;
            AX_ENGINE_IO_BUFFER_T*      pOutputs;
            AX_U32                      nOutputSize;
            AX_U32                      nBatchSize;
            AX_ENGINE_IO_SETTING_T*     pIoSetting;
            AX_U64                      u64Reserved[10];
        } AX_ENGINE_IO_T;
    """
    )
else:
    engine_cffi.cdef(
        """
        typedef struct _AX_ENGINE_IO_META_T
        {
            AX_CHAR*                    pName;
            AX_S32*                     pShape;
            AX_U8                       nShapeSize;
            AX_ENGINE_TENSOR_LAYOUT_T   eLayout;
            AX_ENGINE_MEMORY_TYPE_T     eMemoryType;
            AX_ENGINE_DATA_TYPE_T       eDataType;
            AX_ENGINE_IO_META_EX_T*     pExtraMeta;
            AX_U32                      nSize;
            AX_U32                      nQuantizationValue;
            AX_S32*                     pStride;
            AX_U64 u64Reserved[11];
        } AX_ENGINE_IO_META_T;

        typedef struct _AX_ENGINE_IO_INFO_T
        {
            AX_ENGINE_IO_META_T*        pInputs;
            AX_U32                      nInputSize;
            AX_ENGINE_IO_META_T*        pOutputs;
            AX_U32                      nOutputSize;
            AX_U32                      nMaxBatchSize;
            AX_BOOL                     bDynamicBatchSize;
            AX_U64                      u64Reserved[13];
        } AX_ENGINE_IO_INFO_T;

        typedef struct _AX_ENGINE_IO_BUFFER_T
        {
            AX_U64                      phyAddr;
            AX_VOID*                    pVirAddr;
            AX_U32                      nSize;
            AX_S32*                     pStride;
            AX_U8                       nStrideSize;
            AX_U64                      u64Reserved[13];
        } AX_ENGINE_IO_BUFFER_T;

        typedef struct _AX_ENGINE_IO_T
        {
            AX_ENGINE_IO_BUFFER_T*      pInputs;
            AX_U32                      nInputSize;
            AX_ENGINE_IO_BUFFER_T*      pOutputs;
            AX_U32                      nOutputSize;
            AX_U32                      nBatchSize;
            AX_ENGINE_IO_SETTING_T*     pIoSetting;
            AX_U64                      u64Reserved[12];
        } AX_ENGINE_IO_T;
    """
    )

# ax_engine_api.h
engine_cffi.cdef(
    """
    const AX_CHAR* AX_ENGINE_GetVersion(AX_VOID);

    AX_VOID AX_ENGINE_NPUReset(AX_VOID);
    AX_S32 AX_ENGINE_Init(AX_ENGINE_NPU_ATTR_T* pNpuAttr);
    AX_S32 AX_ENGINE_GetVNPUAttr(AX_ENGINE_NPU_ATTR_T* pNpuAttr);
    AX_S32 AX_ENGINE_Deinit(AX_VOID);

    AX_S32 AX_ENGINE_GetModelType(const AX_VOID* pData, AX_U32 nDataSize, AX_ENGINE_MODEL_TYPE_T* pModelType);

    AX_S32 AX_ENGINE_CreateHandleV2(uint64_t** pHandle, const AX_VOID* pData, AX_U32 nDataSize, AX_ENGINE_HANDLE_EXTRA_T* pExtraParam);
    AX_S32 AX_ENGINE_DestroyHandle(uint64_t* nHandle);

    AX_S32 AX_ENGINE_GetIOInfo(uint64_t* nHandle, AX_ENGINE_IO_INFO_T** pIO);
    AX_S32 AX_ENGINE_GetGroupIOInfoCount(uint64_t* nHandle, AX_U32* pCount);
    AX_S32 AX_ENGINE_GetGroupIOInfo(uint64_t* nHandle, AX_U32 nIndex, AX_ENGINE_IO_INFO_T** pIO);

    AX_S32 AX_ENGINE_GetHandleModelType(uint64_t* nHandle, AX_ENGINE_MODEL_TYPE_T* pModelType);

    AX_S32 AX_ENGINE_CreateContextV2(uint64_t* nHandle, uint64_t** pContext);

    AX_S32 AX_ENGINE_RunSyncV2(uint64_t* handle, uint64_t* context, AX_ENGINE_IO_T* pIO);
    AX_S32 AX_ENGINE_RunGroupIOSync(uint64_t* handle, uint64_t* context, AX_U32 nIndex, AX_ENGINE_IO_T* pIO);

    AX_S32 AX_ENGINE_SetAffinity(uint64_t* nHandle, AX_ENGINE_NPU_SET_T nNpuSet);
    AX_S32 AX_ENGINE_GetAffinity(uint64_t* nHandle, AX_ENGINE_NPU_SET_T* pNpuSet);

    AX_S32 AX_ENGINE_GetCMMUsage(uint64_t* nHandle, AX_ENGINE_CMM_INFO_T* pCMMInfo);

    const AX_CHAR* AX_ENGINE_GetModelToolsVersion(uint64_t* nHandle);

    // internal use api, remember no question
    AX_S32 AX_ENGINE_GetTotalOps();
"""
)

engine_name = "ax_engine"
engine_path = ctypes.util.find_library(engine_name)
assert (
    engine_path is not None
), f"Failed to find library {engine_name}. Please ensure it is installed and in the library path."

engine_lib = engine_cffi.dlopen(engine_path)
assert engine_lib is not None, f"Failed to load library {engine_path}. Please ensure it is installed and in the library path."
