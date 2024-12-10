import ctypes.util
import os

from .ax_session import InferenceSession as AXInferenceSession
from .axcl_session import InferenceSession as AXCLInferenceSession


def InferenceSession(path_or_bytes: str | bytes | os.PathLike, device_id: int = -1, *args, **kwargs):
    is_axcl = False
    if device_id >= 0:
        if ctypes.util.find_library('axcl_rt') is not None:
            is_axcl = True
        elif ctypes.util.find_library('ax_engine') is not None:
            is_axcl = False
        else:
            print("axcl_rt not found, please install axcl_host driver")

    if is_axcl:
        print(f"Using axclrt backend, device_no: {device_id}")
        return AXCLInferenceSession(path_or_bytes, device_id)
    else:
        print("Using ax backend with onboard npu")
        return AXInferenceSession(path_or_bytes)
