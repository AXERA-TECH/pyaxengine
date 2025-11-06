# PyAXEngine

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/AXERA-TECH/pyaxengine/main/LICENSE)

## 简介

**PyAXEngine** 基于 cffi 模块实现了 Axera NPU Runtime 的 Python API，其 Python API 与 ONNXRuntime 高度兼(相)容(似)，并同时支持开发板和M.2算力卡形态，方便开源社区开发者使用
Python 脚本快速构建 NPU 推理脚本

支持芯片

- AX650N
- AX630C

环境版本

- python >= 3.8
- cffi >= 1.0.0
- ml-dtypes >= 0.1.0
- numpy >= 1.22.0

*需要注意的是，如果您的开发环境是算力卡，那么更建议您优先考虑使用 [pyAXCL](https://github.com/AXERA-TECH/pyaxcl) 进行项目开发；pyAXCL 项目完整包含了算力卡形态的全部 API，更适合用于正式部署；PyAXEngine 项目更适合算法工程师进行快速原型验证，且用于计算卡环境时，PyAXEngine 不能调用编解码等模块(不是 PyAXEngine 的设计目标)。*

*AX650 SDK 2.18，AX620E SDK 3.12 以前的版本不支持 bf16，llm 模型会有返回 unknown 的 dtype问题，请注意升级*

*如果您评估认为不知道如何升级 SDK，也可以提交 issue 索要下载，不需要更新完整 SDK，只更新 libax_engine.so 即可*

## 快速上手

基于社区开发板 **爱芯派Pro(AX650N)** 进行展示

### 获取 wheel 包并安装

- [下载链接](https://github.com/AXERA-TECH/pyaxengine/releases/latest)
- 将 `axengine-x.x.x-py3-none-any.whl` 拷贝到开发板上，执行 `pip install axengine-x.x.x-py3-none-any.whl` 安装

### 简单示例

当前示例需要分别依赖 PIL 和 OpenCV，可以用 `pip install pillow opencv-python-headless` 安装。其中 `opencv-python-headless` 是 OpenCV 的 headless 版本，不依赖 GUI(非 headless 的版本需要依赖 OpenGL ES，运行环境中并没有)。

```python
将 [classification.py](https://github.com/AXERA-TECH/pyaxengine/blob/main/examples/classification.py) 拷贝到开发板上并执行。
```

```bash
root@ax650:~/samples# python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg
[INFO] Available providers:  ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.10.1s
[INFO] Model type: 0 (single core)
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
  ------------------------------------------------------
  Top 5 Predictions:
    Class Index: 282, Score: 9.774
    Class Index: 278, Score: 8.981
    Class Index: 277, Score: 8.453
    Class Index: 281, Score: 8.321
    Class Index: 287, Score: 7.924
  ------------------------------------------------------
  min =   0.890 ms   max =   22.417 ms   avg =   1.119 ms
  ------------------------------------------------------
```

示例也演示了如何选择计算设备：这意味着既可以在 **AX650/AX630C** 等开发板上运行，也可以在 AX650 M.2 算力卡上运行。

切换计算设备的方式是通过 `-p` 参数指定，如 `-p AxEngineExecutionProvider` 表示使用开发板上的 NPU 进行推理，而 `-p AXCLRTExecutionProvider` 表示使用 M.2 算力卡进行推理。
注意：在使用 M.2 算力卡进行推理时，需要将算力卡插入宿主机上，并且已经安装驱动，详见： [axcl](https://axcl-docs.readthedocs.io/zh-cn/latest/)。

```bash
root@ax650:~/samples# python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -p AXCLRTExecutionProvider
[INFO] Available providers:  ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Using provider: AXCLRTExecutionProvider
[INFO] SOC Name: AX650N
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
  ------------------------------------------------------
  Top 5 Predictions:
    Class Index: 282, Score: 9.774
    Class Index: 278, Score: 8.981
    Class Index: 277, Score: 8.453
    Class Index: 281, Score: 8.321
    Class Index: 287, Score: 7.924
  ------------------------------------------------------
  min =   1.587 ms   max =   12.624 ms   avg =   1.718 ms
  ------------------------------------------------------
root@ax650:~/samples# python3 classification.py -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -p AxEngineExecutionProvider
[INFO] Available providers:  ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Using provider: AxEngineExecutionProvider
[INFO] Chip type: ChipType.MC50
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Engine version: 2.10.1s
[INFO] Model type: 0 (single core)
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
  ------------------------------------------------------
  Top 5 Predictions:
    Class Index: 282, Score: 9.774
    Class Index: 278, Score: 8.981
    Class Index: 277, Score: 8.453
    Class Index: 281, Score: 8.321
    Class Index: 287, Score: 7.924
  ------------------------------------------------------
  min =   0.897 ms   max =   22.542 ms   avg =   1.125 ms
  ------------------------------------------------------
```

## 社区贡献者

- [zylo117](https://github.com/zylo117): 提供了基于 cffi 的 AXCL Runtime Python API 实现
- [nnn](https://github.com/nnn112358)，[HongJie Li](https://github.com/techshoww) 和 [Shinichi Tanaka](https://github.com/s1tnk) 报告 cffi 的使用问题，[Shinichi Tanaka](https://github.com/s1tnk) 提供了解决方案
- [yuyun](https://github.com/yuyun2000): 修复了加载模型时会在系统内存重复占用内存的bug

## 关联项目

- [ax-samples](https://github.com/AXERA-TECH/ax-samples)
- [ax-llm](https://github.com/AXERA-TECH/ax-llm)
- [Pulsar2](https://pulsar2-docs.readthedocs.io/zh-cn/latest/)
- [AXCL](https://axcl-docs.readthedocs.io/zh-cn/latest/)
- [pyAXCL](https://github.com/AXERA-TECH/pyaxcl)

## 技术讨论

- Github issues
- QQ 群: 139953715
