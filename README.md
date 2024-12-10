# PyAXEngine

[![License](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://raw.githubusercontent.com/AXERA-TECH/pyaxengine/main/LICENSE)

## 简介

**PyAXEngine** 基于 cffi 模块实现了 Axera NPU Runtime 的 Python API，并同时支持开发板和M.2算力卡形态，方便开源社区开发者使用 Python 脚本快速构建 NPU 推理脚本

支持芯片

- AX650N
- AX630C

环境版本

- python >= 3.8
- cffi >= 1.0.0
- ml-dtypes >= 0.1.0

## 快速上手

基于社区开发板 **爱芯派Pro(AX650N)** 进行展示

### 获取 wheel 包并安装

- [下载链接](https://github.com/AXERA-TECH/pyaxengine/releases/download/0.0.1rc2/axengine-0.0.1-py3-none-any.whl)
- 将 `axengine-x.x.x-py3-none-any.whl` 拷贝到开发板上，执行 `pip install axengine-x.x.x-py3-none-any.whl` 安装

### 简单示例

将 [classification.py](https://github.com/AXERA-TECH/pyaxengine/blob/main/examples/classification.py) 拷贝到开发板上并执行。

```
# AX650开发板模式
root@ax650:~/samples# python3 classification.py
[INFO] Chip type: ChipType.AX650
[INFO] Engine version: 2.7.2a
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Model type: 0 (single core)
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
Top 5 Predictions:
Class Index: 282, Score: 9.77352523803711
Class Index: 278, Score: 8.981077194213867
Class Index: 277, Score: 8.452778816223145
Class Index: 281, Score: 8.320704460144043
Class Index: 287, Score: 7.924479961395264

# AX650 M.2算力卡模式
root@ax650:~/samples# python3 classification.py -b axcl
[INFO] SOC Name: AX650N
[INFO] Runtime version: 1.0.0
[INFO] VNPU type: VNPUType.DISABLED
[INFO] Compiler version: 1.2-patch2 7e6b2b5f
grp_id: 0
input size: 1
	name: input
		shape: 1 x 224 x 224 x 3
output size: 1
	name: output
		shape: 1 x 1000
[INFO] cost time in host to device: 0.617ms, inference: 1.087ms, device to host: 0.266ms
Top 5 Predictions:
Class Index: 282, Score: 9.77352523803711
Class Index: 278, Score: 8.981077194213867
Class Index: 277, Score: 8.452778816223145
Class Index: 281, Score: 8.320704460144043
Class Index: 287, Score: 7.924479961395264
```

## 关联项目

- [ax-samples](https://github.com/AXERA-TECH/ax-samples)
- [ax-llm](https://github.com/AXERA-TECH/ax-llm)
- [pulsar2](https://pulsar2-docs.readthedocs.io/zh-cn/latest/)

## 技术讨论

- Github issues
- QQ 群: 139953715
