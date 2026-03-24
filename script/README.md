# 硬件测试手册

本文档提供 PyAXEngine 硬件测试的完整说明，包含 4 个核心测试用例。

## 硬件要求

### 支持的硬件平台

- **开发板**: AX650N / AX630C（如爱芯派Pro）
- **算力卡**: AX650 M.2 算力卡

### 软件依赖

- Python >= 3.8
- axengine 已安装
- 测试模型文件（.axmodel 格式）

### 驱动要求

- 开发板: 需要安装 libax_engine.so
- 算力卡: 需要安装 AXCL 驱动，参考 [AXCL 文档](https://axcl-docs.readthedocs.io/zh-cn/latest/)

---

## 测试 1: AxEngine 基础功能测试

### 测试目的

验证 AxEngineExecutionProvider 在开发板上的基本功能，包括会话创建和模型信息获取。

### 硬件要求

- AX650N 或 AX630C 开发板
- 已安装 libax_engine.so

### 环境配置

```bash
# 确认 axengine 已安装
pip show axengine

# 准备测试模型
# 模型路径示例: /opt/data/npu/models/mobilenetv2.axmodel
```

### 运行命令

```bash
cd script
python3 test_axengine_basic.py <模型路径>
```

示例:
```bash
python3 test_axengine_basic.py /opt/data/npu/models/mobilenetv2.axmodel
```

### 预期输出

```
[INFO] Testing AxEngine with model: /opt/data/npu/models/mobilenetv2.axmodel
[INFO] Available providers: ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Successfully created session with AxEngineExecutionProvider
[INFO] Model inputs: 1, outputs: 1
```

退出码: 0（成功）

### 故障排查

**问题**: `[ERROR] AxEngineExecutionProvider not available`
- 检查是否在开发板上运行
- 确认 libax_engine.so 已正确安装
- 运行 `ldd` 检查库依赖

**问题**: `[ERROR] Failed to create session`
- 确认模型文件路径正确
- 检查模型文件是否为有效的 .axmodel 格式
- 确认模型与芯片型号匹配（AX650N/AX630C）

---

## 测试 2: AXCLRT 基础功能测试

### 测试目的

验证 AXCLRTExecutionProvider 在算力卡上的基本功能。

### 硬件要求

- AX650 M.2 算力卡
- 已安装 AXCL 驱动

### 环境配置

```bash
# 确认算力卡已识别
lspci | grep AXERA

# 确认 AXCL 驱动已加载
lsmod | grep axcl

# 确认 axengine 已安装
pip show axengine
```

### 运行命令

```bash
cd script
python3 test_axclrt_basic.py <模型路径>
```

示例:
```bash
python3 test_axclrt_basic.py /opt/data/npu/models/mobilenetv2.axmodel
```

### 预期输出

```
[INFO] Testing AXCLRT with model: /opt/data/npu/models/mobilenetv2.axmodel
[INFO] Available providers: ['AXCLRTExecutionProvider', 'AxEngineExecutionProvider']
[INFO] Successfully created session with AXCLRTExecutionProvider
[INFO] Model inputs: 1, outputs: 1
```

退出码: 0（成功）

### 故障排查

**问题**: `[ERROR] AXCLRTExecutionProvider not available`
- 检查算力卡是否正确插入
- 确认 AXCL 驱动已安装: `lsmod | grep axcl`
- 重新安装驱动或重启系统

**问题**: `[ERROR] Failed to create session`
- 检查模型文件路径
- 确认算力卡有足够内存
- 查看系统日志: `dmesg | tail`

---

## 测试 3: AxEngine 会话创建集成测试

### 测试目的

验证 AxEngineExecutionProvider 的会话创建和 provider 查询功能。

### 硬件要求

- AX650N 或 AX630C 开发板

### 环境配置

```bash
# 安装 pytest
pip install pytest

# 准备测试模型
# 在项目根目录放置 model.axmodel
```

### 运行命令

```bash
# 在项目根目录运行
pytest tests/test_integration.py::TestHardwareIntegration::test_axengine_session_creation -v
```

### 预期输出

```
tests/test_integration.py::TestHardwareIntegration::test_axengine_session_creation PASSED [100%]
```

### 故障排查

**问题**: `FileNotFoundError: model.axmodel`
- 在项目根目录创建或链接 model.axmodel
- 使用软链接: `ln -s /opt/data/npu/models/mobilenetv2.axmodel model.axmodel`

**问题**: 断言失败 `assert sess.get_providers() == 'AxEngineExecutionProvider'`
- 检查 provider 是否正确初始化
- 确认开发板环境正常

---

## 测试 4: 推理运行集成测试

### 测试目的

验证完整的推理流程，包括输入准备、推理执行和输出获取。

### 硬件要求

- AX650N 或 AX630C 开发板，或 AX650 M.2 算力卡

### 环境配置

```bash
# 安装依赖
pip install pytest numpy

# 准备测试模型
# 在项目根目录放置 model.axmodel
```

### 运行命令

```bash
# 在项目根目录运行
pytest tests/test_integration.py::TestHardwareIntegration::test_inference_run -v
```

### 预期输出

```
tests/test_integration.py::TestHardwareIntegration::test_inference_run PASSED [100%]
```

### 故障排查

**问题**: `FileNotFoundError: model.axmodel`
- 参考测试 3 的解决方案

**问题**: 推理失败或输出为空
- 检查输入数据形状是否与模型匹配
- 确认模型输入类型为 float32
- 检查 NPU 内存是否充足

**问题**: 数值异常
- 确认 SDK 版本支持模型的数据类型（bf16 需要 AX650 SDK >= 2.18）
- 检查模型编译版本与运行时版本是否匹配

---

## 批量运行所有硬件测试

```bash
# 运行所有硬件测试
pytest tests/test_integration.py -m hardware -v

# 跳过硬件测试（仅运行单元测试）
pytest -m "not hardware"
```

## 常见问题

### SDK 版本问题

AX650 SDK 2.18 和 AX620E SDK 3.12 之前的版本不支持 bf16，可能导致 LLM 模型返回 unknown dtype。

解决方案:
- 升级 SDK 到最新版本
- 或仅更新 libax_engine.so

### 算力卡 vs 开发板选择

- **开发板模式**: 使用 AxEngineExecutionProvider，适合快速原型验证
- **算力卡模式**: 使用 AXCLRTExecutionProvider，适合生产部署

如果主要使用算力卡，建议使用 [pyAXCL](https://github.com/AXERA-TECH/pyaxcl) 获得完整 API 支持。

## 技术支持

- GitHub Issues: https://github.com/AXERA-TECH/pyaxengine/issues
- QQ 群: 139953715
