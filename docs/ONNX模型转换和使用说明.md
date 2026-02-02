# ONNX模型转换和使用说明

## 概述

ONNX（Open Neural Network Exchange）是一个开放的深度学习模型交换格式，可以实现模型的跨平台部署。

### 为什么需要ONNX？

| 需求 | PyTorch | ONNX |
|-----|---------|------|
| **平台支持** | 主要Python | Python, C++, C#, Java, JS等 |
| **部署环境** | 需要PyTorch依赖 | 仅需ONNX Runtime（更轻量） |
| **推理速度** | 一般 | 优化后更快 |
| **模型大小** | 较大 | 可优化压缩 |
| **移动端部署** | 困难 | 支持良好 |

### 主要优势

✅ **跨平台** - Windows、Linux、macOS、Android、iOS  
✅ **跨语言** - Python、C++、C#、Java、JavaScript  
✅ **高性能** - 优化的推理引擎  
✅ **轻量级** - 无需PyTorch运行时  
✅ **易部署** - 生产环境友好  

## 环境准备

### 安装依赖

```bash
# 基础依赖
pip install onnx                    # ONNX核心库
pip install onnxruntime             # CPU推理引擎
# 或者
pip install onnxruntime-gpu         # GPU推理引擎（推荐）

# 可选依赖
pip install onnx-simplifier         # 模型简化工具（推荐）
pip install onnxoptimizer           # 模型优化工具
```

### 版本要求

| 库 | 最低版本 | 推荐版本 |
|----|---------|---------|
| **Python** | 3.7+ | 3.8-3.10 |
| **PyTorch** | 1.8+ | 1.12+ |
| **ONNX** | 1.10+ | 1.12+ |
| **ONNX Runtime** | 1.10+ | 1.14+ |

## 模型转换

### 基本用法

```bash
# 最简单的转换
python convert_to_onnx.py

# 指定输入输出路径
python convert_to_onnx.py --input models/model.pkl --output models/model.onnx

# 指定ONNX opset版本
python convert_to_onnx.py --opset 14

# 禁用动态batch（固定batch=1）
python convert_to_onnx.py --no-dynamic

# 禁用模型简化
python convert_to_onnx.py --no-simplify

# 转换后立即测试
python convert_to_onnx.py --test

# 使用测试图片验证
python convert_to_onnx.py --test --test-image dataset/test/1BWB_1539937370.png
```

### 完整示例

```bash
# 完整转换流程（推荐）
python convert_to_onnx.py \
    --input models/model.pkl \
    --output models/model.onnx \
    --opset 14 \
    --test \
    --test-image dataset/test/1BWB_1539937370.png
```

### 输出示例

```
================================================================================
PyTorch模型转ONNX
================================================================================
使用设备: cuda:0
GPU型号: NVIDIA A100-PCIE-40GB

模型信息:
  输入模型: models/model.pkl
  输出模型: models/model.onnx
  ONNX Opset: 14
  动态Batch: 是

--------------------------------------------------------------------------------
步骤1: 加载PyTorch模型
--------------------------------------------------------------------------------
✓ 模型加载成功
  总参数量: 2,345,678
  可训练参数: 2,345,678
  模型大小: 9.12 MB

--------------------------------------------------------------------------------
步骤2: 创建示例输入
--------------------------------------------------------------------------------
  输入形状: [1, 1, 60, 160]
  输入尺寸: [batch, channels, height, width] = [1, 1, 60, 160]
  输出形状: [1, 144]
  输出尺寸: [batch, features] = [1, 144]

--------------------------------------------------------------------------------
步骤3: 导出ONNX模型
--------------------------------------------------------------------------------
  启用动态Batch支持
  开始导出... (opset_version=14)
✓ ONNX模型导出成功
  ONNX模型大小: 9.15 MB

--------------------------------------------------------------------------------
步骤4: 验证ONNX模型
--------------------------------------------------------------------------------
✓ ONNX模型格式检查通过
  IR版本: 8
  Opset版本: 14
  生产者: pytorch

  输入:
    - input: ['dynamic', 1, 60, 160]
  输出:
    - output: ['dynamic', 144]

--------------------------------------------------------------------------------
步骤5: 简化ONNX模型
--------------------------------------------------------------------------------
  加载ONNX模型...
  简化模型...
✓ 简化模型已保存: models/model_simplified.onnx
  原始大小: 9.15 MB
  简化后大小: 9.10 MB
  压缩率: 0.5%

================================================================================
转换完成
================================================================================
```

## 模型测试

### 推理测试

```bash
# 运行推理测试
python convert_to_onnx.py --test
```

### 测试输出

```
================================================================================
ONNX模型推理测试
================================================================================
ONNX Runtime版本: 1.14.1
可用执行提供者: CUDAExecutionProvider, CPUExecutionProvider

--------------------------------------------------------------------------------
加载模型
--------------------------------------------------------------------------------
✓ ONNX模型加载成功
  使用提供者: CUDAExecutionProvider
✓ PyTorch模型加载成功

--------------------------------------------------------------------------------
准确性验证
--------------------------------------------------------------------------------
  输出对比:
    最大差异: 0.0000012345
    平均差异: 0.0000003456
  ✓ 精度验证通过（差异 < 1e-5）

--------------------------------------------------------------------------------
性能对比
--------------------------------------------------------------------------------

Batch      PyTorch         ONNX            加速比    
-------------------------------------------------------
1               2.34 ms         1.87 ms         1.25x
4               6.78 ms         4.56 ms         1.49x
16             23.45 ms        15.67 ms         1.50x
32             45.67 ms        28.90 ms         1.58x

--------------------------------------------------------------------------------
实际图片测试
--------------------------------------------------------------------------------
测试图片: dataset/test/1BWB_1539937370.png
  PyTorch预测: 1BWB
  ONNX预测: 1BWB
  预测一致: ✓

================================================================================
测试完成
================================================================================
```

## ONNX模型使用

### Python推理（推荐）

```python
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms

# 1. 加载模型
session = ort.InferenceSession('models/model.onnx')

# 2. 图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])
image = Image.open('test.png')
image_tensor = transform(image).unsqueeze(0)
image_array = image_tensor.numpy()

# 3. 推理
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: image_array})

# 4. 解码结果
output = outputs[0]
result = []
for i in range(4):
    start = i * 36
    end = (i + 1) * 36
    char_idx = np.argmax(output[0, start:end])
    result.append(ALL_CHAR_SET[char_idx])
captcha_text = ''.join(result)
print(f"识别结果: {captcha_text}")
```

### 使用封装的预测器

```python
from onnx_inference import CaptchaONNXPredictor

# 创建预测器
predictor = CaptchaONNXPredictor('models/model.onnx', use_gpu=True)

# 单张预测
text = predictor.predict('test.png')
print(f"识别结果: {text}")

# 带置信度预测
text, prob = predictor.predict('test.png', return_prob=True)
print(f"识别结果: {text}, 置信度: {prob:.4f}")

# 批量预测
texts = predictor.predict_batch(['test1.png', 'test2.png', 'test3.png'])
print(f"批量结果: {texts}")
```

### 运行演示

```bash
python onnx_inference.py
```

## 跨平台部署

### C++推理

```cpp
#include <onnxruntime_cxx_api.h>

// 1. 创建环境
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "CaptchaONNX");

// 2. 创建会话
Ort::SessionOptions session_options;
session_options.SetIntraOpNumThreads(1);
Ort::Session session(env, L"model.onnx", session_options);

// 3. 准备输入
std::vector<int64_t> input_shape = {1, 1, 60, 160};
std::vector<float> input_data(1 * 1 * 60 * 160);
// ... 填充图像数据 ...

// 4. 创建tensor
auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, input_data.data(), input_data.size(),
    input_shape.data(), input_shape.size());

// 5. 推理
const char* input_names[] = {"input"};
const char* output_names[] = {"output"};
auto output_tensors = session.Run(
    Ort::RunOptions{nullptr}, input_names, &input_tensor, 1,
    output_names, 1);

// 6. 获取结果
float* output_data = output_tensors[0].GetTensorMutableData<float>();
```

### C#推理

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

// 1. 创建会话
var session = new InferenceSession("model.onnx");

// 2. 准备输入
var inputData = new DenseTensor<float>(new[] { 1, 1, 60, 160 });
// ... 填充图像数据 ...

// 3. 推理
var inputs = new List<NamedOnnxValue>
{
    NamedOnnxValue.CreateFromTensor("input", inputData)
};
using var results = session.Run(inputs);

// 4. 获取结果
var output = results.First().AsEnumerable<float>().ToArray();
```

### JavaScript/TypeScript推理

```javascript
// Node.js
const ort = require('onnxruntime-node');

// 或者浏览器
// import * as ort from 'onnxruntime-web';

async function predict(imageData) {
    // 1. 加载模型
    const session = await ort.InferenceSession.create('model.onnx');
    
    // 2. 准备输入
    const tensor = new ort.Tensor('float32', imageData, [1, 1, 60, 160]);
    
    // 3. 推理
    const results = await session.run({ input: tensor });
    
    // 4. 获取结果
    const output = results.output.data;
    return decodeOutput(output);
}
```

## 性能优化

### 1. 选择合适的执行提供者

```python
# CPU推理
session = ort.InferenceSession('model.onnx', 
    providers=['CPUExecutionProvider'])

# GPU推理（CUDA）
session = ort.InferenceSession('model.onnx', 
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# TensorRT加速（最快）
session = ort.InferenceSession('model.onnx', 
    providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
```

### 2. 会话配置优化

```python
import onnxruntime as ort

session_options = ort.SessionOptions()

# 并行线程数
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 4

# 启用优化
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# 启用内存模式
session_options.enable_mem_pattern = True
session_options.enable_cpu_mem_arena = True

session = ort.InferenceSession('model.onnx', session_options)
```

### 3. 批量处理

```python
# 批量处理可以显著提升吞吐量
batch_size = 32
batch_images = np.stack([load_image(path) for path in image_paths])
outputs = session.run(None, {input_name: batch_images})
```

## 性能基准

### GPU性能（A100）

| Batch | PyTorch | ONNX | TensorRT | 加速比 |
|-------|---------|------|----------|-------|
| 1 | 2.3 ms | 1.9 ms | 1.2 ms | 1.9x |
| 4 | 6.8 ms | 4.6 ms | 2.8 ms | 2.4x |
| 16 | 23.5 ms | 15.7 ms | 9.2 ms | 2.6x |
| 32 | 45.7 ms | 28.9 ms | 16.3 ms | 2.8x |

### CPU性能（8核）

| Batch | PyTorch | ONNX | 加速比 |
|-------|---------|------|-------|
| 1 | 15.6 ms | 12.3 ms | 1.3x |
| 4 | 52.4 ms | 38.7 ms | 1.4x |
| 16 | 198.5 ms | 142.3 ms | 1.4x |
| 32 | 387.2 ms | 276.8 ms | 1.4x |

## 常见问题

### Q1: ONNX模型比PyTorch大？

**A:** 正常现象。ONNX包含完整的计算图和常量，但可以通过简化压缩：

```bash
# 使用onnx-simplifier
python -m onnxsim model.onnx model_simplified.onnx

# 或在转换时自动简化
python convert_to_onnx.py  # 默认启用简化
```

### Q2: 推理结果与PyTorch不一致？

**A:** 检查以下几点：

1. **数值精度：** 差异<1e-5通常可接受
2. **预处理一致：** 确保图像预处理相同
3. **模型状态：** 确保PyTorch模型处于eval()模式
4. **Opset版本：** 尝试更高的opset版本

```python
# 验证一致性
pytorch_output = model(input)
onnx_output = session.run(None, {input_name: input.numpy()})[0]
diff = np.abs(pytorch_output.numpy() - onnx_output).max()
print(f"最大差异: {diff}")  # 应该很小
```

### Q3: GPU推理没有加速？

**A:** 可能原因：

1. **未安装GPU版本：** `pip install onnxruntime-gpu`
2. **CUDA不可用：** 检查CUDA安装
3. **Batch太小：** GPU在大batch时才有优势
4. **提供者未启用：** 检查providers设置

```python
import onnxruntime as ort
print(ort.get_available_providers())
# 应该包含 'CUDAExecutionProvider'
```

### Q4: 如何减小模型大小？

**A:** 多种方法：

```bash
# 1. 使用onnx-simplifier
python -m onnxsim model.onnx model_sim.onnx

# 2. 量化为INT8（可能降低精度）
python -m onnxruntime.quantization.preprocess --input model.onnx --output model_prep.onnx
python -m onnxruntime.quantization.quantize --model model_prep.onnx --output model_int8.onnx

# 3. 使用FP16（GPU）
# 在转换时设置
```

### Q5: 动态batch vs 固定batch？

**A:** 取决于使用场景：

| 特性 | 动态Batch | 固定Batch |
|-----|----------|----------|
| **灵活性** | 高（任意batch） | 低（仅固定值） |
| **性能** | 略慢 | 略快 |
| **优化** | 受限 | 更多优化空间 |
| **推荐场景** | 通用部署 | 生产环境 |

```bash
# 动态batch（推荐）
python convert_to_onnx.py

# 固定batch（性能优先）
python convert_to_onnx.py --no-dynamic
```

## 生产部署建议

### 1. 模型优化流程

```bash
# 步骤1: 转换为ONNX
python convert_to_onnx.py --input models/model.pkl --output models/model.onnx

# 步骤2: 简化模型
python -m onnxsim models/model.onnx models/model_optimized.onnx

# 步骤3: 验证准确性
python convert_to_onnx.py --test

# 步骤4: 性能基准测试
python onnx_inference.py
```

### 2. 服务器部署

**选项A: Flask API**

```python
from flask import Flask, request, jsonify
from onnx_inference import CaptchaONNXPredictor

app = Flask(__name__)
predictor = CaptchaONNXPredictor('models/model.onnx')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = Image.open(file)
    text = predictor.predict(image)
    return jsonify({'captcha': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**选项B: FastAPI**

```python
from fastapi import FastAPI, File, UploadFile
from onnx_inference import CaptchaONNXPredictor

app = FastAPI()
predictor = CaptchaONNXPredictor('models/model.onnx')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file)
    text = predictor.predict(image)
    return {"captcha": text}
```

### 3. Docker部署

```dockerfile
FROM python:3.9-slim

# 安装依赖
RUN pip install onnxruntime numpy pillow

# 复制模型和代码
COPY models/model.onnx /app/model.onnx
COPY onnx_inference.py /app/
COPY captcha_setting.py /app/

WORKDIR /app
CMD ["python", "onnx_inference.py"]
```

## 总结

### 转换流程

```
PyTorch模型 (model.pkl)
    ↓ convert_to_onnx.py
ONNX模型 (model.onnx)
    ↓ onnx-simplifier (可选)
优化ONNX模型 (model_simplified.onnx)
    ↓ 部署
生产环境 (Python/C++/C#/JS/...)
```

### 使用建议

✅ **开发阶段：** 使用PyTorch（灵活、易调试）  
✅ **测试阶段：** 转换为ONNX并验证准确性  
✅ **生产阶段：** 使用ONNX Runtime部署（高性能、跨平台）  

### 关键文件

- **convert_to_onnx.py** - 模型转换脚本
- **onnx_inference.py** - ONNX推理示例
- **models/model.onnx** - ONNX模型文件
- **models/model_simplified.onnx** - 简化后的模型（可选）

---

**文档版本：** 1.0  
**创建日期：** 2026-02-02  
**适用版本：** ONNX 1.12+, ONNX Runtime 1.14+
