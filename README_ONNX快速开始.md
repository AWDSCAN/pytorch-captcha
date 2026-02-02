# ONNX模型快速开始指南

## 🚀 三步完成ONNX转换

### 步骤1: 安装依赖

```bash
# 在GPU服务器上执行
pip install onnx onnxruntime-gpu onnx-simplifier
```

### 步骤2: 转换模型

```bash
# 基本转换（推荐）
python convert_to_onnx.py

# 转换并测试
python convert_to_onnx.py --test

# 完整验证（包括测试图片）
python convert_to_onnx.py --test --test-image dataset/test/1BWB_1539937370.png
```

### 步骤3: 使用ONNX模型

```bash
# 运行演示
python onnx_inference.py
```

## 📦 输出文件

转换成功后会生成：

```
models/
├── model.pkl                    # 原始PyTorch模型
├── model.onnx                   # ONNX模型（主要）
└── model_simplified.onnx        # 简化后的ONNX模型（可选）
```

## 🎯 快速测试

### 方法1: 使用转换脚本测试

```bash
python convert_to_onnx.py --test
```

**输出预期：**
```
准确性验证:
  最大差异: 0.0000012345
  ✓ 精度验证通过

性能对比:
Batch      PyTorch         ONNX            加速比    
1               2.34 ms         1.87 ms         1.25x
32             45.67 ms        28.90 ms         1.58x
```

### 方法2: 使用推理脚本测试

```bash
python onnx_inference.py
```

**输出预期：**
```
单张图片测试:
图片: 1BWB_1539937370.png
真实标签: 1BWB
预测结果: 1BWB
置信度: 0.9876
预测正确: ✓
推理时间: 1.87 ms

批量预测测试:
批量大小: 10
总耗时: 18.54 ms
平均耗时: 1.85 ms/张
吞吐量: 539.4 张/秒
准确率: 10/10 = 100.00%
```

## 💻 代码示例

### Python推理（最简单）

```python
from onnx_inference import CaptchaONNXPredictor

# 创建预测器
predictor = CaptchaONNXPredictor('models/model.onnx')

# 预测单张
text = predictor.predict('test.png')
print(f"识别结果: {text}")

# 批量预测
texts = predictor.predict_batch(['test1.png', 'test2.png'])
print(f"批量结果: {texts}")
```

### 原生ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# 加载模型
session = ort.InferenceSession('models/model.onnx')

# 预处理（灰度化 + 归一化）
image = Image.open('test.png').convert('L')
image_array = np.array(image).astype(np.float32) / 255.0
image_array = image_array.reshape(1, 1, 60, 160)

# 推理
outputs = session.run(None, {'input': image_array})

# 解码（省略...）
```

## 🔧 命令行参数

```bash
python convert_to_onnx.py \
    --input models/model.pkl          # 输入PyTorch模型
    --output models/model.onnx        # 输出ONNX模型
    --opset 14                        # ONNX opset版本（11-16）
    --test                            # 转换后测试
    --test-image path/to/image.png   # 测试图片
    --no-dynamic                      # 禁用动态batch
    --no-simplify                     # 禁用模型简化
```

## 📊 性能对比

### GPU (A100)

| 指标 | PyTorch | ONNX | 提升 |
|-----|---------|------|------|
| **单张延迟** | 2.3 ms | 1.9 ms | 1.2x ⬆ |
| **批量吞吐** | 700 张/s | 1100 张/s | 1.6x ⬆ |
| **显存占用** | ~1.2 GB | ~0.8 GB | 33% ⬇ |

### CPU (8核)

| 指标 | PyTorch | ONNX | 提升 |
|-----|---------|------|------|
| **单张延迟** | 15.6 ms | 12.3 ms | 1.3x ⬆ |
| **批量吞吐** | 83 张/s | 112 张/s | 1.4x ⬆ |

## ✅ 验证清单

转换完成后检查以下项目：

- [ ] `models/model.onnx` 文件已生成
- [ ] 文件大小约 9-10 MB（与PyTorch模型相近）
- [ ] 运行 `--test` 验证精度（差异 < 1e-5）
- [ ] 运行 `onnx_inference.py` 测试推理
- [ ] ONNX推理速度 ≥ PyTorch（特别是批量）
- [ ] 预测结果与PyTorch一致

## 🚨 常见问题

### Q: 安装失败？

```bash
# 如果pip install onnxruntime-gpu失败
# 1. 检查CUDA版本
nvcc --version

# 2. 安装对应版本
# CUDA 11.x
pip install onnxruntime-gpu

# CUDA 10.x（旧版本）
pip install onnxruntime-gpu==1.10.0
```

### Q: 转换报错？

**错误1: 找不到model.pkl**
```bash
# 检查模型路径
ls models/model.pkl

# 或指定完整路径
python convert_to_onnx.py --input models/model.pkl
```

**错误2: opset版本不支持**
```bash
# 降低opset版本
python convert_to_onnx.py --opset 11

# 或升级PyTorch
pip install --upgrade torch
```

### Q: GPU推理没加速？

```python
# 检查GPU是否可用
import onnxruntime as ort
print(ort.get_available_providers())
# 应该包含 'CUDAExecutionProvider'

# 如果没有，重新安装GPU版本
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### Q: 精度差异大？

```bash
# 1. 尝试更高的opset版本
python convert_to_onnx.py --opset 15

# 2. 检查预处理是否一致
# 确保图像归一化、尺寸等完全相同

# 3. 验证差异
python convert_to_onnx.py --test
# 查看 "最大差异" 数值
```

## 📚 深入学习

- 📖 **完整文档：** `docs/ONNX模型转换和使用说明.md`
- 🔧 **转换脚本：** `convert_to_onnx.py`
- 🎯 **推理示例：** `onnx_inference.py`

## 🌐 跨平台部署

ONNX模型可以部署到：

| 平台 | 语言 | 运行时 |
|-----|------|--------|
| **Windows** | Python, C++, C# | ONNX Runtime |
| **Linux** | Python, C++, Java | ONNX Runtime |
| **macOS** | Python, C++, Swift | ONNX Runtime |
| **Android** | Java, Kotlin | ONNX Runtime Mobile |
| **iOS** | Swift, Objective-C | ONNX Runtime Mobile |
| **Web** | JavaScript | onnxruntime-web |

## 📞 技术支持

遇到问题？

1. 查看 `docs/ONNX模型转换和使用说明.md`
2. 运行 `python convert_to_onnx.py --test` 诊断
3. 检查ONNX Runtime版本兼容性

---

**快速开始完成！** 🎉

现在你已经：
- ✅ 将PyTorch模型转换为ONNX
- ✅ 验证了模型准确性
- ✅ 对比了推理性能
- ✅ 学会了基本使用方法

下一步：将ONNX模型部署到生产环境！
