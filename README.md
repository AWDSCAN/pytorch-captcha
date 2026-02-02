深度学习识别验证码
=========

本项目致力于使用神经网络来识别各种验证码。

## 环境要求

### ✅ 推荐配置
- **Python**: 3.10 / 3.11 / 3.12
- **PyTorch**: 2.0+ / 2.1+ / 2.2+ / 2.3+
- **CUDA**: 11.8 / 12.1（如果使用GPU）

### 最低要求
- Python 3.8+
- PyTorch 1.6+（支持混合精度训练）

### GPU支持
- ✅ 完全支持所有NVIDIA显卡（N卡）
- ✅ 特别优化A100等高端显卡
- 💻 支持CPU训练（速度较慢）

> 详细版本兼容性说明请查看：[docs/GPU支持说明.md](docs/GPU支持说明.md)

特性
===
- __端到端，不需要做更多的图片预处理（比如图片字符切割、图片尺寸归一化、图片字符标记、字符图片特征提取）__
- __验证码包括数字、大写字母、小写__
- __采用自己生成的验证码来作为神经网络的训练集合、测试集合、预测集合__
- __纯四位数字，验证码识别率高达 99.9999 %__
- __四位数字 + 大写字符，验证码识别率约 96 %__
- __深度学习框架pytorch + 验证码生成器ImageCaptcha__
- __🚀 优化版支持GPU加速，A100训练速度提升10-15倍__
- __✨ 训练可视化：进度条、训练曲线、实时监控__ ⭐
- __📊 准确率可视化：训练/测试准确率实时追踪、详细错误分析__ ⭐⭐


原理
===

- __训练集合生成__

    使用常用的 Python 验证码生成库 ImageCaptcha，生成 10w 个验证码，并且都自动标记好;
    如果需要识别其他的验证码也同样的道理，寻找对应的验证码生成算法自动生成已经标记好的训练集合或者手动对标记，需要上万级别的数量，纯手工需要一定的时间，再或者可以借助一些网络的打码平台进行标记

- __训练卷积神经网络__
    构建一个多层的卷积网络，进行多标签分类模型的训练
    标记的每个字符都做 one-hot 编码
    批量输入图片集合和标记数据，大概15个Epoch后，准确率已经达到 96% 以上


验证码识别率展示
========
![](https://raw.githubusercontent.com/dee1024/pytorch-captcha-recognition/master/docs/number.png)
![](https://raw.githubusercontent.com/dee1024/pytorch-captcha-recognition/master/docs/number2.png)


快速开始
====

### 步骤一：环境安装

#### 1.1 安装Python依赖

**方式A：一键安装（推荐）**
```bash
pip install -r requirements.txt
```

**方式B：手动安装**
```bash
# 基础依赖
pip install captcha pillow numpy

# 可视化依赖（强烈推荐）
pip install tqdm matplotlib

# 安装PyTorch（CPU版本）
pip install torch torchvision torchaudio

# 安装PyTorch（GPU版本 - CUDA 11.8）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装PyTorch（GPU版本 - CUDA 12.1）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 1.2 检查GPU环境（可选）
```bash
python check_gpu.py
```
会自动检测GPU型号、显存、CUDA版本，并给出训练建议。

### 步骤二：生成验证码

支持两种生成方式：
- **Simple模式**：使用captcha库（简洁风格）
- **Custom模式**：自定义PIL生成（带干扰线和干扰点）
- **Mixed模式**：随机混合两种方式（推荐）⭐

```bash
# 直接运行（默认mixed模式，生成100张）
python captcha_gen.py
```

执行以上命令，会在目录 `dataset/train/` 下生成多张验证码图片。

**修改生成参数：** 编辑 `captcha_gen.py` 文件
```python
count = 100000      # 修改生成数量（推荐10万张）
mode = 'mixed'      # 生成模式：'simple', 'custom', 'mixed'
path = captcha_setting.TRAIN_DATASET_PATH  # 修改保存路径
```

> 详细使用说明请查看：[docs/验证码生成说明.md](docs/验证码生成说明.md)

### 步骤三：训练模型

#### 3.1 标准版训练（支持所有设备）
```bash
python captcha_train.py
```
- 自动检测并使用GPU（如果可用）
- 训练150轮，约需数小时
- 生成 `model.pkl` 模型文件
- **✨ 新增可视化功能：**
  - ✅ tqdm进度条（实时显示训练进度）
  - ✅ 自动生成训练曲线图
  - ✅ 保存训练历史（JSON）
  - ✅ 详细训练日志
  - ✅ ETA时间预估

#### 3.2 A100优化版训练（推荐高端GPU）⭐
```bash
python captcha_train_a100_optimized.py
```
- 支持所有NVIDIA GPU（特别优化A100/H100）
- TF32 + 混合精度加速
- A100上150轮仅需约50分钟
- **✨ 增强可视化：**
  - ✅ 进度条 + 4张训练曲线图
  - ✅ GPU显存监控
  - ✅ 累计时间统计
- 详见：[docs/A100优化指南.md](docs/A100优化指南.md)

### 步骤四：测试模型
```bash
python captcha_test.py
```
可以在控制台看到详细的测试报告：
- ✅ 整体准确率统计
- ✅ 字符级别准确率
- ✅ 位置错误分析
- ✅ 字符混淆统计（例如0和O）
- ✅ 自动生成4张可视化图表

如果准确率较低，回到步骤二，生成更多的图片集合再次训练。

### 步骤五：使用模型做预测
```bash
python captcha_predict.py
```
可以在控制台看到预测输出的结果。
    
## 项目结构

```
pytorch-captcha-recognition/
│
├── 📄 核心Python文件
│   ├── captcha_cnn_model.py              # CNN模型定义（7层卷积网络）
│   ├── captcha_train.py                  # 标准训练脚本
│   ├── captcha_train_a100_optimized.py   # A100优化训练脚本
│   ├── captcha_test.py                   # 模型测试脚本
│   ├── captcha_predict.py                # 验证码预测脚本
│   ├── captcha_gen.py                    # 验证码生成脚本
│   ├── captcha_setting.py                # 全局配置文件
│   ├── my_dataset.py                     # 数据集加载器
│   ├── one_hot_encoding.py               # 编码解码工具
│   └── check_gpu.py                      # GPU环境检测工具
│
├── 📁 dataset/                            # 数据集目录
│   ├── train/                            # 训练集（10w+张图片）
│   ├── test/                             # 测试集
│   └── predict/                          # 预测集
│
├── 📁 docs/                              # 📚 文档目录（所有说明文档）
│   ├── 版本兼容性说明.md                # Python/PyTorch版本要求
│   ├── 项目结构说明.md                  # 项目结构详细说明
│   ├── OPTIMIZATION_README.md           # 模型优化详细说明
│   ├── GPU支持说明.md                   # GPU支持情况说明
│   ├── A100优化指南.md                  # A100专属优化指南
│   ├── 优化对比.md                      # 优化前后对比表格
│   ├── number.png                       # 示例图片1
│   └── number2.png                      # 示例图片2
│
├── 📄 配置文件
│   ├── README.md                        # 项目主文档（本文件）
│   ├── LICENSE                          # 开源协议
│   └── .gitignore                       # Git忽略规则
│
└── 🎯 训练生成（运行后产生）
    ├── model.pkl                        # 训练好的模型
    └── model_epoch_*.pkl                # 训练检查点
```

> 详细的项目结构说明请查看：[docs/项目结构说明.md](docs/项目结构说明.md)

## 📚 文档索引

| 文档 | 说明 | 推荐阅读 |
|------|------|---------|
| [快速开始.md](docs/快速开始.md) | 5分钟快速上手指南 | ⭐⭐⭐ 必读 |
| [版本兼容性说明.md](docs/版本兼容性说明.md) | Python/PyTorch版本要求和安装指南 | ⭐⭐⭐ 必读 |
| [验证码生成说明.md](docs/验证码生成说明.md) | 验证码生成器使用说明（两种模式） | ⭐⭐⭐ 必读 |
| **[算力优化说明.md](docs/算力优化说明.md)** | **多GPU算力优化，训练速度提升5-10倍** | **⭐⭐⭐ 强烈推荐** 🚀 |
| **[训练可视化说明.md](docs/训练可视化说明.md)** | **训练进度条和曲线可视化** | **⭐⭐⭐ 必读** |
| **[准确率可视化说明.md](docs/准确率可视化说明.md)** | **训练/测试准确率可视化** | **⭐⭐⭐ 必读** |
| [项目结构说明.md](docs/项目结构说明.md) | 项目目录结构和文件详解 | ⭐⭐⭐ |
| [OPTIMIZATION_README.md](docs/OPTIMIZATION_README.md) | 模型优化详细说明（7层卷积、余弦学习率等） | ⭐⭐⭐ |
| [GPU支持说明.md](docs/GPU支持说明.md) | GPU支持情况、硬件兼容性说明 | ⭐⭐ |
| [A100优化指南.md](docs/A100优化指南.md) | A100/H100等高端GPU优化指南 | ⭐⭐（GPU用户） |
| [优化对比.md](docs/优化对比.md) | 优化前后性能对比表格 | ⭐ |

## 性能对比

| 配置 | 每轮耗时 | 150轮总耗时 | 准确率 |
|------|---------|-----------|--------|
| CPU (Intel i7) | ~120分钟 | ~300小时 | 96% |
| GTX 1060 | ~8分钟 | ~20小时 | 96% |
| RTX 3090 | ~3分钟 | ~7.5小时 | 96% |
| A100 (标准) | ~2分钟 | ~5小时 | 96% |
| **A100 (优化)** | **~20秒** | **~50分钟** | **96%** ⭐ |

## 常见问题

### Q: 支持哪些GPU？
A: 支持所有NVIDIA显卡，包括GTX、RTX、Tesla、A100、H100等。不支持AMD显卡。

### Q: 没有GPU可以训练吗？
A: 可以，但训练速度较慢（约300小时）。建议至少使用GTX 1060以上显卡。

### Q: A100显卡使用哪个版本？
A: 强烈推荐使用 `captcha_train_a100_optimized.py`，性能提升10-15倍。

### Q: 准确率如何提升？
A: 
1. 增加训练数据量（10w+）
2. 使用优化版训练脚本
3. 训练更多轮次
4. 针对0和O混淆增加样本

贡献
===
我们期待你的 pull requests !

作者
===
* __原作者__: Dee Qiu <coolcooldee@gmail.com>
* __优化版本__: 添加7层卷积、GPU加速、A100优化等特性

其它
===
* __Github项目交流QQ群__ 570997546

声明
===
本项目仅用于交流学习