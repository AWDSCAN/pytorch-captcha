# -*- coding: UTF-8 -*-
"""
ONNX模型推理示例
展示如何使用ONNX模型进行验证码识别
"""
import numpy as np
import onnxruntime as ort
from PIL import Image
import torchvision.transforms as transforms
import captcha_setting
import os
import time

class CaptchaONNXPredictor:
    """ONNX验证码识别器"""
    
    def __init__(self, model_path='models/model.onnx', use_gpu=True):
        """
        初始化ONNX预测器
        
        参数:
            model_path: ONNX模型路径
            use_gpu: 是否使用GPU（如果可用）
        """
        self.model_path = model_path
        
        # 设置执行提供者
        providers = []
        if use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # 加载模型
        self.session = ort.InferenceSession(model_path, providers=providers)
        
        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        print(f"✓ ONNX模型加载成功: {model_path}")
        print(f"  执行提供者: {self.session.get_providers()[0]}")
    
    def preprocess(self, image_path):
        """
        图像预处理
        
        参数:
            image_path: 图片路径或PIL.Image对象
        
        返回:
            numpy数组 [1, 1, H, W]
        """
        if isinstance(image_path, str):
            image = Image.open(image_path)
        else:
            image = image_path
        
        # 转换为tensor并添加batch维度
        image_tensor = self.transform(image).unsqueeze(0)
        
        # 转为numpy
        return image_tensor.numpy()
    
    def decode(self, output):
        """
        解码模型输出为验证码文本
        
        参数:
            output: 模型输出 [1, 144]
        
        返回:
            验证码文本字符串
        """
        result = []
        for i in range(captcha_setting.MAX_CAPTCHA):
            start = i * captcha_setting.ALL_CHAR_SET_LEN
            end = (i + 1) * captcha_setting.ALL_CHAR_SET_LEN
            char_idx = np.argmax(output[0, start:end])
            result.append(captcha_setting.ALL_CHAR_SET[char_idx])
        return ''.join(result)
    
    def predict(self, image_path, return_prob=False):
        """
        预测验证码
        
        参数:
            image_path: 图片路径
            return_prob: 是否返回概率
        
        返回:
            验证码文本或(文本, 概率)
        """
        # 预处理
        image_array = self.preprocess(image_path)
        
        # 推理
        outputs = self.session.run([self.output_name], {self.input_name: image_array})
        output = outputs[0]
        
        # 解码
        text = self.decode(output)
        
        if return_prob:
            # 计算平均置信度
            probs = []
            for i in range(captcha_setting.MAX_CAPTCHA):
                start = i * captcha_setting.ALL_CHAR_SET_LEN
                end = (i + 1) * captcha_setting.ALL_CHAR_SET_LEN
                # 使用softmax计算概率
                logits = output[0, start:end]
                exp_logits = np.exp(logits - np.max(logits))
                softmax_probs = exp_logits / exp_logits.sum()
                probs.append(softmax_probs.max())
            
            avg_prob = np.mean(probs)
            return text, avg_prob
        else:
            return text
    
    def predict_batch(self, image_paths):
        """
        批量预测
        
        参数:
            image_paths: 图片路径列表
        
        返回:
            验证码文本列表
        """
        # 批量预处理
        images = []
        for path in image_paths:
            images.append(self.preprocess(path))
        
        # 合并为batch
        batch_array = np.concatenate(images, axis=0)
        
        # 批量推理
        outputs = self.session.run([self.output_name], {self.input_name: batch_array})
        output = outputs[0]
        
        # 批量解码
        results = []
        for i in range(len(image_paths)):
            text = self.decode(output[i:i+1])
            results.append(text)
        
        return results

def demo():
    """演示ONNX推理"""
    print("=" * 80)
    print("ONNX验证码识别演示")
    print("=" * 80)
    
    # 检查模型是否存在
    model_path = 'models/model.onnx'
    if not os.path.exists(model_path):
        print(f"错误：ONNX模型不存在 - {model_path}")
        print("请先运行: python convert_to_onnx.py")
        return
    
    # 创建预测器
    predictor = CaptchaONNXPredictor(model_path, use_gpu=True)
    
    # 找测试图片
    test_dir = captcha_setting.TEST_DATASET_PATH
    if not os.path.exists(test_dir):
        test_dir = captcha_setting.TRAIN_DATASET_PATH
    
    if not os.path.exists(test_dir):
        print(f"错误：找不到测试图片目录")
        return
    
    # 获取图片列表
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if not image_files:
        print(f"错误：目录中没有图片")
        return
    
    # 测试单张图片
    print("\n" + "-" * 80)
    print("单张图片测试")
    print("-" * 80)
    
    test_file = image_files[0]
    test_path = os.path.join(test_dir, test_file)
    true_label = test_file.split('_')[0]
    
    # 预测
    start = time.time()
    pred_text, prob = predictor.predict(test_path, return_prob=True)
    elapsed = (time.time() - start) * 1000
    
    print(f"图片: {test_file}")
    print(f"真实标签: {true_label}")
    print(f"预测结果: {pred_text}")
    print(f"置信度: {prob:.4f}")
    print(f"预测正确: {'✓' if pred_text == true_label else '✗'}")
    print(f"推理时间: {elapsed:.2f} ms")
    
    # 批量测试
    print("\n" + "-" * 80)
    print("批量预测测试")
    print("-" * 80)
    
    batch_size = 10
    batch_files = image_files[:batch_size]
    batch_paths = [os.path.join(test_dir, f) for f in batch_files]
    true_labels = [f.split('_')[0] for f in batch_files]
    
    # 预测
    start = time.time()
    pred_texts = predictor.predict_batch(batch_paths)
    elapsed = (time.time() - start) * 1000
    
    print(f"批量大小: {batch_size}")
    print(f"总耗时: {elapsed:.2f} ms")
    print(f"平均耗时: {elapsed/batch_size:.2f} ms/张")
    print(f"吞吐量: {batch_size/(elapsed/1000):.1f} 张/秒")
    
    correct = sum(1 for true, pred in zip(true_labels, pred_texts) if true == pred)
    print(f"准确率: {correct}/{batch_size} = {correct/batch_size*100:.2f}%")
    
    print("\n预测结果:")
    for i, (file, true, pred) in enumerate(zip(batch_files, true_labels, pred_texts), 1):
        status = '✓' if true == pred else '✗'
        print(f"  {i:2d}. {file:30s} | 真实={true} | 预测={pred} | {status}")
    
    # 性能测试
    print("\n" + "-" * 80)
    print("性能基准测试")
    print("-" * 80)
    
    num_warmup = 10
    num_iterations = 100
    
    # 预热
    for _ in range(num_warmup):
        _ = predictor.predict(test_path)
    
    # 测试不同batch size
    print(f"\n{'Batch Size':<15} {'吞吐量 (张/秒)':<20} {'平均延迟 (ms)':<20}")
    print("-" * 55)
    
    for batch_size in [1, 4, 8, 16, 32]:
        batch_paths = [os.path.join(test_dir, f) for f in image_files[:batch_size]]
        
        start = time.time()
        for _ in range(num_iterations):
            _ = predictor.predict_batch(batch_paths)
        elapsed = time.time() - start
        
        throughput = (num_iterations * batch_size) / elapsed
        latency = (elapsed / num_iterations) * 1000
        
        print(f"{batch_size:<15} {throughput:>15.1f}     {latency:>15.2f}")
    
    print("\n" + "=" * 80)
    print("演示完成")
    print("=" * 80)

if __name__ == '__main__':
    demo()
