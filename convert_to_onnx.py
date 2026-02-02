# -*- coding: UTF-8 -*-
"""
PyTorch模型转ONNX脚本
将 models/model.pkl 转换为跨平台的 ONNX 模型
"""
import torch
import torch.onnx
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import captcha_setting
from captcha_cnn_model import CNN
import os
import time

def convert_to_onnx(
    pytorch_model_path='models/model.pkl',
    onnx_model_path='models/model.onnx',
    opset_version=14,
    dynamic_batch=True,
    simplify=True
):
    """
    将PyTorch模型转换为ONNX格式
    
    参数:
        pytorch_model_path: PyTorch模型路径
        onnx_model_path: 输出ONNX模型路径
        opset_version: ONNX opset版本（推荐11-16）
        dynamic_batch: 是否支持动态batch size
        simplify: 是否简化ONNX模型（需要安装onnx-simplifier）
    """
    print("=" * 80)
    print("PyTorch模型转ONNX")
    print("=" * 80)
    
    # 检测设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 检查PyTorch模型是否存在
    if not os.path.exists(pytorch_model_path):
        # 尝试当前目录
        pytorch_model_path = 'model.pkl'
        if not os.path.exists(pytorch_model_path):
            print(f"错误：找不到模型文件")
            return False
    
    print(f"\n模型信息:")
    print(f"  输入模型: {pytorch_model_path}")
    print(f"  输出模型: {onnx_model_path}")
    print(f"  ONNX Opset: {opset_version}")
    print(f"  动态Batch: {'是' if dynamic_batch else '否'}")
    
    # 创建输出目录
    output_dir = os.path.dirname(onnx_model_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建目录: {output_dir}")
    
    print("\n" + "-" * 80)
    print("步骤1: 加载PyTorch模型")
    print("-" * 80)
    
    # 加载模型
    model = CNN()
    model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    model.to(device)
    model.eval()  # 设置为评估模式
    print("✓ 模型加载成功")
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {os.path.getsize(pytorch_model_path) / 1024 / 1024:.2f} MB")
    
    print("\n" + "-" * 80)
    print("步骤2: 创建示例输入")
    print("-" * 80)
    
    # 创建示例输入（灰度图）
    # 输入形状: [batch_size, channels, height, width]
    batch_size = 1
    channels = 1
    height = captcha_setting.IMAGE_HEIGHT
    width = captcha_setting.IMAGE_WIDTH
    
    dummy_input = torch.randn(batch_size, channels, height, width, device=device)
    print(f"  输入形状: {list(dummy_input.shape)}")
    print(f"  输入尺寸: [batch, channels, height, width] = [{batch_size}, {channels}, {height}, {width}]")
    
    # 测试前向传播
    with torch.no_grad():
        dummy_output = model(dummy_input)
    print(f"  输出形状: {list(dummy_output.shape)}")
    print(f"  输出尺寸: [batch, features] = [{dummy_output.shape[0]}, {dummy_output.shape[1]}]")
    
    print("\n" + "-" * 80)
    print("步骤3: 导出ONNX模型")
    print("-" * 80)
    
    # 定义输入/输出名称
    input_names = ['input']
    output_names = ['output']
    
    # 动态axes配置（支持动态batch size）
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        print("  启用动态Batch支持")
    else:
        dynamic_axes = None
        print("  使用固定Batch")
    
    # 导出ONNX
    print(f"  开始导出... (opset_version={opset_version})")
    
    try:
        torch.onnx.export(
            model,                          # 模型
            dummy_input,                    # 示例输入
            onnx_model_path,               # 输出路径
            export_params=True,            # 导出参数
            opset_version=opset_version,   # ONNX版本
            do_constant_folding=True,      # 常量折叠优化
            input_names=input_names,       # 输入名称
            output_names=output_names,     # 输出名称
            dynamic_axes=dynamic_axes,     # 动态axes
            verbose=False                  # 不打印详细信息
        )
        print("✓ ONNX模型导出成功")
        onnx_size = os.path.getsize(onnx_model_path) / 1024 / 1024
        print(f"  ONNX模型大小: {onnx_size:.2f} MB")
    except Exception as e:
        print(f"✗ 导出失败: {e}")
        return False
    
    print("\n" + "-" * 80)
    print("步骤4: 验证ONNX模型")
    print("-" * 80)
    
    try:
        import onnx
        
        # 加载ONNX模型
        onnx_model = onnx.load(onnx_model_path)
        
        # 检查模型格式
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX模型格式检查通过")
        
        # 打印模型信息
        print(f"  IR版本: {onnx_model.ir_version}")
        print(f"  Opset版本: {onnx_model.opset_import[0].version}")
        print(f"  生产者: {onnx_model.producer_name}")
        
        # 打印输入输出信息
        graph = onnx_model.graph
        print(f"\n  输入:")
        for input_tensor in graph.input:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"    - {input_tensor.name}: {shape}")
        
        print(f"  输出:")
        for output_tensor in graph.output:
            shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"    - {output_tensor.name}: {shape}")
        
    except ImportError:
        print("⚠ 未安装onnx包，跳过验证")
        print("  安装命令: pip install onnx")
    except Exception as e:
        print(f"✗ 验证失败: {e}")
        return False
    
    # 简化ONNX模型（可选）
    if simplify:
        print("\n" + "-" * 80)
        print("步骤5: 简化ONNX模型")
        print("-" * 80)
        
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("  加载ONNX模型...")
            onnx_model = onnx.load(onnx_model_path)
            
            print("  简化模型...")
            model_simplified, check = onnx_simplify(onnx_model)
            
            if check:
                simplified_path = onnx_model_path.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, simplified_path)
                print(f"✓ 简化模型已保存: {simplified_path}")
                
                original_size = os.path.getsize(onnx_model_path) / 1024 / 1024
                simplified_size = os.path.getsize(simplified_path) / 1024 / 1024
                print(f"  原始大小: {original_size:.2f} MB")
                print(f"  简化后大小: {simplified_size:.2f} MB")
                print(f"  压缩率: {(1 - simplified_size/original_size)*100:.1f}%")
            else:
                print("⚠ 简化验证失败，使用原始模型")
        except ImportError:
            print("⚠ 未安装onnx-simplifier，跳过简化")
            print("  安装命令: pip install onnx-simplifier")
        except Exception as e:
            print(f"⚠ 简化失败: {e}")
    
    print("\n" + "=" * 80)
    print("转换完成")
    print("=" * 80)
    return True

def test_onnx_inference(
    onnx_model_path='models/model.onnx',
    pytorch_model_path='models/model.pkl',
    test_image_path=None,
    batch_sizes=[1, 4, 16, 32]
):
    """
    测试ONNX模型推理并与PyTorch对比
    """
    print("\n" + "=" * 80)
    print("ONNX模型推理测试")
    print("=" * 80)
    
    # 检查ONNX Runtime
    try:
        import onnxruntime as ort
        print(f"ONNX Runtime版本: {ort.__version__}")
        
        # 检查可用的执行提供者
        available_providers = ort.get_available_providers()
        print(f"可用执行提供者: {', '.join(available_providers)}")
    except ImportError:
        print("错误：未安装onnxruntime")
        print("安装命令: pip install onnxruntime  # CPU版本")
        print("         pip install onnxruntime-gpu  # GPU版本")
        return False
    
    print("\n" + "-" * 80)
    print("加载模型")
    print("-" * 80)
    
    # 加载ONNX模型
    try:
        # 优先使用GPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else ['CPUExecutionProvider']
        ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
        print(f"✓ ONNX模型加载成功")
        print(f"  使用提供者: {ort_session.get_providers()[0]}")
    except Exception as e:
        print(f"✗ ONNX模型加载失败: {e}")
        return False
    
    # 加载PyTorch模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pytorch_model = CNN()
    pytorch_model.load_state_dict(torch.load(pytorch_model_path, map_location=device))
    pytorch_model.to(device)
    pytorch_model.eval()
    print(f"✓ PyTorch模型加载成功")
    
    print("\n" + "-" * 80)
    print("准确性验证")
    print("-" * 80)
    
    # 创建测试输入
    test_input = torch.randn(1, 1, captcha_setting.IMAGE_HEIGHT, captcha_setting.IMAGE_WIDTH)
    
    # PyTorch推理
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input.to(device)).cpu().numpy()
    
    # ONNX推理
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
    onnx_output = ort_session.run(None, ort_inputs)[0]
    
    # 对比结果
    max_diff = np.abs(pytorch_output - onnx_output).max()
    mean_diff = np.abs(pytorch_output - onnx_output).mean()
    
    print(f"  输出对比:")
    print(f"    最大差异: {max_diff:.10f}")
    print(f"    平均差异: {mean_diff:.10f}")
    
    if max_diff < 1e-5:
        print("  ✓ 精度验证通过（差异 < 1e-5）")
    elif max_diff < 1e-3:
        print("  ✓ 精度验证通过（差异 < 1e-3，可接受）")
    else:
        print(f"  ⚠ 精度差异较大（差异 = {max_diff}）")
    
    print("\n" + "-" * 80)
    print("性能对比")
    print("-" * 80)
    
    # 预热
    for _ in range(10):
        _ = ort_session.run(None, ort_inputs)
    
    print(f"\n{'Batch':<10} {'PyTorch':<15} {'ONNX':<15} {'加速比':<10}")
    print("-" * 55)
    
    for batch_size in batch_sizes:
        # 创建输入
        test_input = torch.randn(batch_size, 1, captcha_setting.IMAGE_HEIGHT, captcha_setting.IMAGE_WIDTH)
        
        # PyTorch推理时间
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = pytorch_model(test_input.to(device))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        pytorch_time = (time.time() - start) / 100 * 1000  # ms
        
        # ONNX推理时间
        ort_inputs = {ort_session.get_inputs()[0].name: test_input.numpy()}
        
        start = time.time()
        for _ in range(100):
            _ = ort_session.run(None, ort_inputs)
        onnx_time = (time.time() - start) / 100 * 1000  # ms
        
        speedup = pytorch_time / onnx_time
        print(f"{batch_size:<10} {pytorch_time:>10.2f} ms   {onnx_time:>10.2f} ms   {speedup:>8.2f}x")
    
    # 如果提供了测试图片，进行实际预测
    if test_image_path and os.path.exists(test_image_path):
        print("\n" + "-" * 80)
        print("实际图片测试")
        print("-" * 80)
        print(f"测试图片: {test_image_path}")
        
        # 图像预处理
        transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        image = Image.open(test_image_path)
        image_tensor = transform(image).unsqueeze(0)
        
        # PyTorch预测
        with torch.no_grad():
            pytorch_pred = pytorch_model(image_tensor.to(device)).cpu().numpy()
        
        # ONNX预测
        ort_inputs = {ort_session.get_inputs()[0].name: image_tensor.numpy()}
        onnx_pred = ort_session.run(None, ort_inputs)[0]
        
        # 解码
        def decode_output(output):
            result = []
            for i in range(4):
                start = i * captcha_setting.ALL_CHAR_SET_LEN
                end = (i + 1) * captcha_setting.ALL_CHAR_SET_LEN
                char_idx = np.argmax(output[0, start:end])
                result.append(captcha_setting.ALL_CHAR_SET[char_idx])
            return ''.join(result)
        
        pytorch_text = decode_output(pytorch_pred)
        onnx_text = decode_output(onnx_pred)
        
        print(f"  PyTorch预测: {pytorch_text}")
        print(f"  ONNX预测: {onnx_text}")
        print(f"  预测一致: {'✓' if pytorch_text == onnx_text else '✗'}")
    
    print("\n" + "=" * 80)
    print("测试完成")
    print("=" * 80)
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PyTorch模型转ONNX')
    parser.add_argument('--input', type=str, default='models/model.pkl', help='输入PyTorch模型路径')
    parser.add_argument('--output', type=str, default='models/model.onnx', help='输出ONNX模型路径')
    parser.add_argument('--opset', type=int, default=14, help='ONNX opset版本 (11-16)')
    parser.add_argument('--no-dynamic', action='store_true', help='禁用动态batch')
    parser.add_argument('--no-simplify', action='store_true', help='禁用模型简化')
    parser.add_argument('--test', action='store_true', help='转换后测试推理')
    parser.add_argument('--test-image', type=str, default=None, help='测试图片路径')
    
    args = parser.parse_args()
    
    # 转换模型
    success = convert_to_onnx(
        pytorch_model_path=args.input,
        onnx_model_path=args.output,
        opset_version=args.opset,
        dynamic_batch=not args.no_dynamic,
        simplify=not args.no_simplify
    )
    
    if not success:
        print("\n转换失败！")
        return
    
    # 测试推理
    if args.test:
        test_onnx_inference(
            onnx_model_path=args.output,
            pytorch_model_path=args.input,
            test_image_path=args.test_image
        )
    
    print("\n使用说明:")
    print("-" * 80)
    print("Python推理:")
    print("  import onnxruntime as ort")
    print("  session = ort.InferenceSession('models/model.onnx')")
    print("  outputs = session.run(None, {'input': image_array})")
    print("")
    print("其他语言:")
    print("  - C++: 使用 ONNX Runtime C++ API")
    print("  - C#: 使用 Microsoft.ML.OnnxRuntime NuGet包")
    print("  - Java: 使用 ONNX Runtime Java API")
    print("  - JavaScript: 使用 onnxruntime-web")
    print("=" * 80)

if __name__ == '__main__':
    main()
