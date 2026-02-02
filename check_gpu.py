# -*- coding: UTF-8 -*-
"""
GPU环境检测脚本
快速检查CUDA和GPU信息
"""
import torch

def check_gpu_environment():
    """检查GPU环境配置"""
    print("=" * 60)
    print("PyTorch & CUDA 环境检测")
    print("=" * 60)
    
    # PyTorch版本
    print(f"\n【PyTorch信息】")
    print(f"  PyTorch版本: {torch.__version__}")
    print(f"  CUDA编译版本: {torch.version.cuda if torch.version.cuda else '未安装'}")
    print(f"  cuDNN版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else '未安装'}")
    
    # CUDA可用性
    print(f"\n【CUDA状态】")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA可用: {'✅ 是' if cuda_available else '❌ 否'}")
    
    if not cuda_available:
        print("\n⚠️  警告: 未检测到CUDA，将使用CPU训练（速度会很慢）")
        print("   解决方案:")
        print("   1. 检查是否安装了NVIDIA显卡驱动")
        print("   2. 安装CUDA版本的PyTorch:")
        print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return
    
    # GPU数量和信息
    gpu_count = torch.cuda.device_count()
    print(f"  可用GPU数量: {gpu_count}")
    
    print(f"\n【GPU详细信息】")
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    计算能力: {props.major}.{props.minor}")
        print(f"    总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"    多处理器数: {props.multi_processor_count}")
        
        # 架构识别
        compute_cap = f"{props.major}.{props.minor}"
        if props.major == 6:
            arch = "Pascal (GTX 10系列)"
        elif props.major == 7 and props.minor == 0:
            arch = "Volta (V100)"
        elif props.major == 7 and props.minor == 5:
            arch = "Turing (RTX 20系列, T4)"
        elif props.major == 8 and props.minor == 0:
            arch = "Ampere (A100, A30)"
            print(f"    🚀 架构: {arch} - 支持TF32加速!")
        elif props.major == 8 and props.minor == 6:
            arch = "Ampere (RTX 30系列, A10)"
            print(f"    架构: {arch}")
        elif props.major == 8 and props.minor == 9:
            arch = "Ada Lovelace (RTX 40系列)"
            print(f"    架构: {arch}")
        elif props.major == 9:
            arch = "Hopper (H100)"
            print(f"    🚀 架构: {arch} - 最新架构!")
        else:
            arch = f"未知架构 (计算能力 {compute_cap})"
            print(f"    架构: {arch}")
        
        # 显存使用情况
        torch.cuda.set_device(i)
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
        mem_free = (props.total_memory - torch.cuda.memory_reserved(i)) / 1024**3
        print(f"    当前显存使用: {mem_allocated:.2f} GB")
        print(f"    显存保留: {mem_reserved:.2f} GB")
        print(f"    显存可用: {mem_free:.2f} GB")
    
    # 特性支持
    print(f"\n【高级特性支持】")
    
    # TF32支持（仅Ampere及以上）
    tf32_supported = torch.cuda.get_device_properties(0).major >= 8
    print(f"  TF32加速: {'✅ 支持' if tf32_supported else '❌ 不支持（需要Ampere架构及以上）'}")
    if tf32_supported:
        print(f"    当前状态: {'已启用' if torch.backends.cuda.matmul.allow_tf32 else '未启用'}")
    
    # AMP支持
    amp_supported = hasattr(torch.cuda.amp, 'autocast')
    print(f"  混合精度(AMP): {'✅ 支持' if amp_supported else '❌ 不支持'}")
    
    # cuDNN
    cudnn_enabled = torch.backends.cudnn.enabled
    print(f"  cuDNN加速: {'✅ 已启用' if cudnn_enabled else '❌ 未启用'}")
    print(f"  cuDNN benchmark: {'已启用' if torch.backends.cudnn.benchmark else '未启用（建议启用）'}")
    
    # 推荐配置
    print(f"\n【训练建议】")
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if "A100" in torch.cuda.get_device_name(0):
        print("  🎯 检测到A100显卡！")
        print("  强烈建议使用: captcha_train_a100_optimized.py")
        print(f"  推荐batch_size: 128-256")
        print("  预计训练速度: 150轮约50分钟")
    elif "H100" in torch.cuda.get_device_name(0):
        print("  🎯 检测到H100显卡！")
        print("  强烈建议使用: captcha_train_a100_optimized.py")
        print(f"  推荐batch_size: 256-512")
        print("  预计训练速度: 150轮约30分钟")
    elif total_memory >= 20:
        print(f"  检测到大显存GPU ({total_memory:.0f}GB)")
        print("  建议使用: captcha_train_a100_optimized.py")
        print(f"  推荐batch_size: {min(256, int(total_memory * 6))}")
    elif total_memory >= 10:
        print(f"  检测到中等显存GPU ({total_memory:.0f}GB)")
        print("  可以使用标准版: captcha_train.py")
        print(f"  推荐batch_size: 64-128")
    else:
        print(f"  检测到小显存GPU ({total_memory:.0f}GB)")
        print("  建议使用标准版: captcha_train.py")
        print(f"  推荐batch_size: 32-64")
    
    # 性能测试
    print(f"\n【性能测试】")
    print("  正在进行简单性能测试...")
    
    device = torch.device("cuda:0")
    
    # 矩阵乘法测试
    import time
    size = 2048
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 预热
    for _ in range(3):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    
    # 测试
    start = time.time()
    iterations = 20
    for _ in range(iterations):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tflops = (2 * size**3 * iterations) / (elapsed * 1e12)
    print(f"  矩阵乘法性能: {tflops:.2f} TFLOPS")
    
    # 参考性能
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" in gpu_name:
        print(f"  A100理论峰值: ~19.5 TFLOPS (FP32), ~312 TFLOPS (TF32)")
        if tflops > 10:
            print(f"  性能评估: ✅ 优秀")
        elif tflops > 5:
            print(f"  性能评估: ⚠️  良好，但可以更好")
        else:
            print(f"  性能评估: ⚠️  偏低，检查驱动和CUDA配置")
    
    print(f"\n{'=' * 60}")
    print("检测完成！")
    print("=" * 60)

if __name__ == '__main__':
    try:
        check_gpu_environment()
    except Exception as e:
        print(f"检测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
