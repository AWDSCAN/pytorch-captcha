# -*- coding: UTF-8 -*-
"""
针对A100优化的训练脚本 - 带可视化
- 混合精度训练（AMP）
- 梯度累积
- 多GPU支持
- TF32加速
- 进度条和训练曲线可视化
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import my_dataset
from captcha_cnn_model import CNN
import time
import os

# 优化后的超参数 - 充分利用3张A100算力
num_epochs = 150
batch_size = 512  # 大幅增加batch_size，充分利用显存（从64提升到512）
learning_rate = 0.0002
gradient_accumulation_steps = 1  # 显存充足，不需要梯度累积
num_workers = 12  # 数据加载并行线程数（推荐：GPU数量 * 4）

def print_gpu_info():
    """打印GPU信息"""
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  显存总量: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"cuDNN启用: {torch.backends.cudnn.enabled}")
    else:
        print("警告：未检测到CUDA设备")

def main():
    # 检测GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')
    print_gpu_info()
    
    # A100优化：启用TF32加速（Ampere架构特性）
    if torch.cuda.is_available():
        # TF32可以在A100上提供高达10x的加速
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ TF32加速已启用（A100专属优化）")
        
        # 启用cudnn benchmark自动寻找最优算法
        torch.backends.cudnn.benchmark = True
        print("✓ cuDNN自动调优已启用")
    
    # 初始化模型
    cnn = CNN()
    
    # 多GPU支持（如果有多张A100）
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 张GPU进行数据并行训练")
        cnn = nn.DataParallel(cnn)
    
    cnn.to(device)
    cnn.train()
    print('✓ 初始化网络完成')
    
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    
    # 余弦学习率衰减调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 混合精度训练（AMP）- A100的Tensor Core优化
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("✓ 混合精度训练（AMP）已启用")

    # 训练模型 - 使用优化的DataLoader
    train_dataloader = my_dataset.get_train_data_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    total_steps = len(train_dataloader)
    
    print(f"\n开始训练:")
    print(f"  总轮数: {num_epochs}")
    print(f"  Batch大小: {batch_size} (优化: 从64增加到{batch_size})")
    print(f"  数据加载线程: {num_workers}")
    print(f"  初始学习率: {learning_rate}")
    print(f"  训练样本数: {len(train_dataloader.dataset)}")
    print(f"  每轮步数: {total_steps}")
    print(f"  梯度累积步数: {gradient_accumulation_steps}")
    print(f"  实际batch大小: {batch_size * gradient_accumulation_steps}\n")
    
    # 导入可视化工具
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
        print("提示: 安装tqdm可以显示进度条 (pip install tqdm)")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
    
    # 训练历史记录
    from datetime import datetime
    import json
    history = {
        'epoch': [],
        'loss': [],
        'avg_loss': [],
        'lr': [],
        'gpu_memory': [],
        'time': []
    }
    train_start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = 'training_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        epoch_start_time = time.time()
        
        # 使用进度条
        if HAS_TQDM:
            pbar = tqdm(enumerate(train_dataloader), total=total_steps,
                       desc=f'Epoch {epoch+1}/{num_epochs}',
                       ncols=120,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            data_iter = pbar
        else:
            data_iter = enumerate(train_dataloader)
        
        for i, (images, labels) in data_iter:
            # 将数据移至GPU（non_blocking=True异步传输，提升效率）
            images = Variable(images).to(device, non_blocking=True)
            labels = Variable(labels.float()).to(device, non_blocking=True)
            
            # 混合精度前向传播
            with autocast(enabled=use_amp):
                predict_labels = cnn(images)
                loss = criterion(predict_labels, labels)
                # 梯度累积
                loss = loss / gradient_accumulation_steps
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (i + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # 更新进度条
            if HAS_TQDM:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss_so_far = epoch_loss / (i + 1)
                
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    pbar.set_postfix({
                        'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'AvgLoss': f'{avg_loss_so_far:.4f}',
                        'LR': f'{current_lr:.6f}',
                        'GPU': f'{mem_allocated:.2f}/{mem_reserved:.2f}GB'
                    })
                else:
                    pbar.set_postfix({
                        'Loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'AvgLoss': f'{avg_loss_so_far:.4f}',
                        'LR': f'{current_lr:.6f}'
                    })
            
            if (i+1) % 100 == 0:
                # 保存时去除DataParallel包装
                model_to_save = cnn.module if hasattr(cnn, 'module') else cnn
                torch.save(model_to_save.state_dict(), "./model.pkl")
        
        if HAS_TQDM:
            pbar.close()
        
        # 每轮结束后更新学习率
        scheduler.step()
        avg_loss = epoch_loss / total_steps
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # 记录训练历史
        history['epoch'].append(epoch + 1)
        history['loss'].append(loss.item() * gradient_accumulation_steps)
        history['avg_loss'].append(avg_loss)
        history['lr'].append(current_lr)
        if torch.cuda.is_available():
            history['gpu_memory'].append(torch.cuda.memory_allocated(device) / 1024**3)
        history['time'].append(time.time() - train_start_time)
        
        # 打印Epoch总结
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成")
        print(f"  平均Loss: {avg_loss:.6f}")
        print(f"  当前Loss: {loss.item() * gradient_accumulation_steps:.6f}")
        print(f"  学习率: {current_lr:.6e}")
        print(f"  耗时: {epoch_time:.2f}秒")
        
        if torch.cuda.is_available():
            mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
            mem_peak = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  显存使用: {mem_allocated:.2f}GB (峰值: {mem_peak:.2f}GB)")
        
        # 预估剩余时间
        if epoch < num_epochs - 1:
            avg_time_per_epoch = (time.time() - train_start_time) / (epoch + 1)
            remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            print(f"  预计剩余: {hours}小时 {minutes}分钟")
        print(f"{'='*80}\n")
        
        # 每10轮保存模型和绘制曲线
        if (epoch+1) % 10 == 0:
            model_to_save = cnn.module if hasattr(cnn, 'module') else cnn
            torch.save(model_to_save.state_dict(), f"./model_epoch_{epoch+1}.pkl")
            print(f"✓ 已保存Epoch {epoch+1}检查点\n")
            
            # 绘制训练曲线
            if HAS_MATPLOTLIB and len(history['epoch']) > 0:
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Loss曲线
                    axes[0, 0].plot(history['epoch'], history['avg_loss'], 'b-', linewidth=2)
                    axes[0, 0].set_xlabel('Epoch')
                    axes[0, 0].set_ylabel('Average Loss')
                    axes[0, 0].set_title('训练Loss曲线')
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # 学习率曲线
                    axes[0, 1].plot(history['epoch'], history['lr'], 'r-', linewidth=2)
                    axes[0, 1].set_xlabel('Epoch')
                    axes[0, 1].set_ylabel('Learning Rate')
                    axes[0, 1].set_title('学习率变化')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # GPU显存使用
                    if len(history['gpu_memory']) > 0:
                        axes[1, 0].plot(history['epoch'], history['gpu_memory'], 'g-', linewidth=2)
                        axes[1, 0].set_xlabel('Epoch')
                        axes[1, 0].set_ylabel('GPU Memory (GB)')
                        axes[1, 0].set_title('GPU显存使用')
                        axes[1, 0].grid(True, alpha=0.3)
                    
                    # 训练时间
                    axes[1, 1].plot(history['epoch'], [t/3600 for t in history['time']], 'm-', linewidth=2)
                    axes[1, 1].set_xlabel('Epoch')
                    axes[1, 1].set_ylabel('Time (hours)')
                    axes[1, 1].set_title('累计训练时间')
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plot_file = os.path.join(log_dir, f'training_curves_{timestamp}.png')
                    plt.savefig(plot_file, dpi=100, bbox_inches='tight')
                    plt.close()
                    print(f"✓ 训练曲线已更新: {plot_file}\n")
                except Exception as e:
                    print(f"绘图错误: {e}\n")
    
    # 保存最终模型
    model_to_save = cnn.module if hasattr(cnn, 'module') else cnn
    torch.save(model_to_save.state_dict(), "./model.pkl")
    print("✓ 训练完成，最终模型已保存")
    
    # 保存训练历史
    history_file = os.path.join(log_dir, f'training_history_{timestamp}.json')
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"✓ 训练历史已保存: {history_file}")
    
    # 最终绘制曲线
    if HAS_MATPLOTLIB and len(history['epoch']) > 0:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            axes[0, 0].plot(history['epoch'], history['avg_loss'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Average Loss')
            axes[0, 0].set_title('训练Loss曲线')
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 1].plot(history['epoch'], history['lr'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_title('学习率变化')
            axes[0, 1].grid(True, alpha=0.3)
            
            if len(history['gpu_memory']) > 0:
                axes[1, 0].plot(history['epoch'], history['gpu_memory'], 'g-', linewidth=2)
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('GPU Memory (GB)')
                axes[1, 0].set_title('GPU显存使用')
                axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 1].plot(history['epoch'], [t/3600 for t in history['time']], 'm-', linewidth=2)
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Time (hours)')
            axes[1, 1].set_title('累计训练时间')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            final_plot = os.path.join(log_dir, f'training_curves_final_{timestamp}.png')
            plt.savefig(final_plot, dpi=100, bbox_inches='tight')
            plt.close()
            print(f"✓ 最终训练曲线: {final_plot}")
        except Exception as e:
            print(f"最终绘图错误: {e}")
    
    # 训练总结
    total_time = time.time() - train_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 80)
    print("训练总结")
    print("=" * 80)
    print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
    print(f"训练轮数: {len(history['epoch'])} / {num_epochs}")
    if len(history['avg_loss']) > 0:
        print(f"最终Loss: {history['avg_loss'][-1]:.6f}")
        print(f"最低Loss: {min(history['avg_loss']):.6f} (Epoch {history['avg_loss'].index(min(history['avg_loss'])) + 1})")
        print(f"最终学习率: {history['lr'][-1]:.6e}")
    
    # 显示最终显存使用
    if torch.cuda.is_available():
        print(f"\n最终显存统计:")
        print(f"  已分配: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        print(f"  已保留: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"  峰值: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
    
    print("=" * 80)

if __name__ == '__main__':
    main()
