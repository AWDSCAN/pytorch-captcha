# -*- coding: UTF-8 -*-
"""
A100超级优化训练脚本 - 充分利用多GPU算力
- 使用DistributedDataParallel（DDP）替代DataParallel（更高效）
- 大batch size（512）充分利用显存
- 优化数据加载（多线程、pin_memory、预加载）
- 混合精度训练（AMP）
- TF32加速
"""
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.autograd import Variable
from torch.cuda.amp import autocast, GradScaler
import my_dataset
from captcha_cnn_model import CNN
import time
import os

# 超参数 - 充分利用多GPU算力
num_epochs = 150
batch_size = 512  # 每个GPU的batch_size（总batch=512*3=1536）
learning_rate = 0.0002
num_workers = 12  # 数据加载线程数

def setup_ddp(rank, world_size):
    """初始化DDP"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理DDP"""
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    """DDP训练函数"""
    print(f"[GPU {rank}] 初始化...")
    
    # 设置DDP
    setup_ddp(rank, world_size)
    
    # A100优化：启用TF32和cuDNN benchmark
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # 创建模型并移到GPU
    model = CNN().to(rank)
    model = DDP(model, device_ids=[rank])
    
    # 优化器和调度器
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 数据加载器（使用DistributedSampler）
    import captcha_setting
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from PIL import Image
    import one_hot_encoding as ohe
    
    class mydataset(torch.utils.data.Dataset):
        def __init__(self, folder, transform=None):
            self.train_image_file_paths = [os.path.join(folder, f) for f in os.listdir(folder)]
            self.transform = transform

        def __len__(self):
            return len(self.train_image_file_paths)

        def __getitem__(self, idx):
            image_root = self.train_image_file_paths[idx]
            image_name = image_root.split(os.path.sep)[-1]
            image = Image.open(image_root)
            if self.transform is not None:
                image = self.transform(image)
            label = ohe.encode(image_name.split('_')[0])
            return image, label
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    dataset = mydataset(captcha_setting.TRAIN_DATASET_PATH, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    if rank == 0:
        print(f"\n{'='*80}")
        print("DDP训练配置:")
        print(f"  GPU数量: {world_size}")
        print(f"  每GPU Batch大小: {batch_size}")
        print(f"  总Batch大小: {batch_size * world_size}")
        print(f"  数据加载线程: {num_workers} (每GPU)")
        print(f"  总训练样本: {len(dataset)}")
        print(f"  每轮步数: {len(train_dataloader)}")
        print(f"  总轮数: {num_epochs}")
        print(f"{'='*80}\n")
    
    # 训练循环
    try:
        from tqdm import tqdm
        HAS_TQDM = True
    except ImportError:
        HAS_TQDM = False
    
    model.train()
    train_start = time.time()
    
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 重要：确保每个epoch的shuffle不同
        epoch_start = time.time()
        epoch_loss = 0
        
        if rank == 0 and HAS_TQDM:
            pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', ncols=120)
            data_iter = pbar
        else:
            data_iter = train_dataloader
        
        for i, (images, labels) in enumerate(data_iter):
            images = images.to(rank, non_blocking=True)
            labels = labels.float().to(rank, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            
            # 更新进度条
            if rank == 0 and HAS_TQDM:
                current_lr = optimizer.param_groups[0]['lr']
                avg_loss = epoch_loss / (i + 1)
                mem_gb = torch.cuda.memory_allocated(rank) / 1024**3
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AvgLoss': f'{avg_loss:.4f}',
                    'LR': f'{current_lr:.6f}',
                    'GPU': f'{mem_gb:.2f}GB'
                })
            
            # 定期保存
            if rank == 0 and (i+1) % 100 == 0:
                torch.save(model.module.state_dict(), "./model.pkl")
        
        if rank == 0 and HAS_TQDM:
            pbar.close()
        
        scheduler.step()
        
        # Epoch统计
        avg_loss = epoch_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start
        
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\n{'='*80}")
            print(f"Epoch [{epoch+1}/{num_epochs}] 完成")
            print(f"  平均Loss: {avg_loss:.6f}")
            print(f"  学习率: {current_lr:.6e}")
            print(f"  耗时: {epoch_time:.2f}秒")
            
            mem_allocated = torch.cuda.memory_allocated(rank) / 1024**3
            mem_peak = torch.cuda.max_memory_allocated(rank) / 1024**3
            print(f"  GPU{rank}显存: {mem_allocated:.2f}GB (峰值: {mem_peak:.2f}GB)")
            
            # 预估剩余时间
            if epoch < num_epochs - 1:
                elapsed = time.time() - train_start
                avg_time = elapsed / (epoch + 1)
                remaining = avg_time * (num_epochs - epoch - 1)
                hours = int(remaining // 3600)
                minutes = int((remaining % 3600) // 60)
                print(f"  预计剩余: {hours}小时 {minutes}分钟")
            print(f"{'='*80}\n")
        
        # 每10轮保存检查点
        if rank == 0 and (epoch+1) % 10 == 0:
            torch.save(model.module.state_dict(), f"./model_epoch_{epoch+1}.pkl")
            print(f"✓ 已保存Epoch {epoch+1}检查点\n")
    
    # 保存最终模型
    if rank == 0:
        torch.save(model.module.state_dict(), "./model.pkl")
        total_time = time.time() - train_start
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 80)
        print("DDP训练完成")
        print("=" * 80)
        print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
        print(f"最终模型已保存: model.pkl")
        print("=" * 80)
    
    cleanup_ddp()

def main():
    """主函数"""
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        print("错误：未检测到GPU")
        return
    
    print("=" * 80)
    print("A100超级优化训练 - DistributedDataParallel")
    print("=" * 80)
    print(f"检测到 {world_size} 张GPU")
    for i in range(world_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"    显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    print(f"总Batch大小: {batch_size} × {world_size} = {batch_size * world_size}")
    print("=" * 80 + "\n")
    
    # 启动DDP训练
    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == '__main__':
    main()
