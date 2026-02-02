# -*- coding: UTF-8 -*-
"""
优化的训练脚本 - 带可视化和进度条
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import my_dataset
from captcha_cnn_model import CNN
from tqdm import tqdm
import time
import json
import os
from datetime import datetime
import numpy as np
import captcha_setting

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 中文显示
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("提示: 安装matplotlib可以生成训练曲线图 (pip install matplotlib)")

# 优化后的超参数
num_epochs = 150
batch_size = 128  # 增加batch_size以提升训练效率
learning_rate = 0.0002
num_workers = 8  # 数据加载并行线程数

def calculate_accuracy(model, dataloader, device, max_samples=1000):
    """
    计算模型准确率
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 最多评估的样本数（避免测试时间过长）
    
    Returns:
        准确率（百分比）
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            if total >= max_samples:
                break
            
            images = Variable(images).to(device)
            predict_labels = model(images)
            predict_labels_cpu = predict_labels.cpu()
            
            # 逐个样本计算
            for idx in range(images.size(0)):
                if total >= max_samples:
                    break
                
                # 解码预测结果
                predicted_text = ''
                for i in range(captcha_setting.MAX_CAPTCHA):
                    start_idx = i * captcha_setting.ALL_CHAR_SET_LEN
                    end_idx = (i + 1) * captcha_setting.ALL_CHAR_SET_LEN
                    char_idx = np.argmax(predict_labels_cpu[idx, start_idx:end_idx].data.numpy())
                    predicted_text += captcha_setting.ALL_CHAR_SET[char_idx]
                
                # 解码真实标签
                import one_hot_encoding
                true_text = one_hot_encoding.decode(labels.numpy()[idx])
                
                if predicted_text == true_text:
                    correct += 1
                total += 1
    
    model.train()
    accuracy = 100.0 * correct / total if total > 0 else 0
    return accuracy

class TrainingVisualizer:
    """训练可视化器"""
    
    def __init__(self, num_epochs, save_dir='training_logs'):
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.history = {
            'epoch': [],
            'loss': [],
            'avg_loss': [],
            'lr': [],
            'train_acc': [],
            'test_acc': [],
            'time': []
        }
        self.start_time = time.time()
        
        # 创建日志目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 生成时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(save_dir, f'training_log_{self.timestamp}.txt')
        
        # 写入训练开始信息
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"训练开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总轮数: {num_epochs}\n")
            f.write(f"Batch大小: {batch_size}\n")
            f.write(f"初始学习率: {learning_rate}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message):
        """记录日志"""
        with open(self.log_file, 'a', encoding='utf-8') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    
    def update(self, epoch, loss, avg_loss, lr, train_acc=None, test_acc=None):
        """更新训练历史"""
        self.history['epoch'].append(epoch)
        self.history['loss'].append(loss)
        self.history['avg_loss'].append(avg_loss)
        self.history['lr'].append(lr)
        self.history['train_acc'].append(train_acc if train_acc is not None else 0)
        self.history['test_acc'].append(test_acc if test_acc is not None else 0)
        self.history['time'].append(time.time() - self.start_time)
    
    def plot_curves(self):
        """绘制训练曲线"""
        if not HAS_MATPLOTLIB or len(self.history['epoch']) == 0:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制Loss曲线
            ax1 = axes[0, 0]
            epochs = self.history['epoch']
            ax1.plot(epochs, self.history['avg_loss'], 'b-', label='平均Loss', linewidth=2)
            ax1.set_xlabel('Epoch', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('训练Loss曲线', fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # 绘制学习率曲线
            ax2 = axes[0, 1]
            ax2.plot(epochs, self.history['lr'], 'r-', label='学习率', linewidth=2)
            ax2.set_xlabel('Epoch', fontsize=12)
            ax2.set_ylabel('Learning Rate', fontsize=12)
            ax2.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # 绘制准确率曲线
            ax3 = axes[1, 0]
            if any(self.history['train_acc']):
                ax3.plot(epochs, self.history['train_acc'], 'g-', label='训练准确率', linewidth=2)
            if any(self.history['test_acc']):
                ax3.plot(epochs, self.history['test_acc'], 'm-', label='测试准确率', linewidth=2)
            ax3.set_xlabel('Epoch', fontsize=12)
            ax3.set_ylabel('Accuracy (%)', fontsize=12)
            ax3.set_title('准确率曲线', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            ax3.set_ylim([0, 105])  # 设置y轴范围0-105%
            
            # 绘制Loss vs Accuracy对比
            ax4 = axes[1, 1]
            ax4_twin = ax4.twinx()
            line1 = ax4.plot(epochs, self.history['avg_loss'], 'b-', label='Loss', linewidth=2)
            if any(self.history['test_acc']):
                line2 = ax4_twin.plot(epochs, self.history['test_acc'], 'r-', label='测试准确率', linewidth=2)
            ax4.set_xlabel('Epoch', fontsize=12)
            ax4.set_ylabel('Loss', fontsize=12, color='b')
            ax4_twin.set_ylabel('Accuracy (%)', fontsize=12, color='r')
            ax4.set_title('Loss vs Accuracy', fontsize=14, fontweight='bold')
            ax4.tick_params(axis='y', labelcolor='b')
            ax4_twin.tick_params(axis='y', labelcolor='r')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            plot_file = os.path.join(self.save_dir, f'training_curves_{self.timestamp}.png')
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            return plot_file
        except Exception as e:
            print(f"绘图错误: {e}")
            return None
    
    def save_history(self):
        """保存训练历史到JSON"""
        history_file = os.path.join(self.save_dir, f'training_history_{self.timestamp}.json')
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        return history_file
    
    def print_summary(self):
        """打印训练总结"""
        if len(self.history['epoch']) == 0:
            return
        
        total_time = time.time() - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        print("\n" + "=" * 80)
        print("训练总结")
        print("=" * 80)
        print(f"总训练时间: {hours}小时 {minutes}分钟 {seconds}秒")
        print(f"训练轮数: {len(self.history['epoch'])} / {self.num_epochs}")
        print(f"最终Loss: {self.history['avg_loss'][-1]:.6f}")
        print(f"最低Loss: {min(self.history['avg_loss']):.6f} (Epoch {self.history['avg_loss'].index(min(self.history['avg_loss'])) + 1})")
        print(f"最终学习率: {self.history['lr'][-1]:.6e}")
        
        # 准确率统计
        if any(self.history['test_acc']):
            test_accs = [acc for acc in self.history['test_acc'] if acc > 0]
            if test_accs:
                print(f"最终测试准确率: {self.history['test_acc'][-1]:.2f}%")
                print(f"最高测试准确率: {max(test_accs):.2f}% (Epoch {self.history['test_acc'].index(max(test_accs)) + 1})")
        
        if any(self.history['train_acc']):
            train_accs = [acc for acc in self.history['train_acc'] if acc > 0]
            if train_accs:
                print(f"最终训练准确率: {self.history['train_acc'][-1]:.2f}%")
        
        print("=" * 80)

def main():
    # 检测GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("验证码识别模型训练")
    print("=" * 80)
    print(f"使用设备: {device}")
    if torch.cuda.is_available():
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"训练轮数: {num_epochs}")
    print(f"Batch大小: {batch_size}")
    print(f"初始学习率: {learning_rate}")
    print("=" * 80 + "\n")
    
    # 初始化可视化器
    visualizer = TrainingVisualizer(num_epochs)
    visualizer.log("训练开始")
    
    cnn = CNN()
    cnn.to(device)
    cnn.train()
    print('✓ 网络初始化完成\n')
    
    criterion = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    
    # 余弦学习率衰减调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 训练模型 - 使用优化的DataLoader
    train_dataloader = my_dataset.get_train_data_loader(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_dataloader = my_dataset.get_test_data_loader(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )
    total_steps = len(train_dataloader)
    
    print(f"训练集样本数: {len(train_dataloader.dataset)}")
    print(f"每轮步数: {total_steps}\n")
    
    # 训练循环
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_start_time = time.time()
        
        # 使用tqdm进度条
        pbar = tqdm(enumerate(train_dataloader), total=total_steps, 
                   desc=f'Epoch {epoch+1}/{num_epochs}',
                   ncols=100,
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for i, (images, labels) in pbar:
            # 将数据移至GPU
            images = Variable(images).to(device)
            labels = Variable(labels.float()).to(device)
            
            predict_labels = cnn(images)
            loss = criterion(predict_labels, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条显示
            current_lr = optimizer.param_groups[0]['lr']
            avg_loss_so_far = epoch_loss / (i + 1)
            
            # 显示GPU显存（如果可用）
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated(device) / 1024**3
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AvgLoss': f'{avg_loss_so_far:.4f}',
                    'LR': f'{current_lr:.6f}',
                    'GPU': f'{mem_allocated:.2f}GB'
                })
            else:
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'AvgLoss': f'{avg_loss_so_far:.4f}',
                    'LR': f'{current_lr:.6f}'
                })
            
            # 定期保存模型
            if (i+1) % 100 == 0:
                torch.save(cnn.state_dict(), "./model.pkl")
        
        # Epoch结束
        pbar.close()
        scheduler.step()
        
        avg_loss = epoch_loss / total_steps
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start_time
        
        # 计算准确率（每5轮或最后一轮）
        train_acc = None
        test_acc = None
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            print("\n正在计算准确率...")
            # 计算训练集准确率（抽样）
            train_acc = calculate_accuracy(cnn, train_dataloader, device, max_samples=500)
            # 计算测试集准确率（全部）
            test_acc = calculate_accuracy(cnn, test_dataloader, device, max_samples=float('inf'))
        
        # 更新训练历史
        visualizer.update(epoch + 1, loss.item(), avg_loss, current_lr, train_acc, test_acc)
        
        # 打印Epoch总结
        print(f"\n{'='*80}")
        print(f"Epoch [{epoch+1}/{num_epochs}] 完成")
        print(f"  平均Loss: {avg_loss:.6f}")
        print(f"  当前Loss: {loss.item():.6f}")
        print(f"  学习率: {current_lr:.6e}")
        if train_acc is not None:
            print(f"  训练准确率: {train_acc:.2f}% (抽样500张)")
        if test_acc is not None:
            print(f"  测试准确率: {test_acc:.2f}%")
        print(f"  耗时: {epoch_time:.2f}秒")
        
        # 预估剩余时间
        if epoch < num_epochs - 1:
            avg_time_per_epoch = (time.time() - visualizer.start_time) / (epoch + 1)
            remaining_time = avg_time_per_epoch * (num_epochs - epoch - 1)
            hours = int(remaining_time // 3600)
            minutes = int((remaining_time % 3600) // 60)
            print(f"  预计剩余时间: {hours}小时 {minutes}分钟")
        print(f"{'='*80}\n")
        
        # 记录日志
        visualizer.log(f"Epoch {epoch+1}/{num_epochs} - AvgLoss: {avg_loss:.6f}, LR: {current_lr:.6e}, Time: {epoch_time:.2f}s")
        
        # 每10轮保存模型和绘制曲线
        if (epoch+1) % 10 == 0:
            torch.save(cnn.state_dict(), f"./model_epoch_{epoch+1}.pkl")
            print(f"✓ 已保存Epoch {epoch+1}检查点\n")
            visualizer.log(f"保存检查点: model_epoch_{epoch+1}.pkl")
            
            # 绘制训练曲线
            if HAS_MATPLOTLIB:
                plot_file = visualizer.plot_curves()
                if plot_file:
                    print(f"✓ 训练曲线已更新: {plot_file}\n")
    
    # 训练完成
    torch.save(cnn.state_dict(), "./model.pkl")
    visualizer.log("训练完成，模型已保存")
    
    # 保存训练历史
    history_file = visualizer.save_history()
    print(f"✓ 训练历史已保存: {history_file}")
    
    # 最终绘制曲线
    if HAS_MATPLOTLIB:
        plot_file = visualizer.plot_curves()
        if plot_file:
            print(f"✓ 最终训练曲线: {plot_file}")
    
    # 打印训练总结
    visualizer.print_summary()
    visualizer.log("训练任务结束")

if __name__ == '__main__':
    main()


