# -*- coding: UTF-8 -*-
"""
模型测试脚本 - 带可视化和详细分析
"""
import numpy as np
import torch
from torch.autograd import Variable
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN
import one_hot_encoding
from collections import Counter

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

def main():
    # 检测GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("模型测试")
    print("=" * 80)
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl', map_location=device))
    cnn.to(device)
    print("✓ 模型加载完成\n")

    test_dataloader = my_dataset.get_test_data_loader()
    total_samples = len(test_dataloader.dataset)
    print(f"测试集样本数: {total_samples}\n")

    correct = 0
    total = 0
    error_samples = []  # 记录错误样本
    char_errors = Counter()  # 记录每个字符的错误次数
    position_errors = [0, 0, 0, 0]  # 记录每个位置的错误次数
    
    # 使用进度条
    if HAS_TQDM:
        pbar = tqdm(test_dataloader, desc='测试进度', ncols=100)
        data_iter = pbar
    else:
        data_iter = test_dataloader
        print("开始测试...")
    
    with torch.no_grad():
        for images, labels in data_iter:
            images = Variable(images).to(device)
            predict_label = cnn(images)
            
            # 将预测结果移回CPU进行处理
            predict_label_cpu = predict_label.cpu()
            labels_cpu = labels.cpu().numpy()
            
            # 遍历batch中的每个样本
            batch_size = labels.size(0)
            for idx in range(batch_size):
                c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[idx, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[idx, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[idx, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[idx, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
                predict_label_str = '%s%s%s%s' % (c0, c1, c2, c3)
                true_label = one_hot_encoding.decode(labels_cpu[idx])
                total += 1
                
                if predict_label_str == true_label:
                    correct += 1
                else:
                    error_samples.append((true_label, predict_label_str))
                    
                    # 分析每个字符的错误
                    for pos, (true_char, pred_char) in enumerate(zip(true_label, predict_label_str)):
                        if true_char != pred_char:
                            char_errors[f"{true_char}->{pred_char}"] += 1
                            position_errors[pos] += 1
                    
                    if len(error_samples) <= 10:
                        print(f"\n错误识别: 真实={true_label}, 预测={predict_label_str}")
            
            # 更新进度条
            if HAS_TQDM:
                current_acc = 100.0 * correct / total
                pbar.set_postfix({'准确率': f'{current_acc:.2f}%'})
    
    if HAS_TQDM:
        pbar.close()
    
    # 计算最终准确率
    accuracy = 100.0 * correct / total
    error_rate = 100.0 * len(error_samples) / total
    
    # 打印测试结果
    print(f"\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"总样本数: {total}")
    print(f"正确识别: {correct}")
    print(f"错误识别: {len(error_samples)}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"错误率: {error_rate:.2f}%")
    print("=" * 80)
    
    # 字符级别准确率
    char_correct = sum(position_errors)
    char_total = total * 4
    char_accuracy = 100.0 * (char_total - char_correct) / char_total
    print(f"\n字符级别统计:")
    print(f"  字符总数: {char_total}")
    print(f"  字符正确: {char_total - char_correct}")
    print(f"  字符准确率: {char_accuracy:.2f}%")
    
    # 每个位置的错误分析
    print(f"\n位置错误分析:")
    for i, errors in enumerate(position_errors):
        error_rate_pos = 100.0 * errors / total
        print(f"  位置 {i+1}: {errors} 个错误 ({error_rate_pos:.2f}%)")
    
    # 最常见的错误
    if error_samples:
        print(f"\n最常见的字符混淆 (Top 10):")
        for (error_type, count) in char_errors.most_common(10):
            print(f"  {error_type}: {count} 次")
    
    # 绘制可视化图表
    if HAS_MATPLOTLIB and total > 0:
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. 准确率饼图
            ax1 = axes[0, 0]
            sizes = [correct, len(error_samples)]
            labels = [f'正确\n{correct}个\n({accuracy:.2f}%)', 
                     f'错误\n{len(error_samples)}个\n({error_rate:.2f}%)']
            colors = ['#66c2a5', '#fc8d62']
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax1.set_title('整体识别准确率', fontsize=14, fontweight='bold')
            
            # 2. 位置错误分布
            ax2 = axes[0, 1]
            positions = ['位置1', '位置2', '位置3', '位置4']
            ax2.bar(positions, position_errors, color='#8da0cb')
            ax2.set_ylabel('错误次数', fontsize=12)
            ax2.set_title('各位置错误分布', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            for i, v in enumerate(position_errors):
                ax2.text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            # 3. Top 10 混淆字符
            ax3 = axes[1, 0]
            if char_errors:
                top_errors = char_errors.most_common(10)
                error_labels = [e[0] for e in top_errors]
                error_counts = [e[1] for e in top_errors]
                ax3.barh(range(len(error_labels)), error_counts, color='#e78ac3')
                ax3.set_yticks(range(len(error_labels)))
                ax3.set_yticklabels(error_labels)
                ax3.set_xlabel('错误次数', fontsize=12)
                ax3.set_title('Top 10 字符混淆', fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3)
                ax3.invert_yaxis()
            
            # 4. 准确率对比
            ax4 = axes[1, 1]
            metrics = ['整体准确率', '字符准确率']
            values = [accuracy, char_accuracy]
            colors_bar = ['#66c2a5', '#8da0cb']
            bars = ax4.bar(metrics, values, color=colors_bar)
            ax4.set_ylabel('准确率 (%)', fontsize=12)
            ax4.set_title('准确率对比', fontsize=14, fontweight='bold')
            ax4.set_ylim([0, 105])
            ax4.grid(True, alpha=0.3)
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图片
            import os
            log_dir = 'training_logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(log_dir, f'test_results_{timestamp}.png')
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ 测试结果可视化已保存: {plot_file}")
        except Exception as e:
            print(f"\n绘图错误: {e}")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()


