# -*- coding: UTF-8 -*-
"""
训练集模型测试脚本 - 验证模型在训练数据上的效果
测试 models/model.pkl 在 dataset/train 上的识别准确率
"""
import os
import numpy as np
import torch
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms as transforms
import captcha_setting
from captcha_cnn_model import CNN
import one_hot_encoding as ohe
from collections import Counter
import random

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

def predict_single_image(model, image_path, device):
    """
    预测单张图片
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    # 读取图片
    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # 添加batch维度
    
    # 预测
    with torch.no_grad():
        image = Variable(image).to(device)
        predict_label = model(image)
        predict_label_cpu = predict_label.cpu()
    
    # 解码预测结果
    c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
    
    predict_label_str = '%s%s%s%s' % (c0, c1, c2, c3)
    print("预测结果：",predict_label_str)
    return predict_label_str

def extract_label_from_filename(filename):
    """
    从文件名提取真实标签
    文件名格式: ABCD_timestamp.png
    """
    return filename.split('_')[0]

def main():
    # 检测GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("训练集模型测试")
    print("=" * 80)
    print(f'使用设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    
    # 加载模型
    model_path = 'models/model.pkl'
    if not os.path.exists(model_path):
        # 尝试当前目录
        model_path = 'model.pkl'
    
    print(f'模型路径: {model_path}')
    
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path, map_location=device))
    cnn.to(device)
    print("✓ 模型加载完成\n")

    # 获取训练集图片列表
    train_dir = captcha_setting.TRAIN_DATASET_PATH
    if not os.path.exists(train_dir):
        print(f"错误：训练集目录不存在 - {train_dir}")
        return
    
    image_files = [f for f in os.listdir(train_dir) if f.endswith('.png') or f.endswith('.jpg')]
    total_samples = len(image_files)
    
    print(f"训练集目录: {train_dir}")
    print(f"训练集样本数: {total_samples}\n")
    
    if total_samples == 0:
        print("错误：训练集为空")
        return
    
    # 统计变量
    correct = 0
    total = 0
    error_samples = []  # 记录错误样本
    char_errors = Counter()  # 记录每个字符的错误次数
    position_errors = [0, 0, 0, 0]  # 记录每个位置的错误次数
    
    # 进度条
    if HAS_TQDM:
        pbar = tqdm(image_files, desc='测试进度', ncols=120)
        data_iter = pbar
    else:
        data_iter = image_files
        print("开始测试...")
    
    # 遍历所有图片
    for filename in data_iter:
        image_path = os.path.join(train_dir, filename)
        true_label = extract_label_from_filename(filename)
        
        # 预测
        try:
            predict_label = predict_single_image(cnn, image_path, device)
            total += 1
            
            if predict_label == true_label:
                correct += 1
            else:
                error_samples.append((filename, true_label, predict_label))
                
                # 分析每个字符的错误
                for pos, (true_char, pred_char) in enumerate(zip(true_label, predict_label)):
                    if true_char != pred_char:
                        char_errors[f"{true_char}->{pred_char}"] += 1
                        position_errors[pos] += 1
                
                # 打印前10个错误
                if len(error_samples) <= 10:
                    print(f"\n错误识别: 文件={filename}, 真实={true_label}, 预测={predict_label}")
        
        except Exception as e:
            print(f"\n处理图片失败: {filename}, 错误: {e}")
            continue
        
        # 更新进度条
        if HAS_TQDM:
            current_acc = 100.0 * correct / total if total > 0 else 0
            pbar.set_postfix({'准确率': f'{current_acc:.2f}%', '错误': len(error_samples)})
    
    if HAS_TQDM:
        pbar.close()
    
    # 计算最终准确率
    if total == 0:
        print("\n错误：没有成功处理任何图片")
        return
    
    accuracy = 100.0 * correct / total
    error_rate = 100.0 * len(error_samples) / total
    
    # 打印测试结果
    print(f"\n" + "=" * 80)
    print("测试结果")
    print("=" * 80)
    print(f"测试集: 训练集 (dataset/train)")
    print(f"模型: {model_path}")
    print("-" * 80)
    print(f"总样本数: {total}")
    print(f"正确识别: {correct}")
    print(f"错误识别: {len(error_samples)}")
    print(f"准确率: {accuracy:.2f}%")
    print(f"错误率: {error_rate:.2f}%")
    print("=" * 80)
    
    # 字符级别准确率
    char_total = total * 4
    char_correct_count = sum(position_errors)
    char_accuracy = 100.0 * (char_total - char_correct_count) / char_total
    print(f"\n字符级别统计:")
    print(f"  字符总数: {char_total}")
    print(f"  字符正确: {char_total - char_correct_count}")
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
    
    # 随机显示一些错误案例
    if error_samples:
        sample_errors = random.sample(error_samples, min(20, len(error_samples)))
        print(f"\n随机错误样本 (最多20个):")
        for i, (filename, true_label, predict_label) in enumerate(sample_errors, 1):
            print(f"  {i:2d}. {filename:30s} | 真实={true_label} | 预测={predict_label}")
    
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
            explode = (0.05, 0) if len(error_samples) > 0 else (0, 0)
            ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                   startangle=90, explode=explode)
            ax1.set_title('整体识别准确率（训练集）', fontsize=14, fontweight='bold')
            
            # 2. 位置错误分布
            ax2 = axes[0, 1]
            positions = ['位置1', '位置2', '位置3', '位置4']
            bars = ax2.bar(positions, position_errors, color='#8da0cb')
            ax2.set_ylabel('错误次数', fontsize=12)
            ax2.set_title('各位置错误分布', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
            for i, (bar, v) in enumerate(zip(bars, position_errors)):
                if v > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2, v + max(position_errors)*0.02,
                            str(v), ha='center', va='bottom', fontweight='bold')
            
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
                ax3.grid(True, alpha=0.3, axis='x')
                ax3.invert_yaxis()
                for i, (val, label) in enumerate(zip(error_counts, error_labels)):
                    ax3.text(val + max(error_counts)*0.02, i, str(val), 
                            va='center', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, '无字符混淆\n（完美识别！）', 
                        ha='center', va='center', fontsize=16, fontweight='bold')
                ax3.set_xlim([0, 1])
                ax3.set_ylim([0, 1])
                ax3.axis('off')
            
            # 4. 准确率对比
            ax4 = axes[1, 1]
            metrics = ['整体准确率', '字符准确率']
            values = [accuracy, char_accuracy]
            colors_bar = ['#66c2a5', '#8da0cb']
            bars = ax4.bar(metrics, values, color=colors_bar, width=0.6)
            ax4.set_ylabel('准确率 (%)', fontsize=12)
            ax4.set_title('准确率对比（训练集）', fontsize=14, fontweight='bold')
            ax4.set_ylim([0, 105])
            ax4.grid(True, alpha=0.3, axis='y')
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}%', ha='center', va='bottom', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # 保存图片
            log_dir = 'training_logs'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(log_dir, f'train_dataset_test_{timestamp}.png')
            plt.savefig(plot_file, dpi=100, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ 测试结果可视化已保存: {plot_file}")
        except Exception as e:
            print(f"\n绘图错误: {e}")
    
    # 性能评估
    print(f"\n" + "=" * 80)
    print("性能评估")
    print("=" * 80)
    if accuracy >= 99.5:
        print("🌟 优秀！模型在训练集上接近完美（≥99.5%）")
    elif accuracy >= 98.0:
        print("✅ 很好！模型在训练集上表现优秀（≥98%）")
    elif accuracy >= 95.0:
        print("✓ 良好！模型在训练集上表现良好（≥95%）")
    elif accuracy >= 90.0:
        print("⚠ 一般。模型在训练集上表现一般（≥90%）")
    else:
        print("❌ 欠拟合！模型在训练集上准确率过低（<90%）")
    
    print(f"\n对比参考:")
    print(f"  训练集准确率: {accuracy:.2f}%")
    print(f"  建议: 运行 captcha_test.py 查看测试集准确率")
    print(f"  健康指标: 训练集准确率 ≈ 测试集准确率 + 1-3%")
    print("=" * 80)

if __name__ == '__main__':
    main()
