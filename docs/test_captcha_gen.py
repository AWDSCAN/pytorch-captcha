# -*- coding: UTF-8 -*-
"""
验证码生成器测试脚本
用于快速测试两种生成模式的效果
"""
import os
from captcha_gen import gen_captcha_text_and_image

def test_generate_samples():
    """生成3种模式的样本进行对比"""
    
    # 创建测试目录
    test_dir = 'test_samples'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    print("=" * 60)
    print("验证码生成器测试")
    print("=" * 60)
    
    modes = {
        'simple': '简单模式（captcha库）',
        'custom': '复杂模式（自定义PIL+干扰）',
        'mixed': '混合模式（随机选择）'
    }
    
    # 每种模式生成5张样本
    samples_per_mode = 5
    
    for mode, description in modes.items():
        print(f"\n{description}:")
        print("-" * 60)
        
        for i in range(samples_per_mode):
            text, image = gen_captcha_text_and_image(mode=mode)
            filename = f"{mode}_{text}_{i+1}.png"
            filepath = os.path.join(test_dir, filename)
            image.save(filepath)
            print(f"  ✓ 已生成: {filename} (验证码: {text})")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print(f"样本已保存到: {test_dir}/")
    print(f"共生成: {len(modes) * samples_per_mode} 张样本")
    print("=" * 60)
    print("\n建议：")
    print("1. 查看生成的图片对比三种模式的风格")
    print("2. simple模式：字符清晰，无干扰")
    print("3. custom模式：带干扰线和干扰点")
    print("4. mixed模式：随机混合以上两种")
    print("5. 推荐使用mixed模式训练，提高模型泛化能力")

def test_batch_generate():
    """测试批量生成性能"""
    import time
    
    print("\n" + "=" * 60)
    print("批量生成性能测试")
    print("=" * 60)
    
    test_count = 100
    modes = ['simple', 'custom', 'mixed']
    
    for mode in modes:
        start_time = time.time()
        
        for _ in range(test_count):
            text, image = gen_captcha_text_and_image(mode=mode)
        
        elapsed = time.time() - start_time
        speed = test_count / elapsed
        
        print(f"{mode:8s} 模式: {test_count}张耗时 {elapsed:.2f}秒, 速度: {speed:.1f}张/秒")
    
    print("=" * 60)

if __name__ == '__main__':
    # 测试1：生成样本对比
    test_generate_samples()
    
    # 测试2：性能测试（可选）
    print("\n是否进行性能测试？(y/n): ", end='')
    try:
        choice = input().strip().lower()
        if choice == 'y':
            test_batch_generate()
    except:
        print("跳过性能测试")
