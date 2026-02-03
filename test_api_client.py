# -*- coding: UTF-8 -*-
"""
FastAPI验证码识别服务测试客户端
演示如何调用API进行验证码识别
"""
import requests
import base64
import os
import time
from PIL import Image

# API配置
API_BASE_URL = "http://localhost:8000"  # 修改为实际的API地址

def test_health():
    """测试健康检查"""
    print("=" * 80)
    print("1. 测试健康检查")
    print("=" * 80)
    
    response = requests.get(f"{API_BASE_URL}/health")
    print(f"状态码: {response.status_code}")
    print(f"响应: {response.json()}")
    print()

def test_predict_file(image_path):
    """测试文件上传方式"""
    print("=" * 80)
    print("2. 测试文件上传识别")
    print("=" * 80)
    print(f"图片路径: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"错误：图片不存在 - {image_path}")
        return
    
    # 准备文件
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/png')}
        
        # 发送请求
        start = time.time()
        response = requests.post(f"{API_BASE_URL}/predict", files=files)
        elapsed = (time.time() - start) * 1000
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"响应: {result}")
    print(f"总耗时: {elapsed:.2f} ms")
    
    if result.get('success'):
        print(f"\n识别结果: {result['captcha']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print()

def test_predict_base64(image_path):
    """测试Base64方式"""
    print("=" * 80)
    print("3. 测试Base64识别")
    print("=" * 80)
    print(f"图片路径: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"错误：图片不存在 - {image_path}")
        return
    
    # 读取图片并转为Base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
    
    # 发送请求（JSON方式）
    start = time.time()
    response = requests.post(
        f"{API_BASE_URL}/predict/base64/json",
        json={'image_base64': image_base64}
    )
    elapsed = (time.time() - start) * 1000
    
    print(f"状态码: {response.status_code}")
    result = response.json()
    print(f"响应: {result}")
    print(f"总耗时: {elapsed:.2f} ms")
    
    if result.get('success'):
        print(f"\n识别结果: {result['captcha']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"推理时间: {result['inference_time_ms']:.2f} ms")
    print()

def test_batch_predict(image_dir, num_images=20):
    """测试批量识别"""
    print("=" * 80)
    print("4. 测试批量识别性能")
    print("=" * 80)
    
    if not os.path.exists(image_dir):
        print(f"错误：目录不存在 - {image_dir}")
        return
    
    # 获取图片列表
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"错误：目录中没有图片")
        return
    
    image_files = image_files[:num_images]
    print(f"测试图片数量: {len(image_files)}")
    
    # 批量识别（使用Base64方式）
    correct = 0
    total = 0
    total_time = 0
    
    for filename in image_files:
        image_path = os.path.join(image_dir, filename)
        true_label = filename.split('_')[0]
        
        # 读取图片并转为Base64
        with open(image_path, 'rb') as f:
            image_data = f.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 使用Base64 JSON方式识别
        start = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict/base64/json",
            json={'image_base64': image_base64}
        )
        elapsed = (time.time() - start) * 1000
        total_time += elapsed
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                predicted = result['captcha']
                total += 1
                if predicted == true_label:
                    correct += 1
                    status = '✓'
                else:
                    status = '✗'
                
                print(f"{filename:30s} | 真实={true_label} | 预测={predicted} | {status} | {elapsed:.2f}ms")
    
    print("\n" + "-" * 80)
    print(f"总数: {total}")
    print(f"正确: {correct}")
    print(f"准确率: {correct/total*100:.2f}%" if total > 0 else "N/A")
    print(f"总耗时: {total_time:.2f} ms")
    print(f"平均耗时: {total_time/total:.2f} ms/张" if total > 0 else "N/A")
    print(f"吞吐量: {total/(total_time/1000):.1f} 张/秒" if total > 0 else "N/A")
    print()

def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("FastAPI验证码识别服务测试")
    print("=" * 80)
    print(f"API地址: {API_BASE_URL}")
    print()
    
    # 1. 健康检查
    try:
        test_health()
    except Exception as e:
        print(f"健康检查失败: {e}")
        print("请确保API服务已启动: python captcha_api.py")
        return
    
    # 查找测试图片
    # 初始化为空列表，用于存储前10张图片的路径
    test_images = []  # 变量名改为复数更贴合多图片场景
    target_count = 100  # 明确需要获取的前N张图片数量
    
    for test_dir in ['dataset/test', 'dataset/train']:
        if os.path.exists(test_dir):
            # 筛选目录下所有.png格式图片
            png_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
            if png_images:
                # 提取前10张（如果图片不足10张，则取全部）
                top_10_images = png_images[:target_count]
                # 拼接完整路径并添加到结果列表中
                for img in top_10_images:
                    test_images.append(os.path.join(test_dir, img))
                # 若已获取到10张，直接终止循环（可选，根据需求决定是否继续遍历下一个目录）
                if len(test_images) >= target_count:
                    test_images = test_images[:target_count]  # 确保不超过10张
                    break
    
    # 为了兼容后续代码，取第一张图片作为test_image
    test_image = test_images[0] if test_images else None
    
    if not test_image:
        print("错误：找不到测试图片")
        return
    
    # # 2. 测试文件上传
    # try:
    #     test_predict_file(test_image)
    # except Exception as e:
    #     print(f"文件上传测试失败: {e}\n")
    
    # 3. 测试Base64
    try:
        test_predict_base64(test_image)
    except Exception as e:
        print(f"Base64测试失败: {e}\n")
    
    # 4. 测试批量识别
    test_dir = 'dataset/test' if os.path.exists('dataset/test') else 'dataset/train'
    if os.path.exists(test_dir):
        try:
            test_batch_predict(test_dir, num_images=10)
        except Exception as e:
            print(f"批量测试失败: {e}\n")
    
    print("=" * 80)
    print("测试完成")
    print("=" * 80)

if __name__ == '__main__':
    main()
