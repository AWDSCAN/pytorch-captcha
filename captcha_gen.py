# -*- coding: UTF-8 -*-
"""
验证码生成器 - 支持两种生成方式
1. captcha库生成（简单风格）
2. 自定义PIL生成（复杂风格，带干扰线和干扰点）
"""
from captcha.image import ImageCaptcha  # pip install captcha
from PIL import Image, ImageDraw, ImageFont
import random
import time
import captcha_setting
import os


class CustomCaptchaGenerator:
    """自定义验证码生成器 - 使用PIL生成带干扰的验证码"""
    
    def __init__(self, width=160, height=60, font_size=40):
        """
        初始化验证码生成器
        
        Args:
            width (int): 图片宽度（默认使用captcha_setting中的配置）
            height (int): 图片高度（默认使用captcha_setting中的配置）
            font_size (int): 字体大小
        """
        self.width = width
        self.height = height
        self.font_size = font_size
        self.chars = captcha_setting.ALL_CHAR_SET  # 使用项目统一的字符集
        self.max_captcha = captcha_setting.MAX_CAPTCHA  # 验证码长度
    
    def _random_color(self, min_val=0, max_val=128):
        """生成随机颜色"""
        return (
            random.randint(min_val, max_val),
            random.randint(min_val, max_val),
            random.randint(min_val, max_val)
        )
    
    def generate_captcha(self):
        """
        生成验证码图片和文本
        
        Returns:
            tuple: (PIL.Image对象, 验证码文本字符串)
        """
        # 随机生成验证码文本
        captcha_text = ''.join(random.choice(self.chars) for _ in range(self.max_captcha))
        
        # 创建图片背景
        bgcolor = (255, 255, 255)  # 白色背景
        image = Image.new('RGB', (self.width, self.height), bgcolor)
        draw = ImageDraw.Draw(image)
        
        # 加载字体（使用默认字体）
        try:
            font = ImageFont.load_default().font_variant(size=self.font_size)
        except:
            # 兼容旧版本PIL
            font = ImageFont.load_default()
        
        # 绘制干扰线（5-8条）
        line_count = random.randint(5, 8)
        for _ in range(line_count):
            x1 = random.randint(0, self.width)
            y1 = random.randint(0, self.height)
            x2 = random.randint(0, self.width)
            y2 = random.randint(0, self.height)
            linecolor = self._random_color(0, 128)
            draw.line((x1, y1, x2, y2), fill=linecolor, width=random.randint(1, 3))
        
        # 绘制干扰点（100-300个）
        dot_count = random.randint(100, 300)
        for _ in range(dot_count):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            dotcolor = self._random_color(0, 128)
            draw.point((x, y), fill=dotcolor)
        
        # 计算字符间距
        char_width = self.width // (self.max_captcha + 1)
        
        # 绘制验证码文字（带阴影和旋转效果）
        for i, char in enumerate(captcha_text):
            # 随机字体颜色（深色）
            fontcolor = self._random_color(0, 128)
            
            # 随机位置偏移
            x_offset = random.randint(-5, 5)
            y_offset = random.randint(-5, 5)
            x_pos = char_width * (i + 1) + x_offset
            y_pos = (self.height - self.font_size) // 2 + y_offset
            
            # 绘制阴影（可选）
            if random.random() > 0.5:
                shadow_offset = random.randint(1, 3)
                shadow_color = (100, 100, 100)
                draw.text(
                    (x_pos + shadow_offset, y_pos + shadow_offset),
                    char,
                    font=font,
                    fill=shadow_color
                )
            
            # 绘制字符
            draw.text((x_pos, y_pos), char, font=font, fill=fontcolor)
        
        return image, captcha_text


def random_captcha():
    """生成随机验证码文本"""
    captcha_text = []
    for i in range(captcha_setting.MAX_CAPTCHA):
        c = random.choice(captcha_setting.ALL_CHAR_SET)
        captcha_text.append(c)
    return ''.join(captcha_text)


def gen_captcha_text_and_image_simple():
    """
    使用captcha库生成验证码（简单风格）
    
    Returns:
        tuple: (验证码文本, PIL.Image对象)
    """
    image = ImageCaptcha(width=captcha_setting.IMAGE_WIDTH, height=captcha_setting.IMAGE_HEIGHT)
    captcha_text = random_captcha()
    captcha_image = Image.open(image.generate(captcha_text))
    return captcha_text, captcha_image


def gen_captcha_text_and_image_custom():
    """
    使用自定义生成器生成验证码（复杂风格，带干扰）
    
    Returns:
        tuple: (验证码文本, PIL.Image对象)
    """
    generator = CustomCaptchaGenerator(
        width=captcha_setting.IMAGE_WIDTH,
        height=captcha_setting.IMAGE_HEIGHT,
        font_size=40
    )
    captcha_image, captcha_text = generator.generate_captcha()
    return captcha_text, captcha_image


def gen_captcha_text_and_image(mode='mixed'):
    """
    生成验证码（统一接口）
    
    Args:
        mode (str): 生成模式
            - 'simple': 仅使用captcha库生成
            - 'custom': 仅使用自定义生成器
            - 'mixed': 随机混合使用两种方式（默认）
    
    Returns:
        tuple: (验证码文本, PIL.Image对象)
    """
    if mode == 'simple':
        return gen_captcha_text_and_image_simple()
    elif mode == 'custom':
        return gen_captcha_text_and_image_custom()
    elif mode == 'mixed':
        # 随机选择生成方式
        if random.random() > 0.5:
            return gen_captcha_text_and_image_simple()
        else:
            return gen_captcha_text_and_image_custom()
    else:
        raise ValueError(f"不支持的生成模式: {mode}，请使用 'simple', 'custom' 或 'mixed'")


if __name__ == '__main__':
    # 配置参数
    count = 100  # 生成数量
    mode = 'mixed'  # 生成模式: 'simple', 'custom', 'mixed'
    path = captcha_setting.TRAIN_DATASET_PATH  # 通过改变此处目录，以生成训练、测试和预测用的验证码集
    
    print("=" * 60)
    print("验证码生成器")
    print("=" * 60)
    print(f"生成模式: {mode}")
    print(f"  - simple: 使用captcha库（简单风格）")
    print(f"  - custom: 使用自定义生成器（复杂风格）")
    print(f"  - mixed: 随机混合两种方式")
    print(f"生成数量: {count}")
    print(f"保存路径: {path}")
    print(f"图片尺寸: {captcha_setting.IMAGE_WIDTH}x{captcha_setting.IMAGE_HEIGHT}")
    print(f"验证码长度: {captcha_setting.MAX_CAPTCHA}位")
    print(f"字符集: {len(captcha_setting.ALL_CHAR_SET)}个字符（数字+大写字母）")
    print("=" * 60)
    
    # 创建目录
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")
    
    # 统计两种生成方式的数量
    simple_count = 0
    custom_count = 0
    
    # 生成验证码
    print("\n开始生成验证码...")
    for i in range(count):
        # 记录当前使用的生成方式
        current_mode = mode
        if mode == 'mixed':
            current_mode = 'simple' if random.random() > 0.5 else 'custom'
        
        if current_mode == 'simple':
            simple_count += 1
        else:
            custom_count += 1
        
        # 生成验证码
        now = str(int(time.time() * 1000000))  # 使用微秒时间戳避免重复
        text, image = gen_captcha_text_and_image(mode)
        
        # 保存文件
        filename = f"{text}_{now}.png"
        filepath = os.path.join(path, filename)
        image.save(filepath)
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == count:
            print(f"进度: {i+1}/{count} - 已保存: {filename}")
    
    print("\n" + "=" * 60)
    print("生成完成！")
    print(f"总计: {count} 张")
    if mode == 'mixed':
        print(f"  - captcha库生成: {simple_count} 张")
        print(f"  - 自定义生成: {custom_count} 张")
    print(f"保存路径: {path}")
    print("=" * 60)

