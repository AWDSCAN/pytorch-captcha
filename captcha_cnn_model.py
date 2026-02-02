# -*- coding: UTF-8 -*-
import torch.nn as nn
import captcha_setting

# 优化的CNN模型 (7层卷积，3次池化，保持8倍下采样)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第1层卷积 - 输入1通道，输出32通道，第一次池化
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 60x160 -> 30x80
        
        # 第2层卷积 - 32->64通道，不池化
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        # 第3层卷积 - 64->64通道，第二次池化
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 30x80 -> 15x40
        
        # 第4层卷积 - 64->128通道，不池化
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        # 第5层卷积 - 128->128通道，不池化
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        # 第6层卷积 - 128->128通道，第三次池化
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 15x40 -> 7x20 (实际是7.5x20)
        
        # 第7层卷积 - 128->128通道，不池化
        self.layer7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        
        # 全连接层，只在这里使用dropout
        self.fc = nn.Sequential(
            nn.Linear((captcha_setting.IMAGE_WIDTH//8)*(captcha_setting.IMAGE_HEIGHT//8)*128, 1024),
            nn.Dropout(0.5),  # 只在全连接层使用dropout
            nn.ReLU())
        
        self.rfc = nn.Sequential(
            nn.Linear(1024, captcha_setting.MAX_CAPTCHA*captcha_setting.ALL_CHAR_SET_LEN),
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out

