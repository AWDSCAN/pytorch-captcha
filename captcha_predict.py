# -*- coding: UTF-8 -*-
import numpy as np
import torch
from torch.autograd import Variable
#from visdom import Visdom # pip install Visdom
import captcha_setting
import my_dataset
from captcha_cnn_model import CNN

def main():
    # 检测GPU是否可用
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'使用设备: {device}')
    
    cnn = CNN()
    cnn.eval()
    cnn.load_state_dict(torch.load('model.pkl', map_location=device))
    cnn.to(device)  # 将模型移至GPU
    print("模型加载完成")

    predict_dataloader = my_dataset.get_predict_data_loader()

    #vis = Visdom()
    with torch.no_grad():  # 预测时不需要计算梯度
        for i, (images, labels) in enumerate(predict_dataloader):
            images = Variable(images).to(device)
            predict_label = cnn(images)
            
            # 将预测结果移回CPU进行处理
            predict_label_cpu = predict_label.cpu()

            c0 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, 2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
            c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label_cpu[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]

            c = '%s%s%s%s' % (c0, c1, c2, c3)
            print(c)
            #vis.images(image, opts=dict(caption=c))

if __name__ == '__main__':
    main()


