#!/usr/bin/env python


# # 构建AlexNet模型
# 具体实现(由于原文是使用两块GPU进行并行运算,此处我们之分析一半的模型)

# In[2]:


import torch
import torch.nn as nn


# In[1]:


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 提取特征
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),   # input[3, 224, 224] output[48, 55, 55]
            nn.ReLU(inplace=True),   # # inPlace=True， 增加计算量减少内存使用的一个方法
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]  
        )
        # 将全连接层作为一个整体，通过Dropout使其防止过拟合，一般放在全连接层和全连接层之间
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
            
        )
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # 展平处理，从channel维度开始展平，（第一个维度为channel）
        x = self.classifier(x)
        return x


# In[5]:

if __name__ == '__main__':

    model = AlexNet()
    print(model)


# In[ ]:




