# 机构：人工智能研究所
# 人员：东
# 时间：2026/5/31 19:01

import torch
import torch.nn as nn
import config


class AlexNet(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(AlexNet, self).__init__()

        # 将前面的特征提取层直接按顺序打包好，这样forward就少了一些
        self.features = nn.Sequential(
            nn.Conv2d(config.IN_CHANNELS, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.DROPOUT),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),

            nn.Dropout(config.DROPOUT),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    model = AlexNet()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    print(y.shape)







