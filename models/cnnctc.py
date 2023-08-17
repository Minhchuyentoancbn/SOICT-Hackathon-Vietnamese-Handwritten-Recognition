import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNNCTC(nn.Module):
    """
    CNN + CTC + Sliding Window
    """
    def __init__(self, class_num):
        """
        Arguments:
        ----------
        class_num: int
            The number of classes to predict
        """
        super(CNNCTC, self).__init__()
        self.class_num = class_num

        feature = [
            nn.Conv2d(3, 50, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(inplace=True),
            nn.Conv2d(50, 100, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.Conv2d(100, 100, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.1),
            nn.BatchNorm2d(100),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(100, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.Conv2d(200, 200, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.2),
            nn.BatchNorm2d(200),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(200, 250, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(250),
            nn.ReLU(inplace=True),
            nn.Conv2d(250, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.Conv2d(300, 300, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.3),
            nn.BatchNorm2d(300),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(300, 350, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(350),
            nn.ReLU(inplace=True),
            nn.Conv2d(350, 400, stride=1, kernel_size=3, padding=1),
            # nn.Dropout(p=0.4),
            # nn.Conv2d(400, 400, stride=1, kernel_size=3, padding=1),
            nn.Dropout(p=0.4),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, stride=2)
        ]

        classifier = [
            nn.Dropout(p=0.5),
            nn.Conv2d(400, self.class_num, kernel_size=(4, 1), stride=1),
        ]

        self.feature = nn.Sequential(*feature)
        self.classifier = nn.Sequential(*classifier)


    def forward(self, x):  # x: N, C, H, W
        N, C, H, W = x.size()
        x = self.feature(x)
        x = self.classifier(x)  # N, class_num, 1, W
        x = x.squeeze(2)  # N, class_num, W
        x = x.permute(2, 0, 1)  # W, N, class_num
        return x