import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .resnet import resnet50


class Model(nn.Module):
    def __init__(
            self,
            last_conv_stride=2,
            last_conv_dilation=1,
            max_or_avg='avg',
            dropout_rate=0,
            num_classes=751,
    ):
        super(Model, self).__init__()

        self.base = resnet50(
            pretrained=True,
            last_conv_stride=last_conv_stride,
            last_conv_dilation=last_conv_dilation
        )

        self.pool = F.adaptive_avg_pool2d if max_or_avg == 'avg' else F.adaptive_max_pool2d
        self.dropout = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(2048, num_classes)
        init.normal(self.fc.weight, std=0.001)
        init.constant(self.fc.bias, 0)
        print('Model Structure:\n', self)

    def forward(self, x):
        x = self.base(x)
        x = self.pool(x, 1)
        feat = x.view(x.size(0), -1)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return feat, logits
