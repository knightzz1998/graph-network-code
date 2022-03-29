# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 10:53
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : models.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, num_features: object, num_classes: object, hidden_channels: object) -> object:
        super(MLP, self).__init__()
        # 设置随机种子
        torch.manual_seed(1234)
        self.lin1 = nn.Linear(num_features, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        """
        :param x: x [num_nodes, num_features]
        :return:
        """
        x = self.lin1(x)  # [num_nodes, hidden_channels]
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)  # [num_nodes, num_classes]
        return x
