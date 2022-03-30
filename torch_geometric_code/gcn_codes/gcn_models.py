# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 16:47
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : gcn_models.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(nn.Module):
    def __init__(self, num_features: object, num_classes: object, hidden_channels: object) -> object:
        super(GCN, self).__init__()
        # 设置随机种子
        torch.manual_seed(1234)
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        """
        :param edge_index: 边的下标
        :param x: x [num_nodes, num_features]
        :return:
        """
        x = self.conv1(x, edge_index)  # [num_nodes, hidden_channels]
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)  # [num_nodes, num_classes]
        return x