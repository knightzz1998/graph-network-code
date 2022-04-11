# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 15:04
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : EdgeNet.py
# @Software: PyCharm

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class EdgeNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels:
        :param out_channels:
        """
        super(EdgeNet, self).__init__()
        self.conv1 = GCNConv(in_channels=in_channels, out_channels=128)
        self.conv2 = GCNConv(in_channels=128, out_channels=out_channels)

    def encode(self, x, edge_index):
        """
        生成节点表征(节点的特征表示)
        :param x: 节点矩阵 [2708, 1433]
        :param edge_index: 边下标 [2, ]
        :return:
        """
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index):
        """
        根据边两端节点的表征生成边为真的几率（odds）
        :param z:
        :param pos_edge_index: 例如 : [2, 527]
        :param neg_edge_index: 例如 : [2, 527]
        :return:
        """
        # 拼接正向边和负向边
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # [2, 1054]
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  #

    def decode_all(self, z):
        """
        推理阶段 : 对所有的节点预测存在边的几率
        :param z:
        :return:
        """
        # 矩阵相乘
        prob_obj = z @ z.t()
        return (prob_obj > 0).nozero(as_tuple=False).t()
