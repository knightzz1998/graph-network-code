# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 9:39
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : model.py
# @Software: PyCharm

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr="add", flow="source_to_target")
        # aggr : 聚合方式 => add
        # flow="source_to_target" : 表示消息从源节点传播到目标节点
        self.linear = torch.nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x, edge_index):
        """
        :param x: [num_nodes, in_channels]
        :param edge_index: [2, num_edges]
        :return:
        """
        # step 1 : 向邻接矩阵添加自环边
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # step 2 : 对节点表征做线性变换
        x = self.linear(x)
        # step 3 : 计算归一化系数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # step 4-5 : 开启消息传播
        return self.propagate(edge_index=edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """
        :param x_j: [num_edges, out_channels]
        :param norm: 归一化系数
        :return:
        """
        return norm.view(-1, 1) * x_j
