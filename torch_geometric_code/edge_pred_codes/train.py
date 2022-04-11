# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 15:04
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : train.py
# @Software: PyCharm


import torch
import torch.nn as nn
import torch.nn.functional as F
import os.path as osp
from torch_geometric.utils import negative_sampling
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import train_test_split_edges
from pre_process import get_link_labels


def train(data, model, optimizer):
    model.train()
    # 负向采样
    neg_edge_index = negative_sampling(edge_index=data.train_pos_edge_index,
                                       num_nodes=data.num_nodes)
    # 梯度归零
    optimizer.zero_grad()
    # 编码器, 生成节点表征
    z = model.encode(data.x, data.train_pos_edge_index)

    # TODO : 不太懂下面的步骤什么意思

    # 解码
    link_logits = model.decode(z, data.train_pos_edge_index, neg_edge_index)
    # 获取完整训练集的标签
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index).to(data.x.device)
    # 计算目标和输入之间的二进制交叉熵
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    loss.backward()
    optimizer.step()
    return loss
