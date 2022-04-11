# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 15:04
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : pre_process.py
# @Software: PyCharm

import torch
import torch.functional as F


def get_link_labels(pos_edge_index, neg_edge_index):
    """
    函数用于生成完整训练集的标签。
    :param pos_edge_index:
    :param neg_edge_index:
    :return:
    """
    # 统计正向边和负向边的数量
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    # 获取节点标签
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1
    return link_labels
