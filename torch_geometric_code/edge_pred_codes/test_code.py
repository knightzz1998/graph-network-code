# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 17:52
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : test.py
# @Software: PyCharm

import torch
from sklearn.metrics import roc_auc_score

from pre_process import get_link_labels
import torch.nn.functional as F


@torch.no_grad()
def test(data, model):
    model.eval()
    # 对数据进行编码
    z = model.encode(data.x, data.train_pos_edge_index)

    results = []
    for prefix in ['val', 'test']:
        # 获取正向边下标和反向边下标
        pos_edge_index = data[f'{prefix}_pos_edge_index']
        neg_edge_index = data[f'{prefix}_neg_edge_index']

        # 具体是干嘛的?
        link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        link_probs = torch.sigmoid(link_logits)
        link_labels = get_link_labels(pos_edge_index, neg_edge_index).to(data.x.device)
        results.append(roc_auc_score(link_labels.cpu(), link_probs.cpu()))
    return results
