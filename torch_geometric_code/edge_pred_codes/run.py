# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 18:08
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : run.py
# @Software: PyCharm
import torch
import os.path as osp
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import train_test_split_edges
from tqdm import trange, tqdm

from EdgeNet import EdgeNet
from train import train
from test_code import test


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = 'Cora'
    # __file__表示显示文件当前的位置
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
    print(path)
    dataset = Planetoid(path, dataset, transform=NormalizeFeatures())
    data = dataset[0]
    ground_truth_edge_index = data.edge_index.to(device)
    # 设置训练节点和测试验证节点为None
    data.train_mask = data.val_mask = data.test_mask = None
    # 分割数据集
    data = train_test_split_edges(data)
    data = data.to(device)
    # 构建模型
    model = EdgeNet(dataset.num_features, 64).to(device)
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    best_val_auc = test_auc = 0
    loop = tqdm(range(100), total=100)
    for epoch in loop:
        loss = train(data, model, optimizer)
        val_auc, tmp_test_auc = test(data, model)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            test_auc = tmp_test_auc

        # 打印结果
        loop.set_description(f'Epoch: [{epoch+1}/{100}]')
        loop.set_postfix(loss=loss.item(), val_auc=val_auc, test_auc=test_auc)


if __name__ == '__main__':
    main()
