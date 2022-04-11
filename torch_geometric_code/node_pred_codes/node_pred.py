# -*- coding: utf-8 -*-
# @Time    : 2022/4/7 16:37
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : node_pred.py
# @Software: PyCharm
from os import path

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import ReLU
from torch_geometric.nn import GATConv, Sequential
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import argparse

from torch_geometric.utils import f1_score
from tqdm import tqdm, trange


class GAT(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels_list):
        """
        :param num_features: 数据集的特征数量
        :param num_classes: 图的类别数
        :param hidden_channels:
        """
        super(GAT, self).__init__()
        # 设置随机数种子
        torch.manual_seed(12345)
        # 拼接输入特征数和隐藏层数 : [input_features, hidden_channel , ...]
        hns = [num_features] + hidden_channels_list
        conv_list = []
        #
        for idx in range(len(hidden_channels_list)):
            # [input_features, hidden_channels_1]
            # [hidden_channels_1, hidden_channels_2]
            # ...
            # [hidden_channels_n-1, hidden_channels_n]
            conv_list.append((GATConv(in_channels=hns[idx], out_channels=hns[idx + 1]), 'x, edge_index -> x'))
            conv_list.append(ReLU(inplace=True))

        # 整合 多层网络
        self.conv_seq = Sequential('x, edge_index', conv_list)
        # linear
        self.linear = nn.Linear(hidden_channels_list[-1], num_classes)  # [input_features, num_classes]

    def forward(self, x, edge_index):
        x = self.conv_seq(x, edge_index)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear(x)
        return x


def pre_process():
    # 读取数据
    datasets = Planetoid(root="K:/CodeWorkSpace/DeeplApp/graph-network-code/datasets", name="Cora",
                         transform=NormalizeFeatures())
    return datasets


parser = argparse.ArgumentParser(description="node_pred.py")
parser.add_argument("-num_features", default=12, help="用户输入特征大小")
parser.add_argument("-hidden_channels_list", default=3, help="GATConv层数")
parser.add_argument("-epochs", default=30, help="训练轮数")
parser.add_argument("-best_model_path", default="model/best_model.pth", help="模型保存路径")


def train(args, model):
    """
    训练模型
    :param args:
    :param model:
    :return:
    """
    # 加载预训练模型
    if path.exists(args.best_model_path):
        model.load_state_dict(torch.load(args.best_model_path))

    # 定义损失函数
    criterion = nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for epoch in range(args.epochs):
        best_loss = float('inf')
        loop = tqdm(range(300), total=300)
        for iteration in loop:
            model.train()
            # 初始化梯度
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            # 计算loss
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            # 反向传播
            loss.backward()
            # 更新梯度
            optimizer.step()

            # 更新信息
            loop.set_description(f'Epoch [{epoch}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())
            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(model.state_dict(), args.best_model_path)


def test(args, model, data):
    """
    训练模型
    :param args:
    :param model:
    :return:
    """
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    accuracy = int(test_correct.sum()) / int(data.test_mask.sum())
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 获取训练数据
    datasets = pre_process()
    data = datasets[0].to(device)
    # 定义模型
    model = GAT(datasets.num_features, datasets.num_classes, [200, 100]).to(device)
    train(args, model)
    test(args, model)
