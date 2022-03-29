# -*- coding: utf-8 -*-
# @Time    : 2022/3/28 16:15
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : graph_info.py
# @Software: PyCharm


def getGraphInfo(data):
    """
    打印图的信息
    :param data:
    :return:
    """
    print("=================================================")
    print("节点数 : {} ".format(data.num_nodes))
    print("边的数量 : {} ".format(data.num_edges))
    print("平均节点的度 : {} ".format(data.num_edges / data.num_nodes))
    print("可训练的节点数 : {} ".format(data.num_edges / data.train_mask.sum()))
    print("包含的独立节点数 : {} ".format(data.contains_isolated_nodes()))
    print("包含的环形节点数 : {} ".format(data.contains_self_loops()))
    print("是否是无向图 : {} ".format(data.is_undirected()))
    print("=================================================")


def getDatasetInfo(dataset):
    """
    打印数据集信息
    :param dataset:
    :return:
    """
    print("=================================================")
    print("数据集 : {}".format(dataset))
    print("数据集中图的数量 : {}".format(len(dataset)))
    print("数据集的特征数量 : {}".format(dataset.num_features))
    print("数据集的类别数量 : {}".format(dataset.num_classes))
    print("=================================================")