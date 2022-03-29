# -*- coding: utf-8 -*-
# @Time    : 2022/3/29 10:23
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : draw_graph.py
# @Software: PyCharm

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import warnings


def visualize(out, color):
    """
    :param out: out 是图神经网络模型的参数
    :param color:
    :return:
    """
    # 消除 matplotlib 的警告
    warnings.filterwarnings("ignore", module="matplotlib")
    # ====================
    z = TSNE(n_components=2).fit_transform(out.detach().cpu().numpy())
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()
