# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 17:34
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : graph_base.py
# @Software: PyCharm

import pandas as pd
import networkx as nx

edges = pd.DataFrame()
# 起点
edges["sources"] = [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]
# 终点
edges["targets"] = [2, 4, 5, 3, 1, 2, 5, 1, 5, 1, 3, 4]
# 权重
edges["weights"] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# 设置图
G = nx.from_pandas_edgelist(edges, source='sources', target='targets', edge_attr='weights')

# 计算度
print("度 : ", nx.degree(G))

# 连通分量 : 节点
print("连通分量 : ", list(nx.connected_components(G)))

# 图直径, 图中最短路径的最大值
print("图直径 : ", nx.diameter(G))

# 度中心性
print("度中心性 : ", nx.degree_centrality(G))

# 特征向量中心性
print("特征向量中心性 : ", nx.eigenvector_centrality(G))

# 中介中心性 : Betweenness
print("中介中心性 : Betweenness : ", nx.betweenness_centrality(G))

# 连接中心性 : Closeness
print("连接中心性 : Closeness : ", nx.closeness_centrality(G))

# PageRank值
print("PageRank : ", nx.pagerank(G))

# HITS
print("HITS : ", nx.hits(G))
