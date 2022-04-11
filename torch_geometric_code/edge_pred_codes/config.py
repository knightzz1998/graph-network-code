# -*- coding: utf-8 -*-
# @Time    : 2022/4/11 18:49
# @Author  : 王天赐
# @Email   : 15565946702@163.com
# @File    : config.py
# @Software: PyCharm

import yaml

# 参考 : https://www.cnblogs.com/klb561/p/10085328.html

config = yaml.load(open('config/config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
print(config)
print(type(config))
print(config['dataset_path'])