#coding=utf-8
import falconn
import numpy as np
import os
base_path = './数据增强_整理版'
dim = 2048
# 获得数组
my_feature = np.load(os.path.join(base_path, 'feature.npy'))
my_class_name = np.load(os.path.join(base_path, 'class_name.npy'))
# 获取数组数量
trainNum=len(my_feature)
# 获得默认参数
p=falconn.get_default_parameters(trainNum, dim)
t=falconn.LSHIndex(p)
dataset = my_feature
# 生成hash
t.setup(dataset)
q=t.construct_query_pool()
print q