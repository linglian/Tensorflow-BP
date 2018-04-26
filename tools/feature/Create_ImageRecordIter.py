# coding=utf-8

import sys
sys.path.insert(0, '/home/lol/dl/mxnet/python')
import mxnet as mx
import numpy as np
import cv2
import os
root = '/home/lol/dl/dlbp/image/lst'

batch_size = 1

# 创建训练集iter
train_iter = mx.io.ImageRecordIter(
    path_imgrec=os.path.join(root, 'train.rec'),  # rec文件路径
    data_shape=(3, 32, 32),  # 期望的数据形状，注意：
    # 即使图片不是这个尺寸，也可以在此被自动转换
    batch_size=batch_size,  # 每次传入10条数据
)

# 创建内部测试集iter
val_iter = mx.io.ImageRecordIter(
    path_imgrec=os.path.join(root, 'test.rec'),  # rec文件路径
    data_shape=(3, 32, 32),
    batch_size=batch_size,  # 必须与上面的batch_size相等，否则不能对应
)

batch = train_iter.next()  # 导入一个样本batch
print batch

data = batch.data[0]

import logging

logging.getLogger().setLevel(logging.DEBUG)  # 设置logger输出级别
# 删除此句则无法输出训练状态


def get_model():
    sym, arg_params,aux_params = mx.model.load_checkpoint('full-resnet-152',0)

    mod= mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
    mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod

model = get_model()


model.fit(
    train_data=train_iter,  # 训练集
    eval_data=val_iter,  # 验证集
    batch_end_callback=mx.callback.Speedometer(batch_size, 200),
    num_epoch=0
    # 监视训练状态
    # 每200个iteration后输出一次
)
