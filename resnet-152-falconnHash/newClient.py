#coding=utf-8
from multiprocessing.connection import Client
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import os
import numpy as np
import time
from multiprocessing.connection import Listener
import gc

img = None
k = 20
my_id = 99

# 创建文件夹
def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

# 在string中从前往后找str
def find_last(string, str):
    last_position=-1
    while True:
        position=string.find(str, last_position+1)
        if position==-1:
            return last_position
        last_position=position

# 获得两张图片的cos相似度（相似度范围0-1， 越大越相似）
def getDistOfCos(f, t):
    up = np.sum(np.multiply(f, t))
    ff = np.sqrt(np.sum(np.multiply(f, f)))
    tt = np.sqrt(np.sum(np.multiply(t, t)))
    down = ff * tt
    return up / down

# 分析服务端传回来的数据
# return 分析后的结果（数组形式），如果为None， 则图片地址有误
#   [[色卡id，相似图片地址], [色卡id，相似图片地址], ... , [色卡id，相似图片地址]]
def check(fal, tList):
    if tList is None:
        return None
    else:
        # 用于存取是否已经存在此色卡
        imgList = {}

        # 存取得到的结果
        resultList = []
        temp_click = 0

        # 找k个相似色卡
        near_k = k

        # 遍历相似图片（选出k个不同色卡）
        for i in tList:
            # 获得相似度， 放大100倍
            gailv = int(getDistOfCos(fal, i[0]) * 100)
            # 判断是否有该色卡
            if imgList.has_key(i[1]) == False:
                imgList[i[1]] = [i[1], i[2]]
                resultList.append([i[1], i[2]])
            if len(imgList) >= k: # 判断是否找齐near_k个
                break
        return resultList

def checkImage(imagePath, number=20, port=99):
    c = Client('./server%d.temp' % port, authkey=b'lee123456')
    arg = ['-f', imagePath, '-k', number]
    # 将信息传送给服务端
    c.send(arg)
    # 等待服务端处理结果
    ar = c.recv()
    is_Ok = False
    for i in ar:
        if i == 'Next':
            is_Ok = True
        elif is_Ok:
            is_Ok = False
            return check(i[0], i[1])
    
if __name__ == '__main__':
    filepath = None
    is_l = False
    is_m = False
    try:
        arg = sys.argv[1:]
        opts, args = getopt.getopt(sys.argv[1:], 'f:zt:k:sp', ['help', 'train'])
        for op, value in opts:
            if op == '-f':
                img = value
            elif op == '-k':
                k = int(value)
            elif op == '-p':
                my_id = int(value)
        c = Client('./server%d.temp' % my_id, authkey=b'lee123456')
        # 将信息传送给服务端
        c.send(arg)
        # 等待服务端处理结果
        ar = c.recv()
        is_Ok = False
        for i in ar:
            if i == 'Next':
                is_Ok = True
            elif is_Ok:
                is_Ok = False
                print(check(i[0], i[1]))
    except EOFError:
        print('Connection closed, Please Reset Server.')
