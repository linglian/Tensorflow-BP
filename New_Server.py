# coding=utf-8
import multiprocessing
from multiprocessing.connection import Listener
from multiprocessing import Pool
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import os
import cv2
import numpy as np
import time
from PIL import Image
import gc
from sklearn.decomposition import PCA

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='test_train_Server.log',
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

# 公共区
path = '/media/lee/data/macropic/已整理宏观图/'
is_pool = True

# 数据增强区
tilesPerImage = 16
rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
rotate45degree = [45, 135, 270]
thresholdGLOABL = 0.42

# 提取特征区
dim = 2048
beishu = 8
num_round = 0
mxnetpath = '/home/lee/mxnet/python'
sys.path.insert(0, mxnetpath)
prefix = "full-resnet-152"
layer = 'pool1_output'

# 全局变量
global my_arr, my_id, big_class

def removeFile(name):
    if os.path.exists(name):
        os.remove(name)

# 获取处理后的图片
# 参数 img： 图片数组
def getImage(img):
    # img=cv2.imread(img, 1)
    if img is not None:
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=cv2.resize(img, (224, 224))
        img=np.swapaxes(img, 0, 2)
        img=np.swapaxes(img, 1, 2)
        img=img[np.newaxis, :]
        return img
    else:
        return None

# 获取图片的特征
# 参数 img：图片地址
# 参数 f_mod： 模型
def getFeatures(img, f_mod = None):
    img=getImage(img)
    if img is not None:
        f=f_mod.predict(img)
        f=np.ravel(f)
        temp = []
        for k in range(0, len(f), beishu):
            t = []
            for l in range(0, beishu):
                t.append(f[k + l])
            temp.append(t)
        pca = PCA(n_components=1, copy=False)
        f = pca.fit_transform(temp)
        f = np.ravel(f).astype(np.float32)
        return f
    else:
        return None

# 初始化hash
def init_hash():
    global my_arr, my_id, big_class
    # 获得数组
    my_arr = np.load(os.path.join(path, 'array.npy'))
    my_id = np.load(os.path.join(path, 'id.npy'))
    f = open(os.path.join(path, 'big_class.txt'),'r')  
    a = f.read()  
    big_class = eval(a)  
    f.close()
    # 获取数组数量
    trainNum=len(my_arr)
    # 获得默认参数
    p=falconn.get_default_parameters(trainNum, dim)
    t=falconn.LSHIndex(p)
    dataset = my_arr
    # 生成hash
    logging.info('Start Hash setup')
    t.setup(dataset)
    if is_pool:
        q=t.construct_query_pool()
    else:
        q=t.construct_query_object()
    return q


# 初始化mxnet
def init_mxnet(GPUid = 0):
    import mxnet as mx
    model=mx.model.FeedForward.load(
        prefix, num_round, ctx = mx.gpu(GPUid), numpy_batch_size = 1)
    internals=model.symbol.get_internals()
    fea_symbol=internals[layer]
    feature_extractor=mx.model.FeedForward(ctx = mx.gpu(GPUid), symbol = fea_symbol, numpy_batch_size = 1,
                                             arg_params = model.arg_params, aux_params = model.aux_params, allow_extra_params = True)
    init_mod=feature_extractor
    return feature_extractor

# 初始化
def init():
    t1 = time.time()
    mod=init_mxnet()
    q = init_hash()
    logging.info('Speed Time %.02f' % (time.time() - t1))
    return (mod, q)

# 获得Cos距离
def getDistOfCos(f, t):
    up=np.sum(np.multiply(f, t))
    ff=np.sqrt(np.sum(np.multiply(f, f)))
    tt=np.sqrt(np.sum(np.multiply(t, t)))
    down=ff * tt
    return up / down

# 获取图片的特征，并进行hash计算
def getTest(img, mod, q, k = 20):
    fal = getFeatures(img, f_mod = mod)
    if fal is not None:
        tList = np.array(q.find_k_nearest_neighbors(fal, k))
        return fal, tList
    else:
        return fal, None

"""将图片切分
    im: 所要处理的图片
    deg: 选择的角度
"""

def im_crotate_image_square(im, deg):
    import math
    im2 = im.rotate(deg, expand=1)
    im = im.rotate(deg, expand=0)

    width, height = im.size
    if width == height:
        im = im.crop((0, 0, width, int(height * 0.9)))
        width, height = im.size

    rads = math.radians(deg)
    new_width = width - (im2.size[0] - width)

    left = top = int((width - new_width) / 2)
    right = bottom = int((width + new_width) / 2)

    im = im.crop((left, top, right, bottom))
    width, height = im.size
    if width == 0 or height == 0:
        return im2
    return im


"""数据增强
    im: 需要增强的图片
    temp_list: 需要存在哪个数组
"""


def splite_img(imgfile):
    import random
    try:
        temp_list = []
        # 打开图片
        im = Image.open(imgfile)
        # 获得原始图片大小
        w, h = im.size
        # 变换形状224， 224
        temp_im = cv2.resize(np.array(im), (224, 224))
        # 增加原始图片
        temp_list.append(temp_im)
        # 删除图片上下尺子的影响
        im = im.crop((0, int(h * 0.1), w, int(h * 0.9)))

        dx = dy = 224
        # 将图片增强tilesPerImage份
        for i in range(1, tilesPerImage + 1):
            # 获得图片大小
            # 获取图片截取大小
            if i < (tilesPerImage / 360) * 100 and w > 300:
                dx = 224
            if (tilesPerImage / 360) * 100 < i < (tilesPerImage / 360) * 200 and w > 500:
                dx = 320
            if (tilesPerImage / 360) * 200 < i < (tilesPerImage / 360) * 300 and w > 800:
                dx = 640
            if i < (tilesPerImage / 360) * 100 and h > 300:
                dy = 224
            if (tilesPerImage / 360) * 100 < i < (tilesPerImage / 360) * 200 and h > 500:
                dy = 320
            if (tilesPerImage / 360) * 200 < i < (tilesPerImage / 360) * 300 and h > 800:
                dy = 640

            # 随机获得图片区域图片
            x = random.randint(0, w - dx - 1)
            y = random.randint(0, h - dy - 1)

            # 将图片截取指定大小
            im_cropped = im.crop((x, y, x + dx + 1, y + dy + 1))

            if i % 2 == 0:  # 将图片旋转 90\180
                im_cropped = im_cropped.transpose(
                    random.choice(rotateAction))
            if i % 2 == 0 and i > (tilesPerImage / 360) * 300:  # 将图片旋转1-45角度
                roate_drgree = random.choice(rotate45degree)
                im_cropped = im_crotate_image_square(
                    im_cropped, roate_drgree)

            # 将处理后的图片转为224、224
            im_cropped = cv2.resize(
                np.array(im_cropped), (224, 224))

            # 将处理后的图片按照，图片特征， 色卡id，图片地址进行存入数组
            temp_list.append(im_cropped)
        return temp_list
    except Exception as msg:
        logging.error('Bad Image: %s B %s ' % (imgfile, msg))
        return None

# 客户端发来的请求进行处理(最好需要几个就设置多少k，不然影响速度)
# return 图片特征，最近的数组(k * max_img长度的数组, 相似度从近到远)
def make_work(conn, mod, q):
    global my_arr, my_id, big_class
    try:
        msg=conn.recv()
        logging.info(msg)
        opts, args=getopt.getopt(msg, 'f:zt:k:sp', ['help', 'train'])
        img=None
        rank=20
        img_type=0
        is_save=True
        msg=[]
        filepath=None
        is_pcl=False
        for op, value in opts:
            if op == '-f':
                img=value
            elif op == '-z':
                return 'Close'
            elif op == '-k':
                rank=int(value)
            elif op == '-s':
                is_save=False
            elif op == '--train':
                return "train"
            elif op == '-p':
                is_pcl=True
            elif op == '-t':
                img_type=value
            elif op == '--help':
                msg.append(' ')
                msg.append('Usage:')
                msg.append('  Client [options]')
                msg.append(' ')
                msg.append('General Options:')
                msg.append('-f <path>\t\t Set test image path')
                msg.append('-z \t\t\t Close server')
                msg.append('-k <number>\t\t Set rank')
                msg.append('-s \t\t\t No Save image of rank K')
                msg.append(
                    '-t <number>\t\t Set image type if you want to know test type')
                return msg
        if img is None:
            msg.append('Must set Image Path use -f')
            return msg
        ti_time= time.time()
        img_list = splite_img(img)
        logging.info('Splite = %d' % len(img_list))
        main_list = []
        main_class_map = {}
        for t_img in img_list:
            fal, tList = getTest(t_img, mod, q, k = rank)
            if tList is None:
                msg.append('Bad Img Path')
                return msg
            else:
                # 计算大类
                ks = {}
                for i in tList:
                    for j in big_class[my_id[i]]:
                        if ks.has_key(j):
                            ks[j] += 1;
                        else:
                            ks[j] = 1;
                my_big_class_num = 0
                my_big_class = {}
                
                flag = True
                while flag:
                    flag = False
                    for i in ks:
                        if my_big_class_num == 0:
                            my_big_class_num = ks[i]
                            my_big_class = {}
                            my_big_class[i] = True
                        elif ks[i] > my_big_class_num + int(rank / 2):
                            my_big_class_num = ks[i]
                            my_big_class = {}
                            my_big_class[i] = True
                            flag = True
                            break
                        elif ks[i] >= my_big_class_num - int(rank / 2):
                            my_big_class[i] = True

                for i in my_big_class:
                    if main_class_map.has_key(i):
                        main_class_map[i] += 1;
                    else:
                        main_class_map[i] = 1;

                logging.info('######### Find Big Class')
                logging.info(my_big_class)
                imgList = {}
                # 遍历获得识别结果
                for i in tList:
                    # 判断是否为一个大类
                    flag = False
                    for j in my_big_class:
                        for k in big_class[my_id[i]]:
                            if j == k:
                                flag = True
                                break
                        if flag:
                            break
                    if flag:
                        # 计算概率，并添加
                        if imgList.has_key(my_id[i]):
                            imgList[my_id[i]]= False
                        else:
                            main_list.append(my_id[i])
                            imgList[my_id[i]]= True
                        if len(imgList) >= rank:
                            break
        main_big_class_num = 0
        main_big_class = {};
        print(main_class_map)
        flag = True
        while flag:
            flag = False
            for i in main_class_map:
                if main_big_class_num == 0:
                    main_big_class_num = main_class_map[i]
                    main_big_class = {}
                    main_big_class[i] = True
                elif main_class_map[i] > main_big_class_num + int(tilesPerImage / 2):
                    main_big_class_num = main_class_map[i]
                    main_big_class = {}
                    main_big_class[i] = True
                    flag = True
                    break
                elif main_class_map[i] >= main_big_class_num - int(tilesPerImage / 2):
                    main_big_class[i] = True
        print(main_big_class)
        id_Map = {}
        id_List = []
        for i in main_list:
            flag = False
            for j in main_big_class:
                for k in big_class[i]:
                    if j == k:
                        flag = True
                        break
                if flag:
                    break
            if flag:
                if id_Map.has_key(i):
                    id_Map[i] = False
                else:
                    id_Map[i] = True
                    id_List.append(i)
        msg.append('Next')
        msg.append(id_List)
        msg.append('Test Image Spend Time: %.2lf s' %
                    (time.time() - ti_time))
        return msg
    except EOFError:
        logging.info('Connection closed')
        return None


# 运行Server，一直监听接口
def run_server(address, authkey, mod, q):
    serv = Listener(address, authkey =authkey)
    while True:
        try:
            client= serv.accept()
            msg= make_work(client, mod, q)
            if msg == 'Close': # 关闭监听
                serv.close()
                return "Close"
            else:
                client.send(msg)
        except Exception:
            traceback.print_exc()
    serv.close()


if __name__ == '__main__':
    opts, args= getopt.getopt(sys.argv[1:], 'f:x:p:k:b:d:t:')
    server_id= 99
    for op, value in opts:
        if op == '-f': # 设置四个NPY文件所在文件夹路径
            path = value
        elif op == '-p': # 设置运行时的id，同于通信
            server_id = int(value)
        elif op == '-b': # 设置最大PCA倍数
            beishu = int(value)
        elif op == '-d': # 设置最大图片数量
            dim = int(value)
        elif op == '-x': # 设置mxnet/python所在路径
            mxnetpath = value
            sys.path.insert(0, mxnetpath)
        elif op == '-t': # 设置增强数量
            tilesPerImage = int(value)
    while True:
        logging.info('Start Init')
        mod, q = init()
        logging.info('End Init')
        logging.info('Start Run')
        run_server('/usr/local/server%d.temp' % server_id, b'lee123456', mod, q)
        logging.info('Stop Run')
