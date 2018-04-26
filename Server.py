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

path = '/home/lol/dl/Image/feature_train.npy'
mxnetpath = '/home/lol/dl/mxnet/python'
sys.path.insert(0, mxnetpath)
num_round = 0
prefix = "full-resnet-152"
layer = 'pool1_output'
is_pool = True
dim = 2048
reportTime = 500
max_img = 0
splite_num = 144
rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
npy_name = 'knn_splite.npy'

def removeFile(name):
    if os.path.exists(name):
        os.remove(name)

# 进行训练兼数据增强
def spliteAllOfPath(mod=None):
    if mod is None:
        mod = init_mxnet()  # 初始化mxnet
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    # 循环扫描大类文件夹
    tilesPerImage = splite_num
    for imgDir in subfolders:
        splits_resamples(facescrub_root=os.path.join(path, imgDir),
                            tilesPerImage=tilesPerImage,
                            mod=mod)

# 将大类文件夹内容进行分类


def splits_resamples(facescrub_root, tilesPerImage=360, mod=None):
    t_time = time.time()
    # 设置大类文件夹
    fold = facescrub_root

    # 获得所有色卡文件夹，并执行
    subfolders = [folder for folder in os.listdir(
        fold) if os.path.isdir(os.path.join(fold, folder))]
    temp_Process(subfolders, fold, mod)
    return fold

# 对色卡文件夹进行数据增强


def temp_Process(subfolders, fold, mod):
    # 设置增强多少张
    tilesPerImage = splite_num

    # 循环所有色卡
    for subfolder in subfolders:
        # 获取图片文件列表（以.JPG为结尾的文件）
        imgsfiles = [os.path.join(fold, subfolder, img)
                        for img in os.listdir(os.path.join(fold, subfolder)) if img.endswith('.JPG')]

        temp_list = []

        ''' 旧策略
        # 检测该文件夹色卡文件夹是否存在npy_name， 有则不训练该文件夹数据
        if not_double and os.path.exists(os.path.join(fold, subfolder, npy_name)):
            logging.info('Has %s' % os.path.join(
                fold, subfolder, npy_name))
            continue
        '''

        #''' 新策略
        # 检测该文件夹色卡文件夹是否存在npy_name， 有则不训练该文件夹数据
        if os.path.exists(os.path.join(fold, subfolder, npy_name)):
            try:
                if len(np.load(os.path.join(fold, subfolder, npy_name))) / tilesPerImage == len(imgsfiles):
                    continue
                else:
                    logging.info('Has new images insert, Then remove file' +
                                 os.path.join(fold, subfolder, npy_name))
                    removeFile(os.path.join(fold, subfolder, npy_name))
            except Exception:
                logging.error('npy_name: %s' %
                              os.path.join(fold, subfolder, npy_name))
                removeFile(os.path.join(fold, subfolder, npy_name))
        #'''
        temp_time = time.time()

        # 循环遍历所有的图片文件
        for imgfile in imgsfiles:
            if os.path.exists(imgfile) == False:
                logging.error('Bad Image: %s' % imgfile)
                continue
            try:
                # 打开图片
                im = Image.open(imgfile)
                # 获得原始图片大小
                w, h = im.size
                # 变换形状224， 224
                temp_im = cv2.resize(np.array(im), (224, 224))
                # 增加原始图片
                temp_list.append(
                    [getFeaturesOfSplite(temp_im, mod), subfolder, imgfile])

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
                    temp_list.append(
                        [getFeaturesOfSplite(im_cropped, mod), subfolder, imgfile])
            except Exception:
                logging.error('Bad Image: %s' % imgfile)
                continue
            #break;
        logging.info('Save %s SpeedTime: %0.2f' % (os.path.join(
            fold, subfolder, npy_name), (time.time() - temp_time)))

        #break;
        # 将处理后得到的特征数组存到色卡文件夹下的npy_name
        np.save(os.path.join(fold, subfolder, npy_name), temp_list)
    return 'Good Ending %s    %s' % (fold, subfolders)

# 将所有的npy_name读取到内存


def load_all_beOne(path):
    import time
    import random

    # 获得大类文件夹
    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]

    tt = time.time()
    main_imgArray = []  # 存取读取到的数组

    # 临时变量
    per = 0
    testNum = 0
    num = 0

    # 遍历所有大类文件夹
    for file in subfolders:

        # 获得大类文件夹完整路径
        filepath = os.path.join(path, file)
        logging.info('Start Merge Npy %s' % filepath)

        # 获得所有色卡文件夹
        subfolders2 = [folder for folder in os.listdir(
            filepath) if os.path.isdir(os.path.join(filepath, folder))]

        # 遍历所有色卡文件夹
        for file2 in subfolders2:
            try:
                t1 = time.time()

                # 获得色卡文件夹完整路径
                filepath2 = os.path.join(filepath, file2)

                # 判断是否存在npy_name文件
                if os.path.exists(os.path.join(filepath2, npy_name)):
                    imgArray = np.load(os.path.join(
                        filepath2, npy_name))
                else:
                    continue

                # 记录数组占用大小
                num += sys.getsizeof(imgArray)

                # 如果读取到的imgArray数组为0，则npy_name所在文件夹损坏
                if len(imgArray) == 0:
                    logging.error('Bad Npy: %s' %
                                os.path.join(filepath2, npy_name))
                    continue

                t_time = time.time()

                j = 0  # 记录当前循环得到了几个数组
                n = 0  # 记录已经获取了几个图片
                # 遍历所有数组
                for i in range(0, len(imgArray) / splite_num):
                    # 将获取的数组打乱顺序
                    tempList = imgArray[splite_num * i: splite_num * (i + 1)]
                    random.shuffle(tempList)
                    j = 0
                    for img in tempList:
                        if j <= max_img:
                            #print('Load %d ' % len(img[0]))
                            main_imgArray.append(img.copy())
                        j += 1
                        if j >= splite_num:
                            break
                    # 将多余的数组删除
                    del tempList
                # 将多余的数组删除
                del imgArray
                # 刷新，无用指针清理
                gc.collect()
                #break
            except EOFError:
                logging.error('Bad Folder ' + file + '_' + file2)
        logging.info('End Merge Npy: %d %f s' %
                      (len(main_imgArray), (time.time() - tt)))
    logging.info('Good Job')
    return main_imgArray



# 获取处理后的图片
# 参数 img： 图片地址
def getImage(img):
    img=cv2.imread(img, 1)
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
        return f
    else:
        return None


# 获取处理后的图片
# 参数 img： 图片
def getImageOfSplite(img):
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]
    return img


# 获取图片的特征
# 参数 img：图片地址
# 参数 f_mod： 模型
def getFeaturesOfSplite(img, f_mod=None):
    img = getImageOfSplite(img)
    f = f_mod.predict(img)
    f = np.ravel(f)
    return f

# 初始化hash
def init_hash():
    # 获得数组
    train=np.array(load_all_beOne(path))
    # 获取数组数量
    trainNum=len(train)
    # 获得默认参数
    p=falconn.get_default_parameters(trainNum, dim)
    t=falconn.LSHIndex(p)
    dataset=[np.ravel(x[0]).astype(np.float32) for x in train]
    dataset=np.array(dataset)
    # 生成hash
    logging.info('Start Hash setup')
    t.setup(dataset)
    if is_pool:
        q=t.construct_query_pool()
    else:
        q=t.construct_query_object()
    return (q, train)


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

# 初始化，不包括重载模型
def initWithoutMod():
    q, train=init_hash()
    return (q, train)

# 初始化
def init():
    mod=init_mxnet()
    q, train=init_hash()
    return (mod, q, train)

# 获得Cos距离
def getDistOfCos(f, t):
    up=np.sum(np.multiply(f, t))
    ff=np.sqrt(np.sum(np.multiply(f, f)))
    tt=np.sqrt(np.sum(np.multiply(t, t)))
    down=ff * tt
    return up / down

# 获取图片的特征，并进行hash计算
def getTest(img, mod, train, q, k = 20):
    fal=getFeatures(img, f_mod = mod)
    if fal is not None:
        tList=train[np.array(q.find_k_nearest_neighbors(fal, k))]
        return fal, tList
    else:
        return fal, None

# 客户端发来的请求进行处理(最好需要几个就设置多少k，不然影响速度)
# return 图片特征，最近的数组(k * max_img长度的数组, 相似度从近到远)
def make_work(conn, mod, q, train):
    try:
        while True:
            msg=conn.recv()
            logging.info(msg)
            opts, args=getopt.getopt(msg, 'f:zt:k:sp', ['help', 'train'])
            img=None
            k=20
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
                    k=int(value)
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
            fal, tList = getTest(img, mod, train, q, k =k * max_img)
            if is_pcl: # 是否为批处理
                is_Right= False
                if tList is None:
                    msg.append('Bad Img Path')
                    return msg
                else:
                    ti_time2= time.time()
                    ti= str(int(time.time()))
                    ti= ti + '_' + img
                    imgList= {}
                    nk= 400
                    for i in tList:
                        if is_save:
                            gailv= int(getDistOfCos(fal, i[0]) * 100)
                            if imgList.has_key(i[1]):
                                if imgList[i[1]][0] < gailv:
                                    imgList[i[1]]= [gailv, i[2]]
                            else:
                                imgList[i[1]]= [gailv, i[2]]
                            if len(imgList) >= 400:
                                break
                        if img_type != 0 and img_type == i[1]:
                            nk= len(imgList)
                            is_Right= True
                            break
                msg.append('Next')
                msg.append([imgList, nk])
                msg.append('Test Image Spend Time: %.2lf s' %
                           (time.time() - ti_time))
            else:
                msg.append('Next')
                msg.append([fal, tList])
                msg.append('Test Image Spend Time: %.2lf s' %
                           (time.time() - ti_time))
            return msg
    except EOFError:
        logging.info('Connection closed')
        return None


# 运行Server，一直监听接口
def run_server(address, authkey, mod, q, train):
    serv = Listener(address, authkey =authkey)
    while True:
        try:
            client= serv.accept()
            msg= make_work(client, mod, q, train)
            if msg == 'Close': # 关闭监听
                serv.close()
                return "Close"
            else:
                client.send(msg)
            if msg == 'train': # 开始训练，训练期间将关闭监听
                serv.close()
                spliteAllOfPath(mod);
                q, train= initWithoutMod()
                return "train"
        except Exception:
            traceback.print_exc()
    serv.close()


if __name__ == '__main__':
    opts, args= getopt.getopt(sys.argv[1:], 'f:x:m:p:k:')
    my_id= 99
    for op, value in opts:
        if op == '-f': # 设置大类所在文件夹路径
            path= value
        if op == '-k': # 设置npy_name的文件名
            npy_name= value
        if op == '-p': # 设置运行时的id，同于通信
            my_id= int(value)
        if op == '-m': # 设置最大图片数量
            max_img= int(value)
        elif op == '-x': # 设置mxnet/python所在路径
            mxnetpath= value
            sys.path.insert(0, mxnetpath)
    logging.info('Start Init')
    mod, q, train= init()
    logging.info('End Init')
    logging.info('Start Run')
    while run_server('./server%d.temp' % my_id, b'lee123456', mod, q, train) == "train":
        logging.info('Reset Server')
    logging.info('Stop Run')
