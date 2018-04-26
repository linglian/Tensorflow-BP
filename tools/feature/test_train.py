#coding=utf-8
import numpy as np
import os
import math
import multiprocessing
import sys
import cv2
import logging
sys.path.insert(0, '/home/lol/dl/mxnet/python')
import mxnet as mx

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='test.log',
                filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def EuclideanDistance(form, to):
    return cv2.norm(form, to)


def getTrain(filePath):
    train = np.load(os.path.join(filePath, 'train.npy'))
    return train

def get_image(path):
    # download and show the image
    m = cv2.imread(path, 1)
    img = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
    img = img[np.newaxis, :]  # extend to (n, c, h, w)
    return img

def getFeatures(img, f_mod):
    img = get_image(img)
    f = f_mod.predict(img)
    f = np.ravel(f)
    return f

def init(GPUid=0):
    prefix = "full-resnet-152"
    num_round = 0	
    model = mx.model.FeedForward.load( prefix, num_round, ctx=mx.gpu(GPUid),numpy_batch_size=1)
    internals = model.symbol.get_internals()
    fea_symbol = internals["pool1_output"]	  
    feature_extractor = mx.model.FeedForward( ctx=mx.gpu(GPUid), symbol=fea_symbol, numpy_batch_size=1, \
            arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)

    return feature_extractor


def getClass(img, train):
    import time
    mod = init()
    f = getFeatures(img, mod)
    t1 = time.time()
    logging.info('Start: %s' % img)
    minD = [999, '', ''];
    for k in train:
        for l in k:
            temp = EuclideanDistance(f, l[0])
            if minD[0] > temp:
                minD[0] = temp
                minD[1] = l[1]
                minD[2] = l[2]
            #if trainNow % 500 == 0:
            #    logging.info('Finish Train %d/%d' % (trainNow, trainNum))
    logging.info('End: %f %s %s (Speed time: %f)' % (minD[0], minD[1], minD[2], time.time() - t1))
    return minD

def run():
    import time
    t = time.time()
    filePath = '/home/lol/dl/Image'
    print "Start load npy"
    test = np.load(os.path.join(filePath, 'test.npy'))
    train = np.load(os.path.join(filePath, 'train.npy'))
    print "End load npy"
    testNum = 0
    for i in test:
        for j in i:
            testNum += 1
    trainNum = 0
    for i in train:
        for j in i:
            trainNum += 1
    logging.info('TestNumber: %d TrainNumber: %d' % (testNum, trainNum))
    testNow = 0
    good = 0
    bad = 0
    # print test
    for i in test:
        for j in i:
            t1 = time.time()
            logging.info('Start: %s %s' % (j[1], j[2]))
            minD = [999, '', ''];
            trainNow = 0
            for k in train:
                for l in k:
                    temp = EuclideanDistance(j[0], l[0])
                    if minD[0] > temp:
                        minD[0] = temp
                        minD[1] = l[1]
                        minD[2] = l[2]
                    trainNow += 1
                    #if trainNow % 500 == 0:
                    #    logging.info('Finish Train %d/%d' % (trainNow, trainNum))
            logging.info('End: %f %s %s (Speed time: %f)' % (minD[0], minD[1], minD[2], time.time() - t1))
            testNow += 1
            if j[1] == minD[1]:
                good += 1
            else:
                bad += 1
                logging.warning('Bad is Coming: %s %s != %s %s' %(j[1], [2], minD[1], minD[2]))
            logging.info('Good/Bad %d/%d' % (good, bad))
            logging.info('Finish Test %d/%d' % (testNow, testNum))
    logging.info('Finish All, Speed time: %f' % (time.time() - t))

if __name__=='__main__':
    # print getClass('/home/lol/dl/Image/people/1448/DSC01352.JPG', getTrain('/home/lol/dl/Image'))
    run()