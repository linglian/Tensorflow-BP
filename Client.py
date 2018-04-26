from multiprocessing.connection import Client
import sys
import getopt
import traceback
sys.path.insert(0, '/home/lol/anaconda2/lib/python2.7/site-packages')
import falconn
import os
import cv2
import numpy as np
import time
from multiprocessing.connection import Listener
from PIL import Image
import gc

img = None
k = 20
img_type = 0
is_save = True
msg = []
filepath = None
my_id = 99
path = '/media/lee/data/image'
def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

def find_last(string,str):
    last_position=-1
    while True:
        position=string.find(str,last_position+1)
        if position==-1:
            return last_position
        last_position=position

def getDistOfCos(f, t):
    up = np.sum(np.multiply(f, t))
    ff = np.sqrt(np.sum(np.multiply(f, f)))
    tt = np.sqrt(np.sum(np.multiply(t, t)))
    down = ff * tt
    return up / down

def save_img(imgList, ti):
    if imgList is None:
        return
    imList = []
    for i in imgList:
        m = cv2.imread(i[1], 1)
        if m is not None:
            im = cv2.resize(m, (1024, 1024))
            imList.append([i[0], im, i[2]])
    number = 1
    for i in imList:
        m = i[1]
        if m is not None:
            im = Image.fromarray(m)
            if os.path.exists(path + '/%s/%d%%_%s.JPG' % (ti, i[2], i[0])):
                while os.path.exists(path + '/%s/%d%%_%s(%d).JPG' % (ti, i[2], i[0], number)):
                    number += 1
                im.save(path + '/%s/%d%%_%s(%d).JPG' % (ti, i[2], i[0], number))
                number += 1
            else:
                number = 1
                im.save(path + '/%s/%d%%_%s.JPG' % (ti, i[2], i[0]))

def save_img2(imgList, ti):
    if imgList is None:
        return
    imList = []
    for i in imgList:
        m = cv2.imread(i[1], 1)
        if m is not None:
            im = cv2.resize(m, (1024, 1024))
            imList.append([i[0], im, i[2]])
    number = 1
    for i in imList:
        m = i[1]
        if m is not None:
            im = Image.fromarray(m)
            if os.path.exists(path + '/%s/%d%%_%s.JPG' % (ti, i[2], i[0])):
                while os.path.exists(path + '/%s/%d%%_%s(%d).JPG' % (ti, i[2], i[0], number)):
                    number += 1
                im.save(path + '/%s/%d%%_%s(%d).JPG' % (ti, i[2], i[0], number))
                number += 1
            else:
                number = 1
                im.save(path + '/%s/%d%%_%s.JPG' % (ti, i[2], i[0]))


def getSeKa(str):
    strLen = len(str)
    for i in range(0, strLen):
        if str[i].isdigit() == False:
            return str[0 : i]

def pichuliOfOnce(filepath, path):
    subfolders = [folder for folder in os.listdir(
        filepath) if folder.endswith('.JPG') or folder.endswith('.jpg')]
    print(subfolders)
    for img in subfolders:
        img_type = getSeKa(img)
        print img_type, img
        img2 = os.path.join(filepath, img)
        c = Client('./server%d.temp' % my_id, authkey=b'lee123456')
        c.send(['-f', img2, '-k', '400', '-t', img_type, '-p'])
        ar = c.recv()
        is_Ok = False
        for i in ar:
            if i == 'Next':
                is_Ok = True
            elif is_Ok:
                is_Ok = False
                print('###### Start Save Image %s' % img2)
                ti_time2 = time.time()
                ti = str(int(time.time()))
                ti = img
                imgList = i[0]
                nk = i[1]
                if is_save:
                    ti = ti + '_%d' % nk
                    checkFold(os.path.join(path))
                    if os.path.exists(os.path.join(path, ti)):
                        print('Has Same Folder %s' % os.path.join(path, ti))
                        break
                    checkFold(os.path.join(path, ti))
                    n = 0
                    m = cv2.imread(img2, 1)
                    if m is not None:
                        im = cv2.resize(m, (1024, 1024))
                        im = Image.fromarray(im)
                        im.save(path + '/%s/GT_%s.JPG' % (ti, img2[find_last(img2, '/') + 1: find_last(img2, '.')]))
                    else:
                        print('Bad Image: %s' % i[2])
                if is_save:
                    t_imgList = []
                    for i in imgList:
                        t_imgList.append([i, imgList[i][1], imgList[i][0]])
                    jobs = []
                    print('Need Save %d Images' % len(t_imgList))
                    cutLen = int(len(imgList) / 4)
                    if cutLen != 0:
                        for i in range(0, 4 - 1):
                            save_img(t_imgList[i * cutLen : (i + 1) * cutLen], ti)
                        save_img(t_imgList[3 * cutLen : ], ti)
                    else:
                        save_img(t_imgList, ti)
                if is_save:
                    print('Save Image Spend Time: %.2lf s' % (time.time() - ti_time2))
                    print('Save Image: %s/%s/' % (path, ti))
            else:
                print i

def pichuli(filepath, path):
    subfolders = [folder for folder in os.listdir(
        filepath) if os.path.isdir(os.path.join(filepath, folder))]
    print(subfolders)
    for path2 in subfolders:
        path3 = os.path.join(filepath, path2)
        subfolders2 = [folder for folder in os.listdir(
        path3) if folder.endswith('.JPG') or folder.endswith('.jpg')]
        img_type = path2
        print(subfolders2)
        for img in subfolders2:
            img2 = os.path.join(path3, img)
            c = Client('./server%d.temp' % my_id, authkey=b'lee123456')
            c.send(['-f', img2, '-k', '400', '-t', img_type, '-p'])
            ar = c.recv()
            is_Ok = False
            for i in ar:
                if i == 'Next':
                    is_Ok = True
                elif is_Ok:
                    is_Ok = False
                    print('###### Start Save Image %s' % img2)
                    ti_time2 = time.time()
                    ti = str(int(time.time()))
                    ti = img_type + '_' + img
                    imgList = i[0]
                    nk = i[1]
                    if is_save:
                        ti = ti + '_%d' % nk
                        checkFold(os.path.join(path))
                        checkFold(os.path.join(path, ti))
                        n = 0
                        m = cv2.imread(img2, 1)
                        if m is not None:
                            im = cv2.resize(m, (1024, 1024))
                            im = Image.fromarray(im)
                            im.save(path + '/%s/GT_%s.JPG' % (ti, img2[find_last(img2, '/') + 1: find_last(img2, '.')]))
                        else:
                            print('Bad Image: %s' % i[2])
                    if is_save:
                        t_imgList = []
                        for i in imgList:
                            t_imgList.append([i, imgList[i][1], imgList[i][0]])
                        jobs = []
                        cutLen = int(len(imgList) / 4)
                        if cutLen != 0:
                            for i in range(0, 4 - 1):
                                save_img(t_imgList[i * cutLen : (i + 1) * cutLen], ti)
                            save_img(t_imgList[3 * cutLen : ], ti)
                        else:
                            save_img(t_imgList, ti)
                    if is_save:
                        print('Save Image Spend Time: %.2lf s' % (time.time() - ti_time2))
                        print('Save Image: %s/%s/' % (path, ti))
                else:
                    print i

def danchuli(fal, tList):
    is_Right = False
    if tList is None:
        msg.append('Bad Img Path')
        return msg
    else:
        ti_time2 = time.time()
        ti = str(int(time.time()))
        imgList = {}
        temp_click = 0
        near_k = k
        for i in tList:
            print i[1], img_type
            if is_save:
                gailv = int(getDistOfCos(fal, i[0]) * 100)
                if imgList.has_key(i[1]):
                    if imgList[i[1]][0] < gailv:
                        imgList[i[1]] = [gailv, i[2]]
                else:
                    imgList[i[1]] = [gailv, i[2]]
                if len(imgList) >= k:
                    break
            if img_type != 0 and str(img_type) == i[1]:
                print('Find K = %d' % len(imgList))
                near_k = len(imgList)
                is_Right = True
                break
        if is_save:
            ti = ti + '_%d' % near_k
            checkFold(os.path.join(path))
            checkFold(os.path.join(path, ti))
            n = 0
            m = cv2.imread(img, 1)
            if m is not None:
                im = cv2.resize(m, (1024, 1024))
                im = Image.fromarray(im)
                print('Save Image %s' % path + '/%s/GT_%s.JPG' % (ti, img[find_last(img, '/') + 1: find_last(img, '.')]))
                im.save(path + '/%s/GT_%s.JPG' % (ti, img[find_last(img, '/') + 1: find_last(img, '.')]))
            else:
                print('Bad Image: %s' % i[2])
        if is_save:
            t_imgList = []
            for i in imgList:
                t_imgList.append([i, imgList[i][1], imgList[i][0]])
            jobs = []
            cutLen = int(len(imgList) / 4)
            print('Start Save %d Images' % len(imgList))
            if cutLen != 0:
                for i in range(0, 4 - 1):
                    save_img(t_imgList[i * cutLen : (i + 1) * cutLen], ti)
                save_img(t_imgList[3 * cutLen : ], ti)
            else:
                save_img(t_imgList, ti)
        if is_save:
            print('Save Image Spend Time: %.2lf s' % (time.time() - ti_time2))
            print('Save Image: %s/%s/' % (path, ti))
        if is_Right:
            print('Find Right Image')
        elif img_type != 0:
            print('Find Bad Image')

if __name__ == '__main__':
    filepath = None
    is_l = False
    is_m = False
    try:
        arg = sys.argv[1:]
        opts, args = getopt.getopt(sys.argv[1:], 'f:zt:k:sm:p:l:o:', ['help', 'train'])
        for op, value in opts:
            if op == '-f':
                img = value
            if op == '-o':
                path = value
                print('############ %s' % path)
                arg.remove(op)
                arg.remove(value)
            elif op == '-k':
                k = int(value)
            elif op == '-p':
                my_id = int(value)
            elif op == '-s':
                is_save = False
            elif op == '-m':
                is_m = True
                filepath = value
            elif op == '-l':
                is_l =True
                filepath = value
            elif op == '-t':
                img_type = value
        if filepath is None:
            c = Client('./server%d.temp' % my_id, authkey=b'lee123456')
            c.send(arg)
            ar = c.recv()
            is_Ok = False
            for i in ar:
                if i == 'Next':
                    is_Ok = True
                elif is_Ok:
                    is_Ok = False
                    print('Start Save Image')
                    danchuli(i[0], i[1])
                else:
                    print i
        elif is_m:
            pichuli(filepath, path)
        elif is_l:
            pichuliOfOnce(filepath, path)
    except EOFError:
        print 'Connection closed, Please Reset Server.'
