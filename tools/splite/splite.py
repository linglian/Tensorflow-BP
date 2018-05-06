#coding=utf-8

from PIL import Image
import cv2
import numpy as np
import os
import sys
import getopt

import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='splite.log',
                    filemode='w')
      
# 数据增强区
tilesPerImage = 180
rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]
rotate45degree = [45, 135, 270]
thresholdGLOABL = 0.42

mainFold = '/www/clothes/'
toFold = '/www/增强后的数据/'
"""创建文件夹（如果文件夹不存在的话）
"""

def check_fold(name):
    if not os.path.exists(name):
        os.mkdir(name)

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
        exit_file = imgfile.replace(mainFold, toFold)
        exit_file = exit_file[0: exit_file.find('.')] + '.jpg'
        if os.path.exists(exit_file) is True:
            return
        temp_list = []
        # 打开图片
        im = cv2.imread(imgfile)
        im = Image.fromarray(im)
        # 获得原始图片大小
        w, h = im.size
        # 变换形状224， 224
        temp_im = cv2.resize(np.array(im), (224, 224))
        # 增加原始图片
        temp_imgfile = imgfile.replace(mainFold, toFold);

        t_im = Image.fromarray(temp_im)
        t_im.save(temp_imgfile[0: temp_imgfile.find('.')] + '.jpg', "JPEG")
        # 删除图片上下尺子的影响
        im = im.crop((0, int(h * 0.1), w, int(h * 0.9)))

        dx = dy = 224
        # 将图片增强tilesPerImage份
        for i in range(1, tilesPerImage + 1):
            newname = imgfile.replace('.', '_{:03d}.'.format(i))
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
            # temp_list.append(im_cropped)
            # 将处理后的图片存起来
            newname = newname.replace(mainFold, toFold)
            t_im = Image.fromarray(im_cropped)
            t_im.save(newname[0: newname.find('.')] + '.jpg', "JPEG")
    except Exception as msg:
        logging.error('Bad Image: %s B %s ' % (imgfile, msg))
        print('Bad Image: %s B %s ' % (imgfile, msg))
        return None
 
def start_splite(path, filePath, toPath):
    logging.info('Create Fold %s ' % os.path.join(toPath, filePath));
    print('Create Fold %s ' % os.path.join(toPath, filePath));
    
    check_fold(os.path.join(toPath, filePath));
    folders = [folder for folder in os.listdir(
        os.path.join(path, filePath)) if os.path.isdir(os.path.join(path, filePath, folder))]

    folders2 = [folder for folder in os.listdir(
        os.path.join(path, filePath)) if folder.endswith('.webp')]

    tempFolders = [folder for folder in os.listdir(
        os.path.join(toPath, filePath)) if folder.endswith('.jpg')]

    for folder in folders2:
        # print('Splite Image %s ' % os.path.join(toPath, filePath, folder));
        splite_img(os.path.join(path, filePath, folder))

    for folder in folders:
        start_splite(os.path.join(path, filePath), folder, os.path.join(toPath, filePath))


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'f:s:')
    for op, value in opts:
        # 设置根目录路径
        if op == '-f':
            mainFold = value
        elif op == '-s':
            toFold = value
    start_splite(mainFold, '', toFold);