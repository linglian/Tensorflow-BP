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

def getCrop(image_filepath, is_rotate=False, is_center_crop=True, img_size=(224, 224), mode=10):
    import cv2
    from PIL import Image
    def get_image_rotate(img, angle=0):
        if isinstance(img, Image.Image) is False:
            raise ArithmeticError('img Must be PIL.Image.Image', img)
        w, h = img.size
        rotate35_img_true = img.rotate(angle, expand=True)
        rotate35_img_false = img.rotate(angle, expand=False)
        width, height = rotate35_img_false.size
        width2, height2 = rotate35_img_true.size
        new_width = width - width2 + width
        new_height = height - height2 + height
        left = int((width - new_width) / 2)
        top = int((height - new_height) / 2)
        right = left + new_width
        bottom = top + new_height
        result_img = rotate35_img_false.crop((left, top, right, bottom))
        return result_img
    if mode != 5 and mode != 10:
        raise ArithmeticError('Mode only has 5 or 10', mode)
    try:
        img = Image.fromarray(cv2.imread(image_filepath))
    except Exception:
        raise ValueError('image_filepath is Bad or None', image_filepath)
    w, h = img.size
    # 进行Central-Crop
    if is_center_crop:
        img = img.crop((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)))
        w, h = img.size
    img_array = []
    if is_rotate:
        rotate_array = [-30, -20, -10, 10, 20, 30]
        for angle in rotate_array:
            img_array.append(get_image_rotate(img, angle=angle))
    # 获取Crop-5
    img_array.append( img.crop( (0, 0,
                                               int(w * 0.5), int(h * 0.5))))
    img_array.append( img.crop( (int(w * 0.5), 0,
                                               w, int(h * 0.5))))
    img_array.append( img.crop( (0, int(h * 0.5),
                                               int(w * 0.5), h)))
    img_array.append( img.crop( (int(w * 0.5), int(h * 0.5),
                                               w, h)))
    img_array.append( img.crop( (int(w * 0.25), int(h * 0.25),
                                               int(w * 0.75), int(h * 0.75))))
    # 进而获取Crop-10
    if mode == 10:
        for i in range(5):
            img_array.append(img_array[i].transpose(Image.FLIP_LEFT_RIGHT))
    result_array = []
    for im in img_array:
        result_array.append(cv2.resize(np.array(im), img_size))
    return result_array

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
        img_array = getCrop(imgfile)
        # 将图片增强tilesPerImage份
        for idx, im in enumerate(img_array):
            newname = imgfile.replace('.', '_{:03d}.'.format(idx))
            newname = newname.replace(mainFold, toFold)
            t_im = Image.fromarray(im)
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

    print(folders2)

    for folder in folders2:
        # print('Splite Image %s ' % os.path.join(toPath, filePath, folder));
        splite_img(os.path.join(path, filePath, folder))

    for folder in folders:
        start_splite(os.path.join(path, filePath), folder, os.path.join(toPath, filePath))


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], 'f:s:t:')
    for op, value in opts:
        # 设置根目录路径
        if op == '-f':
            mainFold = value
        elif op == '-s':
            toFold = value
        elif op == '-t':
            tilesPerImage = int(value)
    start_splite(mainFold, '', toFold);