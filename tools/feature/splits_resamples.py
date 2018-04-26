#coding=utf-8
#import sklearn
import numpy as np
#import matplotlib.pyplot as plt
#import skimage
#from skimage import transform as tf
import shutil
# this file is expected to be in {caffe_root}/examples/siamese
facescrub_root = '/home/lol/dl/dlbp/image'
thresholdGLOABL = 0.42
import sys
import os
import Image
import random
import math
subfolders = [folder for folder in os.listdir(
    facescrub_root) if os.path.isdir(os.path.join(facescrub_root, folder))]
print subfolders

dict = {}
for subfolder in subfolders:
    imgsfiles = [os.path.join(facescrub_root, subfolder, img)
                 for img in os.listdir(os.path.join(facescrub_root, subfolder))]
    for img in imgsfiles:
        dict[img] = subfolder
print dict

def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

tilesPerImage = 360
fold = facescrub_root

dx = dy = 224
fold_idx = 1
rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

rotate45degree = [45, 135, 270]
subfolders = [folder for folder in os.listdir(
    fold) if os.path.isdir(os.path.join(fold, folder))]
# print subfolders


def im_crotate_image_square(im, deg):
    im2 = im.rotate(deg, expand=1)
    im = im.rotate(deg, expand=0)

    width, height = im.size
    assert (width == height)

    rads = math.radians(deg)
    new_width = width - (im2.size[0] - width)

    left = top = int((width - new_width) / 2)
    right = bottom = int((width + new_width) / 2)

    return im.crop((left, top, right, bottom))



checkFold(fold + '/examples')

for subfolder in subfolders:
    imgsfiles = [os.path.join(fold, subfolder, img)
                 for img in os.listdir(os.path.join(fold, subfolder))]
    checkFold(fold + '/examples/' + subfolder)
    for imgfile in imgsfiles:

        im = Image.open(imgfile)
        w, h = im.size
        im = im.crop((0, 0, w, int(h * 0.9)))
        #dx = 224
        for i in range(1, tilesPerImage + 1):
            newname = imgfile.replace('.', '_{:03d}.'.format(i))
            # print newname
            w, h = im.size
            # print("Cropping",w,h)
            if i < 100 and w > 300:
                dx = dy = 224
            if 100 < i < 200 and w > 500:
                dx = dy = 320
            if 200 < i < 300 and w > 800:
                dx = dy = 640
            x = random.randint(0, w - dx - 1)
            y = random.randint(0, h - dy - 1)
            #print("Cropping {}: {},{} -> {},{}".format(file, x,y, x+dx, y+dy))
            im_cropped = im.crop((x, y, x + dx, y + dy))
            if i % 2 == 0:  # roate 180,90
                im_cropped = im_cropped.transpose(random.choice(rotateAction))
            if i % 2 == 0 and i > 300:
                roate_drgree = random.choice(rotate45degree)
                im_cropped = im_crotate_image_square(im_cropped, roate_drgree)
            newname = newname.replace(fold, fold + '/examples');
            im_cropped.save(newname)
        # don't remove startImg
        # os.remove(imgfile)
