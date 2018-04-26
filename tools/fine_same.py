#coding=utf-8

import os
import numpy as np
from PIL import Image
import sys
sys.path.append('/home/lol/anaconda2/lib/python2.7/site-packages')
import imagehash as ih
import cv2
import logging
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename='same.log',
                filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

if __name__ == '__main__':
    import getopt
    path = '/home/lol/dl/Image'
    opts, args = getopt.getopt(sys.argv[1:], 'f:')
    for op, value in opts:
        if op == '-f':
            path = value

    ks = {}
    ik = {}

    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    for file in subfolders:
        print('Start %s' % file)
        path2 = os.path.join(path, file)
        subfolders2 = [folder for folder in os.listdir(
            path2) if os.path.isdir(os.path.join(path2, folder))]
        for file2 in subfolders2:
            print('Start %s' % file2)
            path3 = os.path.join(path2, file2)
            if ks.has_key(file2):
                logging.error('######### Error Has Same: %s(%s) %s' % (file, file2, ks[file2]))
            ks[file2] = file
        print('End %s' % file)



    subfolders = [folder for folder in os.listdir(
        path) if os.path.isdir(os.path.join(path, folder))]
    for file in subfolders:
        print('Start %s' % file)
        path2 = os.path.join(path, file)
        subfolders2 = [folder for folder in os.listdir(
            path2) if os.path.isdir(os.path.join(path2, folder))]
        for file2 in subfolders2:
            print('Start %s' % file2)
            path3 = os.path.join(path2, file2)
            subfolders3 = [folder for folder in os.listdir(
                path3) if os.path.join(path3, folder).endswith('.JPG')]
            for file3 in subfolders3:
                path4 = os.path.join(path3, file3)
                m = cv2.imread(path4, 1)
                if m is not None:
                    img = Image.fromarray(m)
                    ihash = ih.average_hash(img)
                    if ik.has_key(ihash) and ik[ihash][0] != file:
                        logging.error('######### Error Has Same Image: %s == %s' % (path4, ik[ihash]))
                    ik[ihash] = [file, file2, file3]
                else:
                    logging.error('Bad Image: %s' % path4)
        print('End %s' % file)
    a = np.array(ik)
    a.save('my_hash.npy')