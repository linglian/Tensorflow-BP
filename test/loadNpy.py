#coding=utf-8
import numpy as np
import os
import sys
from PIL import Image
path = '/home/lol/dl/Image'

i = 0
def checkFile(name):
    if os.path.exists(name):
        os.remove(name)
    os.mknod(name)

def checkFold(name):
    if not os.path.exists(name):
        os.mkdir(name)

def getDistOfCos(f, t):
    up = np.sum(np.multiply(f, t))
    ff = np.sqrt(np.sum(np.multiply(f, f)))
    tt = np.sqrt(np.sum(np.multiply(t, t)))
    down = ff * tt
    return up / down

checkFold(path + '/lst')

checkFile(path + '/lst/test.lst')

out = open(path + '/lst/test.lst', 'w')

checkFile(path + '/lst/train.lst')

out2 = open(path + '/lst/train.lst', 'w')

r = 0.02

main = np.load(os.path.join(path, 'knn_train.npy'))

test = main[:int(len(main) * r)]
train = main[int(len(main) * r):]

print 'Test: %d Train: %d' %(len(test), len(train))
ks = {}
subfolders = [folder for folder in os.listdir(
    path) if os.path.isdir(os.path.join(path, folder))]
for file in subfolders:
    print 'Start %s' % file
    path2 = os.path.join(path, file)
    subfolders2 = [folder for folder in os.listdir(
        path2) if os.path.isdir(os.path.join(path2, folder))]
    for file2 in subfolders2:
        if ks.has_key(file2):
            print '######### Error Has Same: %s(%s) %s' % (file, file2, ks[file2])
        ks[file2] = file
    print 'End %s' % file

key = {}

n = 0

num = len(test) + len(train)

t_n = 0

t = 0

for i in test:
    img = Image.fromarray(i[0])
    img.save(path + '/lst/%s_%s_%s' % (ks[i[1]], i[1], i[2]))
    if not key.has_key(ks[i[1]]):
        key[ks[i[1]]] = n
        n += 1
    out.write('%d\t%d\t%s\n' % (t, key[ks[i[1]]], 'lst/%s_%s_%s' % (ks[i[1]], i[1], i[2])))
    t_n += 1
    if t_n % 500 == 1:
        print 'Finish %d/%d' % (t_n, num)
    t += 1

t = 0
for i in train:
    img = Image.fromarray(i[0])
    img.save(path + '/lst/%s_%s_%s' % (ks[i[1]], i[1], i[2]))
    if not key.has_key(ks[i[1]]):
        key[ks[i[1]]] = n
        n += 1
    if t % 5 == 1:
        out2.write('%d\t%d\t%s\n' % (t, key[ks[i[1]]] + 1, 'lst/%s_%s_%s' % (ks[i[1]], i[1], i[2])))
    else:
        out2.write('%d\t%d\t%s\n' % (t, key[ks[i[1]]], 'lst/%s_%s_%s' % (ks[i[1]], i[1], i[2])))
    t_n += 1
    if t_n % 500 == 1:
        print 'Finish %d/%d' % (t_n, num)
    t += 1

out.close()
out2.close()