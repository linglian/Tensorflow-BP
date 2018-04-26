
# imgPath: facescrub_root/examples/

def splits_resamples(facescrub_root):
    #import sklearn
    import numpy as np
    #import matplotlib.pyplot as plt
    #import skimage
    #from skimage import transform as tf
    import shutil
    # this file is expected to be in {caffe_root}/examples/siamese
    # facescrub_root = '/home/lol/dl/image'
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

    tilesPerImage = 360
    fold = facescrub_root

    dx = dy = 224
    fold_idx = 1
    rotateAction = [Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM,
                    Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270]

    rotate45degree = [45, 135, 270]
    subfolders = [folder for folder in os.listdir(
        fold) if os.path.isdir(os.path.join(fold, folder))]
    print 'files: %s' % subfolders

    checkFold(fold + '/examples')
    for subfolder in subfolders:
        imgsfiles = [os.path.join(fold, subfolder, img)
                     for img in os.listdir(os.path.join(fold, subfolder))]
        print 'Start Directory: %s' % subfolder
        for imgfile in imgsfiles:
            checkFold(fold + '/examples/' + subfolder)
            print 'Start Image: %s' % imgfile
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
                    im_cropped = im_cropped.transpose(
                        random.choice(rotateAction))
                if i % 2 == 0 and i > 300:
                    roate_drgree = random.choice(rotate45degree)
                    im_cropped = im_crotate_image_square(
                        im_cropped, roate_drgree)
                newname = newname.replace(fold, fold + '/examples')
                im_cropped.save(newname)
            # don't remove startImg
            # os.remove(imgfile)


def batch(folder):
    import numpy as np
    import cv2
    import sys
    sys.path.insert(0, '/home/lol/dl/mxnet/python')
    import mxnet as mx
    import argparse
    import time
    import os
    import random
    # folder = '/home/lol/dl/image/examples'
    test_ratio = 0.98
    import multiprocessing
    #folder = './imagenet/tiny-imagenet-200/test/images/'
    with open('full-synset.txt','r') as file:
        synsets = [l.rstrip() for l in file]
    # print synsets
    def handleFolder(GUPid, tasks):
        #synset = [l.strip() for l in open(args.synset).readlines()]
        prefix = "full-resnet-152"
        num_round = 0
        model = mx.model.FeedForward.load(
            prefix, num_round, ctx=mx.gpu(GUPid), numpy_batch_size=1)
        internals = model.symbol.get_internals()
        fea_symbol = internals["fc1_output"]
        feature_extractor = mx.model.FeedForward(ctx=mx.gpu(GUPid), symbol=fea_symbol, numpy_batch_size=1,
                                                arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
        
        #subfolders = [ fold for fold in os.listdir(folder)]
        k = 0
        for subfolder in tasks:
            workspace_folder = os.path.join(folder, subfolder)
            print "extract label####", subfolder, "---GPU: ", GUPid, " process: ", k, "/", len(tasks)
            i = 0
            k += 1
            feature_array = []
            print 'Start: %s' % workspace_folder
            for filename in os.listdir(workspace_folder):
                print 'Start : %s' % filename
                if '.jpg' in filename or '.JPEG' in filename:
                    i += 1
                    m = cv2.imread(os.path.join(workspace_folder, filename), 1)
                    img = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
                    # resize to 224*224 to fit model
                    img = cv2.resize(img, (224, 224))
                    img = np.swapaxes(img, 0, 2)
                    img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
                    img = img[np.newaxis, :]  # extend to (n, c, h, w)
                    f = feature_extractor.predict(img)
                    print 'label: %s #### %d' % (synsets[f[0].argmax()], f[0].argmax())
                    feature_array.append((f[0], subfolder, filename))
                print 'End : %s' % filename
            random.shuffle(feature_array)
            # print len(feature_array)
            print 'Save: %s/%s' % (workspace_folder, "test.npy")
            np.save((os.path.join(workspace_folder, "test.npy")),
                    feature_array[:int(i * test_ratio)])
            print 'Save: %s/%s' % (workspace_folder, "train.npy")
            np.save((os.path.join(workspace_folder, "train.npy")),
                    feature_array[int(i * (test_ratio)):])

    print 'Start'
    t1 = time.time()
    #root_fold = '/home/slu/Downloads/IJB-A/files/'
    # init_ijba_folders(root_fold)
    # GPUvector=[0,1,2,3,0,1,2,3]
    GPUvector = [0]
    nGPU = len(GPUvector)
    # print "GPU",nGPU,"in our training",len(CSV_IJBA)

    jobs = []
    subfolders = [fold for fold in os.listdir(folder)]
    fa = np.array_split(subfolders, nGPU)
    for i in range(nGPU):
        print "##################GPU ", str(i), " take", len(fa[i]), "folders"
    z = 0
    for i in GPUvector:
        mp_kwargs = dict(
            GUPid=i,
            tasks=fa[z]
        )
        p = multiprocessing.Process(target=handleFolder, kwargs=mp_kwargs)
        jobs.append(p)
        p.start()
        z += 1
        print "##################go to next index,start process for:", z, " process"
    for p in jobs:
        print "end of process"
        p.join()
    print "whole process time:", time.time() - t1


def run(path):
    print 'Start SplitsResamples'
    splits_resamples(path)
    print 'End SplitsResamples'
    print 'Start Batch'
    batch(path + '/examples')
    print 'End Batch'


run('/home/lol/dl/dlbp/image')
