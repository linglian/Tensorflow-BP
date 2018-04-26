import numpy as np
import cv2
import sys
sys.path.insert(0, '/home/lol/dl/mxnet/python')
import mxnet as mx
import argparse
import time
import os
import random
folder = '/home/lol/dl/dlbp/image/examples'
test_ratio = 0.67
import multiprocessing

#folder = './imagenet/tiny-imagenet-200/test/images/'
def handleFolder(GUPid,tasks):
    #synset = [l.strip() for l in open(args.synset).readlines()]
    prefix = "full-resnet-152"
    num_round = 0	
    model = mx.model.FeedForward.load( prefix, num_round, ctx=mx.gpu(GUPid),numpy_batch_size=1)
    internals = model.symbol.get_internals()
    fea_symbol = internals["pool1_output"]	  
    feature_extractor = mx.model.FeedForward( ctx=mx.gpu(GUPid), symbol=fea_symbol, numpy_batch_size=1, \
            arg_params=model.arg_params, aux_params=model.aux_params, allow_extra_params=True)
			
    #subfolders = [ fold for fold in os.listdir(folder)]			
    k = 0	
    for subfolder in tasks:
      workspace_folder = os.path.join(folder,subfolder)
      print "extract label####",subfolder,"---GPU: ",GUPid," process: ",k,"/",len(tasks)
      i = 0
      k +=1	  
      feature_array = []
      for filename in os.listdir(workspace_folder):
        if '.jpg'	in filename or '.JPEG' in filename:
          i +=1		
          m = cv2.imread(os.path.join(workspace_folder,filename),1)	  
          img = cv2.cvtColor(m, cv2.COLOR_BGR2RGB)
          img = cv2.resize(img, (224, 224))  # resize to 224*224 to fit model
          img = np.swapaxes(img, 0, 2)
          img = np.swapaxes(img, 1, 2)  # change to (c, h,w) order
          img = img[np.newaxis, :]  # extend to (n, c, h, w)
          f = feature_extractor.predict(img)
          f = np.ravel(f)
          #print f.shape		  
          feature_array.append((f[0],subfolder,filename))
      random.shuffle(feature_array)
      #print len(feature_array)	  
      np.save((os.path.join(workspace_folder,"test.npy")),feature_array[:int(i*test_ratio)])
      np.save((os.path.join(workspace_folder,"train.npy")),feature_array[int(i*(test_ratio)):])
	  


if __name__ == '__main__':
    print 'Start'
    t1 =time.time()
    #root_fold = '/home/slu/Downloads/IJB-A/files/'
    #init_ijba_folders(root_fold)	
    # GPUvector=[0,1,2,3,0,1,2,3]
    GPUvector=[0]	
    nGPU = len(GPUvector)	
    #print "GPU",nGPU,"in our training",len(CSV_IJBA)

    jobs   =[]
    subfolders = [ fold for fold in os.listdir(folder)]		
    fa=np.array_split(subfolders,nGPU)
    for i in range(nGPU):
       print "##################GPU ",str(i)," take",len(fa[i]),"folders"	
    z = 0	   
    for i in GPUvector:	
           mp_kwargs = dict(
            GUPid=i,
            tasks= fa[z]			
            )
           p = multiprocessing.Process(target=handleFolder, kwargs=mp_kwargs)
           jobs.append(p)
           p.start()
           z +=1
           print "##################go to next index,start process for:",z," process"		   
    for p in jobs:
           print "end of process"	
           p.join()
    print "whole process time:",time.time()-t1
