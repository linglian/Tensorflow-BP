import os,urllib
import sys

def download(url,prefix=''):
    filename = prefix+url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.urlretrieve(url,filename)
 
if __name__ == '__main__':
    import sys
    import getopt
    from collections import Counter
    import random
    perfix = 'resnet-152'
    opts, args = getopt.getopt(sys.argv[1:], 'f:p:')
    for op, value in opts:
        if op == '-f':
            perfix = value
        if op == '-p':
            path = value

path='http://data.mxnet.io/models/imagenet-11k/resnet-152/'
download(path + perfix + '-symbol.json','full-')
download(path + perfix + '-0000.params','full-')
# download(path +'synset.txt','full-')
 
# with open('full-synset.txt','r') as file:
#     synsets = [l.rstrip() for l in file]

# # print synsets
# sym, arg_params,aux_params = mx.model.load_checkpoint('full-resnet-152',0)

# mod= mx.mod.Module(symbol=sym, context=mx.gpu())
# mod.bind(for_training=False, data_shapes=[('data', (1,3,224,224))])
# mod.set_params(arg_params, aux_params)

# import cv2
# import numpy as np
# from collections import namedtuple
# Batch= namedtuple('Batch', ['data'])

# print Batch

# def predict(filename, mod, synsets):
#     img = cv2.imread(filename)
#     if img is None:
#         return None
#     img = cv2.resize(img, (224,224))
#     img = np.swapaxes(img,0,2)
#     img = np.swapaxes(img,1,2)
#     img = img[np.newaxis, :] 
    
#     mod.forward(Batch([mx.nd.array(img)]))
#     prob = mod.get_outputs()[0].asnumpy()
#     prob = np.squeeze(prob)
 
#     a = np.argsort(prob)[::-1]    
#     for i in a[0:5]:
#         print('probability=%f, class=%s'%(prob[i], synsets[i]))

# predict('/home/lol/dl/image/strawberry/strawberry.jpg', mod, synsets)