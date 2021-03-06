#coding=utf-8

from tools import create_label_list
import sys

models_path = '/Users/lol/DeepLearn/models'

def make_TFRecord(list_file_path, tfrecord_file_path=None):
    from datasets import dataset_utils
    from PIL import Image
    import tensorflow as tf
    import time
    
    # 默认情况下保存为列表名_tfrecord.tfrecords
    if tfrecord_file_path is None:
        tfrecord_file_path = create_label_list.getFileName(list_file_path).split('.')[0] + '_tfrecord.tfrecords'

    # 图片列表文件
    list_file = open(list_file_path)

    # 获取每行，并且分隔
    lines = [line.split() for line in list_file]

    list_file.close()

    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_file_path)

    with tf.Graph().as_default():
        # 读取格式信息
        decode_jpeg_data = tf.placeholder(dtype=tf.string)
        decode_jpeg = tf.image.decode_jpeg(decode_jpeg_data, channels=3)
        with tf.Session('') as sess:
            num = 0
            max_num = len(lines)
            for line in lines:
                # 处理路径名，去除空格的影响
                if len(line) != 2:
                    for index, i in enumerate(line):
                        if index != 0 and index < len(line) - 1:
                            line[0] = line[0] + ' ' + i
                    line[1] = line[len(line) - 1]
                # 判断是否为数字
                if line[1].isdigit():
                    image_data = tf.gfile.FastGFile(line[0], 'rb').read()
                    image = sess.run(decode_jpeg, feed_dict={decode_jpeg_data: image_data})
                    height, width = image.shape[0], image.shape[1]
                    example = dataset_utils.image_to_tfexample(
                        image_data, b'jpg', height, width, int(line[1]))
                    tfrecord_writer.write(example.SerializeToString())
                    num = num + 1
                    if num % 1000 == 0:
                        print('Finish %d/%d' % (num, max_num))
                        # 防止每个tfrecord太大，进行分割
                        tfrecord_writer.close()
                        tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_file_path.replace('.', '_%d.' % int(num / 1000)))
            tfrecord_writer.close()
        
    
if __name__ == '__main__':
    import sys
    import os
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'f:t:s:')

    list_file_path = 'list.txt'
    tfrecord_file_path = None

    for op, val in opts:
        if op == '-f':
            list_file_path = val
        elif op == '-t':
            tfrecord_file_path = val
        elif op == '-s':
            models_path = val
        
    sys.path.insert(0, models_path + '/research/slim/') #把后面的路径插入到系统路径中 idx=0
    print('%s is Loaded' % (models_path + '/research/slim/'))
    make_TFRecord(list_file_path, tfrecord_file_path)




