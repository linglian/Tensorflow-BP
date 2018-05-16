import os
import tensorflow as tf

slim = tf.contrib.slim

def get_dataset(dataset_dir, num_samples, num_classes, labels_to_names_path=None, file_pattern='*.tfrecord'):
    file_pattern = os.path.join(dataset_dir, file_pattern)
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    items_to_descriptions = {
        'image': 'A color image of varying size.',
        'label': 'A single integer between 0 and ' + str(num_classes - 1),
    }
    labels_to_names = None
    if labels_to_names_path is not None:
        fd = open(labels_to_names_path)
        labels_to_names = {i : line.strip() for i, line in enumerate(fd)}
        fd.close()
    return slim.dataset.Dataset(
            data_sources=file_pattern,
            reader=tf.TFRecordReader,
            decoder=decoder,
            num_samples=num_samples,
            items_to_descriptions=items_to_descriptions,
            num_classes=num_classes,
            labels_to_names=labels_to_names)

"""
python train_image_classifier.py \
    --train_dir=/www/power_clothes/train_logs_02 \
    --dataset_dir=/www/power_clothes_train \
    --num_samples=7525907 \
    --num_classes=100000 \
    --model_name=inception_resnet_v2 \
    --checkpoint_path=/home/slu/tasks/inception_resnet_v2_2016_08_30.ckpt \
    --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \
    --learning_rate=0.01

"""
