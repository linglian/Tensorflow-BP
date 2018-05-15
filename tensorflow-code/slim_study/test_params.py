#coding=utf-8

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# 字符串类型
tf.app.flags.DEFINE_string(
    'name',
    'lingling',
    'The name of the computer users'
)

# bool类型只写--is_handsome的话，默认为True，所以我们这里可以设置不写的话，默认为False
# 如果想设置为False，只能写--is_handsome=false
tf.app.flags.DEFINE_boolean(
    'is_handsome',
    False,
    'Which one is Handsome'
)

# 数字类型
tf.app.flags.DEFINE_integer(
    'age',
    18,
    'Which one\'s age'
)

# 浮点数类型
tf.app.flags.DEFINE_float(
    'weight',
    98.2,
    'Which one\'s Weight'
)

def main(argv):
    print '{} {} Handsome'.format(FLAGS.name, 'is' if FLAGS.is_handsome else 'not is')

if __name__ == '__main__':
    import sys
    tf.app.run(main=main, argv=sys.argv[1:])