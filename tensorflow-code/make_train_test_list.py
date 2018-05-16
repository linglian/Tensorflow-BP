#coding=utf-8

import random

train_file_path = 'train_list.txt'
test_file_path = 'test_list.txt'
list_file_path = 'list.txt'

train_radio = 0.95

if __name__ == '__main__':
    
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'r:t:l:b:')

    for op, value in opts:
        if op == '-r':
            train_file_path = value
        elif op == '-t':
            test_file_path = value
        elif op == '-l':
            list_file_path = value
        elif op == '-b':
            train_radio = float(value)
    
    train_file = open(train_file_path, 'w')
    test_file = open(test_file_path, 'w')
    list_file = open(list_file_path, 'r')

    files_list = list_file.readlines()

    list_file.close()

    # 将数据打乱
    random.shuffle(files_list)

    # 获取训练数据数量
    train_size = int(len(files_list) * train_radio)

    # 写入训练数据
    for file in files_list[:train_size]:
        train_file.write(file)
    
    train_file.close()

    # 写入验证数据
    for file in files_list[train_size:]:
        test_file.write(file)
    
    test_file.close()
        

