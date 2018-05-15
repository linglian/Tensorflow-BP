#coding=utf-8

from tools import create_label_list

# 将要存入的文件
list_file_path = 'list.txt'
file = None

"""将所需文件按照label格式存入到文件中

Raises:
    ValueError -- file_path为空时发出
"""

def make_label(file_path):
    if file_path is None:
        raise ValueError('Please Check File\'s Path Not Be None', file_path)
    
    id_val = create_label_list.getFloderOfFile(file_path)
    file.write('{} {}\n'.format(file_path, id_val))

if __name__ == '__main__':
    
    import sys
    import getopt

    opts, args = getopt.getopt(sys.argv[1:], 'f:l:t:')

    path = None
    check_file_format = 'jpg'
    
    for op, value in opts:
        if op == '-f':
           path = value 
        elif op == '-l':
            list_file_path = value
        elif op == '-t':
            check_file_format = value

    if path is None:
        raise ValueError('Must Enter Path Use -f', path)
        
    file = open(list_file_path, 'w')

    create_label_list.traverse_floder(path, make_label, check_file_format)
    
    file.close()