#coding=utf-8

"""获取文件名
path: 完整的文件路径

Returns:
    str -- 返回文件名
"""

def getFileName(path):
    try:
        return path.split('/')[-1];
    except Exception as msg:
        raise ValueError('Bad Path', path)

"""获取文件的父路径
path: 完整的文件路径
index: 返回第几级父文件夹

Returns:
    str -- 返回父文件夹文字
"""


def getFloderOfFile(path, index=1):
    return path.split('/')[-1 - index]

"""遍历所有文件夹，并且对每个符合格式的文件运行函数
base_folder: 根路径
dothing_func: 执行函数
check_file_format: 文件格式
"""

def traverse_floder(base_folder, dothing_func, check_file_format='jpg', is_log=True):
    import os

    if is_log:
        print('Check %s' % base_folder)

    # 获取base_folder目录下的所有文件
    floders_list = [folder for folder in
        os.listdir(base_folder) if
        os.path.isdir(os.path.join(base_folder, folder))]
    
    # 获取base_folder目录下的check_file_format后缀的文件
    files_list = [file for file in
        os.listdir(base_folder) if
        check_file_format is None or file.endswith(check_file_format)]

    # 执行给定函数操作
    for file in files_list:
        dothing_func(os.path.join(base_folder, file))

    # 递归遍历所有文件夹
    for floder in floders_list:
        traverse_floder(os.path.join(base_folder, floder), dothing_func, check_file_format)