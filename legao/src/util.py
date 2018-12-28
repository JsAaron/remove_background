import os
from os import path


# 得到指定目录
def get_specified_dir(data_dir, dir_name):
    return os.path.join(data_dir, dir_name)


# 创建目录
def create_dir(d):
    if not os.path.exists(d):
        # print("creating dir: %s" % d)
        os.mkdir(d)
        if not os.path.exists(d):
            raise Exception("Failed to create dir: %s" % d)


# 创建指定目录
def create_specified_dir(dir_path, dir_name):
    if dir_path and dir_name:
        new_dir = get_specified_dir(dir_path, dir_name)
        create_dir(new_dir)
        return new_dir
    else:
        create_dir(dir_path)
        return dir_path
