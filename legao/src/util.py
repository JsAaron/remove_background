import os
from os import path
import cv2
from config import config

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


# 清理目录
def clear_dir(dir_name):
    files = os.listdir(dir_name)
    for file_name in files:
        path_to_remove = os.path.join(dir_name, file_name)
        if os.path.isdir(path_to_remove):
            clear_dir(path_to_remove)
            os.rmdir(path_to_remove)
        else:
            os.remove(path_to_remove)


#  target目录
def asscess_dir(target_dir):
    if os.path.exists(target_dir):
        clear_dir(target_dir)
    else:
        create_specified_dir(target_dir,"")



# 获取所有图片名
def get_image_names_from_dir(data_dir,type):
    _files = os.listdir(data_dir)
    image_files = []
    for f in _files:
        if f.lower().endswith('.' + type):
            image_files.append(os.path.join(data_dir, f))
    return image_files


#获取图片
def clipImage(path):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    setH = config["bottomY"]
    setH = h
    img = img[config["topY"] : setH, 0:w]
    return img


#获取文件名
def get_file_name(img_file):
    return os.path.split(img_file)[1]



def get_data_dir(img_file):
    return os.path.split(os.path.split(img_file)[0])[0]



def get_png_name_for_jpeg(img_file):
    img_file = os.path.split(img_file)[1]
    img_file = os.path.splitext(img_file)[0]
    return img_file + '.png'
