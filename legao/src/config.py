import os
from os import path

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)

config = {
    # 一次分解的数量
    "interval": 30,
    # cut图缩放比例
    "resize": 1024,
    #图片切割的Y轴开始
    "start_y": 1100,
    #图片切割的Y轴结束
    "end_y": 2000,
    # 需要处理的图片目录
    "original_dir": os.path.join(root_path, 'original'),
    # 处理后的目录位置
    "target_dir": os.path.join(root_path, 'target'),
    #可选（图片错误保存目录(不填，默认放到original_dir内部)
    "error_dir": ""
}
