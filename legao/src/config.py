import os
from os import path

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)
config = {
    # debug模式
    "isDebug": 0,
    # cut图缩放比例
    "resize": 1024,
    # 截图的Y坐标
    # 左上角坐标(600 - 1800)像素点的高度
    "topY": 1100,
    "bottomY": 2000,
    "root_path":root_path,
    # # 需要处理的图片目录
    "original_dir": os.path.join(root_path,'original'),
    # # 处理后的目录位置
    "target_dir": os.path.join(root_path,'target'),
    # 临时目录
    "temp_dir": os.path.join( os.path.join(root_path,'original'), "temp")
}
