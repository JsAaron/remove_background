import os
from os import path

cur_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(cur_path)

# 全局接口
# 可以在入口处被覆盖
config = {
    # 是否采集数据(尺寸，颜色) 默认采集 True,  不采集 False
    "collectData": True,
    # 分割图片 all：是全部采集(页面所有零件) / full：只采集完整的(去掉边界可能不完整的)
    "splitMode": "full",
    # 一次分解的数量
    "interval": 30,
    # 内部处理切图的缩放比例
    "resize": 1000,
    #图片切割的Y轴开始
    "start_y": 1100,
    #图片切割的Y轴结束
    "end_y": 2000,
    # 需要处理的图片目录
    "original_dir": os.path.join(root_path, 'test'),
    # 处理后的目录位置
    "target_dir": os.path.join(root_path, 'target'),
    #可选（图片错误保存目录(如不填，默认放到original_dir内部)
    "error_dir": ""
}
