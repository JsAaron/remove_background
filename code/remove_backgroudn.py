import argparse
import os
from os import path
import cv2
import numpy as np
import random
from multiprocessing import Pool

import tkinter as tk
from PIL import Image, ImageTk

# debug模式
isDebug = 0

# cut图缩放比例
resize = 1024

# 截图的Y坐标
# 左上角坐标(600 到 1800)像素点的高度
topY = 100
bottomY = 1900

# # 需要处理的图片目录
original_dir = os.path.dirname(os.path.realpath(__file__)) + "/original-4"

# # 处理后的目录位置
target_dir = os.path.dirname(os.path.realpath(__file__)) + "/target/2.png"

# 临时目录
temp_dir = os.path.dirname(os.path.realpath(__file__)) + "/temp"

#################
# 工具函数
#################

img = cv2.imread(target_dir)

rows, cols, channels = img.shape
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_blue = np.array([100, 20, 30])
upper_blue = np.array([190, 100, 100])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

erode = cv2.erode(mask, None, iterations=1)
dilate = cv2.dilate(erode, None, iterations=1)

#遍历替换
for i in range(rows):
    for j in range(cols):
        if dilate[i, j] == 255:
            # img[i, j] = (102, 127, 139)
            img[i, j] = (0, 255, 255)

green = np.uint8([[[36, 34, 39]]])
hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)

# print(hsv_green)

cv2.imshow('res', img)

cv2.waitKey(0)
cv2.destroyAllWindows()