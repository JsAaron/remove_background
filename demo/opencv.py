import os
import cv2
from os import path   
from PIL import Image
import  numpy as np

d = path.dirname(__file__) 
rootPath = os.path.dirname(d)



# 模式

# 1 只切图
# 2 切图后，去掉背景色
mode=2

# 指定切图的高度
# 默认按照中间位置切
cutHeight=700

# 需要处理的图片目录
original_dir = os.path.join(rootPath,"images/original/")

# 处理后的目录位置
new_dir = os.path.join(rootPath,"images/new/")



img = cv2.imread(os.path.join(original_dir,"2.jpg"))
cv2.imshow("image", img)
# rows,cols,channels = img.shape

# hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# lower_blue=np.array([120,0,0])
# upper_blue=np.array([220,200,100])

# mask = cv2.inRange(hsv, lower_blue, upper_blue)
# cv2.imshow('Mask', mask)