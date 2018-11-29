import os
import cv2
import numpy as np
from os import path   
from PIL import Image


d = path.dirname(__file__) 
rootPath = os.path.dirname(d)

# 模式

# 1 只切图
# 2 切图后，去掉背景色
mode=2

# 需要处理的图片目录
original_dir = os.path.join(rootPath,"images/original/")

# 处理后的目录位置
new_dir = os.path.join(rootPath,"images/new/")

path = "d:/project/git/remove_background/images/new/1.png"
path2 = "d:/project/git/remove_background/images/bg.png"
path3 = "d:/project/git/remove_background/images/original/6.jpg"

# im = Image.open(path3)
# orgw,orgh = im.size
# # 指定切图的高度
# # 默认按照中间位置切
# cutHeight=700
# x = 0
# y = orgh/2-cutHeight/2
# region = im.crop((x, y, orgw, y+cutHeight))
# ext,name=os.path.split(path3)  
# newPath = new_dir+name
# region.save(newPath)
# cutImage = cv2.imread(newPath)
# cv2.namedWindow("binary",cv2.WINDOW_NORMAL)
# cv2.imshow("binary", cutImage)  

image = cv2.imread(path3)
h,w = image.shape[:2]
# 左上角坐标(700:1500)，右下角坐标(0:w）的图像。
img = image[700:1500, 0:w]
#转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 高斯去噪
blurred = cv2.GaussianBlur(gray, (9, 9),0)

gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

blurred = cv2.GaussianBlur(gradient, (9, 9),0)
(_, thresh) = cv2.threshold(blurred, 110, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',closed)


# #高斯滤波
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
# #自适应二值化方法
# blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
# # blurred=cv2.copyMakeBorder(blurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))

# #canny边缘检测
# edged = cv2.Canny(blurred, 10, 100)

# cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)


# # 确保至少有一个轮廓被找到
# if len(cnts) > 0:
#     # 将轮廓按大小降序排序
#     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
#     # 对排序后的轮廓循环处理
#     for c in cnts:
#         # 获取近似的轮廓
#         peri = cv2.arcLength(c, True)
#         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#         # 如果近似轮廓有四个顶点，那么就认为找到了答题卡
#         if len(approx) == 4:
#             docCnt = approx
#             break



# #求二值图
# ret, threshImage = cv2.threshold(gray,127,255,cv2.THRESH_BINARY) 
# # 找到轮廓
# image ,contours,hierarchy = cv2.findContours(threshImage,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# newImg = cv2.imread(path2)
# newImg = cv2.resize(newImg, (w,h))

# # 画图
# cv2.drawContours(newImg, contours, -1, (0,0,0), 3)


# cv2.imwrite('messigray.png',newImg)





cv2.waitKey()

