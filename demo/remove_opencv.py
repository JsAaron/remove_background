
import cv2
import numpy as np
import os
from os import path   
from PIL import Image


 #获取图片
def get_image(path):      
    img = cv2.imread(path)
    h,w = img.shape[:2]
    # 左上角坐标(700:1500)，右下角坐标(0:w）的图像。
    img = img[500:1800, 0:w]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img,gray


# 高斯去噪(去除图像中的噪点)
def Gaussian_Blur(gray):    
    blurred = cv2.GaussianBlur(gray, (9, 9),0)
    return blurred


# 梯度
def Sobel_gradient(blurred): 
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    return gradX, gradY, gradient


# 阈值
def Thresh_and_blur(gradient):  #设定阈值
    blurred = cv2.GaussianBlur(gradient, (9, 9),0)
    (_, thresh) = cv2.threshold(gradient, 10, 255, cv2.THRESH_BINARY)
    return thresh


# 建立一个椭圆核函数
def image_morphology(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    return closed


# cv2.RETR_EXTERNAL,             #表示只检测外轮廓
# cv2.RETR_CCOMP,                #建立两个等级的轮廓,上一层是边界
# cv2.RETR_LIST,                 #检测的轮廓不建立等级关系
# cv2.RETR_TREE,                 #建立一个等级树结构的轮廓
# cv2.CHAIN_APPROX_NONE,         #存储所有的轮廓点，相邻的两个点的像素位置差不超过1
def findcnts_and_box_point(closed,original_img):
    # 这里opencv3返回的是三个参数
    (img, contours, hierarchy) = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h, w, _ = original_img.shape
    c_max = 0
    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cx<w/3 and area>100000:
           c_max=i
           print(i)

    # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(contours[c_max])
    box = np.int0(cv2.boxPoints(rect))
    return box


# 裁剪
def drawcnts_and_cut(original_img, box):
    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0, 0, 255), 3)
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    crop_img = original_img[y1:y1+hight, x1:x1+width]
    return draw_img, crop_img


def walk(originalPath,newPath):
    original_img, gray = get_image(originalPath)
    blurred = Gaussian_Blur(gray)
    gradX, gradY, gradient = Sobel_gradient(blurred)
    thresh = Thresh_and_blur(gradient)
    closed = image_morphology(thresh)
    box = findcnts_and_box_point(closed,original_img)
    draw_img, crop_img = drawcnts_and_cut(original_img,box)
    cv2.imwrite(newPath, draw_img)


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    nameList = []
    pathDir =  os.listdir(filepath)
    for name in pathDir:
        nameList.append(name)     
    return nameList


def cycleFigure(original_dir,new_dir):
    nameList = eachFile(original_dir)
    image_list = []
    for item in nameList:
        originalPath = os.path.join(original_dir,item)
        name,ext=os.path.splitext(item)  
        newPath = os.path.join(new_dir,name) + ".png"
        walk(originalPath,newPath)


#根目录
rootPath = os.path.dirname(path.dirname(__file__))

# ==================执行入口=======================

# 需要处理的图片目录
original_dir = os.path.join(rootPath,"images/original/")

# 处理后的目录位置
new_dir = os.path.join(rootPath,"images/new/")

# 开始处理
cycleFigure(original_dir,new_dir)