import os
import cv2
from os import path   
from PIL import Image
import  numpy as np

d = path.dirname(__file__) 
rootPath = os.path.dirname(d)

# 需要处理的图片目录
original_dir = os.path.join(rootPath,"images/original/")

# 处理后的目录位置
new_dir = os.path.join(rootPath,"images/new/")

path = "d:/project/git/remove_background/images/new/1.png"
path2 = "d:/project/git/remove_background/images/bg.png"




 #获取图片
def get_image(path):      
    img = cv2.imread(path3)
    h,w = img.shape[:2]
    # 左上角坐标(700:1500)，右下角坐标(0:w）的图像。
    img = img[700:1500, 0:w]
    # 灰度化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img,gray


# 高斯去噪(去除图像中的噪点)
"""
高斯模糊本质上是低通滤波器:
输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和

高斯矩阵的尺寸和标准差:
(9, 9)表示高斯矩阵的长与宽，标准差取0时OpenCV会根据高斯矩阵的尺寸自己计算。
高斯矩阵的尺寸越大，标准差越大，处理过的图像模糊程度越大。
"""
def Gaussian_Blur(gray):   
    blurred = cv2.GaussianBlur(gray, (9, 9),0)
    return blurred


# 计算梯度
"""
索比尔算子来计算x、y方向梯度
"""
def Sobel_gradient(blurred): 
  gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
  gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)
  gradient = cv2.subtract(gradX, gradY)
  gradient = cv2.convertScaleAbs(gradient)
  return gradX, gradY, gradient


#设定阈值
"""
    cv2.threshold(src,thresh,maxval,type[,dst])->retval,dst (二元值的灰度图)
    src：  一般输入灰度图
	thresh:阈值，
	maxval:在二元阈值THRESH_BINARY和
	       逆二元阈值THRESH_BINARY_INV中使用的最大值 
	type:  使用的阈值类型
    返回值  retval其实就是阈值 
"""
def Thresh_and_blur(gradient): 
    blurred = cv2.GaussianBlur(gradient, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    return thresh


"""
建立一个椭圆核函数
执行图像形态学
"""
def image_morphology(thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    return closed


def findcnts_and_box_point(closed):
    # 这里opencv3返回的是三个参数
    (_, cnts, _) = cv2.findContours(closed.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # 计算最大轮廓的旋转包围盒
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
    return box


def drawcnts_and_cut(original_img, box): #目标图像裁剪
    # 因为这个函数有极强的破坏性，所有需要在img.copy()上画
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



path3 = "d:/project/git/remove_background/images/original/6.jpg"
save_path = r'./cat_save.png'

original_img, gray = get_image(path3)
blurred = Gaussian_Blur(gray)
gradX, gradY, gradient = Sobel_gradient(blurred)
thresh = Thresh_and_blur(gradient)
closed = image_morphology(thresh)
box = findcnts_and_box_point(closed)
draw_img, crop_img = drawcnts_and_cut(original_img,box)



cv2.imwrite(save_path, crop_img)


# cv2.namedWindow('GaussianBlur', cv2.WINDOW_NORMAL)
# cv2.imshow('GaussianBlur', closed)
cv2.waitKey(20171219)





# #进行二值化
# retval, threshImg = cv2.threshold(grayImg, 110, 255, cv2.THRESH_BINARY)
# adaptiveThresh = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 49, 0)

# # 寻找轮廓
# # 轮廓存储模式为简单模式cv2.CHAIN_APPROX_SIMPLE，如果设置为 cv2.CHAIN_APPROX_NONE，所有的边界点都会被存储
# # cv2.RETR_TREE：提取所有轮廓，并重新建立轮廓层次结构
# image, contours, hierarchy = cv2.findContours(adaptiveThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# # 画出轮廓  #此时是将轮廓绘制到了原始图像
# img = cv2.drawContours(cv2.imread(path2), contours, -1, (0,255,0), 3) 

# cv2.namedWindow('Thresh', cv2.WINDOW_NORMAL)
# cv2.imshow('Thresh',img)  #显示原始图像

# cv2.waitKey(0)
# cv2.destroyAllWindows()

