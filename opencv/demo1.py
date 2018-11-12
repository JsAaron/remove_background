# == https://stackoverflow.com/questions/29313667/how-do-i-remove-the-background-from-this-kind-of-image

import cv2
import numpy as np
from matplotlib import pyplot as plt
from util import getImagePath

#== Parameters =======================================================================
BLUR = 21
# 阈值
CANNY_THRESH_1 = 20
CANNY_THRESH_2 = 100

MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0,0.0,1.0) # In BGR format


#== Processing =======================================================================

#-- Read image -----------------------------------------------------------------------
img = cv2.imread(getImagePath("8.jpg"))
# 灰度 降维度 256 =>1600万  
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# 边缘检测
edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
edges = cv2.dilate(edges, None)
# 有助于分离两个连接的对象，腐蚀消除了白色的噪音，缩小了我们的对象
edges = cv2.erode(edges, None)
# cv2.imshow('edges',edges)     

# 轮廓检测
contour_info = []

# 输入图像、层次类型和轮廓逼近方法
# RETR_LIST 检测的轮廓不建立等级关
# cv2.CHAIN_APPROX_NONE 存储所有的轮廓点

# return 修改后的图像、图像的轮廓以及它们的层次
(image, contours, hierarchy) = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
# 最大区域
max_contour = contour_info[0]

mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))


mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
mask_stack = np.dstack([mask]*3)


mask_stack  = mask_stack.astype('float32') / 255.0          # Use float matrices, 
img         = img.astype('float32') / 255.0 

masked = (mask_stack * img) + ((1-mask_stack) * MASK_COLOR)
masked = (masked * 255).astype('uint8')                     # Convert back to 8-bit 

cv2.imshow('img', masked)    
# (B, G, R)
# c_blue, c_green,c_red  = cv2.split(masked)
# merged = cv2.merge([c_blue, c_green,c_red])
# img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))
# plt.imshow(img_a)


# cv2.imshow("Red", masked)
# cv2.imshow("Green", c_green)
# cv2.imshow("Blue", c_blue)

# plt.imsave('D:/project/git/python/images/girl_3.png', img_a)

cv2.waitKey(0)



