import cv2
import argparse
import numpy as np

from util import getImagePath
imageURl = getImagePath("1.jpg")


#== Parameters =======================================================================
BLUR = 19
CANNY_THRESH_1 = 25
CANNY_THRESH_2 = 255
MASK_DILATE_ITER = 5
MASK_ERODE_ITER = 13
MASK_COLOR = (0.0,0.0,1.0)

#-- Read image -----------------------------------------------------------------------
img = cv2.imread(imageURl)
gray = cv2.GaussianBlur(img, (5, 5), 1)
gray = cv2.cvtColor(gray,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)

gray = 255 - cv2.threshold(gray, 0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

edges = cv2.dilate(gray, kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21)) , iterations = 1)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
edges = cv2.GaussianBlur(edges, (1, 1), 0)

#-- Find contours in edges, sort by area ---------------------------------------------
contour_info = []
_, contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]

#-- Create empty mask, draw filled polygon on it corresponding to largest contour ----
# Mask is black, polygon is white
mask = np.zeros(edges.shape)
cv2.fillConvexPoly(mask, max_contour[0], (255))

#-- Smooth mask, then blur it --------------------------------------------------------
#mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
#mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)

#-- Blend masked img into MASK_COLOR background --------------------------------------
img         = img.astype('float32') / 255.0                 #  for easy blending
#cv2.imwrite('mask.png', mask)

c_red, c_green, c_blue = cv2.split(img)
img_a = cv2.merge((c_red, c_green, c_blue, mask.astype('float32') / 255.0))

# cv2.imwrite(args.file_out, img_a*255)

cv2.imshow('img',  img_a*255)                                   # Display
cv2.waitKey()