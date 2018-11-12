#!/usr/bin/python3
# 2018.01.20 20:58:12 CST
# 2018.01.20 21:24:29 CST
import cv2
import numpy as np
from util import getImagePath

## (1) Read
img = cv2.imread(getImagePath("1.jpg"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## (2) Threshold
th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

## (3) Find the min-area contour
_, cnts, _ = cv2.findContours(threshed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea)
for cnt in cnts:
    if cv2.contourArea(cnt) > 100:
        break

## (4) Create mask and do bitwise-op
mask = np.zeros(img.shape[:2],np.uint8)
cv2.drawContours(mask, [cnt],-1, 255, -1)
dst = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('img', dst)                                   # Display
cv2.waitKey()

## Save it
# cv2.imwrite("dst.png", dst)