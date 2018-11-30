import cv2
import numpy as np

img=np.zeros((3,3),dtype=np.uint8)
img=cv2.cvtColor(img,cv2.COLOR_BAYER_BG2BGR)
print(img)