import cv2
import numpy as np
from util import getImagePath

#Create a black image
img = cv2.imread(getImagePath("3.jpg"))

print(img)