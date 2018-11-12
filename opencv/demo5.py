
import cv2
import numpy as np
from util import getImagePath
imageURl = getImagePath("maze.png")

image = cv2.imread(imageURl)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)


arr = np.arange(24).reshape(2, 3, 4)

print(arr)