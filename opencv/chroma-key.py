from util import getImagePath
from scipy.misc import imread
import matplotlib.pyplot as plt

img = imread(getImagePath("atst.png"), mode='RGBA')

norm_factor = 255
red_ratio = img[:, :, 0] / norm_factor
green_ratio = img[:, :, 1] / norm_factor
blue_ratio = img[:, :, 2] / norm_factor

red_vs_green = (red_ratio - green_ratio) + .3
blue_vs_green = (blue_ratio - green_ratio) + .3


red_vs_green[red_vs_green < 0] = 0
blue_vs_green[blue_vs_green < 0] = 0

alpha = (red_vs_green + blue_vs_green) * 255
alpha[alpha > 50] = 255

img[:, :, 3] = alpha


fig, ax = plt.subplots(1, 7)
ax[0].imshow(red_ratio)
ax[1].imshow(green_ratio)
ax[2].imshow(blue_ratio)
ax[3].imshow(red_vs_green)
ax[4].imshow(blue_vs_green)
ax[5].imshow(alpha)
ax[6].imshow(img)
plt.show()