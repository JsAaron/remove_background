from PIL import Image, ImageDraw
from util import getImagePath
from colorsys import rgb_to_hsv
import util
import colorsys
import sys
import os
imageURl = getImagePath("8.jpg")


def get_min_max(value,min,max):
    r = value[0]
    b = value[1]
    g = value[2]
    a = value[3]


# 获取颜色的范围
def get_color_range(image):
    w,h = image.size
    # 取4个点，算背景的平均值
    topLeft = image.getpixel((10,10)) 
    topRight = image.getpixel((w-10,10)) 
    bottomLeft = image.getpixel((10,h-10)) 
    bottomRight = image.getpixel((w-10,h-10)) 
    # (45, 46, 48, 255) (47, 48, 50, 255) (54, 56, 55, 255) (54, 56, 55, 255)
    min=[45,46,48]
    max=[54,56,55]
    return (min,max)


# 切图
def cutFigure(url):
  # 指定宽高，位置中间
  w = 1000 
  h = 700
  name, ext = os.path.splitext(url)
  im = Image.open(url)
  if im.mode != "RGBA":
      im = im.convert('RGBA')
  orgw,orgh = im.size
  # 中间位置
  x = orgw/2-w/2
  y = orgh/2-h/2
  
  region = im.crop((x, y, x+w, y+h))
  # region.save(name + '.png')
  # return Image.open(name + '.png')
  return region


# 替换颜色
def chromaKey(im,min,max):
    min_hsv = util.rgb2hsv(min[0], min[1], min[2])
    max_hsv = util.rgb2hsv(max[0], max[1], max[2])

    pix = im.load()
    width, height = im.size
    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            h, s, v = util.rgb2hsv(r, g, b)
            # h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)

            min_h, min_s, min_v = min_hsv
            max_h, max_s, max_v = max_hsv

            if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
                pix[x, y] = (0, 0, 0, 0)
   
     im.show()


# ChromaKey(imageURl)
if __name__ == '__main__':
    image = cutFigure(imageURl)
    min,max=get_color_range(image)
    chromaKey(image,min,max)
