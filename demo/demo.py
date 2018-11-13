from PIL import Image, ImageDraw
from util import getImagePath
from colorsys import rgb_to_hsv
import util
import colorsys
import sys
import os
from colorthief import ColorThief

imageURl = getImagePath("legao/1.jpg")



# 获取颜色的范围
def get_color_range(image):
    w,h = image.size
    # 取4个点，算背景的平均值
    topLeft = image.getpixel((10,10)) 
    topRight = image.getpixel((w-10,10)) 
    bottomLeft = image.getpixel((10,h-10)) 
    bottomRight = image.getpixel((w-10,h-10))
    maxR = max(topLeft[0],topRight[0],bottomLeft[0],bottomRight[0])
    maxG = max(topLeft[1],topRight[1],bottomLeft[1],bottomRight[1])
    maxB = max(topLeft[2],topRight[2],bottomLeft[2],bottomRight[2])
    return ([0,0,0],[maxR+30,maxG+30,maxB+30])


# 切图
def cutFigure(url):
  # 指定宽高，位置中间
  h = 700
#   name, ext = os.path.splitext(url)
  im = Image.open(url)
  if im.mode != "RGBA":
      im = im.convert('RGBA')
  orgw,orgh = im.size
  # 中间位置
  x = 0
  y = orgh/2-h/2
  region = im.crop((x, y, orgw, y+h))
  # region.save(name + '.png')
  # return Image.open(name + '.png')
  return region


# 替换颜色
def chromaKey(im,minPixel,maxPixel):
    min_r, min_g, min_b = minPixel
    max_r, max_g, max_b = maxPixel
    width, height = im.size
    for x in range(width):
        for y in range(height):
           r,g,b,alpha = im.getpixel((x,y))
           if min_r <= r <= max_r and min_g <= g <= max_g and min_b <= b <= max_b:
               im.putpixel((x,y),(0,0,0,0))
               
    return im




if __name__ == '__main__':
    image = cutFigure(imageURl)
    minPixel,maxPixel=get_color_range(image)
    chromaImage = chromaKey(image,minPixel,maxPixel)
    name, ext = os.path.splitext(imageURl)
    color_thief = ColorThief(imageURl)
    dominant_color = color_thief.get_color(quality=1)
    newIm = Image.new("RGB", (640, 480), dominant_color)
    newIm.show()