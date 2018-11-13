from PIL import Image, ImageDraw
from util import getImagePath
from colorsys import rgb_to_hsv
import util
import colorsys
import sys
import os

imageURl = getImagePath("legao/6.jpg")



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


# 获取主要颜色
def get_dominant_color(image):
    count,r,g,b = image.getcolors()
    print(count)
    # max_score = 0.0001
    # dominant_color = None
    # for count,(r,g,b) in image.getcolors(image.size[0]*image.size[1]):
    #     # 转为HSV标准
    #     saturation = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)[1]
    #     y = min(abs(r*2104+g*4130+b*802+4096+131072)>>13,235)
    #     y = (y-16.0)/(235-16)
    #     #忽略高亮色
    #     if y > 0.9:
    #         continue
    #     score = (saturation+0.1)*count
    #     if score > max_score:
    #         max_score = score
    #         dominant_color = (r,g,b)
    # return dominant_color

if __name__ == '__main__':
    image = cutFigure(imageURl)
    minPixel,maxPixel=get_color_range(image)
    chromaImage = chromaKey(image,minPixel,maxPixel)
    name, ext = os.path.splitext(imageURl)
    # chromaImage.save(name + '.png')
    get_dominant_color(chromaImage)

    # newIm = Image.new("RGB", (640, 480), mainColor)
    # chromaImage.show()