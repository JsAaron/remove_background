from PIL import Image, ImageDraw
from util import getImagePath
from colorsys import rgb_to_hsv
import util
import colorsys
import sys
import os
import cv2
from colorthief import ColorThief


def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    maxc = max(r, g, b)
    minc = min(r, g, b)
    v = maxc
    if minc == maxc:
        return 0.0, 0.0, v
    s = (maxc-minc) / maxc
    rc = (maxc-r) / (maxc-minc)
    gc = (maxc-g) / (maxc-minc)
    bc = (maxc-b) / (maxc-minc)
    if r == maxc:
        h = bc-gc
    elif g == maxc:
        h = 2.0+rc-bc
    else:
        h = 4.0+gc-rc
    h = (h/6.0) % 1.0
    return h, s, v


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

    mixR = min(topLeft[0],topRight[0],bottomLeft[0],bottomRight[0])
    mixG = min(topLeft[1],topRight[1],bottomLeft[1],bottomRight[1])
    mixB = min(topLeft[2],topRight[2],bottomLeft[2],bottomRight[2])
    return ([mixR,mixG,mixB],[maxR,maxG,maxB])


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
def chromaKey(image,min_range,man_range):
    member=[]
    pix = image.load()
    width, height = image.size
    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            h_ratio, s_ratio, v_ratio = rgb_to_hsv(r , g , b)
            h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)
            min_h, min_s, min_v = min_range
            max_h, max_s, max_v = man_range
            # print(h, s, v)

            if min_h <= h <= max_h and min_v <= v <= max_v:
                pix[x, y] = (0, 0, 0, 0)
            # else:
            #    member.append([h,s,v])


    # print(member)
    # image.show()
    return image


def start(i):
    imageURl =getImagePath("legao/"+ str(i) + ".jpg")

    image = cutFigure(imageURl)
    name, ext = os.path.splitext(imageURl)  
    image.save(name + '.cut.png')
    
    color_thief = ColorThief(name + '.cut.png')
    r,b,g = color_thief.get_color(quality=1)

    h_ratio, s_ratio, v_ratio = rgb_to_hsv(r,b,g)
    h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255) 
    print(imageURl,h, s, v)
    min_range = (120, 0, 0)
    max_range = (220, 200, 100)

    newImage=chromaKey(image,min_range,max_range)
    newImage.save(name + '.mask.png')



if __name__ == '__main__':
    for i in range(7):
        if i>0:
            start(i)

