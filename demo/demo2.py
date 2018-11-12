"""
Removes greenscreen from an image.
Usage: python greenscreen_remove.py image.jpg
"""

from PIL import Image
import sys
import os
from util import getImagePath
import colorsys


file_path = getImagePath("1.jpg")

def rgb_to_hsv(r, g, b):
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


GREEN_RANGE_MIN_HSV = (0, 0, 0)
GREEN_RANGE_MAX_HSV = (255, 150, 150)

def main(file_path):
    # Load image and convert it to RGBA, so it contains alpha channel
    name, ext = os.path.splitext(file_path)
    im = Image.open(file_path)
    if im.mode != "RGB":
        im = im.convert('RGB')

    # im.thumbnail((00, 200))

    # # Go through all pixels and turn each 'green' pixel to transparent
    pix = im.load()
    width, height = im.size


    for x in range(width):
        for y in range(height):
            
            # 获取每个像素点的rgb
            # r, g, b = im.getpixel((x,y))
            # print(im.getpixel((x,y)))
            # im.putpixel((x,y),((69, 70, 69)))
            # print(im.getpixel((x,y)))

            # h_ratio, s_ratio, v_ratio = rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            # h, s, v = (h_ratio * 360, s_ratio * 255, v_ratio * 255)
  


            # min_h, min_s, min_v = GREEN_RANGE_MIN_HSV
            # max_h, max_s, max_v = GREEN_RANGE_MAX_HSV
            # if min_h <= h <= max_h and min_s <= s <= max_s and min_v <= v <= max_v:
            #     pix[x, y] = (0, 0, 0, 0)
  
    
    im.save(name + '.png')


def get_accent_color(path):
    im = Image.open(path)
    if im.mode != "RGB":
        im = im.convert('RGB')

    # delta_h = 0.3
    # avg_h = sum(t[0] for t in[colorsys.rgb_to_hsv(*im.getpixel((x,y))) for x in range(im.size[0]) for y in range(im.size[1])])/(im.size[0]*im.size[1])
    # beyond = filter(lambda x: abs(colorsys.rgb_to_hsv(*x)[0]- avg_h)>delta_h ,[im.getpixel((x,y)) for x in range(im.size[0]) for y in range(im.size[1])])
    # if len(beyond):
    #     r = sum(e[0] for e in beyond)/len(beyond)
    #     g = sum(e[1] for e in beyond)/len(beyond)
    #     b = sum(e[2] for e in beyond)/len(beyond)
    #     for i in range(im.size[0]/2):
    #         for j in range(im.size[1]/10):
    #             im.putpixel((i,j), (r,g,b))
    #     im.save('res'+path)
    #     return r, g, b
    # return None


def get_dominant_color(image):
    image = image.convert('RGBA')
    image.thumbnail((200, 200))
    max_score = None
    dominant_color = None
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue
        # 参数取值都是在[0, 1]范围内的浮点数。所以传入RGB参数的时候还需要额外做一个除以255的操作。
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)
        # 忽略高亮色
        if y > 0.9:
            continue
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = (r, g, b)

    return dominant_color

if __name__ == '__main__':
    # get_dominant_color(Image.open(file_path)) 

    # get_accent_color(file_path)
    main(file_path)