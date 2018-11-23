import os
from PIL import Image
from colorthief import ColorThief

# 模式
# 1 只切图
# 2 切图后，去掉背景色
mode=2

# 指定切图的高度
# 默认按照中间位置切
cutHeight=700

# 需要处理的图片目录
original_dir = "d:/project/git/remove_background/images/original/"

# 处理后的目录位置
new_dir = "d:/project/git/remove_background/images/new/"



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


# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    nameList = []
    pathDir =  os.listdir(filepath)
    for name in pathDir:
        nameList.append(name)     
    return nameList


# 切图
def cutFigure(url):
  im = Image.open(url)
  if im.mode != "RGBA":
      im = im.convert('RGBA')
  orgw,orgh = im.size
  x = 0
  y = orgh/2-cutHeight/2
  region = im.crop((x, y, orgw, y+cutHeight))
  return region


# 循环切图
def cycleFigure(nameList):
    image_list = []
    for item in nameList:
        image = cutFigure(os.path.join(original_dir,item))
        name,ext=os.path.splitext(item)  
        newPath = os.path.join(new_dir,name) + ".png"
        if mode==1:
           image.save(newPath)
        else:
            image_list.append({
                "image":image,
                "path":newPath
            })
    return image_list


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
            if min_h <= h <= max_h and min_v <= v <= max_v:
                pix[x, y] = (0, 0, 0, 0)
    return image


# 去背景色
def remove_background(image_list):
    for item in image_list: 
        min_range = (120, 0, 0)
        max_range = (220, 200, 100)
        newImage=chromaKey(item["image"],min_range,max_range)
        newImage.save(item["path"])


if __name__ == '__main__':
    # mode=1 只切图
    image_list = cycleFigure( eachFile(original_dir))
    #  去背景色
    if mode==2:
       remove_background(image_list)
