import argparse
import os
from os import path
import numpy as np
import random
from multiprocessing import Pool
import tkinter as tk
from PIL import Image, ImageTk
import json
import shutil
import copy


import cv2
import matplotlib.pyplot as plt
import imutils
from imutils.perspective import four_point_transform


from color import ColorThief
from config import config
from util import create_dir,create_specified_dir,get_specified_dir,\
                 asscess_dir,clear_dir,get_image_paths_from_dir,\
                 get_file_path, copy_move_file, split_path, get_png_name_for_jpeg, get_jpeg_name_for_png

from preprocess import preprocessImage

# debug模式
isDebug = 0

#################
# 工具函数
#################


# 获取文件的父目录
def get_data_dir(img_file):
    return os.path.split(os.path.split(img_file)[0])[0]


def showImage(img):
    numb = str(random.random())
    cv2.namedWindow(numb, cv2.WINDOW_NORMAL)
    cv2.imshow(numb, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def shrink_mask(mask, kernel_size, n_iter=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, kernel, iterations=n_iter)


# 找到preprocess目录下的图片
def get_preprocess_img_name(img_file):
    ds_dir = os.path.join(get_data_dir(img_file), "1_preprocess")
    return os.path.join(ds_dir, get_png_name_for_jpeg(img_file))


##########################
#  处理第二步
##########################


# 通过图片的原地址，转化成新的目录+png
def get_mask_file_name(img_file):
    data_dir = get_data_dir(img_file)
    masks_dir = get_specified_dir(data_dir, "2_masks")
    png_name = get_png_name_for_jpeg(img_file)
    return os.path.join(masks_dir, png_name)


#粗略的调节对比度和亮度
def contrast_brightness_image(src1, a, g):
    h, w, ch = src1.shape
    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    return dst


def on_canny(rgb,keyNmae,mix,max):
    mix = cv2.getTrackbarPos("mix",keyNmae) or mix
    max =  cv2.getTrackbarPos("max",keyNmae) or max

    split = cv2.split(rgb)
    # Bule ，返回来一个给定形状和类型的用0填充的数组
    acc = np.zeros(split[0].shape, dtype=np.float32)
    # b,g,r
    for img in split:
        # 边缘检测,img 单通道灰度图.找到3种灰度对应的边缘
        # detected_edges = cv2.GaussianBlur(img,(3,3),0)
        blur = cv2.GaussianBlur(img, (3, 3), 0)
        # 图像灰度梯度 高于maxVal被认为是真正的边界，低于minVal的舍弃。
        # 两者之间的值要判断是否与真正的边界相连，相连就保留，不相连舍弃
        edges = cv2.Canny(blur, mix, max)
        acc += edges

    cv2.imshow(keyNmae, acc) 


def on_mask(distances,maskName):
    mix = cv2.getTrackbarPos("mix",maskName)
    max = cv2.getTrackbarPos("max",maskName)
    bg_mask = cv2.threshold(distances,mix, max,cv2.THRESH_BINARY)[1].astype(np.uint8)
    cv2.imshow(maskName, bg_mask)


# img_file 原图
# mask_dst_file 保存目标图
# preproces_path, mask_path
def create_default_mask(preproces_path, mask_path):
    mask_dir, mask_filename = os.path.split(mask_path)
    # 目录文件名 =》xxx.png =>xxx
    mask_title = os.path.splitext(mask_filename)[0]
    # 找到downsampled 第一步处理的图片
    rgb = cv2.imread(get_preprocess_img_name(preproces_path))
    rgb = contrast_brightness_image(rgb, 1, 30)



    # keyNmae = "set-canny"
    # cv2.namedWindow(keyNmae)
    # cv2.createTrackbar("max",keyNmae,0,255,on_canny)
    # cv2.createTrackbar("mix",keyNmae,0,255,on_canny)

    # while(1):
    #     on_canny(rgb,keyNmae,35,92)
    #     k = cv2.waitKey(1)&0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()



    split = cv2.split(rgb)
    # Bule ，返回来一个给定形状和类型的用0填充的数组
    acc = np.zeros(split[0].shape, dtype=np.float32)

    # b,g,r
    for img in split:
        # 边缘检测,img 单通道灰度图.找到3种灰度对应的边缘
        # detected_edges = cv2.GaussianBlur(img,(3,3),0)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        # 图像灰度梯度 高于maxVal被认为是真正的边界，低于minVal的舍弃。
        # 两者之间的值要判断是否与真正的边界相连，相连就保留，不相连舍弃
        edges = cv2.Canny(blur, 23, 70)
        acc += edges


    # 数组的数值被平移或缩放到一个指定的范围，线性归一化
    cv2.normalize(acc, acc, 255, 0, cv2.NORM_MINMAX)

    # 阈值处理
    acc = cv2.threshold(acc, 127, 255, cv2.THRESH_BINARY)[1]

    # 反转颜色
    edges = (255 - acc).astype(np.uint8)

    # 计算2值图象中所有像素离其最近的值为0像素的近似距离
    # src为输入的二值图像。distanceType为计算距离的方式，可以是如下值
    distances = cv2.distanceTransform(edges, cv2.DIST_L2, 5)


    # maskName = "set-mask"
    # cv2.namedWindow(maskName)
    # cv2.createTrackbar("max",maskName,0,255,on_mask)
    # cv2.createTrackbar("mix",maskName,0,255,on_mask)

    # while(1):
    #     on_mask(distances,maskName)
    #     k = cv2.waitKey(1)&0xFF
    #     if k == 27:
    #         break
    # cv2.destroyAllWindows()

    # cv2.normalize(distances, distances, 255, 0, cv2.NORM_MINMAX)
    bg_mask = cv2.threshold(distances,20, 255,cv2.THRESH_BINARY)[1].astype(np.uint8)

    bg_image = rgb.copy()
    # 白色的地方转成黑色
    bg_image[bg_mask != 0] = 0

    # 掩码图像,大小比原图多两个像素点
    ffmask = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), dtype=np.uint8)

    seed_points = np.column_stack(np.where(bg_mask != 0))
    np.random.shuffle(seed_points)
    
    seed = seed_points[0]

    width = rgb.shape[1]
    height = rgb.shape[0]

    # showImage(bg_mask)

    for it in range(5):
        # 颜色替换
        area, _, _, rect = cv2.floodFill(
            rgb,
            ffmask, 
            # 其实填充标记点 
            (seed[1], seed[0]), 
            # 填充值 
            255,
            # 为像素值的下限差值
            loDiff=(4, 4, 4, 4),
            upDiff=(4, 4, 4, 4),
            flags=(4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY))

        bg_mask = cv2.bitwise_or(ffmask[1:1 + height, 1:1 + width], bg_mask)
        seed_points = np.column_stack(np.where(bg_mask != 0))
        if len(seed_points) == 0:
            break

        np.random.shuffle(seed_points)
        seed = seed_points[0]

    bg_mask = shrink_mask(bg_mask, 3, 1)
    bg_image = rgb.copy()
    bg_image[bg_mask != 0] = 0
    cv2.imwrite(mask_path, bg_image)

    return preproces_path


# 生成mask图
def generate_default_masks(pool, preproces_dir, temp_dir):
    mask_dir = create_specified_dir(temp_dir, "2_masks")
    preproces_files = get_image_paths_from_dir(preproces_dir, 'png')

    # 创建目录
    futures = []
    for preproces_path in preproces_files:
        mask_path = get_mask_file_name(preproces_path)
        futures.append(
            pool.apply_async(create_default_mask, (preproces_path, mask_path)))

    for future in futures:
        future.get()

    return mask_dir


##########################
#  处理第三步
##########################


def get_seg_file_name(img_file):
    data_dir = get_data_dir(img_file)
    seg_dir = get_specified_dir(data_dir, "3_segmentation")
    png_name = get_png_name_for_jpeg(img_file)
    return os.path.join(seg_dir, png_name)


# 首先用矩形将要选择的前景区域选定，其中前景区域应该完全包含在矩形框当中。
# 然后算法进行迭代式分割，知道达到效果最佳。但是有时分割结果不好，
# 例如前景当成背景，背景当成前景。测试需要用户修改。
# 用户只需要在非前景区域用鼠标划一下即可。
# mask_dir, seg_path
def create_segmentation(mask_path, seg_file):
    try:
        cut_img = cv2.imread(get_preprocess_img_name(mask_path))
        mask_img = cv2.imread(get_mask_file_name(mask_path))
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)

        # 快速创建图，并且可能是前景
        # cv::GC_BGD  == 0//表示是背景
        # cv::GC_FGD  == 1//表示是前景
        #  cv::GC_PR_BGD  == 2//表示可能是背景
        # cv::GC_PR_FGD  == 3//表示可能是前景
        mask = np.ones(cut_img.shape[:2], np.uint8) * cv2.GC_PR_FGD

        mask[mask_img == 0] = cv2.GC_BGD
        mask[mask_img == 255] = cv2.GC_FGD


        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        # mask ：蒙版图像，指定哪些区域是背景，前景或可能的背景/前景等.它是由下面的标志，
        # cv2.GC_BGD，cv2.GC_FGD，cv2.GC_PR_BGD，cv2.GC_PR_FGD，或简单地将0，1，2，3传递给图像。
        # bdgModel, fgdModel ：算法内部使用的数组,只需要创建两个大小为（1,65）的np.float64类型的0数组
        # iterCount ：算法运行的迭代次数.
        # mode ：cv2.GC_INIT_WITH_RECT或cv2.GC_INIT_WITH_MASK，或者组合起来决定我们是画矩形还是最后的触点.
        cv2.grabCut(cut_img, mask, None, bgdModel, fgdModel, 5,
                    cv2.GC_INIT_WITH_MASK)

        # #0和2做背景
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # 使用蒙板来获取前景区域
        segmented = cut_img * mask2[:, :, np.newaxis]

        # # saving segmentation
        cv2.imwrite(seg_file, segmented)
        return mask_path, True

    except Exception as _:
        return mask_path, False


# preproces_dir, temp_dir
def segment_images(pool, preproces_dir, mask_dir, temp_dir):
    seg_dir = create_specified_dir(temp_dir, "3_segmentation")
    mask_files = get_image_paths_from_dir(mask_dir, 'png')

    futures = []
    for mask_path in mask_files:
        seg_path = get_seg_file_name(mask_path)
        futures.append(
            pool.apply_async(create_segmentation, (mask_path, seg_path)))

    for future in futures:
        future.get()
        # print("提取前景: %s" % name if success else "failed to segment: %s" % name)

    return seg_dir


##########################
#  处理第四步
##########################


# 新文件路径
def get_out_dir_name(img_file, target_dir):
    out_dir = get_specified_dir(target_dir, "")
    dir_name = os.path.splitext(os.path.split(img_file)[1])[0]
    return out_dir, dir_name


# 切割部件
def split_parts_for_image(start_y, preproces_path, out_dir, dir_name,
                          original_dir, collectData, splitMode):
    try:

        # 第三步图片
        segmented_img = cv2.imread(get_seg_file_name(preproces_path))

        # 原始图
        original_img = cv2.imread(
            get_specified_dir(original_dir, dir_name + ".jpg"))

        # 原图宽度
        original_width = original_img.shape[1]
        original_resize = original_width / 1000

        width = segmented_img.shape[1]
        height = segmented_img.shape[0]

        #定义区域
        min_area = 50
        max_area = width * height 

        # BGR=>灰度图
        mask = cv2.cvtColor((segmented_img != 0).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY)

        # 部件图
        partImages = []

        # 边缘部件
        part_remove_left = 0
        part_remove_right = 0

        #参考点标签的坐标合计
        labelPoint = []


        newimage=original_img.copy()

        while True:
            # 将mask转化为1维数组
            # 返回数组mask中值不为零的元素的下标,
            nz = np.nonzero(mask.flatten())[0].flatten()
            if len(nz) == 0:
                break

            nz_i = 0
            found_mask = None
            found_image = None
            while True:
                index = nz[nz_i]
                seed_x = index % width
                # 向下取整
                seed_y = index // width 
                ff_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
                area, _, __, rect = cv2.floodFill(
                    mask,
                    ff_mask,
                    (seed_x, seed_y),
                    255,
                    flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]

                # slicing into found rect
                roi_mask = ff_mask[y + 1:y + 1 + h, x + 1:x + 1 + w]
                found = False

                # 找指定区域的内容
                if min_area < area < max_area:
                    found_mask = roi_mask
                    newX = x * 4
                    newY = (y * 4) + start_y
                    newW = w * 4
                    newH = h * 4

                    # 边界模式，跳出循环
                    if splitMode == "full":
                        if newX <= 0:
                            found_mask = None
                            part_remove_left += 1

                        if newW + newX >= original_width:
                            found_mask = None
                            part_remove_right += 1

                    # 扩散30px截取
                    startX = newX - 30
                    endX = newX + newW + 30
                    startY = newY - 30
                    endY = newY + newH + 30
                    found_image = original_img[startY:endY, startX:endX].copy()
                    found = True

                    out_file = os.path.join(out_dir, "%s.png" % (newX))
                    cv2.imwrite(out_file, found_image)

                    # 数字下标
                    if newW<400 and newH>100 and newH<200 and newY>1800:
                        out_file = os.path.join(out_dir, "%s.png" % (newX))
                        cv2.imwrite(out_file, found_image)
                        pox = preproces_line(out_file,os.path.join(out_dir, "%s.line.png" % (newX)))
                        left_top_x,left_top_y = pox[0]
                        left_bottom_x,left_bottom_y = pox[1]
                        # 计算在图中真实坐标
                        left_top_x += startX 
                        left_top_y += startY 
                        left_bottom_x += startX 
                        left_bottom_y += startY 

                        # 目标的X坐标
                        targetX = getXValue((left_bottom_x, left_bottom_y), (left_top_x, left_top_y), 1050)
                        labelPoint.append([left_bottom_x,left_bottom_y,int(targetX),1050])

                        cv2.line(newimage,(left_bottom_x,left_bottom_y),(int(targetX),1050),(255,255,0),10)
                        found_mask = None

                    # 找到其余的元素
                    if found_mask is not None:
                      centerX = newX + newW/2
                      centerY = newY + newH/2
                      partImages.append([int(centerX),int(centerY),found_image])
                      out_file = os.path.join(out_dir, "%s.png" % (newX))
                      cv2.imwrite(out_file, found_image)

                # clearing found component in the mask
                mask[y:y + h, x:x + w][roi_mask != 0] = 0

                if found:
                    break

                nz_i += 1
                if nz_i >= len(nz):
                    break


        # 获取交叉点坐标
        crosslineDistance(newimage,partImages,labelPoint)
   
        # 输出线图
        cv2.imwrite( os.path.join(out_dir, "%s.png" % (dir_name)), newimage)


        return dir_name + ".png", True
    except Exception as _:
        return dir_name + ".png", False


def split_parts(pool, preproces_dir, g_conf):

    target_dir = g_conf["target_dir"]
    original_dir = g_conf["original_dir"]
    error_dir = g_conf["error_dir"]
    start_y = g_conf["start_y"]
    collectData = g_conf["collectData"]
    splitMode = g_conf["splitMode"]

    preproces_files = get_image_paths_from_dir(preproces_dir, 'png')
    futures = []
    for preproces_path in preproces_files:
        out_dir, dir_name = get_out_dir_name(preproces_path, target_dir)
        futures.append(
            pool.apply_async(split_parts_for_image,
                             (start_y, preproces_path, out_dir, dir_name,
                              original_dir, collectData, splitMode)))

    for future in futures:
        name, success = future.get()
        copyName = get_jpeg_name_for_png(get_specified_dir(original_dir, name))
        if success:
            print("分解成功: ", copyName)
        else:
            distName = get_specified_dir(error_dir,
                                         get_jpeg_name_for_png(name))
            copy_move_file(copyName, distName)
            print("分解失败: ", copyName)


# 开始执行任务
def exec_pool_task(futures, g_conf):
    pool = Pool(4)
    temp_dir = g_conf["temp_dir"]

    # 预处理
    preproces_dir = preprocessImage(pool, futures, g_conf)
    #生成mask图
    mask_dir = generate_default_masks(pool, preproces_dir, temp_dir)
    # #分离前景与背景
    seg_dir = segment_images(pool, preproces_dir, mask_dir, temp_dir)
    # # 分割部件
    split_parts(pool, preproces_dir, g_conf)


def exec_process_image(startCount, image_total_files, g_conf):

    interval = g_conf["interval"]
    temp_dir = g_conf["temp_dir"]

    futures = []

    # 图片总数量
    total_images = len(image_total_files)

    # 退出
    if (startCount >= total_images):
        print(total_images, "张图片，全部分解完毕", sep="")
        return

    # 如果总数低于间隔数
    if total_images < interval:
        interval = total_images

    # 从0开始
    endCount = startCount + interval - 1

    print("===============================")

    #结尾处理
    showEnd = endCount + 1
    if total_images < endCount + 1:
        showEnd = startCount + (total_images - startCount)

    print(
        "共计", total_images, "张图, 开始分解图", startCount + 1, "到", showEnd, sep='')

    for index, value in enumerate(image_total_files):
        if index >= startCount and index <= endCount:
            img = image_total_files[index]
            futures.append(img)

    exec_pool_task(futures, g_conf)

    print("===============================")

    # 检测下一个任务
    # asscess_dir(temp_dir)
    # exec_process_image(endCount + 1, image_total_files, g_conf)


# 执行入口
def legao_main(original_dir="",
               target_dir="",
               interval="",
               error_dir="",
               start_y="",
               collectData="",
               splitMode="",
               resize="",
               end_y=""):

    if original_dir == "":
        print("必须传递原图目录 original_dir")
        return

    if target_dir == "":
        print("必须传递保存目录 target_dir")
        return

    if start_y == "":
        print("必须设置截图的Y轴上部距离(px) start_y")
        return

    if end_y == "":
        print("必须设置截图的Y轴下部距离(px) end_y")
        return

    # 临时保存目录
    temp_dir = get_specified_dir(original_dir, "temp")
    error_dir = error_dir or get_specified_dir(original_dir, "error")

    # 根目录处理
    asscess_dir(target_dir)
    asscess_dir(temp_dir)
    asscess_dir(error_dir)

    original_image_total_files = get_image_paths_from_dir(original_dir, 'jpg')

    g_conf = {
        "collectData": collectData,
        "splitMode": splitMode,
        "interval": interval,
        "resize": resize,
        "start_y": start_y,
        "end_y": end_y,
        "original_dir": original_dir,
        "target_dir": target_dir,
        "temp_dir": temp_dir,
        "error_dir": error_dir
    }

    # 赋全局值
    for prop in g_conf:
        if not g_conf[prop]:
            g_conf[prop] = config[prop]

    modeName = "全部保留"
    if splitMode == "full":
        modeName = "仅保留完整的零件"

    message = [
        ["原图目录: ", original_dir],
        ["临时目录: ", temp_dir],
        ["错误目录: ", error_dir],
        ["完成目录: ", target_dir],
        ["原图截取从Y: ", start_y],
        ["原图截止至Y: ", end_y],
        ["分解量: ", interval],
        ["切割模式: ", modeName],
    ]
    for name in message:
        print(name[0], name[1], sep="")

    #开始任务
    exec_process_image(0, original_image_total_files, g_conf)


def prepare():
    legao_main(
        original_dir=config["original_dir"],
        target_dir=config["target_dir"],
        error_dir=config["error_dir"],
        start_y=config["start_y"],
        resize=config["resize"],
        collectData=config["collectData"],
        splitMode=config["splitMode"],
        interval=config["interval"],
        end_y=config["end_y"])



def preproces_line(out_file,new_file):
    image = cv2.imread(out_file)
    #转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #高斯滤波
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #自适应二值化方法
    blurred=cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,51,2)
    blurred=cv2.copyMakeBorder(blurred,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
    edged = cv2.Canny(blurred, 50, 100)     
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    docCnt = None
    # 确保至少有一个轮廓被找到
    if len(cnts) > 0:
        # 将轮廓按大小降序排序
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        # 对排序后的轮廓循环处理
        for c in cnts:
            # 获取近似的轮廓
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # 如果近似轮廓有四个顶点，那么就认为找到了
            if len(approx) == 4:
                docCnt = approx
                break

    newimage=image.copy()
    for i in docCnt:
        #circle函数为在图像上作图，新建了一个图像用来演示四角选取  b,g,r
        cv2.circle(newimage, (i[0][0],i[0][1]), 0, (240 ,32, 160), -1)

    cv2.imwrite(new_file, newimage)

    image_line = Image.open(new_file)
    if image_line.mode != "RGBA":
        image_line = image_line.convert('RGBA')
    pix = image_line.load()
    width, height = image_line.size

    tempY = 0
    dict_arr = []
    for x in range(width):
        for y in range(height):
            r, g, b, a = pix[x, y]
            if r==160 and g==32 and b == 240:
                if len(dict_arr):
                    if len(dict_arr)>1:
                        return dict_arr
                    if tempY<y:
                        dict_arr.append([x,y])
                    else:
                        dict_arr.insert(0,[x,y]) 
                    break
                else:
                    dict_arr.append([x,y])
                    tempY = y
    return



# d:\project\github\remove_background\legao\target\2876.png
# d:\project\github\remove_background\legao\target\1744.png

def getXValue(p1, p2, y):
    '''
    p1和p2是两个点，y是另外一个点p3的Y坐标值；
    p1,p2和p3在同一条直线上，返回点p3的X坐标值
    p1=(x1,y1)  p2=(x2,y2)
    y记得加上截取的值默认是1050
    '''
    k = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p2[1] - k * p2[0]
    x = (y - b) / k
    return x


def cross_point(line1,line2):#计算交点函数
    x1=line1[0]#取四点坐标
    y1=line1[1]
    x2=line1[2]
    y2=line1[3]
    
    x3=line2[0]
    y3=line2[1]
    x4=line2[2]
    y4=line2[3]
    
    k1=(y2-y1)*1.0/(x2-x1)#计算k1,由于点均为整数，需要进行浮点数转化
    b1=y1*1.0-x1*k1*1.0#整型转浮点型是关键
    if (x4-x3)==0:#L2直线斜率不存在操作
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [x,y]


# 计算元素中心位置，到交叉点的距离
def crosslineDistance(newimage,partImages,labelPoint):
    for item in labelPoint:
        x1,y1,x2,y2 = item
        for part in partImages:
            centerX,centerY,part = part
            line1=[centerX,centerY,4096,centerY]
            line2=[x1, y1,x2,y2]
            # 交叉点坐标
            cross = cross_point(line1, line2)
            if centerX < int(cross[0]):
               cv2.line(newimage,(centerX,centerY),(int(cross[0]),int(cross[1])),(0,255,255),3)
        print(x1)

    # for item in partImages:
    #     centerX,centerY,part = item
    #     line1=[centerX,centerY,4096,centerY]

    #     # 第一个交叉点
    #     x1,y1,x2,y2 = labelPoint[0]
    #     line2=[x1, y1,x2,y2]
    #     # 交叉点坐标
    #     cross = cross_point(line1, line2)
    #     if centerX < int(cross[0]):
    #         cv2.line(newimage,(centerX,centerY),(int(cross[0]),int(cross[1])),(0,255,255),3)
  
    #     # 第一个交叉点
    #     x3,y3,x4,y4 = labelPoint[1]
    #     line2=[x3, y3,x4,y4]
    #     # 交叉点坐标
    #     cross = cross_point(line1, line2)
    #     if centerX < int(cross[0]):
    #         cv2.line(newimage,(centerX+10,centerY+10),(int(cross[0]),int(cross[1])),(0,0,255),3)
  



    return  


if __name__ == "__main__":
    prepare()
    # line1=[2034,1298,4098,1298]
    # line2=[2879, 1984,1805,0]
    # pos = cross_point(line1, line2).
    
    # print(pos)

    # preproces_line("d:\\project\\github\\remove_background\\legao\\target\\1744.png","d:\\project\\github\\remove_background\\legao\\target\\1744.line.png")

