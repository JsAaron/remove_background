import argparse
import os
from os import path
import cv2
import numpy as np
import random
from multiprocessing import Pool
import tkinter as tk
from PIL import Image, ImageTk
import json
import shutil
import copy

from colorthief import ColorThief
from config import config
from util import create_dir,create_specified_dir,get_specified_dir,\
                 asscess_dir,clear_dir,get_image_names_from_dir,clipImage
from preprocess import preprocessImage

# debug模式
isDebug = 0

#################
# 工具函数
#################


# 获取文件的父目录
def get_data_dir(img_file):
    return os.path.split(os.path.split(img_file)[0])[0]


# 转化文件名格式  jpg=>png
def get_png_name_for_jpeg(img_file):
    img_file = os.path.split(img_file)[1]
    img_file = os.path.splitext(img_file)[0]
    return img_file + '.png'


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


# img_file 原图
# mask_dst_file 保存目标图
# preproces_path, mask_path
def create_default_mask(preproces_path, mask_path):
    mask_dir, mask_filename = os.path.split(mask_path)
    # 目录文件名 =》xxx.png =>xxx
    mask_title = os.path.splitext(mask_filename)[0]
    # 找到downsampled 第一步处理的图片
    rgb = cv2.imread(get_preprocess_img_name(preproces_path))
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
        edges = cv2.Canny(blur, 127, 255)
        acc += edges

    # 数组的数值被平移或缩放到一个指定的范围，线性归一化
    cv2.normalize(acc, acc, 255, 0, cv2.NORM_MINMAX)
    # 阈值处理
    acc = cv2.threshold(acc, 30, 255, cv2.THRESH_BINARY)[1]

    if isDebug:
        edge_file = os.path.join(mask_dir, mask_title + "_edge.png")
        cv2.imwrite(edge_file, acc)

    # 探测空区域
    edges = (255 - acc).astype(np.uint8)
    # 计算2值图象中所有像素离其最近的值为0像素的近似距离
    # src为输入的二值图像。distanceType为计算距离的方式，可以是如下值
    # DIST_USER    = ⑴,  //!< User defined distance
    # DIST_L1      = 1,   //!< distance = |x1-x2| + |y1-y2|
    # DIST_L2      = 2,   //!< the simple euclidean distance
    # DIST_C       = 3,   //!< distance = max(|x1-x2|,|y1-y2|)
    # DIST_L12     = 4,   //!< L1-L2 metric: distance = 2(sqrt(1+x*x/2) - 1))
    # DIST_FAIR    = 5,   //!< distance = c^2(|x|/c-log(1+|x|/c)), c = 1.3998
    # DIST_WELSCH  = 6,   //!< distance = c^2/2(1-exp(-(x/c)^2)), c = 2.9846
    # DIST_HUBER   = 7    //!< distance = |x|<c ? x^2/2 : c(|x|-c/2), c=1.345
    # maskSize是蒙板尺寸，只有0,3,5
    # DIST_MASK_3       = 3, //!< mask=3
    # DIST_MASK_5       = 5, //!< mask=5
    # DIST_MASK_PRECISE = 0  //!< mask=0
    distances = cv2.distanceTransform(edges, cv2.DIST_L2, 5)
    bg_mask = cv2.threshold(distances, 60, 255,
                            cv2.THRESH_BINARY)[1].astype(np.uint8)

    cv2.normalize(distances, distances, 255, 0, cv2.NORM_MINMAX)

    if isDebug:
        dist_file = os.path.join(mask_dir, mask_title + "_dist.png")
        cv2.imwrite(dist_file, distances)

    bg_image = rgb.copy()
    bg_image[bg_mask != 0] = 0
    if isDebug:
        bg_mask_file = os.path.join(mask_dir, mask_title + "_bgmask.png")
        cv2.imwrite(bg_mask_file, bg_image)

    ffmask = np.zeros((rgb.shape[0] + 2, rgb.shape[1] + 2), dtype=np.uint8)
    seed_points = np.column_stack(np.where(bg_mask != 0))
    np.random.shuffle(seed_points)
    seed = seed_points[0]
    width = rgb.shape[1]
    height = rgb.shape[0]
    for it in range(10):
        area, _, _, rect = cv2.floodFill(
            rgb,
            ffmask, (seed[1], seed[0]),
            255,
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
    preproces_files = get_image_names_from_dir(preproces_dir, 'png')

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
    mask_files = get_image_names_from_dir(mask_dir, 'png')

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
def split_parts_for_image(preproces_path, out_dir, dir_name, original_dir):
    try:

        # 第三步图片
        segmented_img = cv2.imread(get_seg_file_name(preproces_path))

        # 原始图
        original_img = cv2.imread(
            get_specified_dir(original_dir, dir_name + ".jpg"))

        width = segmented_img.shape[1]
        height = segmented_img.shape[0]

        #定义区域
        min_area = 10
        max_area = width * height / 2

        # BGR=>灰度图
        mask = cv2.cvtColor((segmented_img != 0).astype(np.uint8),
                            cv2.COLOR_BGR2GRAY)

        # 部件图
        partImages = []

        topY = config["topY"]

        while True:
            nz = np.nonzero(mask.flatten())[0].flatten()
            if len(nz) == 0:
                break

            nz_i = 0
            found_mask = None
            found_image = None
            while True:
                index = nz[nz_i]
                seed_x = index % width
                seed_y = index // width

                ff_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
                area, _, __, rect = cv2.floodFill(
                    mask,
                    ff_mask, (seed_x, seed_y),
                    255,
                    flags=cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE)

                x = rect[0]
                y = rect[1]
                w = rect[2]
                h = rect[3]

                # slicing into found rect
                roi_mask = ff_mask[y + 1:y + 1 + h, x + 1:x + 1 + w]
                found = False

                if min_area < area < max_area:
                    found_mask = roi_mask
                    newX = x * 4
                    newY = (y * 4) + topY
                    newW = w * 4
                    newH = h * 4
                    found_image = original_img[newY:newY + newH, newX:newX +
                                               newW].copy()
                    found = True

                # clearing found component in the mask
                mask[y:y + h, x:x + w][roi_mask != 0] = 0

                if found:
                    break

                nz_i += 1
                if nz_i >= len(nz):
                    break

            if found_mask is not None:
                partImages.append(found_image)

        # 如果有多个零件，创建目录保存
        # hasmorepart = len(partImages) > 1
        # if hasmorepart:
        out_dir = get_specified_dir(out_dir, dir_name)

        if os.path.exists(out_dir):
            clear_dir(out_dir)
        else:
            create_dir(out_dir)

        part_index = 0
        for part in partImages:
            title = os.path.splitext(os.path.split(preproces_path)[1])[0]
            file_name = os.path.join("", "%s_%02d.png" % (title, part_index))
            # if hasmorepart:
            out_file = os.path.join(out_dir,
                                    "%s_%02d.png" % (title, part_index))
            # else:
            #     out_file = os.path.join(out_dir, "%s.png" % (title))
            cv2.imwrite(out_file, part)
            color_thief = ColorThief(out_file)
            dominant_color = color_thief.get_color(quality=1)
            h, w = part.shape[:2]
            datas = {
                "name": file_name,
                "w": w,
                "h": h,
                "area": w * h,
                "colour": dominant_color
            }
            filePath = out_dir + "/data.json"
            if (os.path.exists(filePath)):
                fl = open(filePath, 'a')
            else:
                fl = open(filePath, 'w')
            fl.write(json.dumps(datas, ensure_ascii=False, indent=2))
            fl.close()
            part_index += 1

        return dir_name + ".png", True
    except Exception as _:
        return dir_name + ".png", False


def split_parts(pool, preproces_dir, target_dir, original_dir):
    preproces_files = get_image_names_from_dir(preproces_dir, 'png')
    futures = []
    for preproces_path in preproces_files:
        out_dir, dir_name = get_out_dir_name(preproces_path, target_dir)
        futures.append(
            pool.apply_async(
                split_parts_for_image,
                (preproces_path, out_dir, dir_name, original_dir)))

    for future in futures:
        name, success = future.get()
        print("处理成功: %s" % name if success else "处理失败: %s" % name)


#每次处理数量
baseCount = 2


def check_next_task():
    print(1)


def start_pool(futures, original_dir, temp_dir, target_dir):
    # 开始处理
    pool = Pool(4)
    # 预处理
    preproces_dir = preprocessImage(pool, futures, temp_dir)
    # #生成mask图
    mask_dir = generate_default_masks(pool, preproces_dir, temp_dir)
    #分离前景与背景
    seg_dir = segment_images(pool, preproces_dir, mask_dir, temp_dir)
    #分割部件
    split_parts(pool, preproces_dir, target_dir, original_dir)


def pocess_image(startCount, image_total_files, original_dir, temp_dir,
                 target_dir):
    futures = []
    interval = baseCount

    # 退出
    if (startCount >= len(image_total_files)):
        print(len(image_total_files), "张图片，全部处理完毕")
        return

    # 如果总数低于间隔数
    if len(image_total_files) < baseCount:
        interval = len(image_total_files)

    # 从0开始
    endCount = startCount + interval - 1

    print("===============================")
    showEnd = endCount+1
    if len(image_total_files)<endCount+1:
        showEnd = endCount
    print("共计图", len(image_total_files), "张图, 开始处理", startCount+1 , "到",
          showEnd )

    for index, value in enumerate(image_total_files):
        if index >= startCount and index <= endCount:
            img = image_total_files[index]
            futures.append(img)

    start_pool(futures, original_dir, temp_dir, target_dir)

    print("===============================")

    # 检测下一个任务
    asscess_dir(temp_dir)
    pocess_image(endCount+1, image_total_files, original_dir, temp_dir,
                 target_dir)


def prepare():
    temp_dir = config["temp_dir"]
    original_dir = config["original_dir"]
    target_dir = config["target_dir"]

    # 根目录处理
    asscess_dir(target_dir)
    asscess_dir(temp_dir)

    image_total_files = get_image_names_from_dir(original_dir, 'jpg')

    #开始任务
    pocess_image(0, image_total_files, original_dir, temp_dir, target_dir)


if __name__ == "__main__":
    prepare()
