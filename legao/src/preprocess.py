import cv2
from util import create_dir,create_specified_dir,get_specified_dir,\
                 asscess_dir,clear_dir,\
                 get_file_name,get_data_dir,get_png_name_for_jpeg


# 等比缩放
def resize_to_resolution(im, preproces_size):
    if max(im.shape[0], im.shape[1]) > preproces_size:
        if im.shape[0] > im.shape[1]:
            dsize = ((preproces_size * im.shape[1]) // im.shape[0],
                     preproces_size)
        else:
            dsize = (preproces_size,
                     (preproces_size * im.shape[0]) // im.shape[1])
        im = cv2.resize(im, dsize, interpolation=cv2.INTER_AREA)
    return im


#获取图片
def clipImage(path, start_y, end_y):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    setH = end_y
    setH = h
    img = img[start_y:setH, 0:w]
    return img


def preprocess_image(path, temp_dir, start_y, end_y):
    img = clipImage(path, start_y, end_y)
    img_small = resize_to_resolution(img, 1024)
    return img_small, path


# 找到preprocess目录下的图片
def get_preprocess_img_name(img_file):
    ds_dir = os.path.join(get_data_dir(img_file), "1_preprocess")
    return os.path.join(ds_dir, get_png_name_for_jpeg(img_file))


#预处理 缩放图
def preprocessImage(pool, image_files, temp_dir, start_y, end_y):
    futures = []
    preproces_dir = create_specified_dir(temp_dir, "1_preprocess")

    for path in image_files:
        futures.append(
            pool.apply_async(preprocess_image,
                             (path, temp_dir, start_y, end_y)))

    imageObjs = []
    for future in futures:
        img, path = future.get()
        imageObjs.append(img)
        new_file_name = get_specified_dir(preproces_dir,
                                          get_png_name_for_jpeg(path))
        cv2.imwrite(new_file_name, img)

    return preproces_dir