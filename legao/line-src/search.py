import cv2
from util import create_dir,create_specified_dir,get_specified_dir,\
                 asscess_dir,clear_dir,get_image_paths_from_dir,clipImage,\
                 get_file_name


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


def find(img, path, target):
    img_small = resize_to_resolution(img, 1024)
    small_h, w = img_small.shape[:2]

    im = cv2.GaussianBlur(img_small, (1, 1), 0)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, *(20, 30))
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    (_, thresh) = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    cimg, cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow("1",cimg)
    # cv2.waitKey(0)

    valid = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if h == small_h:
            valid.append(x)

    if len(valid) >= 2:
        #  print('检索出更多的图',path)
        new_file_name = get_specified_dir(target, get_file_name(path))
        cv2.imwrite(new_file_name, img_small)
        return img_small

    # cv2.drawContours(img_small, cnts, -1, (0, 0, 255), 3)


def preprocess_image(path, target):
    img = clipImage(path)
    return find(img, path, target)


# 找到有效图
def seachUsefulFigure(pool, original, target):
    print("找到原图...")
    futures = []
    image_files = get_image_paths_from_dir(original)
    for path in image_files:
        futures.append(pool.apply_async(preprocess_image, (path, target)))

    for future in futures:
        future.get()

