import random
import cv2
import numpy as np
import time
import glob
from threading import Thread


def create_mosaic(imgs_path='D:/Datasets/laser_etch/helvetica_images/chars/augments/', img_ext='.png'):
    angle_plus_minus = 30
    prefix = '_aug_'
    annots = []  # (class, center_x, center_y, width, height)
    new_mosaic = True
    new_row = True
    full_path = imgs_path + '*' + img_ext
    files = glob.glob(full_path)
    mosaic = None
    first_img = None
    imgrow = None
    initial_len = len(files)
    class_id = ''
    y_offset = 0
    x_offset = 0
    m = 0
    i = 0
    while i < initial_len:
        if len(files) > 0:
            f = random.randint(0, len(files) - 1)
            file = files.pop(f)
            img = cv2.imread(file)
            rows, _, _ = img.shape
            scale = 0
            if new_mosaic:
                if rows > 100:
                    while rows > 100:
                        rows = rows // 2
                        scale = scale + 2
                    scale = 1 / scale
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                first_img = img.copy()
                yolo_x, yolo_y, yolo_w, yolo_h = get_yolo_box(img, y_offset, x_offset)
                cid = get_class_id_from_filename(file)
                annots.append((cid, yolo_x, yolo_y, yolo_w, yolo_h))
                print("first img")
                debug_img(first_img)
                print("end debug.")
                new_mosaic = False
            if new_row:
                if first_img is None:
                    imgrow = img.copy()
                    yolo_x, yolo_y, yolo_w, yolo_h = get_yolo_box(img, y_offset, x_offset)
                    cid = get_class_id_from_filename(file)
                    annots.append((cid, yolo_x, yolo_y, yolo_w, yolo_h))
                else:
                    scale = first_img.shape[0] / img.shape[0]
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    imgrow = img.copy()

                new_row = False
            else:
                scale = first_img.shape[0] / img.shape[0]
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                if imgrow.shape[1] + img.shape[1] > darksize:
                    print("Debugging img: " + file)
                    debug_img(img)
                    print("end debug")
                    new_row = True
                    top, bot, left, right = 0, 0, 0, darksize - imgrow.shape[1]
                    imgrow = np.pad(imgrow, ((top, bot), (left, right), (0, 0)), mode='constant', constant_values=0)
                    if mosaic is None:
                        mosaic = imgrow.copy()
                        debug_img(mosaic, "mosaic " + str(mosaic.shape))
                    else:
                        mosaic = cv2.vconcat([mosaic, imgrow])
                        debug_img(mosaic, "mosaic " + str(mosaic.shape))
                    if mosaic.shape[0] + img.shape[0] > darksize:
                        top, bot, left, right = 0, darksize - mosaic.shape[0], 0, 0
                        mosaic = np.pad(mosaic, ((top, bot), (left, right), (0, 0)), mode='constant',
                                        constant_values=0)
                        debug_img(mosaic, "padded mosaic " + str(mosaic.shape))
                        cv2.imwrite('mosaics/mosaic' + str(m) + ext)
                        m = m + 1
                        del mosaic
                        mosaic = None
                        new_mosaic = True
                        i = i - 1
                    else:
                        del imgrow
                        imgrow = img.copy()
                        y_offset = mosaic.shape[0]
                        y, x, _ = img.shape
                        new_row = False
                else:
                    imgrow = cv2.hconcat([imgrow, img])

        i = i + 1


def get_yolo_box(npshape, y_offset=0, x_offset=0):
    y, x, _ = npshape
    yolo_w = x / darksize
    yolo_h = y / darksize
    yolo_y = (y_offset + (y / 2)) / darksize
    yolo_x = (x_offset + (x / 2)) / darksize
    return yolo_x, yolo_y, yolo_w, yolo_h


def get_class_id_from_filename(filename):
    prefix = '_aug_'
    x = filename.split(prefix)
    cid = x[1][0]
    return cid


def make_chars_list():
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return chars


def copy_file():
    chars = make_chars_list()
    direct = 'D:/Datasets/laser_etch/helvetica_images/chars/'
    filename = 'xlt_'
    extn = '.png'
    chars = make_chars_list()
    for c in chars:
        file = cv2.imread(direct + filename + c + extn)
        cv2.imwrite(direct + c + '_xlt' + '.png', file)


def debug_img(src, title=""):
    img = src.copy()
    cv2.imshow('debug ' + title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def custom_thresh(src, thresh, maxval):
    img = src.copy()
    h, w, d = img.shape
    for x in range(w):
        for y in range(h):
            for ch in range(d):
                if img[y, x, ch] > thresh:
                    img[y, x, ch] = maxval
                else:
                    pass
    return img


def speckle_noise(src):
    image = src.copy()
    row, col, ch = image.shape
    gauss = np.random.randn(row, col, ch)
    gauss = gauss.reshape(row, col, ch)
    multi = float(random.randint(1, 5))
    multi = multi / 10.
    noisy = image + image * gauss * multi
    return noisy


def salt_n_pepper_per_channel(src):
    image = src.copy()
    row, col, ch = image.shape
    s_vs_p = 0.5
    amount = 0.04
    out = np.copy(image)
    salt = random.randint(127, 255)
    pepper = random.randint(0, 126)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[coords] = salt
    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[coords] = pepper
    return out


def gaussian_blur(src):
    rando = random.randint(1, 2)
    if rando == 1:
        ksize = 3
    else:
        ksize = 5
    blurred = cv2.GaussianBlur(src, (ksize, ksize), cv2.BORDER_DEFAULT)
    return blurred


def gaussian_noise(src):
    img = src.copy()
    row, col, ch = img.shape
    mean = 0.
    rando = random.randint(1, 100) / 500
    var = rando
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = img + gauss
    noisy = noisy.astype(np.uint8)
    return noisy


def dilate_img(src):
    img = src.copy()
    rando = random.randint(0, 2)
    if rando == 0:
        ksize = 3
    elif rando == 1:
        ksize = 5
    else:
        ksize = 7
    dilated = cv2.dilate(img, (ksize, ksize))
    return dilated


def erode_img(src):
    img = src.copy()
    rando = random.randint(0, 2)
    if rando == 0:
        ksize = 3
    elif rando == 1:
        ksize = 5
    else:
        ksize = 7
    eroded = cv2.erode(img, (ksize, ksize))
    return eroded


def resize_img(src):
    img = src.copy()
    scale = float(random.randint(100, 1000)) / 100
    rando = random.randint(0, 2)
    if rando == 0:
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    elif rando == 1:
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        resized = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized


def lighten_dots(src):
    img = src.copy()
    row, col, ch = img.shape
    for r in range(row):
        for c in range(col):
            p = random.randint(0, 1) ** 0.5
            val = 45 ** p
            img[r, c, 0] = img[r, c, 0] + val
            img[r, c, 1] = img[r, c, 1] + val
            img[r, c, 2] = img[r, c, 2] + val
    return img


def increase_one_channel(src):
    img = src.copy()
    row, col, ch = img.shape
    p = random.randint(0, 1000) / 1000.
    val = 10 * p
    rando = random.randint(0, 2)
    for r in range(row):
        for c in range(col):
            if rando == 0:
                img[r, c, 0] = img[r, c, 0] + val
            elif rando == 1:
                img[r, c, 1] = img[r, c, 1] + val
            else:
                img[r, c, 2] = img[r, c, 2] + val
    return img


def blotcher1(src):
    img = src.copy()
    h, w, _ = img.shape
    y = int(h / 2)
    x = int(w / 2)
    scale = float(random.randint(75, 125) / 100.)
    angle = random.randint(0, 360)
    img = np.pad(img, ((x, x), (y, y), (0, 0)), mode='constant', constant_values=127)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP)
    img = cv2.erode(img, (3, 3))
    img = cv2.dilate(img, (3, 3))
    # img = increase_one_channel(img)
    img = lighten_dots(img)
    img = cv2.resize(img, (0, 0), fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    unrot_mat = cv2.getRotationMatrix2D(img_center, -angle, 1.0)
    img = cv2.warpAffine(img, unrot_mat, img.shape[1::-1], flags=cv2.INTER_AREA, borderMode=cv2.BORDER_WRAP)
    img = cv2.erode(img, (3, 3))
    img = cv2.dilate(img, (5, 5))
    img = cv2.erode(img, (3, 3))
    img = cv2.dilate(img, (7, 7))
    img = cv2.erode(img, (3, 3))

    img = cv2.resize(img, (0, 0), fx=10, fy=10, interpolation=cv2.INTER_NEAREST)

    img = gaussian_noise(img)
    img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    img = cv2.GaussianBlur(img, (15, 15), cv2.BORDER_DEFAULT)
    img = cv2.GaussianBlur(img, (31, 31), cv2.BORDER_DEFAULT)

    img = cv2.resize(img, (0, 0), fx=1 / 10, fy=1 / 10, interpolation=cv2.INTER_AREA)
    return img


def sharpen(src, ksize=9):
    img = src.copy()
    img2 = cv2.GaussianBlur(img, (9, 9), cv2.BORDER_DEFAULT)
    img = cv2.addWeighted(img, 1.5, img2, -0.5, 0)
    return img


def sharpen_random(src):
    img = src.copy()
    rando = random.randint(1, 4)
    if rando == 1:
        k = 3
    elif rando == 2:
        k = 5
    elif rando == 3:
        k = 7
    else:
        k = 9
    img2 = cv2.GaussianBlur(img, (k, k), cv2.BORDER_DEFAULT)
    img = cv2.addWeighted(img, 1.5, img2, -0.5, 0)
    return img


def get_limit_pcts(c):
    vpct = 0.33
    hpct = 0.33
    if c == 'I':
        hpct = 0.45
    elif c == '1':
        hpct = 0.4
    elif c == '5':
        vpct = 0.3
    return vpct, hpct


def trim(src, character):
    img = src.copy()
    row, col, ch = img.shape
    trim_lines = np.full((row, col, ch), 0, dtype=np.uint8)
    #  trim from top
    thresh1 = 100
    thresh2 = 255
    limit_pct_vrt, limit_pct_hrz = get_limit_pcts(character)
    limit_top = int(row * limit_pct_vrt)
    limit_bottom = int(row - limit_top)
    limit_left = int(col * limit_pct_hrz)
    limit_right = int(col - limit_left)
    # print("Limits: top, bottom, left, right")
    # print(limit_top)
    # print(limit_bottom)
    # print(limit_left)
    # print(limit_right)
    trim_top = 0
    trim_bottom = row
    trim_left = 0
    trim_right = col
    for r in range(row - 1 - 3):
        # print("r=" + str(r))
        if r < limit_top or r > limit_bottom:
            row1 = img[r, 0:col].copy()
            sigma3 = row1.mean() - 3 * row1.std()
            # print("row1 sigma3=" + str(sigma3))
            # print("row1.std=" + str(row1.std()))
            # print("row1.mean=" + str(row1.mean()))
            if sigma3 > thresh1:  # checks if row 'r' contains pixels of the character (which are very dark)
                row2 = img[r + 1, 0:col].copy()
                row12 = cv2.addWeighted(row1, 0.5, row2, 0.5, 0)
                row3 = img[r + 2, 0:col].copy()
                row4 = img[r + 3, 0:col].copy()
                row34 = cv2.addWeighted(row3, 0.5, row4, 0.5, 0)
                if r > row // 2:
                    grad = row34 - row12
                else:
                    grad = row12 - row34
                gmean = grad.mean()
                gstd = grad.std()
                grad_1sig = gmean + gstd
                # print("grad_1sig=" + str(grad_1sig))
                # print("grad std=" + str(gstd))
                # print("grad mean=" + str(gmean))
                if gmean < thresh2:  # if gradient is low then it must not be an edge
                    if r > row // 2:
                        trim_lines = cv2.line(trim_lines, (0, r + 3), (col, r + 3), (0, 0, 255), 1)
                        trim_bottom = min(trim_bottom, r)
                    else:
                        trim_lines = cv2.line(trim_lines, (0, r), (col, r), (0, 0, 255), 1)
                        trim_top = r
                    # print("Drew line. r=" + str(r))
    for c in range(col - 1 - 3):
        if c < limit_left or c > limit_right:
            # print("c=" + str(c))
            col1 = img[0:row, c].copy()
            sigma3 = col1.mean() - 3 * col1.std()
            # print("col1 sigma3=" + str(sigma3))
            # print("col1.std=" + str(col1.std()))
            # print("col1.mean=" + str(col1.mean()))
            if sigma3 > thresh1:  # checks if col 'c' contains pixels of the character (which are very dark)
                col2 = img[0:row, c + 1].copy()
                col12 = cv2.addWeighted(col1, 0.5, col2, 0.5, 0)
                col3 = img[0:row, c + 2].copy()
                col4 = img[0:row, c + 3].copy()
                col34 = cv2.addWeighted(col3, 0.5, col4, 0.5, 0)
                if c > col // 2:
                    grad = col34 - col12
                else:
                    grad = col12 - col34
                gmean = grad.mean()
                gstd = grad.std()
                grad_1sig = gmean + gstd
                # print("grad_1sig=" + str(grad_1sig))
                # print("grad std=" + str(gstd))
                # print("grad mean=" + str(gmean))
                if gmean < thresh2:  # if gradient is low then it must not be an edge
                    if c > col // 2:
                        trim_lines = cv2.line(trim_lines, (c + 3, 0), (c + 3, row), (0, 0, 255), 1)
                        trim_right = min(trim_right, c)
                    else:
                        trim_lines = cv2.line(trim_lines, (c, 0), (c, row), (0, 0, 255), 1)
                        trim_left = c
                    # print("Drew line. c=" + str(c))
    # img = cv2.addWeighted(img, 1, trim_lines, 1, 0)
    img = img[trim_top:trim_bottom, trim_left:trim_right].copy()
    return img


def augment_images(suffix, n):
    #  Medium blotching, medium texturing, thinner character thickness
    start_time = time.time()
    print("Timer started for suffix: " + suffix)
    chars = make_chars_list()
    for i in range(n):
        for c in chars:
            filename = c + suffix
            full_path = src_dir + filename + ext
            img = cv2.imread(full_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = custom_thresh(img, 127, 127)
            img = blotcher1(img)
            img = sharpen(img)
            img = resize_img(img)
            img = erode_img(img)
            img = cv2.dilate(img, (3, 3), iterations=3)
            img = cv2.erode(img, (3, 3), iterations=3)
            img = gaussian_blur(img)
            img = sharpen_random(img)
            img = trim(img, c)
            cv2.imwrite(save_dir + '_aug_' + filename + "_" + str(i) + '.png', img)
            # print("Saved " + save_dir + '_auged_' + filename + '.png')
    end_time = time.time()
    print(suffix + " process time: " + str(end_time - start_time))


def auger_threads(lst, n):
    threads = []
    threads.append(Thread(target=augment_images, args=(lst[0], n)))
    threads.append(Thread(target=augment_images, args=(lst[1], n)))
    threads.append(Thread(target=augment_images, args=(lst[2], n)))
    threads_start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threads_end_time = time.time()
    threads_time = threads_end_time - threads_start_time
    print("All threads finished. Num of threads: " + str(len(threads)))
    print("All threads process time: " + str(threads_time) + " seconds")


if __name__ == '__main__':
    # random.seed(3)
    darksize = 704
    src_dir = 'D:/Datasets/laser_etch/helvetica_images/chars/'
    save_dir = 'D:/Datasets/laser_etch/helvetica_images/chars/augments/'
    norm_suffix = ''
    lite_suffix = '_lt'
    xlite_suffix = '_xlt'
    suffixes = [norm_suffix, lite_suffix, xlite_suffix]
    ext = '.png'
    num = 30
    # auger_threads(suffixes, num)
    create_mosaic()
