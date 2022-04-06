import random
import cv2
import numpy as np
import math


def make_chars_list():
    chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
             'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    return chars


def copy_file():
    chars = make_chars_list()
    direct = 'D:/Datasets/laser_etch/helvetica_images/chars/'
    filename = 'xlt_'
    ext = '.png'
    chars = make_chars_list()
    for c in chars:
        file = cv2.imread(direct + filename + c + ext)
        cv2.imwrite(direct + c + '_xlt' + '.png', file)
    print("Done")


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
    elif rando == 2:
        ksize = 5
    else:
        ksize = 1
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
    dilated = cv2.dilate(img, (3, 3))
    return dilated


def erode_img(src):
    img = src.copy()
    rando = random.randint(0, 2)
    if rando == 0:
        ksize = 3
    elif rando == 1:
        ksize = 5
    elif rando == 2:
        ksize = 7
    else:
        ksize = 1
        print("ksize = 1")
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
    angle = random.randint(0, 90)
    img = np.pad(img, ((x, x), (y, y), (0, 0)), mode='constant', constant_values=127)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    img = cv2.erode(img, (3, 3))
    img = cv2.dilate(img, (3, 3))
    img = increase_one_channel(img)
    img = lighten_dots(img)
    img = cv2.resize(img, (0, 0), fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_NEAREST)
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    unrot_mat = cv2.getRotationMatrix2D(img_center, -angle, 1.0)
    img = cv2.warpAffine(img, unrot_mat, img.shape[1::-1], flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
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
    rads = angle * (math.pi / 180)
    ox = int(math.cos(rads) * 3)
    oy = int(math.sin(rads) * 3)
    # iters = 5
    # crop = np.full_like(img, 0)
    # crop = cv2.line(crop, (0, x + ox + 1), (img.shape[1], x + ox + 1), (0, 0, 50), 1)
    # crop = cv2.line(crop, (0, x + h + ox + 3), (img.shape[1], x + h + ox + 3), (0, 0, 50), 1)
    # crop = cv2.line(crop, (y - oy, 0), (y - oy, img.shape[0]), (0, 0, 50), 1)
    # crop = cv2.line(crop, (y + w - oy + 2, 0), (y + w - oy + 2, img.shape[0]), (0, 0, 50), 1)
    # crop2 = np.full_like(img, 0)
    # crop2 = cv2.line(crop2, (0, x), (img.shape[1], x), (0, 50, 0), 1)
    # crop2 = cv2.line(crop2, (0, x + h), (img.shape[1], x + h), (0, 50, 0), 1)
    # crop2 = cv2.line(crop2, (y, 0), (y, img.shape[0]), (0, 50, 0), 1)
    # crop2 = cv2.line(crop2, (y + w, 0), (y + w, img.shape[0]), (0, 50, 0), 1)
    # img = img + crop + crop2

    # img = img[x:x+h+7, y-3:y+w+5+3].copy()
    top = x + ox + 1
    bottom = x + h + ox + 6
    left = y - oy
    right = y + w - oy + 2
    img = img[top:bottom, left:right].copy()
    return img


def sharpen(src, ksize=9):
    img = src.copy()
    img2 = cv2.GaussianBlur(img, (9, 9), cv2.BORDER_DEFAULT)
    img = cv2.addWeighted(img, 1.5, img2, -0.5, 0)
    return img


def trim(src):
    img = src.copy()
    row, col, ch = img.shape
    trim_lines = np.full((row, col, ch), 0, dtype=np.uint8)
    #  trim from top
    thresh1 = 100
    thresh2 = 255
    for r in range(row - 1 - 3):
        print("r=" + str(r))
        row1 = img[r, 0:col].copy()
        sigma3 = row1.mean() - 3 * row1.std()
        print("row1 sigma3=" + str(sigma3))
        print("row1.std=" + str(row1.std()))
        print("row1.mean=" + str(row1.mean()))
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
            print("grad_1sig=" + str(grad_1sig))
            print("grad std=" + str(gstd))
            print("grad mean=" + str(gmean))
            if gmean < thresh2:  # if gradient is low then it must not be an edge
                if r > row // 2:
                    trim_lines = cv2.line(trim_lines, (0, r + 3), (col, r + 3), (0, 0, 255), 1)
                else:
                    trim_lines = cv2.line(trim_lines, (0, r), (col, r), (0, 0, 255), 1)

                print("Drew line. r=" + str(r))
    for c in range(col - 1 - 3):
        print("c=" + str(c))
        col1 = img[0:row, c].copy()
        sigma3 = col1.mean() - 3 * col1.std()
        print("col1 sigma3=" + str(sigma3))
        print("col1.std=" + str(col1.std()))
        print("col1.mean=" + str(col1.mean()))
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
            print("grad_1sig=" + str(grad_1sig))
            print("grad std=" + str(gstd))
            print("grad mean=" + str(gmean))
            if gmean < thresh2:  # if gradient is low then it must not be an edge
                if c > col // 2:
                    trim_lines = cv2.line(trim_lines, (c + 3, 0), (c + 3, row), (0, 0, 255), 1)
                else:
                    trim_lines = cv2.line(trim_lines, (c, 0), (c, row), (0, 0, 255), 1)
                print("Drew line. c=" + str(c))
    img = cv2.addWeighted(img, 1, trim_lines, 1, 0)
    return img


def process_images_base():
    chars = make_chars_list()
    for c in chars:
        img = cv2.imread('../' + c + '.png')
        rotated = blotcher1(img)
        gnoise = gaussian_noise(rotated)
        resized = resize_img(gnoise)
        eroded = erode_img(resized)
        dilated = dilate_img(eroded)
        gblur = gaussian_blur(dilated)
        cv2.imwrite('auged_' + c + '.png', gblur)
        cv2.destroyAllWindows()


def process_images_1():
    #  Medium blotching, medium texturing, medium character thickness
    chars = make_chars_list()
    for c in chars:
        img = cv2.imread('../' + c + '.png')
        img = blotcher1(img)
        img = resize_img(img)
        img = erode_img(img)
        img = dilate_img(img)
        img = gaussian_blur(img)
        cv2.imwrite('auged_' + c + '.png', img)


def process_images_2():
    #  Medium blotching, medium texturing, thinner character thickness
    chars = make_chars_list()
    src_dir = 'D:/Datasets/laser_etch/helvetica_images/chars/'
    save_dir = 'D:/Datasets/laser_etch/helvetica_images/chars/augments/'
    norm_suffix = ''
    lite_suffix = '_lt'
    xlite_suffix = '_xlt'
    suffixes = [norm_suffix, lite_suffix, xlite_suffix]
    ext = '.png'
    for suffix in suffixes:
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
            img = dilate_img(img)
            img = cv2.dilate(img, (3, 3), iterations=3)
            img = cv2.erode(img, (3, 3), iterations=3)
            img = gaussian_blur(img)
            img = trim(img)
            cv2.imwrite(save_dir + '_auged_' + filename + '.png', img)
            print("Saved " + save_dir + '_auged_' + filename + '.png')


def testing():
    src_dir = 'D:/Datasets/laser_etch/helvetica_images/chars/'
    filename = 'T.png'
    img = cv2.imread(src_dir + filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    img = custom_thresh(img, 127, 127)

    h, w, _ = img.shape
    y = int(h / 2)
    x = int(w / 2)
    scale = 0.5
    angle = random.randint(0, 360)
    img = np.pad(img, ((x, x), (y, y), (0, 0)), mode='constant', constant_values=127)
    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    iters = 1
    img = cv2.erode(img, (3, 3), iterations=iters)
    img = cv2.dilate(img, (3, 3), iterations=iters)
    img = increase_one_channel(img)
    img = lighten_dots(img)

    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(img_center, angle, 1.0)
    img = cv2.warpAffine(img, rot_mat, img.shape[1::-1], flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))

    img = cv2.resize(img, (0, 0), fx=(1 / scale), fy=(1 / scale), interpolation=cv2.INTER_NEAREST)
    img_center = tuple(np.array(img.shape[1::-1]) / 2)
    unrot_mat = cv2.getRotationMatrix2D(img_center, -angle, 1.0)
    img = cv2.warpAffine(img, unrot_mat, img.shape[1::-1], flags=cv2.INTER_AREA,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
    img = cv2.line(img, (0, x), (img.shape[1], x), (0, 0, 255), 1)
    img = cv2.line(img, (0, x+h), (img.shape[1], x+h), (0, 0, 255), 1)
    img = cv2.line(img, (y, 0), (y, img.shape[0]), (0, 0, 255), 1)
    img = cv2.line(img, (y+w, 0), (y+w, img.shape[0]), (0, 0, 255), 1)
    img = cv2.line(img, (0, x+iters), (img.shape[1], x+iters), (0, 255, 0), 1)
    cv2.imshow('img' + str(angle), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    process_images_2()
