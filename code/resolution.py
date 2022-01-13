import os
from collections import namedtuple

import cv2 as cv
import numpy as np
from scipy import signal
from scipy import stats
from scipy.signal import argrelextrema
from tensorflow import keras


def resize_img(img, size):
    resized_image = cv.resize(img, (size * 2, size))
    return resized_image


def rotate_image(img, degreesCCW=30, scaleFactor=1):
    (oldY, oldX) = img.shape  # note: numpy uses (y,x) convention but most OpenCV functions use (x,y)
    M = cv.getRotationMatrix2D(center=(oldX / 2, oldY / 2), angle=degreesCCW,
                               scale=scaleFactor)  # rotate about center of image.

    newX, newY = oldX * scaleFactor, oldY * scaleFactor
    r = np.deg2rad(degreesCCW)
    newX, newY = (abs(np.sin(r) * newY) + abs(np.cos(r) * newX), abs(np.sin(r) * newX) + abs(np.cos(r) * newY))
    (tx, ty) = ((newX - oldX) / 2, (newY - oldY) / 2)
    M[0, 2] += tx  # third column of matrix holds translation, which takes effect after rotation.
    M[1, 2] += ty

    rotatedImg = cv.warpAffine(img, M, dsize=(int(newX), int(newY)), borderValue=255)

    shift_x = int(tx / 1.5)
    shift_y = int(ty / 1.5)

    return rotatedImg[shift_y:int(newY) - shift_y, shift_x:int(newX) - shift_x]


def change_angle(angle):
    if abs(angle) > 45:
        if angle > 0:
            angle = angle - 90
        else:
            if angle < 0:
                angle = angle + 90
    return angle


def delete_outline(array, idx):
    del_outline = np.argwhere(abs(array[idx] - np.mean(array[idx])) > np.std(array[idx]) + 0.6)
    if len(del_outline) != 0:
        idx = np.delete(idx, del_outline)
    return idx


def cut_with_rect(image, rect):
    template_main_rect = [473, 1305, 300, 300]
    height = template_main_rect[3]
    padding = 30
    width_curr = (rect[2] + rect[3]) / 2
    pos = (template_main_rect[0] + padding, template_main_rect[1] + height + padding)
    k = width_curr / height
    w, h = image.shape[::-1]
    curr_pos = (rect[0], rect[1] + width_curr)
    shift_x = int(curr_pos[0] - pos[0] * k)
    shift_y = int(curr_pos[1] - pos[1] * k)
    n_xw = int(pos[1] * k + shift_x + padding)
    n_xh = int(pos[1] * k + shift_y + padding)

    if (shift_x < 0) | (shift_y < 0):
        new_img = np.full((n_xw, n_xh), 255, dtype=np.uint8)
        x_add = 0
        y_add = 0
        if shift_x < 0:
            x_add = -shift_x
            shift_x = 0
        if shift_y < 0:
            y_add = -shift_y
            shift_y = 0
        if (shift_x > w) or (shift_y > h):
            return new_img[y_add: n_xh + y_add, x_add: n_xw + x_add]
        if (n_xw > w):
            n_xw = w - shift_x
        if (n_xh > h):
            n_xh = h - shift_y

        return image[shift_y: n_xh + shift_y, shift_x: n_xw + shift_x].copy()

    return image[shift_y: n_xh, shift_x: n_xw]


def delete_out(array, idx_min, idx_max):
    for j in range(2):
        if len(idx_max) > 1 and len(idx_min) > 1:
            all_idx = np.hstack((idx_min, idx_max))
            std = []
            for i in range(len(all_idx)):
                part = np.delete(all_idx, i)
                std.append(np.std(array[part]))
            std = np.asarray(std)
            if std.min() != 0 and std.max() != 0:
                if std.min() / std.max() < 0.4:
                    out_idx = std.argmin()
                    if out_idx < len(idx_min):
                        idx_min = np.delete(idx_min, out_idx)
                    else:
                        idx_max = np.delete(idx_max, out_idx - len(idx_min))

    return idx_min, idx_max


def clear_array(array, idx_min, idx_max):
    idx_min, idx_max = delete_out(array, idx_min, idx_max)

    if len(idx_min) != 0:
        idx_min = delete_outline(array, idx_min)
    if len(idx_max) != 0:
        idx_max = delete_outline(array, idx_max)

    min_arg = array[idx_min].argmin()
    max_arg = array[idx_max].argmax()
    mean = (array[idx_min][min_arg] + array[idx_max][max_arg]) / 2

    del_min = np.argwhere(array[idx_min] > mean)
    del_max = np.argwhere(array[idx_max] < mean)

    if len(del_min) != 0:
        idx_min = np.delete(idx_min, del_min)

    if len(del_max) != 0:
        idx_max = np.delete(idx_max, del_max)

    return idx_min, idx_max


def find_thresholds(img, order, axs=None, k=None, ws=None):
    less_part = argrelextrema(img, np.less, order=order)[0]
    great_part = argrelextrema(img, np.greater, order=order)[0]
    diff = 0
    x = np.linspace(0, len(img))
    if (less_part.size != 0) & (great_part.size != 0):
        less_part, great_part = clear_array(img, less_part, great_part)
        great_part = np.append(great_part, img.argmax())
        if axs:
            axs.plot(img)
            axs.plot(less_part, img[less_part], 'x')
            axs.plot(great_part, img[great_part], 'x')
        diff = -1
        if len(less_part) != 0 and len(great_part) != 0:
            diff = (np.mean(img[great_part]) - np.mean(img[less_part])) / (
                    np.mean(img[great_part]) + np.mean(img[less_part])) * 100
            if ws is not None:
                ws.write(k, 2, np.mean(img[less_part]))
                ws.write(k, 3, np.mean(img[great_part]))
    return diff


def find_shift(rect_min, rect_max):
    if rect_min == None or rect_max == None:
        return 0, 0, 0
    temp_min = (552, 1026, 77, 76)
    temp_max = (503, 1335, 300, 300)
    x_max, y_max, w_max, h_max = rect_max
    x_min, y_min, w_min, h_min = rect_min
    scale = abs(x_max - x_min) / abs(temp_max[0] - temp_min[0]) + abs(y_max - y_min) / abs(
        temp_max[1] - temp_min[1]) + (w_max + h_max) / (temp_max[2] + temp_max[3]) + (w_min + h_min) / (
                    temp_min[2] + temp_min[3])
    scale = scale / 4
    center_max = (x_max + w_max / 2, y_max + h_max / 2)
    center_min = (x_min + w_min / 2, y_min + h_min / 2)
    center_max_t = (temp_max[0] + temp_max[2] / 2, temp_max[1] + temp_max[3] / 2)
    center_min_t = (temp_min[0] + temp_min[2] / 2, temp_min[1] + temp_min[3] / 2)
    x_shift = (center_max_t[0] * scale - center_max[0] + center_min_t[0] * scale - center_min[0]) / 2
    y_shift = (center_max_t[1] * scale - center_max[1] + center_min_t[1] * scale - center_min[1]) / 2
    return int(x_shift), int(y_shift), scale


def correct_rectangle(image, rect):
    part = image[rect[1]:rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    white1 = np.argwhere(part == 0)
    correction = int(len(white1) / ((rect[2] + rect[3]) * 2))
    return (rect[0] + correction, rect[1] + correction, rect[2] - correction * 2, rect[3] - correction * 2)


Location = namedtuple("Location", ["id", "x", "y"])

LOCATIONS = [
    Location(0, [203, 204, 205, 528, 659, 791], [79, 209, 210, 211, 211, 342]),
    Location(1, [69, 186, 304, 602, 602, 602], [82, 199, 201, 201, 202, 317]),
    Location(2, [74, 179, 284, 548, 548, 548], [68, 174, 175, 175, 175, 279]),
    Location(3, [69, 163, 258, 492, 493, 493], [70, 164, 165, 165, 166, 259]),
    Location(4, [57, 141, 224, 433, 433, 433], [60, 143, 143, 143, 144, 226]),
    Location(5, [48, 123, 197, 384, 384, 384], [49, 122, 122, 123, 123, 197]),
    Location(6, [102, 102, 102, 268, 334, 401], [47, 113, 113, 113, 113, 179]),
    Location(7, [91, 92, 93, 241, 300, 359], [49, 107, 108, 108, 108, 167]),
    Location(8, [84, 84, 84, 215, 268, 321], [38, 90, 91, 91, 91, 144]),
    Location(9, [76, 76, 76, 194, 241, 287], [40, 86, 86, 87, 87, 134]),
    Location(10, [69, 70, 70, 174, 216, 258], [32, 73, 73, 74, 74, 118])
]


def write_image(image_name, image):
    f = open(image_name, 'wb')
    img_str = cv.imencode('.png', image)[1].tobytes()
    f.write(img_str)
    f.close()


def normalize(pixels):
    pixels = np.asarray(pixels)
    pixels = pixels.astype('float32')
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    # clip pixel values to [-1,1]
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0
    return pixels


class Resolution:
    name = None
    image = None
    image_with_template = None
    thresh = None
    _template_config = None
    _template = None
    _table = None
    _img_w = 0
    _img_h = 0
    _angle = 0
    _rect_min = None
    _rect_max = None
    thresholds = []
    probs = []
    model = None
    vertical_resolution = None
    horizontal_resolution = None
    resolution = None
    x_add = 0
    y_add = 0
    scale = 0
    Error = False

    def __init__(self, filename):

        stream = open(filename, 'rb')
        bytes = bytearray(stream.read())
        array = np.asarray(bytes, dtype=np.uint8)
        bgrImage = cv.imdecode(array, cv.IMREAD_GRAYSCALE)
        self.name = filename[filename.rfind('/') + 1:filename.rfind('.')]
        self.name = self.name.rstrip()
        self.image = bgrImage

        if type(self.image) == None:
            raise FileNotFoundError
        self._template_config = np.loadtxt('template_config.txt', dtype=int)

        self._template = np.loadtxt('template.txt', dtype=int)
        self.model = keras.models.load_model('./model/model_newreg')
        self._table = np.loadtxt('table.txt')
        self._table = self._table.flatten(order='F')
        blured = cv.blur(self.image, (3, 3))
        self._img_w, self._img_h = self.image.shape[::-1]
        ret, self.thresh = cv.threshold(blured, np.mean(blured) - blured.min() / 2 - 2, 255, cv.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        self.thresh = cv.dilate(self.thresh, kernel, iterations=2)
        self.thresh = cv.erode(self.thresh, kernel, iterations=1)

    def find_main_rect(self, both=True):
        thresh_ = cv.bitwise_not(self.thresh)
        contours, hierarchy = cv.findContours(thresh_.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        out = self.thresh.copy()
        max_ = 0
        max_i = 0
        rects = []
        centers = []

        if len(contours) < 2:
            print('no contours')
            self.Error = True
            return None, None
        for i, c in enumerate(contours):
            rect = cv.boundingRect(c)
            x, y, w, h = rect
            if (h > self._img_w * 0.02) & (w > self._img_h * 0.02):
                if abs(h - w) < self._img_w * 0.01:
                    centers.append((x, y))
                    rects.append(rect)
                    cv.rectangle(out, (x, y), (x + w, y + h), 220, 3)

                    if max_ < h + w:
                        max_ = h + w
                        max_i = i

        rect_max = cv.boundingRect(contours[max_i])
        rect_max = correct_rectangle(thresh_, rect_max)
        if not both:
            return rect_max, None

        if both:
            x_max, y_max, w_max, h_max = rect_max

            cv.rectangle(out, (x_max, y_max), (x_max + w_max, y_max + h_max), 10, 2)

            length = (w_max + h_max) / 2
            idx = []
            for i, cntr in enumerate(centers):
                if (abs(x_max - cntr[0]) == 0) & (abs(y_max - cntr[1]) == 0):
                    idx.append(i)
                    continue
                if abs(x_max - cntr[0]) > length / 3:
                    idx.append(i)
                    continue
                if abs(y_max - cntr[1]) > length * 1.5:
                    idx.append(i)
                    continue
                if (rects[i][2] + rects[i][3]) / 2 < length * 0.2:
                    idx.append(i)
                    continue
                if (rects[i][2] + rects[i][3]) / 2 > length * 0.5:
                    idx.append(i)
                    continue
            # cv.imwrite('out.jpg', out)
            for i in reversed(idx):
                rects.pop(i)
                centers.pop(i)  # убрать
            # если остался не один квадрат, выдавать сообщение о невозможности определить
            if len(rects) != 1:
                self.Error = True
                print('Error: not only one quad')
                print(len(rects))

            if len(rects) == 0:
                return rect_max, None

            self._rect_min = rects[0]
            self._rect_min = correct_rectangle(thresh_, self._rect_min)

            self._rect_max = rect_max
            return self._rect_max, self._rect_min

    def _main_rotate(self):
        thresh_ = cv.bitwise_not(self.thresh)
        contours, hierarchy = cv.findContours(thresh_.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_ = 0
        max_i = 0
        rects = []
        centers = []

        for i, c in enumerate(contours):
            rect = cv.minAreaRect(c)
            (x, y), (w, h), a = rect
            if (h > self._img_w * 0.02) & (w > self._img_h * 0.02):
                if abs(h - w) < self._img_w * 0.005:
                    centers.append((x, y))
                    rects.append(rect)

                    if max_ < h + w:
                        max_ = h + w
                        max_i = i

        rect_max = cv.minAreaRect(contours[max_i])
        (x_max, y_max), (w_max, h_max), a_max = rect_max
        length = (w_max + h_max) / 2 * 1.5
        idx = []
        for i, cntr in enumerate(centers):
            if (abs(x_max - cntr[0]) == 0) & (abs(y_max - cntr[1]) == 0):
                idx.append(i)
                continue
            if abs(x_max - cntr[0]) > length:
                idx.append(i)
                continue
            if abs(y_max - cntr[1]) > length:
                idx.append(i)
                continue

        for i in reversed(idx):
            rects.pop(i)
            centers.pop(i)  
        # если остался не один квадрат, выдавать сообщение о невозможности определить
        if len(rects) < 1:
            print('Error')
            self.Error = True
            return None

        rect_min = rects[0]
        (x_min, y_min), (w_min, h_min), a_min = rect_min

        angle = 0
        if y_max - y_min > 0:
            if x_max - x_min < 0:
                angle += 90
        else:
            if x_max - x_min < 0:
                angle += 180
            else:
                angle -= 90
        addition = np.arctan(abs((x_max - x_min) / (y_max - y_min)))
        angle -= addition * 180 / np.pi

        if angle != 0:
            self.image = rotate_image(self.image, angle)
            self.thresh = rotate_image(self.thresh, angle)
            self._img_w, self._img_h = self.image.shape[::-1]

        return angle

    def align_image(self):
        self._main_rotate()
        thresh_ = cv.bitwise_not(self.thresh)

        contours, hierarchy = cv.findContours(thresh_.copy(), cv.RETR_EXTERNAL,
                                              cv.CHAIN_APPROX_SIMPLE)
        angle_l = []

        for i, c in enumerate(contours):
            x, y, w, h = cv.boundingRect(c)
            if (h > 50) & (w > 50):
                rect = cv.minAreaRect(c)
                angle = rect[2]
                if abs(angle) > 45:
                    if angle > 0:
                        angle = angle - 90
                    else:
                        if angle < 0:
                            angle = angle + 90
                angle_l.append(angle)
        self._angle = np.mean(signal.medfilt(angle_l))

        if self._angle != 0:
            self.image = rotate_image(self.image, self._angle)
            self.thresh = rotate_image(self.thresh, self._angle)
            self._img_w, self._img_h = self.image.shape[::-1]

    def find_extreme(self, img, config, k=None, ws=None):

        if img.max() == img.min():
            return -1, -1
        rows, cols = np.shape(img)
        left_diff = 0
        right_diff = 0
        cols_part = int(cols / 2)
        part1 = img[0:-1, 0:cols_part].copy()
        part2 = img[0:-1, cols_part:-1].copy()
        part_cols = int(cols / 2)
        if (config[0] == 0) | (config[0] == 1):
            white1 = np.argwhere(part1 == 255)
            white2 = np.argwhere(part2 == 255)
            if len(white1) > 0.0001 * rows * cols:
                left_diff = -1
            if len(white2) > 0.0001 * rows * cols:
                right_diff = -1
            if cols < self._template[k][2] * 0.5:
                left_diff = -1
                right_diff = -1
            elif cols < self._template[k][2] * 0.8:
                right_diff = -1

        order = (7 - config[0])
        if config[0] == 5:
            order = 3
        if config[0] == 2:
            order = 5
        if config[0] == 0:
            order = 10
            
        if left_diff != -1:
            if config[2] == 1:  # vertical horizontal
                if ws is not None:
                    ws.write(2 * k + 2, 1, str('vertical'))
                left_img = [sum(img[i][j] for i in range(rows)) for j in range(0, part_cols)]  # по столбцам 1
                left_img = 255 - np.asarray(left_img) / rows
                left_diff = find_thresholds(left_img, order, k=2 * k + 2, ws=ws)
            if config[2] == 0:
                if ws is not None:
                    ws.write(2 * k + 1, 1, str('horizontal'))
                left_img = [sum(img[i][j] for j in range(0, part_cols)) for i in range(rows)]  # по строчкам 1
                left_img = 255 - np.asarray(left_img) / cols_part
                left_diff = find_thresholds(left_img, order, k=2 * k + 1, ws=ws)
                
        if right_diff != -1:
            if config[2] == 1:  # vertical horizontal
                if ws is not None:
                    ws.write(2 * k + 1, 1, str('horizontal'))
                right_img = [sum(img[i][j] for j in range(part_cols, cols)) for i in range(rows)]  # по строчкам 2
                right_img = 255 - np.asarray(right_img) / cols_part
                right_diff = find_thresholds(right_img, order, k=2 * k + 1, ws=ws)                
            if config[2] == 0:
                if ws is not None:
                    ws.write(2 * k + 2, 1, str('vertical'))
                right_img = [sum(img[i][j] for i in range(rows)) for j in range(part_cols, cols)]  # по столбцам 2
                right_img = 255 - np.asarray(right_img) / rows
                right_diff = find_thresholds(right_img, order, k=2 * k + 2, ws=ws)

        if config[2] == 1: 
            return left_diff, right_diff
        else:
            return right_diff, left_diff

    def correction_shift_micro(self):
        blured = cv.blur(self.image, (3, 3))
        ret, thresh_t = cv.threshold(blured, np.mean(blured) * 0.9, 255, cv.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        thresh_t = cv.dilate(thresh_t, kernel, iterations=1)
        x_add_t, y_add_t = [], []
        for i, zone in enumerate(self._template):
            if i >= 12 and i <= 14:
                x = zone[0] - self.x_add
                if x < 0:
                    x = 0

                y = zone[1] - self.y_add
                if y < 0:
                    y = 0

                part_thresh = cv.bitwise_not(thresh_t[y: y + zone[3], x: x + zone[2]])
                contours, hierarchy = cv.findContours(part_thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                x_centers = []
                y_centers = []

                list_of_pts = []
                for j, c in enumerate(contours):
                    (x_, y_), (w_, h_), a_ = cv.minAreaRect(c)
                    if w_ > 10 and h_ > 10:
                        # объединение контуров
                        list_of_pts += [pt[0] for pt in c]
                        x_centers.append(int(x_))
                        y_centers.append(int(y_))
                ctr = np.array(list_of_pts).reshape((-1, 1, 2)).astype(np.int32)
                ctr = cv.convexHull(ctr)  # done.
                x_r, y_r, w_r, h_r = cv.boundingRect(ctr)
                if i == 12:
                    x_add_t.append((x_r + w_r / 2) * self.scale - 113)
                    y_add_t.append((y_r + h_r / 2) * self.scale - 58)
                if i == 13:
                    x_add_t.append((x_r + w_r / 2) * self.scale - 106)
                    y_add_t.append((y_r + h_r / 2) * self.scale - 53.5)
                if i == 14:
                    x_add_t.append((x_r + w_r / 2) * self.scale - 92)
                    y_add_t.append((y_r + h_r / 2) * self.scale - 49)

        corr_x, corr_y = 0, 0
        corr_x += int(np.mean(x_add_t))
        corr_y += int(np.mean(y_add_t))
        return corr_x, corr_y

    def correction_shift_macro(self):
        x_add_t, y_add_t = [], []
        for i, zone in enumerate(self._template):
            x = zone[0] - self.x_add
            if x < 0:
                x = 0

            y = zone[1] - self.y_add
            if y < 0:
                y = 0

            white = np.argwhere(self.image[y: y + zone[3], x: x + zone[2]] == 255)

            if len(white) < 10:
                part_thresh = cv.bitwise_not(self.thresh[y: y + zone[3], x: x + zone[2]])
                contours, hierarchy = cv.findContours(part_thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                x_centers = []
                y_centers = []
                x_corr, y_corr = 0, 0

                if i < 11:
                    if len(contours) >= 6:
                        test = self.image.copy()[y: y + zone[3], x: x + zone[2]]
                        for j, c in enumerate(contours):
                            x_, y_, w_, h_ = cv.boundingRect(c)
                            if w_ > 20 and h_ > 20:
                                cv.rectangle(test, (x_, y_), (x_ + w_, y_ + h_), (0, 255, 0), 2)
                                x_centers.append(int(x_ + w_ // 2))
                                y_centers.append(int(y_ + h_ // 2))

                    x_centers.sort()
                    y_centers.sort()

                    if len(x_centers) != 6 or len(y_centers) != 6:
                        x_corr, y_corr = 0, 0
                    if len(x_centers) == 6 and len(y_centers) == 6:
                        x_add = np.mean(np.asarray(LOCATIONS[i].x) * self.scale - np.asarray(x_centers))
                        y_add = np.mean(np.asarray(LOCATIONS[i].y) * self.scale - np.asarray(y_centers))
                        if zone[0] - self.x_add < 0:
                            x_add -= zone[0] - self.x_add
                        if zone[1] - self.y_add < 0:
                            y_add -= zone[1] - self.y_add
                        x_corr, y_corr = int(x_add), int(y_add)
                    x_add_t.append(x_corr)
                    y_add_t.append(y_corr)
        return int(np.mean(x_add_t)), int(np.mean(y_add_t))

    def pre_processing(self):
        self.main_rect, self.min_rect = self.find_main_rect(both=False)
        if self.main_rect != None:
            self.image = cut_with_rect(self.image, self.main_rect)
            self.thresh = cut_with_rect(self.thresh, self.main_rect)
            self._img_w, self._img_h = self.image.shape[::-1]
            self.main_rect, self.min_rect = self.find_main_rect(both=True)

        if self.min_rect != None:
            self.x_add, self.y_add, self.scale = find_shift(self.min_rect, self.main_rect)
            self._template = np.int0(self._template * self.scale)
        self.image_with_template = self.image.copy()  
        return self.Error

    def save_for_ds(self):

        horizontal_idx = -1
        vertical_idx = -1
        if self.name.rfind('Г') != -1 and self.name.rfind('В') != -1:
            horizontal = self.name[self.name.rfind('Г') + 1:self.name.rfind('Г') + 4]
            vertical = self.name[self.name.rfind('В') + 1:self.name.rfind('В') + 4]
            horizontal_idx = int(horizontal[0]) * 6 + int(horizontal[-1]) - 1
            vertical_idx = int(vertical[0]) * 6 + int(vertical[-1]) - 1
        else:
            return 0

        path = os.getcwd()
        print("The current working directory is %s" % path)
        res_path = path + '\\results\\' + self.name

        try:
            os.makedirs(res_path, exist_ok=True)

        except OSError:
            print("Creation of the directory %s failed" % res_path)
        else:
            print("Successfully created the directory %s " % res_path)


        x_corr_micro, y_corr_micro = self.correction_shift_micro()
        for i, zone in enumerate(self._template):
            if i > 11 and i < 24:
                x = zone[0] - self.x_add + x_corr_micro
                if x < 0:
                    x = 0

                y = zone[1] - self.y_add + y_corr_micro
                if y < 0:
                    y = 0
                shiftx, shifty = 0, 0
                if i < 16:
                    tmp, shiftx, shifty = self.check_boders(self.image, x, y, zone)
                    tmp = self.image[y + shifty: y + shifty + zone[3], x + shiftx: x + shiftx + zone[2]]
                    x_corr_micro += int(shiftx * 0.2)
                    y_corr_micro += int(shifty * 0.2)
                    tmp = resize_img(tmp, 115)
                else:
                    tmp = self.image[y: y + zone[3], x: x + zone[2]]
                    tmp = resize_img(tmp, 115)

                img1 = tmp[:115, : 115]
                img2 = tmp[:115, 115:]
                if self._template_config[i, 2] == 0:  # horizontal vertical
                    if horizontal_idx != -1:
                        if i <= horizontal_idx:
                            write_image(res_path + '\\' + f'imgh_{i}result1.png', img1)
                        else:
                            write_image(res_path + '\\' + f'imgh_{i}_result0.png', img1)
                    else:
                        write_image(res_path + '\\' + f'imgh_{i}_resultNone1.png', img1)

                    img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
                    if vertical_idx != -1:
                        if i <= vertical_idx:
                            write_image(res_path + '\\' + f'imgv_{i}_result1.png', img2)
                        else:
                            write_image(res_path + '\\' + f'imgv_{i}_result0.png', img2)
                    else:
                        write_image(res_path + '\\' + f'imgv_{i}_resultNone2.png', img2)

                else:
                    if horizontal_idx != -1:
                        if i <= horizontal_idx:
                            write_image(res_path + '\\' + f'imgh_{i}_result1.png', img2)
                        else:
                            write_image(res_path + '\\' + f'imgh_{i}_result0.png', img2)
                    else:
                        write_image(res_path + '\\' + f'imgh_{i}_resultNone2.png', img2)

                    img1 = cv.rotate(img1, cv.ROTATE_90_CLOCKWISE)
                    if vertical_idx != -1:
                        if i <= vertical_idx:
                            write_image(res_path + '\\' + f'imgv_{i}_result1.png', img1)
                        else:
                            write_image(res_path + '\\' + f'imgv_{i}_result0.png', img1)
                    else:
                        write_image(res_path + '\\' + f'imgv_{i}_resultNone1.png', img1)

    def find_resolution_old(self, threshold=10):
        if len(self.thresholds) == 0:
            thresholds = []
            self.image_with_template = self.image.copy()
            x_corr_macro, y_corr_macro = self.correction_shift_macro()
            x_corr_micro, y_corr_micro = self.correction_shift_micro()

            x_corr_macro = (x_corr_micro + x_corr_macro) // 2
            y_corr_macro = (y_corr_micro + y_corr_macro) // 2
            for i, zone in enumerate(self._template):
                if i < 12:
                    x_corr = x_corr_macro
                    y_corr = y_corr_macro
                else:
                    x_corr = x_corr_micro
                    y_corr = y_corr_micro

                x = zone[0] - self.x_add + x_corr
                if x < 0:
                    x = 0

                y = zone[1] - self.y_add + y_corr
                if y < 0:
                    y = 0
                thresholds.append(
                    self.find_extreme(self.image[y: y + zone[3], x: x + zone[2]], self._template_config[i], i))
                cv.rectangle(self.image_with_template, (x, y), (x + zone[2], y + zone[3]), 20, 2)
            vert_data = []
            horiz_data = []
            for i, thresh in enumerate(thresholds):
                if thresh[0] != -1 and i < 17:
                    vert_data.append((i, thresh[0]))
                if thresh[1] != -1 and i < 17:
                    horiz_data.append((i, thresh[1]))
            vert_data = np.asarray(vert_data)
            horiz_data = np.asarray(horiz_data)
            regress_vert = stats.linregress(vert_data[:, 0], vert_data[:, 1])
            regress_horiz = stats.linregress(horiz_data[:, 0], horiz_data[:, 1])
            vert_predict = np.arange(len(thresholds)) * regress_vert.slope + regress_vert.intercept
            horiz_predict = np.arange(len(thresholds)) * regress_horiz.slope + regress_horiz.intercept
            vert_mask = np.where(vert_predict > -2, 1, 0)
            horiz_mask = np.where(horiz_predict > -2, 1, 0)
            thresholds = np.asarray(thresholds)
            thresholds[:, 0] = np.multiply(thresholds[:, 0], vert_mask)
            thresholds[:, 1] = np.multiply(thresholds[:, 1], horiz_mask)

            self.thresholds = thresholds

        thresholds = self.thresholds
        length = len(self._template)
        last_vert = length
        last_horiz = length
        i = 0
        vert_stop = length
        hor_stop = length
        for t_left, t_right in thresholds:
            if (t_left != -1) & (vert_stop == length):
                if (t_left < threshold) & (i < last_vert):
                    last_vert = i
                if (t_left < 0.9 * threshold):
                    vert_stop = i
                    continue
            if (t_right != -1) & (hor_stop == length):
                if (t_right < threshold) & (i < last_horiz):
                    last_horiz = i
                if (t_right < 0.9 * threshold):
                    hor_stop = i
                    continue
             
            i += 1

        self.vertical_resolution = (self._template_config[last_vert, 0], self._template_config[last_vert, 1])
        self.horizontal_resolution = (self._template_config[last_horiz, 0], self._template_config[last_horiz, 1])
        self.resolution = (self._table[last_horiz] + self._table[last_vert]) / 2
        return thresholds

    def align_line(self, image, x, y, zone, thresh=15, shift=2, rotate=False):
        img = self.image[y: y + zone[3], x: x + zone[2]].copy()
        img = resize_img(img, 115)
        if rotate:
            img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
        top = img[0, :]
        bottom = img[-1, :]
        left = img[:, 0]
        right = img[:, -1]
        left_std = np.std(left)
        right_std = np.std(right)
        move_left = left_std > thresh
        move_right = right_std > thresh
        if abs(shift) > 15:
            return img, shift
        if not move_left and not move_right:
            if rotate:
                img = cv.rotate(img, cv.ROTATE_90_COUNTERCLOCKWISE)
            if shift == 5:
                return img, 0
            return img, shift
        if move_left:
            if move_right:
                img, shift = self.align_line(image, x + abs(shift), y, zone, shift=abs(shift) + 2, rotate=False)
                img, shift = self.align_line(image, x - abs(shift), y, zone, shift=-(abs(shift) + 2), rotate=False)
            else:
                img, shift = self.align_line(image, x - abs(shift), y, zone, shift=-(abs(shift) + 2), rotate=False)
        elif move_right:
            img, shift = self.align_line(image, x + abs(shift), y, zone, shift=abs(shift) + 2, rotate=False)
        return img, shift

    def check_boders(self, image, x, y, zone, shiftx=0, shifty=0):
        img = self.image[y: y + zone[3], x: x + zone[2]].copy()
        img = resize_img(img, 115)
        img, shiftx = self.align_line(image, x, y, zone, rotate=False)
        img, shifty = self.align_line(image, x + shiftx, y, zone, rotate=True)
        return img, shiftx, shifty

    def find_resolution_network(self, threshold=0.5):
        x_corr_micro, y_corr_micro = self.correction_shift_micro()
        data_set_h = []
        data_set_v = []
        self.image_with_template = self.image.copy()
        for i, zone in enumerate(self._template):
            if i >= 12 and i<24:
                x_corr = x_corr_micro
                y_corr = y_corr_micro

                x = zone[0] - self.x_add + x_corr
                if x < 0:
                    x = 0

                y = zone[1] - self.y_add + y_corr
                if y < 0:
                    y = 0
                shiftx, shifty = 0, 0
                if i < 16:
                    tmp, shiftx, shifty = self.check_boders(self.image, x, y, zone)
                    tmp = self.image[y + shifty: y + shifty + zone[3], x + shiftx: x + shiftx + zone[2]]
                    x_corr_micro += int(shiftx * 0.2)
                    y_corr_micro += int(shifty * 0.2)
                    tmp = resize_img(tmp, 115)

                else:
                    tmp = self.image[y: y + zone[3], x: x + zone[2]]
                    tmp = resize_img(tmp, 115)

                cv.rectangle(self.image_with_template, (x + shiftx, y + shifty),
                             (x + zone[2] + shiftx, y + shifty + zone[3]), 20, 2)

                img1 = tmp[:115, : 115]
                img2 = tmp[:115, 115:]

                if (self._template_config[i, 0] == 0) | (self._template_config[i, 0] == 1):
                    white1 = np.argwhere(img1 == 255)
                    white2 = np.argwhere(img2 == 255)
                    if len(white1) > 5:
                        continue
                    if len(white2) > 5:
                        continue

                if self._template_config[i, 3] == 0:
                    img1 = cv.rotate(img1, cv.ROTATE_90_CLOCKWISE)
                    data_set_h.append(normalize(img2.copy()))
                    data_set_v.append(normalize(img1.copy()))

                if self._template_config[i, 2] == 0:
                    img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
                    data_set_h.append(normalize(img1.copy()))
                    data_set_v.append(normalize(img2.copy()))

        data_set_h = np.expand_dims(data_set_h, -1)
        data_set_v = np.expand_dims(data_set_v, -1)
        probs_h = self.model.predict(data_set_h)
        probs_v = self.model.predict(data_set_v)
        self.probs = np.concatenate((probs_h, probs_v))
        horiz = probs_h
        vert = probs_v
        vert_01 = np.where(vert < threshold, 1, 0)
        horiz_01 = np.where(horiz < threshold, 1, 0)
        self.vertical_resolution = (
            self._template_config[vert_01[2:].argmax() + 13][0], self._template_config[vert_01[2:].argmax() + 13][1])
        self.horizontal_resolution = (
            self._template_config[horiz_01[2:].argmax() + 13][0], self._template_config[horiz_01[2:].argmax() + 13][1])  

        return horiz, vert

    def get_table_resolution(self, point):
        i, j = point
        if (i * 6 + j - 1) < len(self._table):
            return self._table[i * 6 + j - 1]
        return None

    def get_image(self):
        return self.image

    def get_image_part(self, i, j):
        return self.image

    def find_mean(self):
        x_corr_macro, y_corr_macro = self.correction_shift_macro()
        mean_part = np.array([344, 471, 562, 192])
        mean_part = np.int0(mean_part * self.scale)
        mean_part[0] = mean_part[0] - self.x_add
        mean_part[1] = mean_part[1] - self.y_add
        cv.rectangle(self.image_with_template, (mean_part[0], mean_part[1]),
                     (mean_part[0] + mean_part[2], mean_part[1] + mean_part[3]), 50, 2)
        return np.mean(self.image[mean_part[1]:mean_part[1] + mean_part[3], mean_part[0]: mean_part[0] + mean_part[2]])


import glob

if __name__ == "__main__":
    resolution = Resolution('D:/Python/resolution/source/моя оценка/1020495_Г3-3_В3-3.bmp')
    resolution.align_image()
    resolution.pre_processing()
    resolution.find_resolution_network()
    all_ = glob.glob(".\\source\\моя оценка\\*")
    print(glob.glob(".\\source\\моя оценка\\*"))
    for pack in all_:
        resolution = Resolution(pack)
        resolution.align_image()
        if not resolution.pre_processing():
            resolution.save_for_ds()
