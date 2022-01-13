import cv2 as cv
import numpy as np

img = cv.imread('quad.jpg', 0)
image = cv.bitwise_not(img)
invert_image = cv.flip(image, 0)
invert_image = cv.bitwise_and(invert_image, image)
invert_image2 = cv.flip(image, 1)
invert_image = cv.bitwise_and(invert_image2, invert_image)

contours, hierarchy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

img_w, img_h = image.shape[::-1]

white1 = np.argwhere(image == 255)
print(len(white1), img_w*img_h)


for i, c in enumerate(contours):
    rect = cv.minAreaRect(c)

    (x, y), (w, h), a = rect
    if (h > 40) & (w > 40):
        box = cv.boxPoints(rect)
        box = np.int0(box)
        cv.drawContours(image, [box], 0, 100, 2)
        print(rect)


cv.imshow('image', img)

contours, hierarchy = cv.findContours(invert_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for i, c in enumerate(contours):
    rect = cv.minAreaRect(c)

    (x, y), (w, h), a = rect
    if (h > 40) & (w > 40):
        box = cv.boxPoints(rect)
        box = np.int0(box)
        print(rect)
        cv.drawContours(invert_image, [box], 0, 100, 2)


#cv.drawContours(invert_image, contours, -1, 100, 3)
cv.imshow('image_inv', invert_image)

cv.waitKey()
