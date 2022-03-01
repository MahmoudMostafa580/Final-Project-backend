import cv2
import mediapipe as mp
import numpy as np

# height of person
person_height_in_centimeter = 175
person_height_in_pixels = round(person_height_in_centimeter * 37.795)

img = cv2.imread('taha3 (1).png')
(H, W, D) = img.shape
print(H)
print(W)
print(D)


def crop():
    for h in range(H - 1):
        for w in range(W - 1):
            if np.any(img[h, w, :] == 255):
                return h


def crop_2():
    for h in range(H - 1, 0, -1):
        for w in range(W - 1):
            if np.any(img[h, w, :] == 255):
                return h


W = W * 2
cropped = img[crop() - 1:crop_2(), 0:W]
resize_cropped = cv2.resize(cropped, (W, person_height_in_pixels))
cv2.imwrite('resized.jpg', resize_cropped)
