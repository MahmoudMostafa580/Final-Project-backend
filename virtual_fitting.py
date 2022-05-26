import cv2
from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile

person_image = "person_image.jpg"
cloth_image = "cloth_image.png"


def remove_bg(person_img, cloth_img):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(person_img)
    result_p = remove(f)
    img_p = Image.open(io.BytesIO(result_p)).convert("RGBA")
    img_p.save("person_no_bg.png")
    person_no_bg = cv2.imread("person_no_bg.png")
    person_copy = person_no_bg.copy()

    (height, width, channel) = person_no_bg.shape
    if height < width:
        person_no_bg = cv2.rotate(person_no_bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    person_mask = convert_crop(person_no_bg)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(cloth_img)
    result_c = remove(f)
    img_c = Image.open(io.BytesIO(result_c)).convert("RGBA")
    img_c.save("cloth_no_bg.png")
    cloth_no_bg = cv2.imread("cloth_no_bg.png")
    cloth_copy = cloth_no_bg.copy()

    (height, width, channel) = cloth_no_bg.shape
    if height < width:
        cloth_no_bg = cv2.rotate(cloth_no_bg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cloth_mask = convert_crop(cloth_no_bg)

    return person_copy, person_mask, cloth_copy, cloth_mask


def convert_crop(img):
    (H, W, D) = img.shape
    # convert image to black and white
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(img != [0, 0, 0], axis=-1)
    img[black_pixels_mask] = [255, 255, 255]
    img[non_black_pixels_mask] = [0, 0, 0]
    h1 = crop(img)
    h2 = crop_2(img)
    img_mask = img[h1:h2, 0:W]

    return img_mask


# crop image from top
def crop(converted_image):
    for h in range(converted_image.shape[0] - 1):
        for w in range(int(converted_image.shape[1] / 4), int(converted_image.shape[1] * (3 / 4)), 1):
            if np.any(converted_image[h, w, :] == 0):
                return h


# crop image from bottom
def crop_2(converted_image):
    for h in range(converted_image.shape[0] - 1, 0, -1):
        for w in range(int(converted_image.shape[1] / 4), int(converted_image.shape[1] * (3 / 4)), 1):
            if np.any(converted_image[h, w, :] == 0):
                return h


def get_person_parts(p_image):
    (HEIGHT, WIDTH, CHANNEL) = p_image.shape
    # height ration between original height of person & height of image
    # height_ratio = person_height / HEIGHT

    # calculate start & end of the 8 parts of human body
    # part1_start = 0
    part1_end = round(HEIGHT * (1 / 8))

    part2_start = part1_end + 1
    part2_end = round(HEIGHT * (2 / 8))

    part3_start = part2_end + 1
    part3_end = round(HEIGHT * (3 / 8))

    part4_start = part3_end + 1
    part4_end = round(HEIGHT * (4 / 8))

    part5_start = part4_end + 1
    part5_end = round(HEIGHT * (5 / 8))

    # part6_start = part5_end + 1
    # part6_end = round(HEIGHT * (6 / 8))

    # part7_start = part6_end + 1
    # part7_end = round(HEIGHT * (7 / 8))

    # part8_start = part7_end + 1
    part8_end = HEIGHT

    # shoulder_line_height = round(part2_start + ((part2_end - part2_start) / 2))

    # calculate needed lengths
    chest_line_height = round(part3_start + ((part3_end - part3_start) / 4))
    waist_line_height = round(part4_start + ((part4_end - part4_start) / 4))
    hip_line_height = part5_start
    upper_body_length = round((part5_start + ((part5_end - part5_start) / 2)) - \
                              (part2_start + ((part2_end - part2_start) / 3)))
    inside_leg_length = round(part8_end - (part5_start + ((part5_end - part5_start) / 2)))

    chest_line_length = 0
    for i in range(WIDTH - 1):
        if np.any(p_image[chest_line_height, i, :] == 0):
            chest_line_length = chest_line_length + 1

    waist_line_length = 0
    for i in range(WIDTH - 1):
        if np.any(p_image[waist_line_height, i, :] == 0):
            waist_line_length = waist_line_length + 1

    hip_line_length = 0
    for i in range(WIDTH - 1):
        if np.any(p_image[hip_line_height, i, :] == 0):
            hip_line_length = hip_line_length + 1

    '''chest_line_length *= height_ratio
    waist_line_length *= height_ratio
    upper_body_length *= height_ratio
    inside_leg_length *= height_ratio
    hip_line_length *= height_ratio'''
    return chest_line_length, waist_line_length, hip_line_length, upper_body_length, inside_leg_length


def get_cloth_parts(c_image):
    (H, W, D) = c_image.shape
    # get waist & hip height for upper clothes
    waist_height = round(H * 7 / 10)
    hip_height = round(H * 9 / 10)
    # get average of waist & hip for lower clothes
    w_h_height = round((waist_height + hip_height) / 2)

    waist_length = 0
    for i in range(W - 1):
        if np.any(c_image[waist_height, i, :] == 0):
            waist_length = waist_length + 1

    hip_length = 0
    for i in range(W - 1):
        if np.any(c_image[hip_height, i, :] == 0):
            hip_length = hip_length + 1

    w_h_length = 0
    for i in range(W - 1):
        if np.any(c_image[w_h_height, i, :] == 0):
            w_h_length = w_h_length + 1

    # get length of lower clothes
    lower_length = H

    return waist_length, hip_length, w_h_length, lower_length


def virtual_fitting(person_image, cloth_image):
    person_no_bg, person_mask, cloth_no_bg, cloth_mask = remove_bg(person_image, cloth_image)
    person = cv2.imread(person_image)
    '''cv2.imshow('person no bg', person_no_bg)
    cv2.imshow('cloth no bg', cloth_no_bg)
    cv2.imshow('person mask', person_mask)
    cv2.imshow('cloth mask', cloth_mask)'''

    chest_p, waist_p, hip_p, upper_p, insideLeg_p = get_person_parts(person_mask)
    waist_c, hip_c, w_h_c, lower_l_c = get_cloth_parts(cloth_mask)
    print(f"chest: {chest_p} upper: {upper_p}")

    cloth_no_bg = cv2.resize(cloth_no_bg, (chest_p, upper_p))
    blended_img = cv2.addWeighted(cloth_no_bg, 1, person[:cloth_no_bg.shape[0], :cloth_no_bg.shape[1], :], 0, 0)
    person[:cloth_no_bg.shape[0], :cloth_no_bg.shape[1], :] = blended_img
    cv2.imshow("out", person)
    cv2.imwrite("out.png", person)
    cv2.waitKey(0)


virtual_fitting(person_image=person_image, cloth_image=cloth_image)
