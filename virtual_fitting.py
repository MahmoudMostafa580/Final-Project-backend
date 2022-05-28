import cv2
from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile

person_image = "mms_front.jpeg"
cloth_image = "cloth_image.png"


def remove_bg(person_img, cloth_img):
    #remove background from person image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(person_img)
    result_p = remove(f)
    img_p = Image.open(io.BytesIO(result_p)).convert("RGBA")
    img_p.save("person_no_bg.png")
    p_no_bg = cv2.imread("person_no_bg.png", cv2.IMREAD_UNCHANGED)
    p_no_alpha = cv2.imread("person_no_bg.png")

    #check if person image is rotated
    (height, width, channel) = p_no_alpha.shape
    if height < width:
        p_no_alpha = cv2.rotate(p_no_alpha, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #get person mask
    p_mask, p_start, p_end = convert_crop(p_no_alpha)
    cv2.imwrite("person mask.png", p_mask)

    #remove background from clothes image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(cloth_img)
    result_c = remove(f)
    img_c = Image.open(io.BytesIO(result_c)).convert("RGBA")
    img_c.save("cloth_no_bg.png")
    c_no_bg = cv2.imread("cloth_no_bg.png", cv2.IMREAD_UNCHANGED)
    c_no_alpha = cv2.imread("cloth_no_bg.png")

    #check if image is rotated
    (height, width, channel) = c_no_alpha.shape
    if height < width:
        c_no_alpha = cv2.rotate(c_no_alpha, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #get cloth mask
    c_mask, c_start, c_end = convert_crop(c_no_alpha)
    cv2.imwrite("cloth mask.png", c_mask)

    return p_no_bg, p_mask, p_start, p_end, c_no_bg, c_mask, c_start, c_end


def convert_crop(img):
    W = img.shape[1]
    # convert image to black and white
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(img != [0, 0, 0], axis=-1)
    img[black_pixels_mask] = [255, 255, 255]
    img[non_black_pixels_mask] = [0, 0, 0]
    h1 = crop(img)
    h2 = crop_2(img)
    #img_mask = img[h1:h2, 0:W]

    return img, h1, h2


# crop image from top
def crop(converted_image):
    for h in range(converted_image.shape[0] - 1):
        for w in range(int(converted_image.shape[1] / 8), int(converted_image.shape[1] * (7 / 8)), 1):
            if np.any(converted_image[h, w, :] == 0):
                return h


# crop image from bottom
def crop_2(converted_image):
    for h in range(converted_image.shape[0] - 1, 0, -1):
        for w in range(int(converted_image.shape[1] / 8), int(converted_image.shape[1] * (7 / 8)), 1):
            if np.any(converted_image[h, w, :] == 0):
                return h


def get_person_parts(p_image, p_start, p_end):
    (_, WIDTH, CHANNEL) = p_image.shape
    HEIGHT = p_end - p_start
    # height ration between original height of person & height of image
    # height_ratio = person_height / HEIGHT
    # calculate start & end of the 8 parts of human body
    # part1_start = 0
    part1_end = round(p_start + HEIGHT * (1 / 8))

    part2_start = part1_end + 1
    part2_end = round(p_start + HEIGHT * (2 / 8))

    part3_start = part2_end + 1
    part3_end = round(p_start + HEIGHT * (3 / 8))

    part4_start = part3_end + 1
    part4_end = round(p_start + HEIGHT * (4 / 8))

    part5_start = part4_end + 1
    part5_end = round(p_start + HEIGHT * (5 / 8))

    # part6_start = part5_end + 1
    # part6_end = round(p_start + HEIGHT * (6 / 8))

    # part7_start = part6_end + 1
    # part7_end = round(p_start + HEIGHT * (7 / 8))

    # part8_start = part7_end + 1
    part8_end = p_end

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
            if hip_line_length == 1:
                hip_start = i

    '''chest_line_length *= height_ratio
    waist_line_length *= height_ratio
    upper_body_length *= height_ratio
    inside_leg_length *= height_ratio
    hip_line_length *= height_ratio'''
    return chest_line_length, waist_line_length, hip_line_length, upper_body_length, inside_leg_length, hip_line_height, hip_start


def get_cloth_parts(c_image, c_start, c_end):
    (_, W, D) = c_image.shape
    H = c_end - c_start
    # get waist & hip height for upper clothes
    waist_height = round(c_start + H * 7 / 10)
    hip_height = round(c_start + H * 9 / 10)
    # get average of waist & hip for lower clothes
    w_h_height = round((waist_height + hip_height) / 2)

    waist_length = 0
    for i in range(W - 1):
        if np.any(c_image[waist_height, i, :] == 0):
            waist_length = waist_length + 1

    hip_length = 0
    space_length = 0
    for i in range(W - 1):
        if np.any(c_image[hip_height, i, :] == 0):
            hip_length = hip_length + 1
        if np.any(c_image[hip_height, i, :] != 0):
            space_length = space_length + 1

    w_h_length = 0
    for i in range(W - 1):
        if np.any(c_image[w_h_height, i, :] == 0):
            w_h_length = w_h_length + 1

    # get length of lower clothes
    lower_length = H

    return waist_length, hip_length, space_length, w_h_length, lower_length


def virtual_fitting(person_image, cloth_image):
    p_no_bg, p_mask, p_start, p_end, c_no_bg, c_mask, c_start, c_end = remove_bg(person_image, cloth_image)

    chest_p, waist_p, hip_p, upper_p, insideLeg_p, hip_height, hip_start = get_person_parts(p_mask, p_start, p_end)
    print(f"hip_p {hip_p} upper_p {upper_p}")

    c_no_bg = cv2.resize(c_no_bg, (hip_p, upper_p-10))

    waist_c, hip_c, space_length, w_h_c, lower_l_c = get_cloth_parts(c_mask, c_start, c_end)

    new_width = hip_p + space_length

    #c_no_bg = cv2.resize(c_no_bg, (new_width, upper_p))

    start_x = hip_start
    end_x = hip_start + c_no_bg.shape[1]
    start_y = hip_height - c_no_bg.shape[0]
    end_y = hip_height
    print(f"start x: {start_x} end x: {end_x} start y: {start_y} end y: {end_y}")

    blended_img = cv2.addWeighted(c_no_bg, 1, p_no_bg[start_y:end_y, start_x:end_x, :], 0, 0)
    p_no_bg[start_y:end_y, start_x:end_x, :] = blended_img

    cv2.imshow("out", p_no_bg)
    cv2.imwrite("out.png", p_no_bg)
    cv2.waitKey(0)


virtual_fitting(person_image=person_image, cloth_image=cloth_image)
