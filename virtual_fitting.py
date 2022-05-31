import cv2
from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile

model_image = "model.jpg"
cloth_image = "cloth_image.png"
category = "t-shirt"


def remove_bg(cloth_img):
    # remove background from person image
    '''ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(person_img)
    result_p = remove(f)
    img_p = Image.open(io.BytesIO(result_p)).convert("RGBA")
    img_p.save("person_no_bg.png")
    p_no_bg = cv2.imread("person_no_bg.png", cv2.IMREAD_UNCHANGED)
    p_no_alpha = cv2.imread("person_no_bg.png")

    # check if person image is rotated
    (height, width, channel) = p_no_alpha.shape
    if height < width:
        p_no_alpha = cv2.rotate(p_no_alpha, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # get person mask
    p_mask, p_start, p_end = convert_person(p_no_alpha)
    cv2.imwrite("person mask.png", p_mask)'''

    # remove background from clothes image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(cloth_img)
    result_c = remove(f)
    img_c = Image.open(io.BytesIO(result_c)).convert("RGBA")
    img_c.save("cloth_no_bg.png")
    c_no_bg = cv2.imread("cloth_no_bg.png", cv2.IMREAD_UNCHANGED)
    c_no_alpha = cv2.imread("cloth_no_bg.png")

    # check if image is rotated
    (height, width, channel) = c_no_alpha.shape
    if height < width:
        c_no_alpha = cv2.rotate(c_no_alpha, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # get cloth mask
    c_mask, s_t, s_b, s_l, s_r = convert_crop_cloth(c_no_alpha)
    c_no_bg = c_no_bg[s_t:s_b, s_l:s_r]

    return c_no_bg, c_mask


def convert_person(img):
    # convert image to black and white
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(img != [0, 0, 0], axis=-1)
    img[black_pixels_mask] = [255, 255, 255]
    img[non_black_pixels_mask] = [0, 0, 0]
    # img_mask = img[h1:h2, 0:W]

    return img


def convert_crop_cloth(img):
    # convert image to black and white
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(img != [0, 0, 0], axis=-1)
    img[black_pixels_mask] = [255, 255, 255]
    img[non_black_pixels_mask] = [0, 0, 0]
    s_t = crop_top(img)
    s_b = crop_bottom(img)
    s_l = crop_left(img)
    s_r = crop_right(img)
    img_mask = img[s_t:s_b, s_l:s_r]

    return img_mask, s_t, s_b, s_l, s_r


# crop image from top
def crop_top(converted_image):
    for h in range(converted_image.shape[0] - 1):
        for w in range(int(converted_image.shape[1] / 8), int(converted_image.shape[1] * (7 / 8)), 1):
            if np.any(converted_image[h, w, :] == 0):
                return h


# crop image from bottom
def crop_bottom(converted_image):
    for h in range(converted_image.shape[0] - 1, 0, -1):
        for w in range(int(converted_image.shape[1] / 8), int(converted_image.shape[1] * (7 / 8)), 1):
            if np.any(converted_image[h, w, :] == 0):
                return h


# crop image from left
def crop_left(converted_image):
    for w in range(converted_image.shape[1] - 1):
        for h in range(converted_image.shape[0]):
            if np.any(converted_image[h, w, :] == 0):
                return w


# crop image from top
def crop_right(converted_image):
    for w in range(converted_image.shape[1] - 1, 0, -1):
        for h in range(converted_image.shape[0]):
            if np.any(converted_image[h, w, :] == 0):
                return w


def get_model_parts():
    hip_height = 425
    waist_height = 370
    upper_cloth_start = 160
    upper_cloth_end = hip_height
    t_shirt_x = 180
    shirt_x = 80
    pants_x = 230
    t_shirt_width = 420 - 180
    shirt_width = 525 - 80
    pants_width = 370 - 230
    lower_cloth_start = waist_height
    lower_cloth_end = 685
    upper_b_length = upper_cloth_end - upper_cloth_start
    lower_b_length = lower_cloth_end - lower_cloth_start

    return upper_b_length, t_shirt_width, t_shirt_x, shirt_width, shirt_x, lower_b_length, pants_width, pants_x, hip_height, waist_height


'''def get_cloth_parts(c_image):
    (H, W, _) = c_image.shape
    # get length of clothes
    cloth_length = H
    # get hip height & length for upper clothes
    hip_height = round(H * 9.5 / 10)
    hip_length = 0
    for i in range(W - 1):
        if np.any(c_image[hip_height, i, :] == 0):
            hip_length = hip_length + 1
    space_length = W - hip_length

    # get waist height & length for lower clothes
    waist_height = round(H * 0.5 / 10)
    waist_length = 0
    for i in range(W - 1):
        if np.any(c_image[waist_height, i, :] == 0):
            waist_length = waist_length + 1

    return cloth_length, hip_length, space_length, waist_length'''

def merge_images(model, c_no_bg, start_x, end_x, start_y, end_y):
    # so let's split the overlay image into its individual channels
    fg_b, fg_g, fg_r, fg_a = cv2.split(c_no_bg)
    # Make the range 0...1 instead of 0...255
    fg_a = fg_a / 255.0
    # Multiply the RGB channels with the alpha channel
    label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])
    # Work on a part of the background only
    part_of_bg = model[start_y:end_y, start_x:end_x, :]
    # Same procedure as before: split the individual channels
    bg_b, bg_g, bg_r = cv2.split(part_of_bg)
    # Merge them back with opposite of the alpha channel
    part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])
    # Add the label and the part of the background
    cv2.add(label_rgb, part_of_bg, part_of_bg)
    # Replace a part of the background
    model[start_y:end_y, start_x:end_x, :] = part_of_bg

    return model


def virtual_fitting(model_image, cloth_image, category):
    model = cv2.imread(model_image)
    c_no_bg, c_mask = remove_bg(cloth_image)
    upper_length, t_shirt_width, t_shirt_x, shirt_width, shirt_x, lower_length, pants_width, pants_x, hip_height, waist_height = get_model_parts()
    if category == "t-shirt":
        c_no_bg = cv2.resize(c_no_bg, (t_shirt_width, upper_length))
        start_x = t_shirt_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = hip_height - c_no_bg.shape[0]
        end_y = hip_height
        model = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)

    elif category == "shirt" or category == "jacket":
        c_no_bg = cv2.resize(c_no_bg, (shirt_width, upper_length))
        start_x = shirt_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = hip_height - c_no_bg.shape[0]
        end_y = hip_height
        model = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)

    elif category == "pants" or category == "shorts":
        c_no_bg = cv2.resize(c_no_bg, (pants_width, lower_length))
        start_x = pants_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = waist_height
        end_y = waist_height + c_no_bg.shape[0]
        model = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)

    cv2.imshow("out", model)
    cv2.imwrite("out.png", model)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


virtual_fitting(model_image=model_image, cloth_image=cloth_image, category=category)
