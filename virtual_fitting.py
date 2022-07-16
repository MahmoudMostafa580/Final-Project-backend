import cv2
from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile


def remove_bg(cloth_img):
    # remove background from clothes image
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    f = np.fromfile(cloth_img)
    result_c = remove(f)
    img_c = Image.open(io.BytesIO(result_c)).convert("RGBA")
    img_c.save("cloth_no_bg.png")
    c_no_bg = cv2.imread("cloth_no_bg.png", cv2.IMREAD_UNCHANGED)
    c_no_alpha = cv2.imread("cloth_no_bg.png")

    # get cloth mask
    c_mask, s_t, s_b, s_l, s_r = convert_crop_cloth(c_no_alpha)
    # crop image
    c_no_bg = c_no_bg[s_t:s_b, s_l:s_r]

    return c_no_bg, c_mask

# apply converting and crop of image
def convert_crop_cloth(img):
    # convert image to black and white
    black_pixels_mask = np.all(img == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(img != [0, 0, 0], axis=-1)
    img[black_pixels_mask] = [255, 255, 255]
    img[non_black_pixels_mask] = [0, 0, 0]

    t = crop_top(img)
    b = crop_bottom(img)
    l = crop_left(img)
    r = crop_right(img)
    img_mask = img[t:b, l:r]

    return img_mask, t, b, l, r


# crop image from top
def crop_top(converted_image):
    for h in range(0, converted_image.shape[0] - 1, 2):
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
    for w in range(converted_image.shape[1]):
        for h in range(converted_image.shape[0]):
            if np.any(converted_image[h, w, :] == 0):
                return w


# crop image from right
def crop_right(converted_image):
    for w in range(converted_image.shape[1] - 1, 0, -2):
        for h in range(converted_image.shape[0]):
            if np.any(converted_image[h, w, :] == 0):
                return w

# Extract specific points from model image
def get_model_parts():
    hip_height = 425
    waist_height = 370
    upper_cloth_start = 160
    upper_cloth_end = hip_height
    t_shirt_x = 180
    shirt_x = 80
    pants_x = 225
    t_shirt_width = 420 - 180
    shirt_width = 525 - 80
    pants_width = 375 - 220
    lower_cloth_start = waist_height
    shorts_length = 530 - lower_cloth_start
    lower_cloth_end = 700
    upper_b_length = upper_cloth_end - upper_cloth_start
    lower_b_length = lower_cloth_end - lower_cloth_start

    return upper_b_length, t_shirt_width, t_shirt_x, shirt_width, shirt_x, lower_b_length, pants_width, pants_x, hip_height, waist_height, shorts_length

# Function to merge model image with cloth image in one image
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

#change model image according to model index
def change_model(model_index):
    if model_index == 1:
        model = 'models\model1.png'
    elif model_index == 2:
        model = 'models\model2.png'
    elif model_index == 3:
        model = 'models\model3.png'
    elif model_index == 4:
        model = 'models\model4.png'
    elif model_index == 5:
        model = 'models\model5.png'
    elif model_index == 6:
        model = 'models\model6.png'
    return model

# Apply virtual fitting according to each category
def virtual_fitting(model, cloth_image, category):
    c_no_bg, c_mask = remove_bg(cloth_image)
    upper_length, t_shirt_width, t_shirt_x, shirt_width, shirt_x, lower_length, pants_width, pants_x, hip_height, waist_height, shorts_length = get_model_parts()
    if category == "T-shirts":
        c_no_bg = cv2.resize(c_no_bg, (t_shirt_width, upper_length))
        start_x = t_shirt_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = hip_height - c_no_bg.shape[0]
        end_y = hip_height
        result = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)

    elif category == "Shirts" or category == "Jackets":
        c_no_bg = cv2.resize(c_no_bg, (shirt_width, upper_length))
        start_x = shirt_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = hip_height - c_no_bg.shape[0]
        end_y = hip_height
        result = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)

    elif category == "Pants":
        c_no_bg = cv2.resize(c_no_bg, (pants_width, lower_length))
        start_x = pants_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = waist_height
        end_y = waist_height + c_no_bg.shape[0]
        result = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)
    elif category == 'Shorts':
        c_no_bg = cv2.resize(c_no_bg, (pants_width, shorts_length))
        start_x = pants_x
        end_x = start_x + c_no_bg.shape[1]
        start_y = waist_height
        end_y = waist_height + c_no_bg.shape[0]
        result = merge_images(model, c_no_bg, start_x, end_x, start_y, end_y)

    return result

# Apply virtual fitting for both upper and lower clothes
def full_outfit(model_index, first_cloth, first_category, second_cloth, second_category):
    model = change_model(model_index)
    model_img = cv2.imread(model)
    model_img = cv2.resize(model_img, (601, 768))
    result_1 = virtual_fitting(model_img, first_cloth, first_category)
    final_result = virtual_fitting(result_1, second_cloth, second_category)

    return final_result
