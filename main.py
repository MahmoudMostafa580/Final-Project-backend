import os

from rembg.bg import remove
import numpy as np
import io
from PIL import Image, ImageFile
from flask import Flask, request, jsonify
import werkzeug

def out_image(input_image):
    # Uncomment the following line if working with truncated image formats (ex. JPEG / JPG)
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # apply rembg algorithm to remove background & convert image to numpy array
    f = np.fromfile(input_image)
    result = remove(f)
    img = Image.open(io.BytesIO(result)).convert("RGB")

    width, height = img.size
    if height < width:
        img = img.transpose(Image.ROTATE_90)

    converted = np.array(img)
    (H, W, D) = converted.shape

    # convert image to black and white
    black_pixels_mask = np.all(converted == [0, 0, 0], axis=-1)
    non_black_pixels_mask = np.any(converted != [0, 0, 0], axis=-1)
    converted[black_pixels_mask] = [255, 255, 255]
    converted[non_black_pixels_mask] = [0, 0, 0]
    h1 = crop(converted)
    h2 = crop_2(converted)
    cropped_img = converted[h1:h2, 0:W]

    return cropped_img

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

def get_parts(image, person_height):
    (HEIGHT, WIDTH, CHANNEL) = image.shape
    # height ration between original height of person & height of image
    height_ratio = person_height / HEIGHT

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
    upper_body_length = (part5_start + ((part5_end - part5_start) / 2)) - \
                        (part2_start + ((part2_end - part2_start) / 3))
    inside_leg_length = part8_end - (part5_start + ((part5_end - part5_start) / 2))

    chest_line_length = 0
    for i in range(WIDTH - 1):
        if np.any(image[chest_line_height, i, :] == 0):
            chest_line_length = chest_line_length + 1

    waist_line_length = 0
    for i in range(WIDTH - 1):
        if np.any(image[waist_line_height, i, :] == 0):
            waist_line_length = waist_line_length + 1

    hip_line_length = 0
    for i in range(WIDTH - 1):
        if np.any(image[hip_line_height, i, :] == 0):
            hip_line_length = hip_line_length + 1

    chest_line_length *= height_ratio
    waist_line_length *= height_ratio
    upper_body_length *= height_ratio
    inside_leg_length *= height_ratio
    hip_line_length *= height_ratio
    return chest_line_length, waist_line_length, hip_line_length, upper_body_length, inside_leg_length

def circum(front_size, side_size):
    pi = 3.14159
    m = (pi / 2) * (1.5 * (front_size + side_size) - (front_size * side_size) ** 0.5)
    return m

def shirt_recommendation(chest_length, waist_length):
    chest_length = chest_length * (1 / 2.54)
    waist_length = waist_length * (1 / 2.54)

    if chest_length < 31:
        recommendation_1 = '< XS'
    elif 31 <= chest_length <= 33:
        recommendation_1 = 'XS'
    elif 34 <= chest_length <= 37:
        recommendation_1 = 'S'
    elif 37 <= chest_length <= 40:
        recommendation_1 = 'M'
    elif 40 <= chest_length <= 44:
        recommendation_1 = 'L'
    elif 44 <= chest_length <= 48:
        recommendation_1 = 'XL'
    elif 48 <= chest_length <= 52:
        recommendation_1 = '2XL'
    elif 53 <= chest_length <= 58:
        recommendation_1 = '3XL'
    elif 58 <= chest_length <= 63:
        recommendation_1 = '4XL'
    else:
        recommendation_1 = 'Can not recommend size'

    if waist_length < 27:
        recommendation_2 = '< XS'
    elif 27 <= waist_length <= 29:
        recommendation_2 = 'XS'
    elif 30 <= waist_length <= 32:
        recommendation_2 = 'S'
    elif 32 <= waist_length <= 35:
        recommendation_2 = 'M'
    elif 35 <= waist_length <= 39:
        recommendation_2 = 'L'
    elif 39 <= waist_length <= 43:
        recommendation_2 = 'XL'
    elif 43 <= waist_length <= 47:
        recommendation_2 = '2XL'
    elif 48 <= waist_length <= 53:
        recommendation_2 = '3XL'
    elif 53 <= waist_length <= 57:
        recommendation_2 = '4XL'
    else:
        recommendation_2 = 'Can not recommend size'

    if recommendation_1 != recommendation_2:
        recommended_size = recommendation_1
    else:
        recommended_size = recommendation_2

    return recommended_size

def t_shirt_recommendation(chest_length, waist_length):
    chest_length = chest_length * (1 / 2.54)
    waist_length = waist_length * (1 / 2.54)

    if chest_length < 31:
        recommendation_1 = '< XS'
    elif 31 <= chest_length <= 33:
        recommendation_1 = 'XS'
    elif 33 < chest_length <= 37:
        recommendation_1 = 'S'
    elif 37 <= chest_length <= 40:
        recommendation_1 = 'M'
    elif 40 <= chest_length <= 44:
        recommendation_1 = 'L'
    elif 44 <= chest_length <= 48:
        recommendation_1 = 'XL'
    elif 48 <= chest_length <= 52:
        recommendation_1 = '2XL'
    elif 52 < chest_length <= 58:
        recommendation_1 = '3XL'
    elif 58 < chest_length <= 63:
        recommendation_1 = '4XL'
    else:
        recommendation_1 = 'Can not recommend size'

    if waist_length < 27:
        recommendation_2 = '< XS'
    elif 27 <= waist_length <= 29:
        recommendation_2 = 'XS'
    elif 30 <= waist_length <= 32:
        recommendation_2 = 'S'
    elif 32 <= waist_length <= 35:
        recommendation_2 = 'M'
    elif 35 <= waist_length <= 39:
        recommendation_2 = 'L'
    elif 39 <= waist_length <= 43:
        recommendation_2 = 'XL'
    elif 43 <= waist_length <= 47:
        recommendation_2 = '2XL'
    elif 48 <= waist_length <= 53:
        recommendation_2 = '3XL'
    elif 53 <= waist_length <= 57:
        recommendation_2 = '4XL'
    else:
        recommendation_2 = 'Can not recommend size'

    if recommendation_1 != recommendation_2:
        recommended_size = recommendation_1
        return recommended_size

def trouser_recommendation(waist_length, hip_length):
    waist_length = waist_length * (1 / 2.54)
    hip_length = hip_length * (1 / 2.54)

    if waist_length < 28:
        recommendation_1 = '< 28'
    elif 28 <= waist_length < 30:
        recommendation_1 = '28'
    elif 30 <= waist_length < 32:
        recommendation_1 = '30'
    elif 32 <= waist_length < 34:
        recommendation_1 = '32'
    elif 34 <= waist_length < 36:
        recommendation_1 = '34'
    elif 36 <= waist_length < 38:
        recommendation_1 = '36'
    elif 38 <= waist_length < 40:
        recommendation_1 = '38'
    elif 40 <= waist_length < 42:
        recommendation_1 = '40'
    elif 42 <= waist_length < 43:
        recommendation_1 = '42'
    elif 44 <= waist_length < 45:
        recommendation_1 = '44'
    elif 45 <= waist_length < 46:
        recommendation_1 = '46'
    elif 46 <= waist_length < 48:
        recommendation_1 = '48'
    elif 48 <= waist_length < 50:
        recommendation_1 = '50'
    elif 50 <= waist_length < 52:
        recommendation_1 = '52'
    elif 52 <= waist_length < 54:
        recommendation_1 = '54'
    else:
        recommendation_1 = 'Can not recommend size'

    if hip_length < 34:
        recommendation_2 = '< 28'
    elif 34 <= hip_length < 36:
        recommendation_2 = '28'
    elif 36 <= hip_length < 38:
        recommendation_2 = '30'
    elif 38 <= hip_length < 40:
        recommendation_2 = '32'
    elif 40 <= hip_length < 42:
        recommendation_2 = '34'
    elif 42 <= hip_length < 44:
        recommendation_2 = '36'
    elif 44 <= hip_length < 47:
        recommendation_2 = '38'
    elif 47 <= hip_length < 48:
        recommendation_2 = '40'
    elif 48 <= hip_length < 49:
        recommendation_2 = '42'
    elif 49 <= hip_length < 50:
        recommendation_2 = '44'
    elif 50 <= hip_length < 52:
        recommendation_2 = '46'
    elif 52 <= hip_length < 54:
        recommendation_2 = '48'
    elif 54 <= hip_length < 56:
        recommendation_2 = '50'
    elif 56 <= hip_length < 58:
        recommendation_2 = '52'
    elif 58 <= hip_length < 60:
        recommendation_2 = '54'
    else:
        recommendation_2 = 'Can not recommend size'

    if recommendation_1 != recommendation_2:
        recommended_size = recommendation_2
        return recommended_size

    recommended_size = recommendation_2

    return recommended_size

def shorts_recommendation(waist_length, hip_length):
    waist_length = waist_length * (1 / 2.54)
    hip_length = hip_length * (1 / 2.54)

    if waist_length < 26:
        recommendation_1 = '< XS'
    elif 26 <= waist_length < 29:
        recommendation_1 = 'XS'
    elif 29 <= waist_length < 32:
        recommendation_1 = 'S'
    elif 32 <= waist_length < 35:
        recommendation_1 = 'M'
    elif 35 <= waist_length < 38:
        recommendation_1 = 'L'
    elif 38 <= waist_length < 43:
        recommendation_1 = 'XL'
    elif 43 <= waist_length < 47:
        recommendation_1 = '2XL'
    elif 47 <= waist_length < 52:
        recommendation_1 = '3XL'
    elif 52 <= waist_length < 57:
        recommendation_1 = '4XL'
    else:
        recommendation_1 = 'Can not recommend size'

    if hip_length < 32:
        recommendation_2 = '< XS'
    elif 32 <= hip_length < 35:
        recommendation_2 = 'XS'
    elif 35 <= hip_length < 37:
        recommendation_2 = 'S'
    elif 37 <= hip_length < 41:
        recommendation_2 = 'M'
    elif 41 <= hip_length < 44:
        recommendation_2 = 'L'
    elif 44 <= hip_length < 47:
        recommendation_2 = 'XL'
    elif 47 <= hip_length < 50:
        recommendation_2 = '2XL'
    elif 50 <= hip_length < 53:
        recommendation_2 = '3XL'
    elif 53 <= hip_length < 58:
        recommendation_2 = '4XL'
    else:
        recommendation_2 = 'Can not recommend size'

    if recommendation_1 != recommendation_2:
        recommended_size = recommendation_2
        return recommended_size

    recommended_size = recommendation_2

    return recommended_size

def jackets_recommendation(chest_length):
    chest_length = chest_length * (1 / 2.54)

    if chest_length < 35:
        recommendation_1 = '< XS'
    elif 35 <= chest_length <= 37:
        recommendation_1 = 'XS'
    elif 37 < chest_length <= 39:
        recommendation_1 = 'S'
    elif 39 < chest_length <= 41:
        recommendation_1 = 'M'
    elif 41 < chest_length <= 43:
        recommendation_1 = 'L'
    elif 43 < chest_length <= 46:
        recommendation_1 = 'XL'
    elif 46 < chest_length <= 48:
        recommendation_1 = '2XL'
    elif 48 < chest_length <= 50.3:
        recommendation_1 = '3XL'
    elif 50.3 < chest_length <= 53:
        recommendation_1 = '4XL'
    else:
        recommendation_1 = 'Can not recommend size'

    return recommendation_1

def sizes(front_image, side_image, person_height, category):
    front_out = out_image(front_image)
    side_out = out_image(side_image)

    front_chest, front_waist, front_hip, upper, leg = get_parts(
        front_out, person_height)
    side_chest, side_waist, side_hip, _, _ = get_parts(side_out, person_height)

    chest_length = circum(front_chest, side_chest)
    waist_length = circum(front_waist, side_waist)
    hip_length = circum(front_hip, side_hip)

    shirt_guide = shirt_recommendation(chest_length, waist_length)
    t_shirt_guide = t_shirt_recommendation(chest_length, waist_length)
    trousers_guide = trouser_recommendation(waist_length, hip_length)
    shorts_guide = shorts_recommendation(waist_length, hip_length)
    jackets_guide = jackets_recommendation(chest_length)

    return shirt_guide, t_shirt_guide, trousers_guide, shorts_guide, jackets_guide


app = Flask(__name__)

@app.route('/upload', methods=["POST"])
def upload():
    if request.method == "POST":
        imageFile = request.files['image']
        fileName = werkzeug.utils.secure_filename(imageFile.filename)
        category = fileName.split('_')[0]
        # print('category:', category)
        imageFile.save(fileName)

        imageFile2 = request.files['imageSide']
        fileName2 = werkzeug.utils.secure_filename(imageFile2.filename)
        height = float(fileName2.split('_')[0])
        # print('height:', height)
        imageFile2.save(fileName2)

        # Call function sizes
        shirts, t_shirts, trousers, shorts, jackets = sizes(fileName, fileName2, height, category)

        print(shirts, t_shirts, trousers, shorts, jackets)

        #remove images after extracting sizes
        if os.path.isfile(f"F:\python\pythonProject\{fileName}"):
            os.remove(fileName)
            print("Front image deleted successfully")
        else:
            print("Error: file not found!")

        if os.path.isfile(f"F:\python\pythonProject\{fileName2}"):
            os.remove(fileName2)
            print("Side image deleted successfully")
        else:
            print("Error: file not found!")

        return jsonify({
            "size": [shirts, t_shirts, trousers, shorts, jackets]
        })


if __name__ == "__main__":
    app.run(debug=False, port=4000)
