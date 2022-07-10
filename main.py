import os

from flask import Flask, request, jsonify, send_file
import werkzeug
from werkzeug.serving import WSGIRequestHandler
import cv2
import size_recommendation as sr

#import virtual_fitting as vf

app = Flask(__name__)


@app.route('/size_recommend', methods=["POST"])
def size_recommend():
    if request.method == "POST":
        #get images from client
        imageFile = request.files['image']
        fileName = werkzeug.utils.secure_filename(imageFile.filename)
        category = fileName.split('_')[0]
        imageFile.save(fileName)

        imageFile2 = request.files['imageSide']
        fileName2 = werkzeug.utils.secure_filename(imageFile2.filename)
        height = float(fileName2.split('_')[0])
        imageFile2.save(fileName2)

        # Call function size recommendation
        shirts, t_shirts, trousers, shorts, jackets = sr.size_recommendation(fileName, fileName2, height, category)

        # remove images after extracting sizes
        if os.path.isfile(f"F:\python\FinalProject\{fileName}"):
            os.remove(fileName)
            print("Front image deleted successfully")
        else:
            print("Error: file not found!")

        if os.path.isfile(f"F:\python\FinalProject\{fileName2}"):
            os.remove(fileName2)
            print("Side image deleted successfully")
        else:
            print("Error: file not found!")

        return jsonify({
            "size": [shirts, t_shirts, trousers, shorts, jackets]
        })


'''@app.route('/virtual', method=["POST"])
def virtual():
    if request.method == "POST":
        first_cloth_image = request.files['image1']
        clothName_1 = werkzeug.utils.secure_filename(first_cloth_image.filename)
        first_cloth_image.save(clothName_1)

        category_1 = request.args.get('category1')

        second_cloth_image = request.files['image2']
        clothName_2 = werkzeug.utils.secure_filename(second_cloth_image.filename)
        second_cloth_image.save(clothName_2)

        category_2 = request.args.get('category2')

        result_img = vf.full_outfit(vf.model, clothName_1, category_1, clothName_2, category_2)
        cv2.imwrite("result.png", result_img)

        # remove images after processing
        if os.path.isfile(f"F:\python\FinalProject\{clothName_1}"):
            os.remove(clothName_1)
            print("First image deleted successfully")
        else:
            print("Error: file not found!")

        if os.path.isfile(f"F:\python\FinalProject\{clothName_2}"):
            os.remove(clothName_2)
            print("Second image deleted successfully")
        else:
            print("Error: file not found!")

        return send_file("result.png", mimetype='image/png')


@app.route("/")
def index():
    return "<h1> Welcome to my Flask server !</h1>"'''


if __name__ == "__main__":
    WSGIRequestHandler.protocol_version = "HTTP/1.1"
    app.run(host='0.0.0.0', port=5000)
