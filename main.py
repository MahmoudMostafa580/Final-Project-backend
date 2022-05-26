import os


from flask import Flask, request, jsonify
import werkzeug
from size_recommendation import size_recommendation

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
        shirts, t_shirts, trousers, shorts, jackets = size_recommendation(fileName, fileName2, height, category)

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
