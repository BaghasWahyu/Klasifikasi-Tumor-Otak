import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template


app = Flask(__name__)

# Komen line dibawah ini untuk mematikan Debug Mode
# Debug Mode
config = {"DEBUG": True}  # run app in debug mode
app.config.from_mapping(config)
#

# Load model yang sudah di training
model = load_model("model.h5")

###################################################
# Adding image pre-processing function
def img_pred(img_path):
    print("Memulai Proses Deteksi")
    img = load_img(img_path)
    opencvImage = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_before = cv2.resize(opencvImage, (150, 150))
    img_after = img_before.reshape(1, 150, 150, 3)
    hasil = model.predict(img_after)
    hasil = np.argmax(hasil, axis=1)[0]

    if hasil == 0:
        return "terdapat Tumor Glioma"
    elif hasil == 1:
        return "tidak terdapat Tumor"
    elif hasil == 2:
        return "terdapat Tumor Meningioma"
    else:
        return "terdapat Tumor Pituitary"


###################################################


@app.route("/", methods=["GET"])
def welcome():
    # Halaman Utama
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        imagefile = request.files["imagefile"]
        if imagefile:
            image_path = "static/" + imagefile.filename
            imagefile.save(image_path)
            return render_template(
                "index.html",
                prediction=img_pred(image_path),
                imageloc=imagefile.filename,
            )
    return render_template("index.html", prediction=img_pred(image_path), imageloc=None)


if __name__ == "__main__":
    app.run(port=8080)
