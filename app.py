from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import time
from datetime import datetime

app = Flask(__name__)

model = load_model("malaria_model.h5", compile=False)

CLASS_NAMES = ['Parasitized', 'Uninfected']

MODEL_NAME = "CustomCNN v2.0"

def prepare_image(image_file):

    img = Image.open(image_file).convert('RGB')

    width, height = img.size
    image_quality = f"{width} x {height} px"

    img = img.resize((64,64))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, image_quality


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['file']

    start_time = time.time()

    img_array, image_quality = prepare_image(file)

    prediction = model.predict(img_array)

    end_time = time.time()

    processing_time = round(end_time - start_time, 3)

    pred_class = np.argmax(prediction)

    result = CLASS_NAMES[pred_class]

    confidence = round(100*np.max(prediction),2)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        model_name=MODEL_NAME,
        processing_time=processing_time,
        image_quality=image_quality,
        timestamp=timestamp
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)