from flask import Flask, redirect, url_for, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import cv2
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Welcome to TrashEdu AI !!</p>"

@app.route("/predict_sampah", methods = ['POST'])
def fromform():
    if 'file' not in request.files:
        resp =  {
                 'status': 400,
                 'error': 'No selected file'
                }
        return jsonify(resp)

    if request.files['file'].filename == '':
        resp =  {
                 'status': 400,
                 'error': 'No selected file'
                }
        return jsonify(resp)
        
    output_class = ["batteries",
                "biological",
                "brown glass",
                "cardboard",
                "clothes", 
                "e waste",
                "glass",
                "green glass",
                "light blubs", 
                "metal", 
                "organic", 
                "paper", 
                "plastic",
                "shoes",
                "trash",
                "white glass"]

    img = cv2.cvtColor(cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) / 255
    resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)

    test_image = np.expand_dims(resized, axis=0)

    model = keras.models.load_model('classifyWastetesting2.h5')
    predicted_array = model.predict(test_image)
    predicted_value = output_class[np.argmax(predicted_array)]
    predicted_accuracy = round(np.max(predicted_array) * 100, 2)

    resp =  {
                 'status': 200,
                 'success': "Your waste material is " + str(predicted_value) + " with " +  str(predicted_accuracy) + " % accuracy"
                }
    return jsonify(resp)

if __name__ == "__main__":
    app.run(debug=True)