import os
from PIL import Image
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load pre-trained model
model = load_model('model\skin_tone_model.h5')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    if file:
        # Create the static folder if it doesn't exist
        if not os.path.exists('static'):
            os.makedirs('static')

        # Save the file in the static directory
        file_path = os.path.join('static', 'image.jpg')
        file.save(file_path)

        # Load and preprocess the image for prediction
        img = image.load_img(file_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        categories = ['dark', 'light', 'lighten', 'mid dark', 'mid light', 'mid-dark', 'mid-light']  # Define categories

        return f'Predicted Skin Tone Class: {categories[predicted_class]}'
    else:
        return 'Error in file upload'


if __name__ == '__main__':
    app.run(debug=True)
