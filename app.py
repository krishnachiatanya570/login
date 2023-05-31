import numpy as np
import cv2
from keras.models import load_model
from flask import Flask, render_template, request

app = Flask(__name__)
model = load_model('my_model.h5')
known_identities = ['CHAITANYA','SATISH','RESMA']  # List of known identities

# Function to perform face recognition
def recognize_face(image):
    # Preprocess the captured image
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(image, (100, 100))
    img_array = resized_image/255.0
    preprocess_image=np.reshape(img_array, (1, 100, 100, 3))

    # Pass the preprocessed image through the trained CNN
    predictions = model.predict(preprocess_image)

    # Identify the person with the highest probability as the match
    predicted_identity = known_identities[np.argmax(predictions)]

    return predicted_identity

@app.route('/')
def login_page():
    return render_template('login.html')

@app.route('/authenticate', methods=['POST'])
def authenticate():
    # Get the uploaded image file from the form data
    image_file = request.files['image']

    # Read the image file using OpenCV
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform face recognition on the captured image
    identity = recognize_face(image)

    # Check if the recognized identity exists in the known_identities list
    if identity in known_identities:
        # Grant access
        return (identity),("Login successful!")
        # return("Login successful!")
    else:
        # Access denied
        return "Access denied!"

if __name__ == '__main__':
    app.run()
