from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io
from PIL import Image
import time

app = Flask(__name__)

model = load_model("drowsiness_detection_model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

EYE_CLOSED_THRESHOLD = 3  
FRAME_COUNT = 0  
last_eye_status = "open" 
start_time = None  

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face_resized = cv2.resize(face, (64, 64))

    face_normalized = face_resized / 255.0
    
    face_final = np.expand_dims(face_normalized, axis=-1)
    face_final = np.expand_dims(face_final, axis=0) 
    return face_final

def process_frame(current_eye_status):

    global FRAME_COUNT, last_eye_status, start_time

    if current_eye_status == "closed":
        if last_eye_status == "closed":
            FRAME_COUNT += 1
        else:
            FRAME_COUNT = 1
            start_time = time.time() 
    else:
        FRAME_COUNT = 0
        start_time = None 

    last_eye_status = current_eye_status

    if FRAME_COUNT >= EYE_CLOSED_THRESHOLD and (time.time() - start_time >= 1.5):
        return "sleepy"
    else:
        return "not sleepy"

@app.route('/process_image', methods=['POST'])
def process_image():
    global FRAME_COUNT, last_eye_status, start_time

    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    processed_image = preprocess_image(image)

    if processed_image is None:
        return jsonify({"error": "No face detected in the image"}), 400

    prediction = model.predict(processed_image)

    if prediction[0][0] > 0.5:
        current_eye_status = "closed"
    else:
        current_eye_status = "open"

    result = process_frame(current_eye_status)

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
