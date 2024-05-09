import csv
import copy
import itertools
from flask import Flask, render_template, Response, request, jsonify
import cv2 as cv
import numpy as np
import mediapipe as mp
from Collect_from_microphone import start_audio_recording, stop_audio_recording, transcribe_audio, detect_emotion
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
import pyaudio

app = Flask(__name__)

# Load model
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

keypoint_classifier = KeyPointClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

# Initialize variables
use_brect = True

# Helper functions for facial emotion recognition
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)

    return image

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)

    if facial_text != "":
        info_text = 'Emotion :' + facial_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    return image

# Route for index page
@app.route('/')
def index():
    return render_template('frontpage.html')

# Route for facial emotion recognition
@app.route('/emotion_recognition')
def emotion_recognition():
    def generate_frames():
        cap = cv.VideoCapture(0)

        while True:
            ret, frame = cap.read()  # read the camera frame
            if not ret:
                break
            else:
                frame = cv.flip(frame, 1)
                debug_image = copy.deepcopy(frame)
                image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = face_mesh.process(image)
                image.flags.writeable = True

                if results.multi_face_landmarks is not None:
                    for face_landmarks in results.multi_face_landmarks:
                        brect = calc_bounding_rect(debug_image, face_landmarks)
                        landmark_list = calc_landmark_list(debug_image, face_landmarks)
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        facial_emotion_id = keypoint_classifier(pre_processed_landmark_list)
                        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                        debug_image = draw_info_text(debug_image, brect, keypoint_classifier_labels[facial_emotion_id])

                ret, buffer = cv.imencode('.jpg', debug_image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

tokenizer = AutoTokenizer.from_pretrained("wesleyacheng/twitter-emotion-classification-with-bert")
model = AutoModelForSequenceClassification.from_pretrained("wesleyacheng/twitter-emotion-classification-with-bert")
twitter_emotion_classifier = pipeline(task='text-classification', model=model, tokenizer=tokenizer)

# Route for audio emotion detection
@app.route('/audio_emotion_recognition')
def audio_emotion_recognition():
    return render_template('audio_emotion_recognition.html')

# Route for starting audio recording
@app.route('/start_audio_recording', methods=['POST'])
def start_audio_recording_route():
    # start_audio_recording()  # Comment out or remove this line to prevent automatic recording
    return jsonify({'message': 'Recording started'})

# Route for stopping audio recording and detecting emotion
@app.route('/stop_audio_recording', methods=['POST'])
def stop_audio_recording_route():
    audio_data, sample_rate = stop_audio_recording()  # Stop recording audio
    transcription = transcribe_audio(audio_data)  # Transcribe audio
    emotion = detect_emotion(transcription)  # Detect emotion
    return jsonify({'emotion': emotion})  # Return detected emotion as JSON response

# Route for text emotion detection page
@app.route('/text_emotion_detection')
def text_emotion_detection():
    return render_template('text_emotion_detection.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text data from the form
    text = request.form['tweet']

    # Perform emotion detection using model
    prediction = twitter_emotion_classifier(text)

    # Render the template with the prediction result
    return render_template('text_emotion_detection.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
