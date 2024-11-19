
import dlib
from imutils import face_utils
from scipy.spatial import distance
import numpy as np
import threading
from flask import Flask, render_template, Response, redirect, url_for
import cv2

app = Flask(__name__)

# Load the pre-trained face detector and the shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./DrosinessApplication/shape_predictor_68_face_landmarks.dat')

# Function to calculate EAR
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define eye landmarks
(left_eye_start, left_eye_end) = (42, 48)
(right_eye_start, right_eye_end) = (36, 42)
EYE_AR_THRESH = 0.20  # EAR threshold for drowsiness
EYE_AR_CONSEC_FRAMES = 15  # Consecutive frames the eye must be below the threshold

# Initialize counters
counter = 0

# Global variable to control video capture
video_capture = None
video_thread = None
running = False

def gen_frames():
    global counter, running, video_capture

    cap = cv2.VideoCapture(0)
    while running:
        success, frame = cap.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            rects = detector(gray, 0)

            for rect in rects:
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)

                # Yawning detection
                mouth = shape[48:68]
                A = np.linalg.norm(mouth[2] - mouth[10])  # 51, 59
                B = np.linalg.norm(mouth[4] - mouth[8])   # 53, 57
                C = np.linalg.norm(mouth[0] - mouth[6])   # 49, 55
                mar = (A + B) / (2.0 * C)

                hull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [hull], -1, (0, 255, 0), 1)

                if mar > 0.75:
                    cv2.putText(frame, "Yawning", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Drowsiness detection
                left_eye = shape[left_eye_start:left_eye_end]
                right_eye = shape[right_eye_start:right_eye_end]

                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    counter += 1
                    if counter >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    counter = 0

                # Draw eye landmarks
                for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start')
def start():
    global running, video_thread
    if not running:
        running = True
        video_thread = threading.Thread(target=gen_frames)
        video_thread.start()
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global running
    running = False
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    global running
    if running:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "Video feed not running"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
