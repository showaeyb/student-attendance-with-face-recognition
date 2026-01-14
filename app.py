import os
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = "secret_attendance_key"

KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"

known_encodings = []
known_names = []

# Load known faces at startup
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, filename)
        image = cv2.imread(path)
        if image is not None:
            # Resize and convert for dlib
            image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb_image)
            if encs:
                known_encodings.append(encs[0])
                name = os.path.splitext(filename)[0].split('_')[0]
                known_names.append(name)

def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(ATTENDANCE_FILE) if os.path.exists(ATTENDANCE_FILE) else pd.DataFrame(columns=["Name", "Date", "Time"])
    
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        new_row = pd.DataFrame([[name, date_str, time_str]], columns=["Name", "Date", "Time"])
        pd.concat([df, new_row], ignore_index=True).to_csv(ATTENDANCE_FILE, index=False)
        return True
    return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/mark')
def mark():
    # 1. Open camera
    cam = cv2.VideoCapture(0)
    # Give camera a moment to adjust light
    for _ in range(5): cam.read() 
    success, frame = cam.read()
    cam.release() # 2. Immediately close camera

    if not success:
        flash("Error: Could not access camera.")
        return redirect(url_for('index'))

    # 3. Process the single frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encodings[0], tolerance=0.5)
        if True in matches:
            best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encodings[0]))
            name = known_names[best_match_index]
            mark_attendance(name)
            return render_template('success.html', name=name) # 4. Show success page
    
    flash("Face not recognized. Please try again.")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)