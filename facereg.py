import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not access the camera.")
    exit()

# Load known faces and encodings
try:
    Amitab_image = face_recognition.load_image_file("/Users/gitika/Desktop/Project/Facial attendenace/Photos/Amitab.png")
    Amitab_encoding = face_recognition.face_encodings(Amitab_image)[0]

    Gitika_image = face_recognition.load_image_file("/Users/gitika/Desktop/Project/Facial attendenace/Photos/Gitika.png")
    Gitika_encoding = face_recognition.face_encodings(Gitika_image)[0]

    Tesla_image = face_recognition.load_image_file("/Users/gitika/Desktop/Project/Facial attendenace/Photos/Tesla.png")
    Tesla_encoding = face_recognition.face_encodings(Tesla_image)[0]
except Exception as e:
    print(f"Error loading images or encodings: {e}")
    exit()

known_face_encoding = [
    Amitab_encoding,
    Gitika_encoding,
    Tesla_encoding
]

known_faces_names = [
    "Amitab",
    "Gitika",
    "Tesla"
]

students = known_faces_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

try:
    f = open(current_date + '.csv', 'w+', newline='')
    lnwriter = csv.writer(f)
except Exception as e:
    print(f"Error opening CSV file: {e}")
    video_capture.release()
    cv2.destroyAllWindows()
    exit()

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)
            
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]

            face_names.append(name)

            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2

                cv2.putText(frame, name + ' Present', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

                if name in students:
                    students.remove(name)
                    print(students)
                    now = datetime.now()  # Update `now` to the current time
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
