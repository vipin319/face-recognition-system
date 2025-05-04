from database.database import init_db, mark_attendance
from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import time
from datetime import datetime

init_db();

with open('data/names.pkl', 'rb') as f:
    Labels = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    Faces = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Faces, Labels)

video = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

already_marked_users = set()
last_output = "Unknown"

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_image = frame[y:y+h, x:x+w, :]
        resized_image = cv2.resize(crop_image, (50, 50)).flatten().reshape(1, -1)

        distances, indices = knn.kneighbors(resized_image)
        if distances[0][0] > 5800:
            output = "Unknown"
        else:
            output = knn.predict(resized_image)[0]

        last_output = output  # Save last detected person

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        current_time = datetime.fromtimestamp(ts).strftime("%H:%M:%S")

        color = (0, 255, 0) if output != "Unknown" else (0, 0, 255)
        cv2.putText(frame, str(output), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, color, 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

    key = cv2.waitKey(1)

    if key == ord('m'):
        if last_output != "Unknown":
            if last_output not in already_marked_users:
                marked = mark_attendance(last_output, date, current_time)
                if marked:
                    already_marked_users.add(last_output)
                    print(f"Marked: {last_output} at {current_time}")
                else:
                    print(f"Already marked: {last_output}")
                    already_marked_users.add(last_output)
            else:
                print(f"Already marked (cached): {last_output}")
        else:
            print("Unknown face — not marking.")

    elif key == ord('q'):
        break

    cv2.imshow("Video capture", frame)

video.release()
cv2.destroyAllWindows()
