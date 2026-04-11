import cv2
import os
import numpy as np
import requests
from deepface import DeepFace
import time

# ===== TELEGRAM =====
BOT_TOKEN = "8458081017:AAFArP6GSiUEc3tNlIm1rEUJGMc-Mk7HBRA" # put your new token
CHAT_ID = "6666968646"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, data={"chat_id": CHAT_ID, "text": msg})

# ===== LOAD DATASET =====
known_faces = []
known_names = []

dataset_path = "dataset"

print("Loading dataset...")

for file in os.listdir(dataset_path):
    img_path = os.path.join(dataset_path, file)
    name = os.path.splitext(file)[0]

    try:
        embedding = DeepFace.represent(
            img_path=img_path,
            model_name='Facenet',
            enforce_detection=False
        )[0]["embedding"]

        known_faces.append(embedding)
        known_names.append(name)

        print(f"Loaded: {name}")

    except:
        print(f"Error loading {name}")

print("Dataset loaded ✅")

# ===== CAMERA =====
cap = cv2.VideoCapture(0)

print("System Started...")

last_alert_time = 0   # for spam control

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.represent(
            frame,
            model_name='Facenet',
            enforce_detection=False
        )

        if result:
            face_embedding = result[0]["embedding"]

            distances = []
            for known in known_faces:
                dist = np.linalg.norm(np.array(known) - np.array(face_embedding))
                distances.append(dist)

            min_dist = min(distances)
            index = distances.index(min_dist)

            # ===== THRESHOLD =====
            if min_dist < 8:
                name = known_names[index]
                print(f"Known: {name}")

                cv2.putText(frame, name, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 2)

            else:
                print("Unknown person 🚨")

                cv2.putText(frame, "Unknown", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)

                # ===== SEND ALERT (NO SPAM) =====
                current_time = time.time()
                if current_time - last_alert_time > 10:
                    send_telegram("🚨 Unknown person detected at home!")
                    last_alert_time = current_time

    except Exception as e:
        print("Error:", e)

    cv2.imshow("Camera", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()