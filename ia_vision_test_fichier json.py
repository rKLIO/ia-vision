import cv2
import time
import json
from deepface import DeepFace

# Chargement du détecteur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
analysis_interval = 5  # secondes
last_analysis_time = 0

print("[INFO] Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    current_time = time.time()
    if current_time - last_analysis_time > analysis_interval and len(faces) > 0:
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]

        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(face_rgb, actions=['age', 'gender', 'emotion'], enforce_detection=False)

            if isinstance(result, list):
                result = result[0]

            # Âge
            age = int(result.get("age", -1))

            # Genre
            gender_name = result.get("gender", "Inconnu")
            if isinstance(gender_name, dict):
                gender_name = max(gender_name, key=gender_name.get)
            gender_value = 0 if str(gender_name).lower() in ["man", "male", "homme"] else 1

            # Émotions (top 3)
            emotions_dict = result.get("emotion", {})
            emotions_list = [
                {
                    "emotionName": emotion,
                    "percentage": int(round(value))
                }
                for emotion, value in sorted(emotions_dict.items(), key=lambda item: item[1], reverse=True)
            ][:3]

            # Construction du JSON
            data_to_save = {
                "age": age,
                "gender": {
                    "genderName": gender_name,
                    "genderValue": gender_value
                },
                "emotions": emotions_list
            }

            print(json.dumps(data_to_save, indent=4, ensure_ascii=False))

            # Affichage de l'émotion dominante sur l'image
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            if emotions_list:
                cv2.putText(frame, emotions_list[0]['emotionName'], (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        except Exception as e:
            print(f"[ERREUR] Analyse échouée : {e}")

        last_analysis_time = current_time

    elif len(faces) > 0:
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Webcam - Analyse DeepFace', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
