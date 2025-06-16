import cv2
import time
import json
from datetime import datetime
from deepface import DeepFace

# Chargement du classifieur Haar cascade pour détection visage
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
analysis_interval = 5  # secondes
last_analysis_time = 0

all_data = []

print("[INFO] Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des visages (on prend le premier détecté)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    current_time = time.time()
    if current_time - last_analysis_time > analysis_interval and len(faces) > 0:
        # On traite uniquement le premier visage détecté
        (x, y, w, h) = faces[0]
        face_img = frame[y:y+h, x:x+w]

        try:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            result = DeepFace.analyze(face_rgb, actions=['age', 'gender', 'emotion'], enforce_detection=False)

            age = result.get("age", "Inconnu")
            gender = result.get("gender", "Inconnu")
            dominant_emotion = result.get("dominant_emotion", "Inconnu")
            emotions = result.get("emotion", {})

            data_to_save = {
                "timestamp": datetime.now().isoformat(),
                "age": age,
                "gender": gender,
                "dominant_emotion": dominant_emotion,
                "emotions": emotions
            }
            all_data.append(data_to_save)

            print(f"[INFO] Données capturées à {data_to_save['timestamp']}")
            print(json.dumps(data_to_save, indent=4, ensure_ascii=False))

            # Dessiner un rectangle vert autour du visage traité
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Afficher l’émotion dominante au-dessus du rectangle
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        except Exception as e:
            print(f"[ERREUR] Analyse DeepFace échouée : {e}")

        last_analysis_time = current_time

    elif len(faces) > 0:
        # Si on ne fait pas l'analyse cette fois, on affiche quand même le rectangle sur le visage traité
        (x, y, w, h) = faces[0]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Affichage en direct
    cv2.imshow('Webcam - Analyse DeepFace', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analyse_faciale_complete_{timestamp_str}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=4, ensure_ascii=False)

        print(f"[INFO] Toutes les données sauvegardées dans : {filename}")
        break

cap.release()
cv2.destroyAllWindows()
