import cv2
import time
from deepface import DeepFace
from fer import FER
import insightface
from insightface.app import FaceAnalysis

# Initialisation
face_detector = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_detector.prepare(ctx_id=0)

emotion_detector = FER()

last_analysis_time = 0
analysis_interval = 5  # secondes

# Accès webcam
cap = cv2.VideoCapture(0)

print("[INFO] Appuie sur Q pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector.get(frame_rgb)

    if faces:
        face = faces[0]  # On traite uniquement le visage principal
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box

        # Dessiner le cadre
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        current_time = time.time()
        if current_time - last_analysis_time > analysis_interval:
            face_crop = frame_rgb[y1:y2, x1:x2]

            # Analyse Emotion (FER)
            emotion = emotion_detector.top_emotion(face_crop)
            emotion_text = emotion[0] if emotion else "Unknown"

            # Analyse Age (InsightFace)
            age = face.age

            # Analyse Genre (DeepFace)
            try:
                deepface_result = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False)
                gender = deepface_result[0]['gender']
            except:
                gender = "Unknown"

            # Affichage terminal
            print(f"\n=== Analyse du Visage ===")
            print(f"Âge estimé   : {age}")
            print(f"Sexe         : {gender}")
            print(f"Émotion      : {emotion_text}")
            print("==========================")

            last_analysis_time = current_time

    # Afficher l'image
    cv2.imshow('Webcam - Analyse IA', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
