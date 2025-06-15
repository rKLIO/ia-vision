import cv2
from deepface import DeepFace
import mediapipe as mp

# Initialisation de MediaPipe
mp_face_detection = mp.solutions.face_detection

# Démarrage de la webcam
cap = cv2.VideoCapture(0)

resultats_analyse = None
image_resultat = None

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6) as face_detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir en RGB pour MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)

        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            x, y = max(0, x), max(0, y)

            # Recadrer le visage
            face_crop = frame[y:y+height, x:x+width]

            try:
                result = DeepFace.analyze(
                    face_crop,
                    actions=['age', 'gender', 'emotion'],
                    enforce_detection=False
                )[0]

                # Résultats dans le terminal
                print("=== Résultats de l'analyse du visage ===")
                print(f"Âge estimé       : {result['age']}")
                print(f"Sexe estimé      : {result['gender']}")
                print(f"Émotion dominante: {result['dominant_emotion']}")
                print("========================================")

                # Affichage dans l'image
                image_resultat = frame.copy()
                cv2.rectangle(image_resultat, (x, y), (x + width, y + height), (0, 255, 0), 2)
                texte = f"{result['gender']} | {result['age']} ans | {result['dominant_emotion']}"
                cv2.putText(image_resultat, texte, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                break

            except Exception as e:
                print("Erreur durant l'analyse :", e)
                continue

cap.release()

# Afficher l’image analysée si disponible
if image_resultat is not None:
    while True:
        cv2.imshow('Résultat Analyse Faciale', image_resultat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print("Aucun visage détecté ou analyse échouée.")

cv2.destroyAllWindows()
