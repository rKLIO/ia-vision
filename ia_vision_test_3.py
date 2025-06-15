import cv2
from deepface import DeepFace
import time

# Webcam
cap = cv2.VideoCapture(0)
last_analysis_time = 0
analysis_interval = 5  # secondes
last_result = None

print("Analyse d’un visage (le plus visible) toutes les 5 secondes...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    if current_time - last_analysis_time > analysis_interval:
        try:
            # Analyse tous les visages détectés
            results = DeepFace.analyze(
                frame,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=False
            )

            # Si plusieurs visages, ne garder que le plus grand (zone = w * h)
            results.sort(key=lambda r: r['region']['w'] * r['region']['h'], reverse=True)
            result = results[0]
            last_result = result
            last_analysis_time = current_time

            # Affichage dans le terminal
            print("\n=== Analyse visage principal ===")
            print(f"Âge estimé       : {result['age']}")
            print(f"Sexe estimé      : {result['gender']}")
            print(f"Émotion dominante: {result['dominant_emotion']}")
            print("================================")

        except Exception as e:
            print("⚠️ Erreur : aucun visage détecté ou problème :", e)
            last_result = None

    # Affichage graphique
    if last_result:
        region = last_result['region']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        # Dessiner un rectangle autour du visage traité
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Texte affiché sur l’image
        infos = f"{last_result['gender']} | {last_result['age']} ans | {last_result['dominant_emotion']}"
        cv2.putText(frame, infos, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)

    cv2.imshow("Analyse visage principal", frame)

    # Touche Q pour quitter
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
