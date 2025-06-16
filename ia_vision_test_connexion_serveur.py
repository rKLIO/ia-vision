import cv2
import time
import json
import ssl
import socket
import threading
from datetime import datetime
from deepface import DeepFace
import tkinter as tk
from tkinter import StringVar

# === Connexion SSL au serveur ===
def connect_to_server(server_ip, port, password, cert_pem):
    context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
    context.load_verify_locations(cadata=cert_pem)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ssl_socket = context.wrap_socket(client_socket, server_hostname="example.com")

    ssl_socket.connect((server_ip, port))
    ssl_socket.send(password.encode('utf-8'))

    print("[INFO] Mot de passe envoyé au serveur...")

    response = ssl_socket.recv(1024).decode('utf-8')
    print(f"[INFO] Réponse du serveur : {response}")

    if "incorrect" in response.lower():
        print("[ERREUR] Mot de passe incorrect. Déconnexion...")
        ssl_socket.close()
        return None

    return ssl_socket

# === Paramètres serveur ===
SERVER_IP = "127.0.0.1"
SERVER_PORT = 12345
SERVER_PASSWORD = "mon_mot_de_passe"
CERTIFICATE_PEM = """-----BEGIN CERTIFICATE-----
...votre certificat ici...
-----END CERTIFICATE-----"""

ssl_conn = connect_to_server(SERVER_IP, SERVER_PORT, SERVER_PASSWORD, CERTIFICATE_PEM)

# === Variables Tkinter ===
root = tk.Tk()
root.title("Analyse Faciale en Temps Réel")

age_var = StringVar()
gender_var = StringVar()
dominant_emotion_var = StringVar()
emotions_var = StringVar()

tk.Label(root, text="Âge :").pack()
tk.Label(root, textvariable=age_var).pack()

tk.Label(root, text="Genre :").pack()
tk.Label(root, textvariable=gender_var).pack()

tk.Label(root, text="Émotion dominante :").pack()
tk.Label(root, textvariable=dominant_emotion_var).pack()

tk.Label(root, text="Toutes émotions :").pack()
tk.Label(root, textvariable=emotions_var, wraplength=400, justify="left").pack()

# === Fonction principale d’analyse faciale ===
def analyse_visage():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    analysis_interval = 5
    last_analysis_time = 0
    all_data = []

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

                # Mise à jour interface
                age_var.set(str(age))
                gender_var.set(str(gender))
                dominant_emotion_var.set(str(dominant_emotion))
                emotions_text = '\n'.join([f"{k}: {round(v, 2)}%" for k, v in emotions.items()])
                emotions_var.set(emotions_text)

                # Affichage visuel
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                print(f"[INFO] Données capturées à {data_to_save['timestamp']}")
                print(json.dumps(data_to_save, indent=4, ensure_ascii=False))

                # Envoi au serveur
                if ssl_conn:
                    ssl_conn.sendall((json.dumps(data_to_save) + "\n").encode('utf-8'))
                    print("[INFO] Données envoyées au serveur.")

            except Exception as e:
                print(f"[ERREUR] Analyse DeepFace échouée : {e}")

            last_analysis_time = current_time

        elif len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Webcam - Analyse DeepFace', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Sauvegarde des données
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analyse_faciale_complete_{timestamp_str}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(all_data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Toutes les données sauvegardées dans : {filename}")
            
            if ssl_conn:
                ssl_conn.close()
                print("[INFO] Connexion serveur fermée.")

            root.quit()
            root.destroy()
            break

    cap.release()
    cv2.destroyAllWindows()

# === Lancer l'analyse dans un thread séparé ===
threading.Thread(target=analyse_visage, daemon=True).start()

# === Boucle principale Tkinter ===
root.mainloop()
