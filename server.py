from flask import Flask, Response, stream_with_context
from flask_cors import CORS
from deepface import DeepFace
import cv2
import time
import json
import threading

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

frame_lock = threading.Lock()
current_frame = None
latest_data = {}
analysis_interval = 2
last_analysis_time = 0
streaming = False

def process_video():
    global current_frame, latest_data, last_analysis_time, streaming

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERREUR] Impossible d’ouvrir la caméra.")
        return

    streaming = True
    print("[INFO] Capture vidéo démarrée")

    while streaming:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        with frame_lock:
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            current_frame = frame.copy()

        current_time = time.time()
        if current_time - last_analysis_time > analysis_interval and len(faces) > 0:

            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]

            try:
                face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                result = DeepFace.analyze(face_rgb, actions=['age', 'gender', 'emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]

                age = int(result.get("age", -1))
                gender = result.get("gender", {})
                gender_name = max(gender, key=gender.get) if isinstance(gender, dict) else gender
                gender_percentage = int(gender[gender_name]) if isinstance(gender, dict) else 0

                emotions = result.get("emotion", {})
                top_emotions = sorted(emotions.items(), key=lambda item: item[1], reverse=True)[:3]
                emotion_list = [{"emotionName": k, "percentage": int(round(v))} for k, v in top_emotions]

                latest_data = {
                    "age": age,
                    "gender": {
                        "genderName": gender_name,
                        "percentage": gender_percentage
                    },
                    "emotions": emotion_list
                }

            except Exception as e:
                print(f"[ERREUR] Analyse échouée : {e}")

            last_analysis_time = current_time

        time.sleep(0.05)

    cap.release()
    print("[INFO] Capture vidéo arrêtée")


@app.route('/video_feed')
def video_feed():
    global streaming

    if not streaming:
        t = threading.Thread(target=process_video)
        t.daemon = True
        t.start()
        time.sleep(1)  # Laisse le temps de récupérer une première frame

    def generate():
        while streaming:
            with frame_lock:
                if current_frame is None:
                    continue
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.05)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/events')
def sse():
    def generate_events():
        global latest_data
        while True:
            if latest_data:
                yield f"data: {json.dumps(latest_data)}\n\n"
                latest_data = {}
            time.sleep(1)

    return Response(stream_with_context(generate_events()), content_type='text/event-stream')


@app.route('/stop_stream')
def stop_stream():
    global streaming
    streaming = False
    return "Stream stopped."


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
