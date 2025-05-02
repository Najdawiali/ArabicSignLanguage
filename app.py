import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import base64
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List

# Load preprocessing objects
with open('./lstm_preprocessing_with_mhmd.pickle', 'rb') as f:
    preproc = pickle.load(f)
    scaler = preproc['scaler']
    label_encoder = preproc['label_encoder']
    timesteps = preproc['timesteps']
    n_features = preproc['n_features']

# Load trained model
model = tf.keras.models.load_model('lstm_model_with_mhmd.h5')

SEQUENCE_LENGTH = timesteps
FEATURE_LENGTH = n_features

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65,
    max_num_hands=2
)

labels_dict = {0:'سلام',1:'صباح الخير',2:'شكراً',3:'انا',4:'انت',5:'سعيد',6:'هو',7:'هي',8:'متى',9:'هم'}



def extract_hand_features(frame):
    data_aux = np.zeros(FEATURE_LENGTH, dtype=np.float32)

    # Convert to RGB for MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W,_  = frame.shape
    results = hands.process(img_rgb)

    # Variables to track hand presence and bounding box
    hands_present = False
    x1, y1, x2, y2 = 0, 0, 0, 0

    if results.multi_hand_landmarks:
        hands_present = True
        all_x = []
        all_y = []

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect all x, y for bounding box
            for landmark in hand_landmarks.landmark:
                all_x.append(landmark.x)
                all_y.append(landmark.y)

            # Normalize like the first version
            landmarks = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)
            landmarks -= landmarks[0]  # Center around wrist
            max_dist = np.max(np.linalg.norm(landmarks, axis=1))
            if max_dist > 0:
                landmarks /= max_dist  # Normalize scale

            # Flatten and insert into data_aux
            base_idx = hand_idx * 42
            flat = landmarks.flatten()
            data_aux[base_idx:base_idx + len(flat)] = flat

        # Calculate bounding box if hands are present
        if all_x and all_y:
            x1 = int(min(all_x) * W) - 10
            y1 = int(min(all_y) * H) - 10
            x2 = int(max(all_x) * W) + 10
            y2 = int(max(all_y) * H) + 10

    return data_aux, frame, hands_present, (x1, y1, x2, y2)


# FastAPI setup
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)


manager = ConnectionManager()


@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    confidence_threshold = 0.55
    current_prediction = None
    prediction_scores = None
    prediction_made = False
    previous_buffer_size = 0
    cooldown_frames = 0
    COOLDOWN_PERIOD = 15

    try:
        while True:
            data = await websocket.receive_text()

            if data == "reset":
                sequence_buffer.clear()
                current_prediction = None
                prediction_scores = None
                prediction_made = False
                cooldown_frames = 0
                await websocket.send_json({
                    "status": "reset",
                    "message": "Prediction reset"
                })
                continue

            try:
                image_data = data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                nparr = np.frombuffer(image_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Frame decode failed"
                    })
                    continue

                features, processed_frame, hands_present, _ = extract_hand_features(frame)

                if prediction_made:
                    cooldown_frames += 1
                    if cooldown_frames >= COOLDOWN_PERIOD:
                        sequence_buffer.clear()
                        current_prediction = None
                        prediction_scores = None
                        prediction_made = False
                        cooldown_frames = 0
                        await websocket.send_json({
                            "status": "auto_reset",
                            "message": "Cooldown finished"
                        })
                        previous_buffer_size = 0
                    continue

                if hands_present and not prediction_made:
                    sequence_buffer.append(features)

                current_buffer_size = len(sequence_buffer)
                if current_buffer_size != previous_buffer_size:
                    await websocket.send_json({
                        "status": "buffer_update",
                        "buffer_status": current_buffer_size,
                        "total_needed": SEQUENCE_LENGTH
                    })
                    previous_buffer_size = current_buffer_size

                if len(sequence_buffer) == SEQUENCE_LENGTH and not prediction_made:
                    sequence_data = np.array(list(sequence_buffer)).reshape(1, SEQUENCE_LENGTH, n_features)
                    prediction_scores = model.predict(sequence_data, verbose=0)[0]
                    predicted_idx = np.argmax(prediction_scores)
                    confidence = prediction_scores[predicted_idx]

                    if confidence > confidence_threshold:
                        current_prediction = labels_dict[predicted_idx]
                        prediction_made = True
                        await websocket.send_json({
                            "status": "prediction",
                            "prediction": current_prediction,
                            "confidence": float(confidence)
                        })

            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Exception: {str(e)}"
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {"message": "Sign Language Recognition API is running."}
