import pickle
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import base64
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import List, Dict, Any
import json
import logging
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("sign_language_api")

# Environment variables with defaults
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
MODEL_PATH = os.getenv("MODEL_PATH", "./lstm_model_with_mhmd.h5")
PREPROC_PATH = os.getenv("PREPROC_PATH", "./lstm_preprocessing_with_mhmd.pickle")
PORT = int(os.getenv("PORT", "8000"))
HOST = os.getenv("HOST", "0.0.0.0")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
COOLDOWN_PERIOD = int(os.getenv("COOLDOWN_PERIOD", "15"))

logger.info(f"Starting application with: MODEL_PATH={MODEL_PATH}, PORT={PORT}")

try:
    # Load preprocessing objects
    with open(PREPROC_PATH, 'rb') as f:
        preproc = pickle.load(f)
        scaler = preproc['scaler']
        label_encoder = preproc['label_encoder']
        timesteps = preproc['timesteps']
        n_features = preproc['n_features']
    
    logger.info(f"Loaded preprocessing data: timesteps={timesteps}, features={n_features}")
    
    # Load trained model
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    
    SEQUENCE_LENGTH = timesteps
    FEATURE_LENGTH = n_features
except Exception as e:
    logger.error(f"Failed to load model or preprocessing data: {e}")
    raise RuntimeError(f"Failed to initialize application: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.65,
    max_num_hands=2
)

labels_dict = {
    0: 'سلام', 1: 'صباح الخير', 2: 'شكراً', 3: 'أنا', 4: 'أنتَ', 5: 'أنتِ', 6: 'هو', 7: 'هي',
    8: 'أنتم', 9: 'هم', 10: 'اسم', 11: 'كيف حالك؟', 12: 'الحمد لله', 13: 'سعيد', 14: 'حزين', 15: 'غاضب',
    16: 'جيد', 17: 'سيء', 18: 'تعبان', 19: 'مريض', 20: 'أرى', 21: 'أقول', 22: 'أتكلم', 23: 'أمشي', 24: 'ذهبت',
    25: 'جاء', 26: 'بيت', 27: 'أكل', 28: 'نام', 29: 'الجامعة', 30: 'اليوم', 31: 'غداً', 32: 'الأحد',
    33: 'الثلاثاء', 34: 'الخميس', 35: 'الجمعة', 36: 'أسبوع', 37: 'شهر', 38: 'سنة', 39: 'متى', 40: 'أعرف',
    41: 'أفكر', 42: 'نسيت', 43: 'أحب', 44: 'أريد', 45: 'يساعد', 46: 'غير مسموح', 47: 'أوافق', 48: 'معاً', 49: 'مختلف'
}

def decode_base64_image(base64_string):
    """Decode base64 string to image"""
    try:
        # Remove header if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64 string
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Failed to decode image")
        
        return img
    except Exception as e:
        logger.error(f"Base64 decode error: {e}")
        return None

def convert_yuv_to_bgr(yuv_data: Dict[str, Any]) -> np.ndarray:
    """
    Convert YUV420 frame data from Flutter camera to BGR format for OpenCV
    """
    try:
        width = yuv_data['width']
        height = yuv_data['height']
        planes = yuv_data['planes']

        if len(planes) < 3:
            raise ValueError(f"Expected 3 planes, got {len(planes)}")

        # Decode base64 data
        y_plane = np.frombuffer(base64.b64decode(planes[0]['bytes']), dtype=np.uint8)
        u_plane = np.frombuffer(base64.b64decode(planes[1]['bytes']), dtype=np.uint8)
        v_plane = np.frombuffer(base64.b64decode(planes[2]['bytes']), dtype=np.uint8)

        # Get bytes per row
        y_bytes_per_row = planes[0]['bytesPerRow']
        u_bytes_per_row = planes[1]['bytesPerRow']
        v_bytes_per_row = planes[2]['bytesPerRow']

        # Ensure the Y plane has enough data
        if y_plane.size < height * y_bytes_per_row:
            raise ValueError(f"Y plane data size mismatch: {y_plane.size} < {height * y_bytes_per_row}")

        # Reshape Y plane
        y = y_plane.reshape((height, y_bytes_per_row))[:height, :width]

        # Calculate UV dimensions (typically half of Y dimensions)
        uv_height = height // 2
        uv_width = width // 2

        # Ensure the U and V planes have enough data
        if u_plane.size < uv_height * u_bytes_per_row:
            raise ValueError(f"U plane data size mismatch: {u_plane.size} < {uv_height * u_bytes_per_row}")
        if v_plane.size < uv_height * v_bytes_per_row:
            raise ValueError(f"V plane data size mismatch: {v_plane.size} < {uv_height * v_bytes_per_row}")

        # Reshape U and V planes
        u = u_plane.reshape((uv_height, u_bytes_per_row))[:uv_height, :uv_width]
        v = v_plane.reshape((uv_height, v_bytes_per_row))[:uv_height, :uv_width]

        # Create YUV image - NV12 format
        yuv = np.zeros((height + uv_height, width), dtype=np.uint8)
        yuv[:height, :width] = y

        # Stack UV planes side by side at the bottom of the Y plane
        uv_combined = np.vstack((u, v))
        if uv_width * 2 <= width:
            yuv[height:, :uv_width * 2] = uv_combined.reshape((uv_height, uv_width * 2))
        else:
            # Handle cases where UV width might be too large
            uv_resized = cv2.resize(uv_combined, (width, uv_height))
            yuv[height:, :width] = uv_resized

        # Convert YUV to BGR - try different color conversion formats if one fails
        try:
            bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        except cv2.error:
            try:
                # Try alternative formats if I420 fails
                bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
            except cv2.error:
                # Last resort: Create a grayscale image from Y plane only
                bgr = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

        return bgr

    except Exception as e:
        logger.error(f"Error converting YUV to BGR: {e}")
        # Return a small blank image as fallback
        return np.zeros((height if 'height' in yuv_data else 180,
                         width if 'width' in yuv_data else 320, 3),
                        dtype=np.uint8)


def extract_hand_features(frame):
    """Extract hand landmark features from a frame"""
    data_aux = np.zeros(FEATURE_LENGTH, dtype=np.float32)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    H, W, _ = frame.shape
    results = hands.process(img_rgb)

    hands_present = False
    x1 = y1 = x2 = y2 = 0

    if results.multi_hand_landmarks:
        hands_present = True
        all_x, all_y = [], []

        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            if hand_idx >= 2:
                break
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_, y_ = [], []
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)
                all_x.append(landmark.x)
                all_y.append(landmark.y)

            if x_ and y_:
                min_x, min_y = min(x_), min(y_)
                base_idx = hand_idx * 42
                for i, landmark in enumerate(hand_landmarks.landmark):
                    data_aux[base_idx + i * 2] = landmark.x - min_x
                    data_aux[base_idx + i * 2 + 1] = landmark.y - min_y

        if all_x and all_y:
            x1 = int(min(all_x) * W) - 10
            y1 = int(min(all_y) * H) - 10
            x2 = int(max(all_x) * W) + 10
            y2 = int(max(all_y) * H) + 10

    data_aux_scaled = scaler.transform([data_aux])[0]
    return data_aux_scaled, frame, hands_present, (x1, y1, x2, y2)


# FastAPI setup
app = FastAPI(
    title="Sign Language Recognition API",
    description="API for recognizing sign language using computer vision",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "predictions_made": 0
        }

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_stats["total_connections"] += 1
        self.connection_stats["active_connections"] = len(self.active_connections)
        logger.info(f"New connection. Active: {self.connection_stats['active_connections']}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.connection_stats["active_connections"] = len(self.active_connections)
        logger.info(f"Connection closed. Active: {self.connection_stats['active_connections']}")

    def log_prediction(self):
        self.connection_stats["predictions_made"] += 1


manager = ConnectionManager()


@app.websocket("/ws/sign-language")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    start_time = time.time()

    sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
    confidence_threshold = CONFIDENCE_THRESHOLD
    current_prediction = None
    prediction_scores = None
    prediction_made = False
    previous_buffer_size = 0
    cooldown_frames = 0
    frames_processed = 0
    
    try:
        while True:
            data = await websocket.receive_text()
            frames_processed += 1
            
            try:
                # Check if the data is JSON or plain text
                if data.startswith('{'):
                    try:
                        message = json.loads(data)
                        message_type = message.get('type', '')
                    except json.JSONDecodeError:
                        message_type = ''
                else:
                    message_type = ''

                # Handle reset command
                if (message_type == 'command' and message.get('command') == 'reset') or data == "reset" or data == "\"reset\"":
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

                # Process frame data
                frame = None
                
                # Case 1: YUV data from mobile app
                if message_type == 'frame':
                    yuv_data = message.get('data', {})
                    frame = convert_yuv_to_bgr(yuv_data)
                
                # Case 2: Base64 encoded JPEG image from web client
                elif data.startswith('data:image/jpeg;base64,'):
                    frame = decode_base64_image(data)
                
                # Error handling if frame conversion failed
                if frame is None or frame.size == 0:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Frame conversion failed"
                    })
                    continue

                # Extract features from the frame
                features, processed_frame, hands_present, _ = extract_hand_features(frame)

                # Handle cooldown period after a prediction
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

                # Add features to buffer if hands are detected
                if hands_present and not prediction_made:
                    sequence_buffer.append(features)

                # Update client about buffer status
                current_buffer_size = len(sequence_buffer)
                if current_buffer_size != previous_buffer_size:
                    await websocket.send_json({
                        "status": "buffer_update",
                        "buffer_status": current_buffer_size,
                        "total_needed": SEQUENCE_LENGTH
                    })
                    previous_buffer_size = current_buffer_size

                # Make prediction when buffer is full
                if len(sequence_buffer) == SEQUENCE_LENGTH and not prediction_made:
                    sequence_data = np.array(list(sequence_buffer)).reshape(1, SEQUENCE_LENGTH, n_features)
                    prediction_scores = model.predict(sequence_data, verbose=0)[0]
                    predicted_idx = np.argmax(prediction_scores)
                    confidence = prediction_scores[predicted_idx]

                    if confidence > confidence_threshold:
                        current_prediction = labels_dict[predicted_idx]
                        prediction_made = True
                        manager.log_prediction()
                        
                        await websocket.send_json({
                            "status": "prediction",
                            "prediction": current_prediction,
                            "confidence": float(confidence)
                        })
                        
                        logger.info(f"Prediction made: {current_prediction} with confidence {confidence:.2f}")

            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                await websocket.send_json({
                    "status": "error",
                    "message": f"Exception: {str(e)}"
                })

    except WebSocketDisconnect:
        elapsed_time = time.time() - start_time
        logger.info(f"WebSocket disconnected after {elapsed_time:.2f} seconds. Processed {frames_processed} frames.")
        manager.disconnect(websocket)


@app.get("/")
async def root():
    """Serve the client application"""
    return FileResponse('static/index.html')


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok", 
        "version": "1.0.0",
        "connections": manager.connection_stats
    }


@app.get("/stats")
async def stats():
    """Return basic usage statistics"""
    return {
        "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0,
        "connections": manager.connection_stats,
        "model_info": {
            "timesteps": SEQUENCE_LENGTH,
            "features": FEATURE_LENGTH,
            "labels": len(labels_dict)
        }
    }


@app.on_event("startup")
async def startup_event():
    """Initialize application state on startup"""
    app.state.start_time = time.time()
    logger.info("Server started successfully")


if __name__ == "__main__":
    print(f"Starting server on {HOST}:{PORT}")
    uvicorn.run("app:app", host=HOST, port=PORT, reload=DEBUG)
