import os
import cv2
import time
import glob
import numpy as np
import tensorflow as tf
from flask import Flask, Response, jsonify
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app) # Allow React frontend to fetch data

# Models
print("Loading YOLOv8 Model...")
yolo_model = YOLO("yolov8n.pt")

print("Loading Custom Helmet CNN...")
CNN_MODEL_PATH = r"c:\Users\gupta\smart_road_safety_system\outputs\custom_helmet_cnn.keras"
cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)

# Global Metrics State
metrics = {
    "total_vehicles": 0,
    "helmet_violations": 0,
    "distracted_drivers": 0,
    "weekly_data": [
        {"name": "Mon", "violations": 12},
        {"name": "Tue", "violations": 19},
        {"name": "Wed", "violations": 15},
        {"name": "Thu", "violations": 8},
        {"name": "Fri", "violations": 22},
        {"name": "Sat", "violations": 30},
        {"name": "Sun", "violations": 25},
    ]
}

def check_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea > 0

def process_frame(frame):
    global metrics
    
    # 1. Run YOLO for Person and Cell Phone
    results = yolo_model(frame, verbose=False)
    persons = []
    phones = []
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            if cls == 0 and conf > 0.3: # Person
                persons.append([x1, y1, x2, y2, conf])
            elif cls == 67 and conf > 0.3: # Cell Phone
                phones.append([x1, y1, x2, y2, conf])
                
    # 2. Process each person for Helmet & Distraction
    for person in persons:
        metrics["total_vehicles"] += 1
        x1, y1, x2, y2, conf = person
        
        # --- Distraction Check (Overlap with phone) ---
        is_distracted = False
        for phone in phones:
            if check_intersection(person[:4], phone[:4]):
                is_distracted = True
                break
                
        if is_distracted:
            metrics["distracted_drivers"] += 1
            
        # --- Helmet Check (Custom CNN on top 40% of body) ---
        head_y = y1
        head_h = int((y2 - y1) * 0.4)
        head_h = min(frame.shape[0] - head_y, head_h)
        
        has_helmet = True
        if head_h > 0 and (x2 - x1) > 0:
            crop = frame[head_y:head_y+head_h, x1:x2]
            if crop.size > 0:
                crop_resized = cv2.resize(crop, (128, 128))
                crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
                crop_array = np.expand_dims(crop_rgb, axis=0)
                
                prediction = cnn_model.predict(crop_array, verbose=0)
                score = prediction[0][0]
                has_helmet = False if score >= 0.5 else True # Class 1 is Without_Helmet
                
        if not has_helmet:
            metrics["helmet_violations"] += 1
            
        # --- Drawing Logic ---
        color = (0, 255, 0) # Green (Safe)
        label = "Safe Rider"
        
        if is_distracted and not has_helmet:
            color = (0, 0, 255) # Red
            label = "No Helmet & Distracted!"
        elif is_distracted:
            color = (0, 165, 255) # Orange
            label = "Distracted (Phone)!"
        elif not has_helmet:
            color = (0, 0, 255) # Red
            label = "No Helmet!"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

def generate_frames():
    """Generator to simulate a live video feed by looping through test images"""
    images_pattern = r"c:\Users\gupta\smart_road_safety_system\test\test_images\*.png"
    image_files = glob.glob(images_pattern)
    
    if not image_files:
        # Fallback to creating a blank frame if no images found
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "No Test Images Found", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        ret, buffer = cv2.imencode('.jpg', blank)
        frame_bytes = buffer.tobytes()
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(1)

    # Loop infinitely through the test images
    while True:
        for img_path in image_files:
            frame = cv2.imread(img_path)
            if frame is not None:
                # Process the frame through ML models
                annotated_frame = process_frame(frame)
                
                # Encode to JPEG
                ret, buffer = cv2.imencode('.jpg', annotated_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
                # Sleep to simulate video framerate
                time.sleep(2.0) # Slower to let the user see the detections

@app.route('/api/video_feed')
def video_feed():
    # Multipart MJPEG stream
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    # Return live metrics
    # Update latest day with current violations to make graph dynamic
    metrics["weekly_data"][-1]["violations"] = metrics["helmet_violations"] + metrics["distracted_drivers"]
    return jsonify(metrics)

if __name__ == '__main__':
    print("Starting Flask Backend on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
