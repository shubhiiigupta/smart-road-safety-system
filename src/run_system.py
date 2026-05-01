import os
import cv2
import argparse
import numpy as np
import tensorflow as tf

MODEL_PATH = r"c:\Users\gupta\smart_road_safety_system\outputs\custom_helmet_cnn.keras"
OUTPUT_DIR = r"c:\Users\gupta\smart_road_safety_system\outputs"
IMG_HEIGHT = 128
IMG_WIDTH = 128

def load_system():
    print("Loading Custom CNN Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("Initializing HOG Pedestrian Detector...")
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    return model, hog

def process_frame(frame, model, hog):
    # Detect people in the frame
    # winStride, padding, scale are parameters that can be tuned
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
    
    for (x, y, w, h) in boxes:
        # Since riders might be sitting, HOG might struggle, but we will process whatever it finds.
        # The CNN was trained on heads/helmets. A heuristic is to crop the upper part of the detected body.
        head_y = y
        head_h = int(h * 0.4) # Top 40% of the body
        
        # Ensure bounds
        head_y = max(0, head_y)
        head_h = min(frame.shape[0] - head_y, head_h)
        
        if head_h <= 0 or w <= 0:
            continue
            
        crop = frame[head_y:head_y+head_h, x:x+w]
        
        if crop.size == 0:
            continue
            
        # Preprocess for Keras
        crop_resized = cv2.resize(crop, (IMG_WIDTH, IMG_HEIGHT))
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        crop_array = np.expand_dims(crop_rgb, axis=0)
        
        # Predict
        prediction = model.predict(crop_array, verbose=0)
        score = prediction[0][0]
        
        # Class 0: With_Helmet, Class 1: Without_Helmet
        pred_cls = 'Without Helmet' if score >= 0.5 else 'With Helmet'
        confidence = score if score >= 0.5 else 1 - score
        
        # Draw bounding boxes
        color = (0, 255, 0) if pred_cls == 'With Helmet' else (0, 0, 255) # BGR
        
        # Draw body box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        
        # Draw head box (the part classified)
        cv2.rectangle(frame, (x, head_y), (x + w, head_y + head_h), color, 2)
        
        # Text label
        label = f"{pred_cls}: {confidence*100:.1f}%"
        cv2.putText(frame, label, (x, head_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return frame

def process_image(img_path, model, hog):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: Could not read image {img_path}")
        return
        
    output_frame = process_frame(frame, model, hog)
    base_name = os.path.basename(img_path)
    out_path = os.path.join(OUTPUT_DIR, f"annotated_{base_name}")
    cv2.imwrite(out_path, output_frame)
    print(f"Saved annotated image to {out_path}")

def process_video(video_path, model, hog):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not read video {video_path}")
        return
        
    base_name = os.path.basename(video_path)
    out_path = os.path.join(OUTPUT_DIR, f"annotated_{base_name}")
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    
    print(f"Processing video {video_path}...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        output_frame = process_frame(frame, model, hog)
        out.write(output_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
    cap.release()
    out.release()
    print(f"Saved annotated video to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Helmet Detection System (No YOLO)")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video")
    args = parser.parse_args()
    
    model, hog = load_system()
    
    ext = os.path.splitext(args.source)[1].lower()
    if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        process_image(args.source, model, hog)
    elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
        process_video(args.source, model, hog)
    else:
        print("Unsupported file format.")

if __name__ == "__main__":
    main()
